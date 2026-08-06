"""Time-LLM style adapter: use a pretrained LLM as a time-series encoder.

Approach inspired by Time-LLM (Jin et al., ICLR 2024):
  1. Project each patch of the time-series into LLM embedding space via
     a learned linear projection (reprogramming layer).
  2. Feed the resulting "token sequence" through the frozen LLM.
  3. Use LLM's hidden states as time-series representation.
  4. Add LoRA adapters on a small subset of LLM layers for cheap fine-tune.
  5. Classification head fuses LLM output with handcrafted feat.

We default to a SMALLER open LLM (Qwen2.5-0.5B) for CPU-feasibility of
development + smoke tests. On GPU pod with 32 GB VRAM per card, scale
up to Qwen2.5-7B or Llama-3-8B in 4-bit + LoRA.

Interface matches `src.teacher.MultiStreamTransformer`:
    forward(lob: (B, 3, 20, T), feat: (B, F)) -> (logits (B, 3), reg (B,))

Memory reference on 32 GB VRAM (single card):
  Qwen2.5-0.5B fp16:    1 GB
  Qwen2.5-1.5B fp16:    3 GB
  Qwen2.5-7B   4bit+LoRA: ~5 GB + ~50M trainable LoRA weights
  Llama-3-8B   4bit+LoRA: ~5 GB + ~50M trainable
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TimeLLMConfig:
    """Config for LLM backbone + LoRA adapters + reprogramming layer."""
    # LLM choice. Default small for dev; bake-off overrides to Qwen2.5-7B.
    llm_repo: str = "Qwen/Qwen2.5-0.5B"
    use_4bit: bool = False              # set True on GPU pod for 7B+
    apply_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    # Input patching
    patch_len: int = 10                 # ticks per patch
    n_patches: int = 5                  # total tokens fed to LLM (5 × 10 = 50 = lob_time_dim)

    # Head
    d_proj: int = 192
    head_hidden: int = 256
    dropout: float = 0.15

    # Training
    batch_size: int = 32                # small for big LLMs
    epochs: int = 20
    lr: float = 5e-5
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1
    reg_loss_weight: float = 0.3
    label_smoothing: float = 0.05
    early_stop_patience: int = 4


class TimeLLMClassifier(nn.Module):
    """LLM-as-encoder classifier for LOB.

    Pipeline:
        lob (B, 3, 20, T) → reduce to (B, 80, T) channels × time
                          → patch into (B, n_patches, patch_len × 80)
                          → project to (B, n_patches, d_llm) via reprogrammer
                          → LLM forward (+ LoRA) → (B, n_patches, d_llm) hidden
                          → mean-pool → (B, d_llm)
                          → project + fuse with feat → classification head
    """

    def __init__(
        self,
        num_feat: int = 34,
        lob_time_dim: int = 50,
        lob_channels: int = 3 * 20,
        cfg: TimeLLMConfig = TimeLLMConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.lob_time_dim = lob_time_dim
        self.lob_channels = lob_channels
        assert lob_time_dim == cfg.patch_len * cfg.n_patches, (
            f"lob_time_dim={lob_time_dim} must equal "
            f"patch_len × n_patches = {cfg.patch_len * cfg.n_patches}"
        )

        self._build_llm()

        # Reprogramming layer: (patch_len × lob_channels) → d_llm
        self.reprogram = nn.Sequential(
            nn.Linear(cfg.patch_len * lob_channels, self.d_llm),
            nn.LayerNorm(self.d_llm),
        )

        # Feat tower
        self.feat_proj = nn.Sequential(
            nn.Linear(num_feat, cfg.d_proj),
            nn.GELU(),
            nn.Linear(cfg.d_proj, cfg.d_proj),
        )

        # LLM output projection
        self.lob_proj = nn.Sequential(
            nn.LayerNorm(self.d_llm),
            nn.Linear(self.d_llm, cfg.d_proj),
            nn.GELU(),
        )

        fused_dim = cfg.d_proj * 2
        self.cls_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 3),
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def _build_llm(self) -> None:
        """Load LLM in fp16/4-bit, attach LoRA adapters."""
        from transformers import AutoModel, AutoConfig, BitsAndBytesConfig
        cfg_kwargs: dict = {}
        if self.cfg.use_4bit:
            cfg_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        self.llm = AutoModel.from_pretrained(self.cfg.llm_repo,
                                              trust_remote_code=True,
                                              **cfg_kwargs)
        llm_config = AutoConfig.from_pretrained(self.cfg.llm_repo,
                                                 trust_remote_code=True)
        self.d_llm = getattr(llm_config, "hidden_size",
                              getattr(llm_config, "d_model", 768))

        if self.cfg.apply_lora:
            from peft import LoraConfig, get_peft_model
            peft_cfg = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=list(self.cfg.lora_target_modules),
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.llm = get_peft_model(self.llm, peft_cfg)
            # By default PEFT freezes non-LoRA params — we want this.
        else:
            for p in self.llm.parameters():
                p.requires_grad = False

    def forward(self, lob: torch.Tensor, feat: torch.Tensor, **_kwargs):
        # **_kwargs absorbs PEFT-injected kwargs (input_ids/attention_mask).
        B = lob.shape[0]
        # (B, 3, 20, T) → (B, 60, T) → (B, T, 60)
        x = lob.reshape(B, self.lob_channels, self.lob_time_dim)
        x = x.permute(0, 2, 1).contiguous()                      # (B, T, C)
        # Patch into (B, n_patches, patch_len × C)
        x = x.reshape(B, self.cfg.n_patches,
                       self.cfg.patch_len * self.lob_channels)
        # Reprogram to LLM embedding space
        tokens = self.reprogram(x)                                # (B, N, d_llm)
        # LLM forward with inputs_embeds (bypass tokenizer)
        out = self.llm(inputs_embeds=tokens, output_hidden_states=False,
                       return_dict=True)
        hidden = out.last_hidden_state                            # (B, N, d_llm)
        lob_pool = hidden.mean(dim=1)                             # (B, d_llm)

        lob_tok = self.lob_proj(lob_pool)
        feat_tok = self.feat_proj(feat)
        fused = torch.cat([lob_tok, feat_tok], dim=-1)
        logits = self.cls_head(fused)
        reg = self.reg_head(fused).squeeze(-1)
        return logits, reg

    def num_params(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
