"""Architecture bake-off module — PatchTST, Mamba, TCN, etc.

All architectures expose the SAME interface as `src.teacher.MultiStreamTransformer`:
    forward(lob: (B,3,20,T), feat: (B,F)) -> (logits (B,3), reg (B,))

This lets `train_teacher` drive any architecture interchangeably via a factory.
"""
from __future__ import annotations

from .patchtst import PatchTST, PatchTSTConfig
from .mamba import MambaClassifier, MambaModelConfig
from .hybrid_mamba_attn import HybridMambaAttnClassifier, HybridMambaAttnConfig
from .tcn import TCNClassifier, TCNConfig
from .chronos_adapter import ChronosClassifier, ChronosAdapterConfig
from .timesfm_adapter import TimesFMClassifier, TimesFMAdapterConfig
from .moment_adapter import MOMENTClassifier, MOMENTAdapterConfig
from .time_llm_adapter import TimeLLMClassifier, TimeLLMConfig
from .multitask_heads import (
    MultiTaskHeads, MultiTaskConfig, MultiTaskWrapper, multitask_loss,
)
from .regime_moe import (
    RegimeMoE, RegimeMoEConfig, build_regime_moe, compute_regime_hard,
)
from .meta_label import (
    MetaConfig, build_meta_dataset, train_meta, combine as meta_combine,
)
from .stacking import StackerConfig, train_stacker, predict_stacked, stack_inputs

__all__ = [
    "PatchTST", "PatchTSTConfig",
    "MambaClassifier", "MambaModelConfig",
    "HybridMambaAttnClassifier", "HybridMambaAttnConfig",
    "TCNClassifier", "TCNConfig",
    "ChronosClassifier", "ChronosAdapterConfig",
    "TimesFMClassifier", "TimesFMAdapterConfig",
    "MOMENTClassifier", "MOMENTAdapterConfig",
    "TimeLLMClassifier", "TimeLLMConfig",
    "MultiTaskHeads", "MultiTaskConfig", "MultiTaskWrapper", "multitask_loss",
    "RegimeMoE", "RegimeMoEConfig", "build_regime_moe", "compute_regime_hard",
    "MetaConfig", "build_meta_dataset", "train_meta", "meta_combine",
    "StackerConfig", "train_stacker", "predict_stacked", "stack_inputs",
]
