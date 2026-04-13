"""Architecture bake-off module — PatchTST, Mamba, TCN, etc.

All architectures expose the SAME interface as `src.teacher.MultiStreamTransformer`:
    forward(lob: (B,3,20,T), feat: (B,F)) -> (logits (B,3), reg (B,))

This lets `train_teacher` drive any architecture interchangeably via a factory.
"""
from __future__ import annotations

from .patchtst import PatchTST, PatchTSTConfig

__all__ = ["PatchTST", "PatchTSTConfig"]
