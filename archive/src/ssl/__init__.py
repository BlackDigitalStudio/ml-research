"""Self-supervised pretraining on raw LOB data.

Train an encoder on masked-prediction over the full Tardis depth corpus
(~64M unlabeled snapshots). Then transfer the encoder to downstream
classifiers in the bake-off, where labeled samples are scarce (~300k).

Empirical win: 3-10× sample efficiency on the labeled task.
With our data sizes this typically translates to +5-10% accuracy at
fine-tune time.
"""
from __future__ import annotations

from .dataset import LOBWindowDataset, MaskedLOBBatch
from .model import PatchTSTBackbone, PatchTSTReconstructor
from .pretrain import pretrain_loop, PretrainConfig

__all__ = [
    "LOBWindowDataset",
    "MaskedLOBBatch",
    "PatchTSTBackbone",
    "PatchTSTReconstructor",
    "pretrain_loop",
    "PretrainConfig",
]
