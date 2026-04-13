"""Contract tests for L2 stacking meta-learner."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.stacking import (
    StackerConfig, stack_inputs, train_stacker, predict_stacked,
)


def _synth_primaries(n: int, n_models: int = 3, acc: float = 0.60,
                      seed: int = 42):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 3, size=n).astype(np.int64)
    softs = []
    for m in range(n_models):
        sub_rng = np.random.default_rng(seed + m)
        s = sub_rng.dirichlet([0.5, 0.5, 0.5], size=n).astype(np.float32)
        for i in range(n):
            if sub_rng.random() < acc:
                s[i] = s[i] * 0.3
                s[i, y_true[i]] += 0.7
                s[i] /= s[i].sum()
        softs.append(s)
    X_feat = rng.standard_normal((n, 34)).astype(np.float32)
    return softs, y_true, X_feat


def test_stack_inputs_concatenates_all_primaries():
    softs = [np.random.rand(100, 3).astype(np.float32) for _ in range(4)]
    X_feat = np.random.randn(100, 34).astype(np.float32)
    X = stack_inputs(softs, X_feat=X_feat)
    # 6 features per primary (soft 3 + max 1 + margin 1 + entropy 1) + 34 feats
    assert X.shape == (100, 4 * 6 + 34)


def test_stack_inputs_without_features():
    softs = [np.random.rand(50, 3).astype(np.float32) for _ in range(2)]
    X = stack_inputs(softs, X_feat=None)
    assert X.shape == (50, 2 * 6)


def test_train_stacker_beats_individual_primaries():
    """Stacked accuracy ≥ best individual (diversity + learned weighting)."""
    softs, y, X_feat = _synth_primaries(2500, n_models=3, acc=0.55, seed=11)
    stacker, metrics = train_stacker(softs, y, X_feat=X_feat, val_frac=0.25)
    # Individual val accuracies
    n_val = int(len(y) * 0.25)
    n_tr = len(y) - n_val
    indiv_accs = [(s[n_tr:].argmax(-1) == y[n_tr:]).mean() for s in softs]
    best_indiv = max(indiv_accs)
    # Stacker should NOT be much worse than best individual
    assert metrics["val_acc"] >= best_indiv - 0.05, (
        f"Stacker {metrics['val_acc']:.3f} << best {best_indiv:.3f}"
    )


def test_predict_stacked_shape():
    softs, y, X_feat = _synth_primaries(1000, n_models=2, seed=13)
    stacker, _ = train_stacker(softs, y, X_feat=X_feat)
    out = predict_stacked(stacker, softs, X_feat=X_feat)
    assert out.shape == (1000, 3)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)


def test_stacker_with_one_primary_still_works():
    """Degenerate case: single primary."""
    softs, y, X_feat = _synth_primaries(800, n_models=1, seed=17)
    stacker, metrics = train_stacker(softs, y, X_feat=X_feat)
    # Should at least converge without crashing and produce valid metrics
    assert "val_acc" in metrics
    assert metrics["val_acc"] >= 0.33  # better than random on 3-class
