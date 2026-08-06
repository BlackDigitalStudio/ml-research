---
name: capture-all-information
description: "Save ALL run artifacts (per-epoch/per-trial metrics, decision points, model weights, predictions, stats) — over-save rather than lose; information is the paid-for asset"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e0fb2797-9e8b-421d-8f45-0ea10e0801e7
---

When running experiments/training, **capture everything**: per-epoch / per-trial
metrics, decision points, model weights/checkpoints, test-set predictions + the
labels/payoffs needed to recompute any metric offline, full feature importances,
val curves, and the run config/manifest. **Prefer writing extra gigabytes over
losing data** — re-running to recover a number costs far more than the storage.

**Why:** the user's framing — "Information is the asset we pay money and time for
and which makes money" (echoes CLAUDE.md "the asset is information"). Losing a stat
means paying for the whole run again. Saving the test predictions + per-sample
maker pnl, for instance, lets you rebuild the entire alpha surface (every
conviction/config cut) WITHOUT retraining.

**How to apply:** by default persist to GCS research_runs/: (1) all Optuna trials
with full params+metrics (not just best), (2) the booster(s), (3) test predictions
aligned to sample ts + per-side maker pnl/fill + rH so any operating-point/config
surface is recomputable offline, (4) ALL feature importances (gain/weight/cover),
(5) a run manifest (args, splits, thresholds, data provenance, code version).
Relates to [[audit-before-long-runs]].

**NON-NEGOTIABLE (user escalated 2026-06-02, "third agent in a row"):**
- EVERY script that trains a model MUST save the weights + predictions + trials to
  GCS *as part of the run* — not as an afterthought. A training script with no
  save_model() call is a bug. Build the saving in before launching.
- NEVER kill / discard an in-flight run to "save compute" — that throws away the
  asset (information). Compute is rented and cheap; the information it produced is
  not. If a design needs changing, let the current run FINISH and save, or at least
  capture partial results, before relaunching. Losing progress = throwing money away.
- "Skip work (e.g. Optuna) to save compute" is the WRONG framing and the question
  should not arise. The principle is: do the work, SAVE every output. Cost of compute
  is never the reason to discard or skip — only the reason to not duplicate.
- Reuse already-trained models/predictions (load the saved booster / preds_*.npz)
  instead of retraining what is fixed and unchanged.
