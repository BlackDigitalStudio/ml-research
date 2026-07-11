# REPO_MAP — what is alive, what is frozen, what is history (as of 2026-07-11, s22 baseline)

Orientation for agents: this repo (`BlackDigitalStudio/ml-research`, ex `scalper-bot`)
is the RESEARCH repository — ledger, research log, frozen measurement protocol, run
machinery, and agent memory live here. The deployed algorithm is mirrored/extracted
to `BlackDigitalStudio/trading_algorithm`; ledger deploy records pin the commits.

## ACTIVE — the research program core
- `CLAUDE.md` — binding operating frame. Read first.
- `research/` — THE program state:
  - `hypotheses.jsonl`, `experiments.jsonl` — append-only event ledgers (never mutate).
  - `../RESEARCH_LOG.md` — narrative log (root level; s22 = current baseline).
  - `runtime/` — run machinery + KNOWN_PITFALLS. Extend in place, no ad-hoc runners.
  - `memory/` — canonical shared agent memory (see its README).
- `scripts/subs60_*.py` — the FROZEN measurement-protocol family (tb3s builds,
  xgb optuna training, recorder-EV). Byte-frozen; changes need prereg + parity ritual.
- `rust_ingest/` — depth parser, feature_builder, live axb_engine + parity harnesses.
  Branch `claude/husdc-rev1` is LOAD-BEARING (lib for build_samples/grid_sim_exitdbg —
  see research/runtime/bins.sh). Never delete that branch.
- `live/` — deployed units (axb engine/exec/boot, systemd), synced to the live VM.

## FROZEN REFERENCE (valid, rarely touched)
- `research/*.md` (HANDOFF_*, PLAN*, CRYPTOLAKE_SCHEMA…) — tier handoffs and schemas.
- `research/TCN_CHECKPOINT.md` — closed-tier doc.
- `scripts/hd1_seq_core.py` — the frozen HM1 §5 confirmatory gate (CLAUDE.md-bound).

## ARCHIVED (physically moved 2026-07-11 — see `archive/README.md`)
Everything historical now lives under `archive/`, moved as-is with git history
(`git log --follow`): the pre-HD strategy stack (`src/`, `main.py`, `systemd/`,
`tests/`, `requirements*`), closed tiers (hd1_*/hd2_*/ha*/grid_*/phase_b/stacker/
SSL/RL/mamba script families incl. the mamba-linked `subs60_{ft_stability,
gru_gridsim,maker_grid}.py`), old run outputs and superseded docs. Not maintained;
do not extend; re-verify any number from there against the ledger before use.

Rule of thumb: if you are about to WRITE code outside `research/runtime/`,
`scripts/subs60_*`, `rust_ingest/`, or `live/` — stop and check you are not
resurrecting a closed tier. If you are about to READ a number out of archived
docs — re-verify against the ledger and current data first (CLAUDE.md rules).
