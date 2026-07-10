---
name: xsym-cross-symbol-run
description: "HD3 rev8 cross-symbol year run (7 symbols, anchored h150, t5-only) LAUNCHED 2026-07-10 on hd2-feats-003 (unit xsym); H_TICKS=1800 is the cross-symbol build protocol, not 1500"
metadata: 
  node_type: memory
  type: project
  originSessionId: e405d5fd-5d12-437d-a431-d396bf143b9a
---

HD3 rev8 (prereg `tb3s-20260710_h150anch_year_xsym_PREREG` + AMEND1): cross-symbol
year measurement of the deployed anchored-h150 policy on BTC, ETH, BNB, LTC, SOL,
XRP, LINK (t5 only, no noA, 4 seeds, W200/T30/EMB2 folds). Launched 2026-07-10
~13:35 UTC as systemd unit `xsym` on hd2-feats-003 (n2-standard-8, europe-west1-b),
orchestrator `/home/delmi/xsym/orchestrate.py`, ETA ~3 days.

Key facts:
- **H_TICKS=1800** is the cross-symbol h150 build parameter (recovered from journald
  unit tb3sym; the DOGE-dedicated script's H=1500 is NOT the cross-symbol protocol).
  Not density-normalized: ~200s path on BTC-class books (~9 ticks/s) vs ~900s on
  DOGE-class (~2/s) — chase run-out marked at touch more often on dense books.
- Rebuilt binaries proven **bit-exact** vs July artifacts (BTC 2025-11-15, all keys):
  feature_builder from master rust_ingest; build_samples + grid_sim_exitdbg from
  **husdc-rev1 branch lib** + frozen master `scripts/{build_samples_husdc,grid_sim_exitdbg}.rs`.
- Artifacts land in `research_runs/maker_labels_tb3s_h150{,anch}/` (dailies, combined,
  anch npz, OPTUNA_IC_{SYM}_qm0_SEED{0..3}.json, PERFOLD_S{s}_{SYM}_qm0_f{k}.npz,
  ENS_{SYM}_t5.json). VM scripts: ~/xsym/{orchestrate,prep_anch_sym,ens_sym}.py.
- LINK has only 246 days in window (119d outage) → ~2 folds, reduced-power cell.
- Max 2 concurrent trainings (31GB RAM bound), slots adjustable via ~/xsym/SLOTS.
- STOP THE VM when the run is collected (billing). Related: [[h150-anchored-year]],
  [[deploy-scope-budgets]], [[scope-bound-claims]].
