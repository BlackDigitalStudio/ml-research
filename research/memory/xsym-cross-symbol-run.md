---
name: xsym-cross-symbol-run
description: "HD3 rev8 MEASURED (2026-07-11) — cross-symbol year surface of anchored h150 at t5; XRP argmax (~DOGE magnitude, jitter-fragile at sd.05), BTC/ETH ensemble-fragile, BNB/LTC/SOL negative, LINK degenerate; both research VMs stopped"
metadata: 
  node_type: memory
  type: project
  originSessionId: e405d5fd-5d12-437d-a431-d396bf143b9a
---

HD3 rev8 complete (ledger `tb3s-20260710_h150anch_year_xsym`, RESEARCH_LOG s23).
Cross-symbol year surface of the deployed anchored-h150 policy, t5, 4 seeds:

- **XRP** per-seed +10.97±3.50 (4/4), ENS +12.68 (LOFO≥+8.36, jitter sd.02 P100 /
  sd.05 P41) — argmax, conditional deploy candidate pending recorder-EV cross-check
  with the sd.05 fragility flag. **BTC** +6.20±2.76 but ENS +3.83 jitter-fragile
  (P0 at sd.05) — ensemble DEGRADES here. **ETH** +4.86±3.83, ENS +7.36, jitter fail.
  **SOL −5.05 / BNB −3.00 / LTC −10.77** ens, P0. **LINK degenerate** (2 folds,
  0-3 trades/seed). Reference DOGE: +8.14 per-seed / +13.35 ENS, P100 at sd.05.
- Key structural finding: **ensemble stabilization is DOGE-specific, not universal**;
  sign does not reduce to book density (LTC/BNB/SOL negative at DOGE-like density).
- BTC/ETH cells carry the H_TICKS=1800 dense-book chase-runout caveat.
- Every number = cell: symbol × CL-year(2025-05..2026-06) × anchored-h150 honest
  maker × t5 × frozen rev7 protocol ([[scope-bound-claims]]).
- VMs: hd2-feats-003 and xsym-32 (gen-lang-client project) both STOPPED 2026-07-11.
  Datasets maker_labels_tb3s_h150{,anch} now cover all 8 symbols (reusable).
Related: [[h150-anchored-year]], [[research-runtime-infra]].
