---
name: h150-deploy-candidate
description: "HD3 rev6 close-out (2026-07-07): h150 honest cell (entry 60s, hold 150s from fill, full features) is the first edge>noise result — DOGE t5 +6.27±0.9 / BTC +9.69±1.8 PASS, ETH FAIL; deploy candidate = DOGE t10 (+BTC) with 4-seed ensemble scoring, pending recorder-EV cross-check + live input wiring"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6287b103-b1f8-4152-bb04-df2d7ec2e6cf
---

**Cell (scope-bound, see [[scope-bound-claims]]):** CL-year ~371d 2025-05→2026-06, honest time-based maker cycle — entry GTX 60s from decision, hold **150s FROM FILL**, pegged never-taker chase 300s, always-last both legs, 0 fee (USDC), 3s decision grid, FULL features (funding/liq/OI/ETH/btc real), exact labels, 4 seeds/symbol (RESEARCH_LOG §21, ledger rev6 2026-07-07).

- **DOGE**: t5 +6.27±0.90 (min +4.97), t10 +3.53 (min +2.49; folds>0 79%, avg worst fold +0.4 → robustness argmax). **BTC**: t5 +9.69±1.82 (min +7.48), annS mean +6.95 — strongest, but winter-heavy folds (avg worst −9.8); flat at 30s semantics, needs the long horizon. **ETH**: FAIL (seed sd ≈ mean, t10 sign flip).
- **Hold sweep** (same trades): 90s +3.20 / 150s +7.19 / 240s +0.57 → reversion pays in ~2-2.5min, decays by 4min; exit timing matters ±60s.
- 30s hold is dead at ANY feature set (all-seeds-negative −2.5±1.2); the currently deployed axb-live (30s config) has no year support.

**How to apply (deploy path, user decision):** DOGE t10 (+ optional BTC t5/t10), **4-seed ensemble scoring** (mean pA/pBg across seeds — harvests the measured seed noise). Prereqs: (1) recorder-EV cross-check of the h150 config on live recorder days; (2) live engine wiring: btc bookTicker, funding (markPrice), OI, ETH aggTrade inputs; DECIDE_S 10.8→3s, ENTRY_WIN_S 12.8→60, HOLD_S 30→150. Ops rule: systemd-run expands `${VAR}` in payloads — bare `$VAR` or script files only.
