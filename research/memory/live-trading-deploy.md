---
name: live-trading-deploy
description: "AxB live trading is DEPLOYED on the Tokyo VM (axb-live systemd unit, DOGEUSDC maker, budget t10, 100% of ~10 USDC deposit) since 2026-07-05"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6287b103-b1f8-4152-bb04-df2d7ec2e6cf
---

**CONFIG SWAP 2026-07-07: now running the h150 variant.** Old 30s `axb-live` (robust2, dead feature cols, year EV ~−2.35) STOPPED+disabled; replaced by **`axb-live-doge`** systemd unit (MODE=live, WORKDIR=/home/delmi/axb_h150, code `axb_h150.py` = same `live/axb_live.py`, commit b042cb8). Config: 4-seed ENSEMBLE from `deploy_h150/DOGE/seed{0..3}`, entry 60s / hold 150s-from-fill / pegged never-taker chase, DECIDE_S=3, budget **t5**, SIZE_FRAC=1.0, leverage 2, FULL features (funding/OI/ETH-lead/BTC-lead all real, wired from markPrice/derivatives-poll REST/ethusdt aggTrade/btcusdt bookTicker). tau seeded from RECORDER ensemble scores (`_recev_h150_DOGE`), not CL. Evidence: CL year t5 +6.27±0.9 (4 seeds), recorder-EV causal t5 +6.59 (10d/76tr, hit 57.9%); see [[h150-deploy-candidate]]. Running live+decision-log simultaneously = the paper-vs-live fill parity test (user's ephemeral ~10 USDC deposit). BTC (`axb-live-btc`, BTCUSDC tick 0.1/lot 0.001) pending its recorder-EV + capital split (two units on one wallet ⇒ SIZE_FRAC must sum ≤1). Old robust2 facts below are the PRIOR deployment, kept for history.

---
**(PRIOR) Live trading** (2026-07-05 → 2026-07-07): frozen `deploy_robust2` DOGE model traded **DOGEUSDC** on VM `scalper-recorder` (Tokyo), systemd unit **`axb-live`**, code `live/axb_live.py` (commit b544d76). 3 closed trades total (+2.63/−15.77/−38.57 bp = ROI ×2 leverage); superseded by h150.

- Signal: DOGEUSDT streams (routed WS: depth20@100ms on /public; aggTrade+forceOrder on /market), decisions every 10.8s == offline 8000/day cadence; same rust feature_builder on a 900s rolling window; day-level causal-rolling tau, buffer seeded axb_seed + recorder-EV days, state in /home/delmi/axb/state.npz.
- Execution: GTX post-only entry at touch (12.8s window), hold to decision+30s, pegged reduce-only maker exit; taker backstop at 300s chase / -50bp; halts at -5%/day or 40 trades/day. Budget **t10**, SIZE_FRAC=1.0 (user: "позиция 100% от текущего депозита, бюджет 10 сделок в день — про селективность"). Leverage 2 (margin headroom only).
- Keys: user's real Binance keys in `/home/delmi/axb/config.env` on the VM (chmod 600; NOT in repo). Also on the user's PC at `C:\Разработки\Binance.txt`. Key is IP-restricted to the VM egress IP **35.221.108.194**, which I reserved as a STATIC GCP address (`scalper-recorder-ip`) — do not release it; VM stop/start keeps it.
- Deposit at launch: **10.27 USDC** in USDⓈ-M futures wallet. Expected EV (recorder-EV, 7d): t10 +10.3 bp/trade gross = net (0 fee).
- Logs: /home/delmi/axb/decisions/YYYYMMDD.jsonl (all decisions, `executed` flag) + /home/delmi/axb/trades/YYYYMMDD.jsonl (order lifecycle); decisions upload hourly to gs://market-data-0998ac51/research_runs/axb_shadow/DOGE/decisions/.
- Parity gate tool: `live/axb_shadow_eval.py` (run on hd2-feats-003) — compare live decisions vs offline pipeline on the same day.
- Earlier shadow unit `axb-shadow` is disabled (same file, MODE=shadow).
- Cryptolake subscription is LAPSED (2026-07-05: all S3 AccessDenied, quota user=unknown) — user chose to proceed without it. See [[cl-recorder-sampling-mismatch]], [[recorder-vm-live]], [[gcp-accounts-billing-migration]].

**SIM-vs-LIVE differences (optimism audit 2026-07-05, by code+data, not notes):**
1. **CL labels were tick-based → NOT 30s.** CL book density 1.45-2.32 snaps/s (measured 4 days across the training year) → "hold 282 ticks" = 121-197s wall-clock, "entry window 120 ticks" = 51-83s. So the year-long CL backtest (+1.3…+3.4 bp) priced a ~2-3-MINUTE-hold strategy; B's training targets carry that horizon. Only rH_30 (A-gate target) is honest time-based 30s. The live 30s semantics are validated ONLY by the 7-day recorder-EV (recorder density ~9/s → 282t=30s genuinely).
2. **Exit ran-out marking:** sim path ends 700 ticks after decision (~45s of chase on recorder data); unfilled pegged exits (51-53% of trades!) are MARKED at last touch, 0 fee — optimistic valuation. Measured honest 5-min-chase rerun (horizon 3000): ran-out drops to ~1.2-1.4% (maker exit IS executable), label delta −0.42/−0.62 bp per trade avg, tails to −10bp. BUT top-score selected trades exit fast → recorder-EV table survives correction: t5 +36.6 / t10 +10.27 unchanged, t20 +6.18, t40 +2.25 (honest artifacts in `research_runs/_recev3k_tmp/`). B's TRAINING targets still carry the mark optimism → fix at retrain.
3. Entry sim is conservative (flow counted only when touch at our level; no queue advance from cancels; always-last). Trades triplication did NOT affect training inputs (dup=1.00 across the year — triplication is a recent-CL-days artifact only).
4. Minor training non-causalities: yA p95 + gstd norm-floor over full period; OOF gate folds = interleaved days (day%4); score CDF refs / axb_seed in-sample.
5. Executor policy (user 2026-07-05): **maker-only exit, NEVER taker** — chase until filled; backstop only catastrophic (86400s / 300bp). Hold = 30s FROM FILL (sim: es=k+to). Realistic trade rate: fill on SELECTED trades is low → expect ~1.5-3 executed/day at t10, not 8-9.
6. USDC-vs-USDT venue gap measured (2 days L2): queue 20x thinner but flow 11x thinner → entry/exit fill proxies IDENTICAL (0.31/0.32, 0.26/0.26); USDC touch capacity ~$1k — fine at $10, matters at scale.

**REVISED YEAR-LONG EXPECTATION (HD3 rev6 + §20a, 2026-07-05):** the live configuration (2-3min-trained B, honest 30s execution) has year EV **−2.35 bp/tr** (only the 2 most recent folds positive, + recorder-EV +10.3/7d = regime-conditional). The historical +3.4..+4.2 year baselines were NOT lookahead-inflated (refuted model-free) — they are draws from a protocol whose year-EV has ±3-5bp structural variance under label noise, mean ~+1±3 → edge never statistically identified; see [[label-matching-lookahead]]. Honest-30s cells consistently negative (−3.8…−2.4, all budgets/training horizons). Retraining on 30s targets does not fix it. Live left running per user (operational validation at $10).
