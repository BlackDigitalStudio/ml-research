---
name: recorder-vm-live
description: "The market-data recorder runs 24/7 on a GCP VM in Tokyo (billing live), not on the Windows PC"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2f125ab3-dff7-4906-8f09-cd7955db840b
---

A 24/7 GCP VM (Tokyo) runs the market-data recorder instead of the Windows
PC (recorder is Linux/systemd-only; a VM avoids all Windows porting).

**As of 2026-06-01 the VM runs Chronos / recorder-v2, NOT the legacy
scalper-bot recorder.** The Cryptolake-matching recorder is the Chronos
codebase = repo `BlackDigitalStudio/crypto-market-recorder` (public; cloned
to `C:\Dev\crypto-market-recorder`), forked from the scalper-bot
`recorder-v2-enterprise` branch (that branch is NOT retained in scalper-bot —
verified: not in any scalper-bot/ml-research branch, history, or local dir).

**CANONICAL BASE — do not hunt for a "fuller" version.** The internal
`recorder-v2-enterprise` branch (commit 66dadd9) had ~2 more spec points than
the published product (P8 second-host redundancy, P15 dYdX — per the product
CHANGELOG, "14 of 16"). That fuller version lived only on the old dev/prod
server (Contabo Tokyo), which is now DEAD — it is unrecoverable. The
published `crypto-market-recorder` repo is therefore the definitive recorder
base going forward; P8/dYdX would have to be rewritten from scratch if ever
needed (not needed for Cryptolake parity). Our changes live on branch
`project/light-books` (OrderBookV2 max_levels prune + configurable seed_limit),
pushed to GitHub.
Chronos ≈ Cryptolake superset: depth_snapshot (L20, parity), depth_diff,
trade, mark_price (+index/funding), funding_settlement, liquidation,
derivatives_poll, book_checkpoint, integrity_gap; 3 µs timestamps; raw
.jsonl.gz archive. Deployed at L20 on **all 8 Cryptolake symbols**
(BNB/BTC/DOGE/ETH/LINK/LTC/SOL/XRP USDT-M futures) + Bybit/OKX/Bitget/Gate.io
BTC trades (bonus); Coinbase/Deribit OFF. All public data — no API keys.
- VM resized to **e2-standard-2 (2 vCPU / 8 GB) + 200 GB disk** for the
  8-symbol load (~7-9 GB/day; load avg ~0.2, comfortable). ~$60/mo.
- On VM: repo `/home/scalper/crypto-market-recorder`, entrypoint
  `chronos_run.py` (BINANCE_SYMBOLS = the 8), data root `.../data`, health
  `/home/scalper/chronos.health`. systemd: `chronos.service` +
  `chronos-gcs-sync.timer` + `chronos-watchdog.timer` + `chronos-retention.timer`
  (3-day local retention — Chronos has no built-in rotation).
  GCS mirror → `gs://recorder-data-asia-0998ac51/chronos/<host>/`.
- Legacy `scalper-recorder.service` is stopped + disabled (old data dir
  `/home/scalper/scalper-bot/data` left on disk; harmless).

**Binance WS stream availability (root-caused 2026-06-01 — NOT a GCP issue).**
Earlier hypothesis "GCP IPs are restricted by Binance" was DISPROVEN by
controlled tests (same result from a residential IP and from two WS libraries).
Real cause: Binance USDⓈ-M `@aggTrade` currently **acks the subscription but
streams no frames**, and `@markPrice` (all variants) likewise — while
`@trade`, `@depth@100ms`, `@bookTicker` stream normally from any IP. The
recorder (and legacy) subscribed to `@aggTrade`, so Binance trades never landed.
**Fix (PR#5, deployed): switch the Binance trade stream `@aggTrade`→`@trade`**
(`normalize_binance_trade` + extractor + dispatch + gateway). Verified live:
16/16 symbols recording trades. `@trade` (per-fill `id`) also maps better to
Cryptolake `raw/trades`. STILL OPEN: `@markPrice`/funding has no WS feed
currently → use REST `/fapi/v1/premiumIndex` (returns 200) if funding/mark
needed; liquidations `@forceOrder` unverified (rare events). All chronos fixes
this session merged to main: PR#1 light books, PR#2 env-config symbols,
PR#3 connection-pool, PR#4 combined streams, PR#5 @trade. No relocation needed.
- Deploy tooling: `scalper-bot` repo `deploy/recorder-vm/chronos/`
  (deploy.sh, startup.sh, chronos_run.py, units).

- **VM**: `scalper-recorder`, `e2-small`, zone `asia-northeast1-b` (Tokyo —
  Binance Futures reachable; `us-*` is geo-blocked). Project
  `project-0998ac51-36ba-445c-bc7` (gcloud account virgin.ship03@gmail.com).
- **Cost ~$14-16/mo** while running. Pause: `gcloud compute instances stop
  scalper-recorder --zone asia-northeast1-b`.
- **Data**: local parquet on the VM (7-day retention) + hourly
  `gcloud storage rsync` to `gs://recorder-data-asia-0998ac51/recorder/<host>/`
  (Tokyo bucket, co-located, additive). This is the recorder's OWN data —
  separate from the Cryptolake data in `gs://blackdigital-scalper-data`, but
  written to the same schema.
- **config.env on the VM uses placeholder Binance keys** — recorder uses only
  public streams + public REST; the user-data stream self-disables on the
  expected listenKey -2014 error. Add real keys only if signed calls are
  needed.
- Deploy/monitor tooling: `deploy/recorder-vm/` (deploy.sh, status.ps1, README).
  systemd: `scalper-recorder.service` (Restart=always), `scalper-gcs-sync.timer`,
  `scalper-watchdog.timer`.
