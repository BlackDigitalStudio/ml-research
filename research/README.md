# research/ — the information asset

The single source of truth for every experiment. The main asset of this
project is **information** (which strategy/methodology was tried, on what
data, with what result). Compute is cheap and rented; the asset is the
record. This directory is designed so the asset survives the loss of any
host (it did: Contabo is gone — see *Infra reality* below).

## Files

| File | Role | Git |
|---|---|---|
| `schema.sql` | Canonical DDL — **the contract**. Every column maps to a documented, expensive lesson. | committed |
| `experiments.jsonl` | Append-only. One line = one immutable result. | committed |
| `hypotheses.jsonl` | Append-only event log. One line = one hypothesis revision. | committed |
| `ledger.py` | Stdlib-only gate: `validate` / `append` / `build-db` / `frontier` / `check`. | committed |
| `research.db` | SQLite, **derived** from the JSONL. Rebuilt, never hand-edited. | gitignored |
| `PLAN.md` | Cheap-hypothesis plan (Phase A deliverable 2). | committed |
| `MAKER_SIM.md` | Operator guide: realistic maker-fill / adverse-selection sim (`grid_sim --flow-paths` + `build_samples` flow/entry_q, native flat L2). See HUSDC rev6/rev7. | committed |

JSONL (not a binary .db, not Parquet) is the source of truth because it is
human-diffable, merge-friendly, zero-dependency, and git-durable. SQLite is
a throwaway query index rebuilt by `ledger.py build-db`.

## The contract (why each mandatory field exists)

A result is **refused** unless it carries its provenance. The refusals are
not bureaucracy — each prevents a specific failure that already cost compute:

- **`fee_regime` ∈ {TAKER, MAKER_FIRST, MAKER}** — TAKER↔MAKER confusion
  produced *three* false positives (DOGE +19.6%→−1.5%, ETH +0.036%→−0.047%,
  phase56 +1.30%→−1.4%). The ledger additionally **refuses to store a
  positive-EV TAKER result as `confirmed`** — it must be `suspect` until
  revalidated MAKER-first.
- **`cache_id` + `data_source` + `n_samples` + `symbols`** — caches die
  (Contabo lost, v3 1.85M OOM-deleted, Cryptolake caches terminated). A
  number without its exact data origin is unreproducible noise.
- **`split_method` ∈ {CPCV_6_2, walkforward_7525, honest_val_test}** —
  these are not comparable; silently mixing them is how the chaos started.
- **`label_def`** — the exact `y=` rule. The "WR 76-85%" disaster was a
  label artifact (`target_pnl>0 ⟺ y!=FLAT`).
- **`repro_cmd`** — the exact command. If it can't be reproduced, say so
  (backfilled rows honestly record "source on lost Contabo host").

Corrections are **never edits**. You append a new row with `supersedes` →
old id and `status` in {refuted, artifact}. The lineage *is* the audit
trail; `v_artifact_chains` makes RESEARCH_LOG §4 queryable.

## Result schema = the owner's metrics

The result columns are exactly the owner-defined set (`Самые важные
метрики`, STRATEGY §7) plus the frontier metrics (RESEARCH_LOG §3):

- Owner 1–5 (mutually exclusive, sum→1.0): `pct_full_tp`, `pct_full_sl`,
  `pct_timeout`, `pct_trailing`, `pct_partial_only` — bucketed from the 12
  `live_sim.TradeOutcome.REASONS` exactly as `scripts/backtest.py` does.
- Owner 6–7: `pnl_gross_{pct,usd}`, `pnl_net_{pct,usd}` (|gross|≥|net|).
- Frontier: `ev_per_trade_pct` (primary), `trades_per_day`,
  `net_return_pct` + `kelly_frac`, `win_rate_pct`, `base_rate_pct`,
  `n_trades`, `sharpe`, `max_dd_pct`, `exit_hist_json` (all 12).

**`kind='alpha'` rows** (prediction-only; execution deferred to a future
RL agent) skip the owner/EV columns and instead carry `alpha_target`,
`horizon_sec`, `rank_ic_oos`, `auc_oos`, `top_decile_absmove_pct`,
`cost_floor_pct`, `decile_monotonic`, `economic_pass_loose` (~0.08%
maker, idealised), `economic_pass_strict` (~0.13% taker + slippage
haircut — the binding flag), `n_eff` (decorrelated). The gate **refuses
to `confirm` an alpha with `economic_pass_strict≠1`** — an RL agent
converts existing alpha but cannot beat the fee/spread floor, so
significant-but-sub-cost signal is worthless (`v_alpha`, `v_alpha_audit`;
see `PLAN.md` → *CURRENT DIRECTION*). `kind=NULL/'strategy'` = the
owner/EV contract above.

**SELECTION POLICY (mandatory — applies to every agent/session).**
During SEARCH (`kind='alpha'`) you select what to carry forward by
**robust marginal Δ in predictive skill vs a declared baseline**
(`baseline_ref`, `delta_ic`; robust = exceeds noise/placebo,
consistent across symbols/folds) — **NOT** by `economic_pass_*`. Every
building block is sub-cost alone until stacked; using the discrete
economic gate as a per-experiment keep/kill manufactures false
negatives (it did 3×: HZ1, HA5-scope, H3). `economic_pass_*` are
recorded distance-to-deploy metrics and a deploy gate **only** for a
final candidate (`kind='strategy'` / a `confirmed` alpha). For
`kind='alpha'`, `refuted` means **Δ within noise/placebo vs
baseline**, never "economic_pass_strict=0". A leak-free signal that is
sub-cost alone (e.g. HA1 ~0.08 rank-IC @30s) is the **baseline to
stack on**, not a dead end.

Backfilled rows have owner 1–7 = NULL: those metrics **were not captured**
at the time. That gap is the chaos this ledger closes — new experiments
must populate them.

## Usage

```bash
# add a result (validates before it touches the file)
python3 research/ledger.py append experiment my_result.json
python3 research/ledger.py append hypothesis my_hyp_rev.json

# rebuild the query index + CI gate (run before every commit / in CI)
python3 research/ledger.py check

# regenerate the RESEARCH_LOG §3 frontier table from data (never hand-type)
python3 research/ledger.py frontier

# query
sqlite3 research/research.db "SELECT * FROM v_frontier;"
sqlite3 research/research.db "SELECT * FROM v_artifact_chains;"
sqlite3 research/research.db "SELECT * FROM v_methodology_audit;"  -- must be empty
```

`v_frontier` shows only rows with `n_trades ≥ 30` and a trustworthy status;
thin-coverage history lives in `v_live_experiments`. `tests/test_ledger.py`
enforces the contract in CI (stdlib-only).

## Infra reality (2026-05-16)

The biggest risk to the asset is host loss. It already happened.

- **Contabo `root@84.247.154.229` — LOST.** Every "LIVE on Contabo" cache
  and the entire `/root/.claude/projects/-root/memory/*.md` deep-dive
  archive are **gone**. STRATEGY.md still references them; treat those
  pointers as historical. This ledger exists so it cannot happen again.
- **This container = planning node.** Stdlib Python only (no
  numpy/pandas/gcloud), no cloud creds, github reachable. Good for the
  ledger; cannot run experiments.
- **GCP `virgin.ship03@gmail.com` = compute account** (project
  `project-0998ac51-36ba-445c-bc7`, MIGRATED 2026-05-26 from `blackdigital.kz`
  / `project-26a24ad0…` whose balance ran low). Cheap 96 vCPU N2 VM quota
  (provisioned per-run, europe-west1). Real compute runs here.
- **GCS = the only persistent data asset.** `gs://market-data-0998ac51`
  (europe-west1, **585 GB**, full verified copy of the old bucket; old
  `gs://blackdigital-scalper-data` is the source, still exists, deletable).
  Layout: `raw/{book,trades,funding,liquidations,open_interest}/exchange=BINANCE_FUTURES/symbol=<SYM>/dt=<DAY>/*.parquet`,
  `features_v1/symbol=<SYM>/dt=<DAY>/{features.npy,indices.npy}`,
  `hd2_cache_v1/{streams,midts}/`, `feats_v2/`, `research_runs/`.
- **⚠️ DATA REALITY — do NOT say "no data" from empty `features_v1` columns.**
  `features_v1/features.npy` (N×59) is **book-only**: ~13 cols are
  placeholder-ZERO (ETH lead-lag, trade-flow/cvd, funding, cancel/spoof,
  cross-exchange). The raw inputs (trades, funding, ETH) **exist in `raw/`** —
  recompute the full set with the Rust `feature_builder`
  (`rust_ingest/src/bin/feature_builder.rs`). Only cross-exchange
  (bybit/okx/bitget/gateio) is genuinely absent (raw = BINANCE_FUTURES only).

Rule: a result is not "saved" until its row is committed to
`experiments.jsonl` and pushed. Artifacts go to GCS; the ledger keeps the
`gs://` path + sha256.
