# research/ — the information asset

The single source of truth for every experiment. The asset of this project is
**information** (what was tried, on what data, with what result — and under what
conditions an approach yields harvestable alpha). Compute is cheap and rented; the
record is what survives. Operating rules live in `CLAUDE.md` (read first);
orientation in `REPO_MAP.md`; current baseline = last section of `RESEARCH_LOG.md`
(s22 as of 2026-07-11).

## Live contents (what to actually use)

| Path | Role |
|---|---|
| `hypotheses.jsonl` | Append-only event log: one line = one hypothesis revision (prereg → measured → …). Never mutate a recorded line; append a new one. |
| `experiments.jsonl` | Append-only event log of results/deploys. Same append-only rule. |
| `runtime/` | **The run machinery**: orchestrators, VM/binary recipes, sizing, `KNOWN_PITFALLS.md`. Read its README before writing ANY run code. |
| `memory/` | Canonical shared agent memory (local auto-memory is a cache; see its README). |
| `vm_runs.jsonl`, `hardware_ledger.jsonl` | Append logs of VM runs / hardware facts. |
| `CRYPTOLAKE_SCHEMA.md` | Raw-data schema reference for the CL year. |
| `HANDOFF_*.md`, `TCN_CHECKPOINT.md`, `HDATA_*`, `HDX_LEVERS.md`, `PLAN.md`, `PLAN_subminute_mamba2.md` | FROZEN tier handoffs/plans — historical context for ledger records, not current direction. `PLAN.md`'s "CURRENT DIRECTION (2026-05-17)" is long superseded (the program moved through HD1/HD2 to the deployed HD3 h150 line). |
| `schema.sql`, `ledger.py`, `build_full_ledger_db.py`, `research.db`, `full_ledger.db`, `hd1_session_rev48_57.db`, `raw_research_data/`, `ev/`, `rev*.json` | The 2026-05-era strict-schema ledger tooling and tier leftovers. The strict per-field contract (`fee_regime`, `cache_id`, `split_method`, `label_def`, `repro_cmd`) encoded real expensive lessons and its *principles* remain binding — but current practice records them inside free-form `statement`/`note`/`summary` fields of the JSONL + `RESEARCH_LOG` narrative instead of the SQL columns. Kept for provenance and for querying the old rows. |

Record-writing rules (binding, from CLAUDE.md):
- Prereg a frozen spec BEFORE running a tier; corrections = new appended
  amendment records, never edits.
- Record every production invocation (env + command) in the record.
- Every quantitative claim is a measured cell (symbol × period × execution ×
  features × protocol) — state the cell.
- Exploratory tiers report the conditional alpha SURFACE as the headline; the
  confirmatory deploy gate is a labelled secondary annotation.

## Infra reality (2026-07-11)

- **Workstation**: Windows box (`C:\Dev\scalper-bot`), gcloud authed as
  `virgin.ship03@gmail.com`. Full toolchain; runs orchestrate GCP VMs.
- **GCP**: `project-0998ac51-…` (bucket owner; global CPU quota 12) and
  `gen-lang-client-0075410383` (delmiron27; global cap 32 vCPU — research VMs,
  e.g. `xsym-32` n2-highmem-32). Recorder + live engine run 24/7 on
  `scalper-recorder` (Tokyo). Billing history: see `memory/gcp-accounts-billing-migration.md`.
- **GCS `gs://market-data-0998ac51` (EUROPE-WEST1) = the persistent data asset**:
  `raw/{book,trades,funding,liquidations,open_interest}/exchange=BINANCE_FUTURES/symbol=<SYM>-USDT-PERP/dt=<DAY>/`
  (8 symbols, CL year 2025-05..2026-06), `research_runs/<dataset families>`,
  recorder day files. Same-region VMs read it free.
- **⚠ features_v1 caveat still true**: old `features_v1/` npy caches are book-only
  (funding/OI/liq/ETH cols zero). Raw inputs EXIST in `raw/` — build with
  `FULLFEAT=1` (see `runtime/README.md`); do not conclude "no data" from zeros.
- History: Contabo host lost 2026-05 (the event that created this ledger);
  Cryptolake subscription lapsed 2026-07 — the CL year in `raw/` is the frozen
  research dataset.

Rule unchanged: a result is not "saved" until its row is committed to the JSONL
and pushed; artifacts go to GCS with their path recorded.
