# ml-research

The research repository of the trading program: **the asset here is information** —
which approach yields harvestable alpha, under what conditions, proven how. Code
exists to produce and reproduce that record. The deployed algorithm itself is
mirrored to `BlackDigitalStudio/trading_algorithm`; ledger deploy records pin its
commits.

## Read in this order
1. **`CLAUDE.md`** — binding operating frame (exploratory-vs-confirmatory framing,
   append-only ledgers, frozen-infra rules, run discipline).
2. **`REPO_MAP.md`** — what is active, what is frozen, what is archived.
3. **`research/README.md`** — the ledger contract (hypotheses/experiments jsonl).
4. **`research/runtime/README.md`** — HOW TO RUN things (canonical invocations,
   VM recipes, sizing) + `KNOWN_PITFALLS.md` before debugging anything.
5. **`RESEARCH_LOG.md`** — narrative history; the LAST section is the current baseline.

## Layout (post-cleanup 2026-07-11)
- `research/` — ledgers, research log machinery, run runtime, shared agent memory.
- `scripts/` — the ACTIVE frozen measurement family (`subs60_*`, `hd1_seq_core.py`
  §5 gate, husdc bin sources). Byte-frozen; changes need prereg + parity ritual.
- `rust_ingest/` — feature/label/sim engines + the live axb_engine and parity
  harnesses. Branch `claude/husdc-rev1` is load-bearing — never delete.
- `live/` — deployed unit files and boot/exec sidecars (synced to the live VM).
- `archive/` — everything historical, moved as-is for provenance. Not maintained.

## Current state (2026-07-11)
Deployed: anchored h150 4-seed ensemble, DOGEUSDC maker t5, bit-exact Rust engine
(RESEARCH_LOG s22 = baseline). In flight: HD3 rev8 cross-symbol year campaign
(ledger `tb3s-20260710_h150anch_year_xsym_PREREG` + amendments).
