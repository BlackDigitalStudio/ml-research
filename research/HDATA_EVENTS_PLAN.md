# HDATA — event-driven structured-events plan (cells EV-A..EV-E)

Design doc for the event-driven sub-cells of **HDATA** (open data-axis
container, `research/hypotheses.jsonl` rev1, `status:active`, branch
`claude/hdata-rev1`). This file plans the data acquisition + event-study
design; the *frozen per-cell pre-registration* (axes, levels, provenance,
cost basis) is appended to `hypotheses.jsonl` / `experiments.jsonl` at run
time per CLAUDE.md mechanics — this file pre-enumerates nothing binding.

**Decided 2026-05-24 (user):** broad axis = ALL structured events as ONE
test (no narrow overlapping hypotheses); sequence **A→B→C→D**, try every
class starting with A; **backtest-first** (validate the reaction surface on
archives before any live-ingestion engineering).

---

## 0. Reframe — why structured events, not more free-text news

EXP-6/7 (HDATA_RESULTS.md §7-8, local) tested **aggregate / free-text**
GDELT: GKG 15-min tone rank_IC ≈ 0 (BTC 15m +0.0031, 240m −0.0052) and
keyword-classified events dir_ret ≈ 0 / win-rate ≈ 0.50. The limit was
**not NLP quality** — free text carries no guaranteed event *type* or
*sign*, and GDELT's 15-min bins + lag are unusable for an instant reaction.

Event-driven is a different object: **structured events** have a known type
and a known sign, and the move is *mechanical forced-flow* (listing arb /
index inclusion, unlock sell-pressure, macro repricing, large on-chain
flow). The actionable timestamp is precise → the reaction is **backtestable
on the 1m klines already in `free_v1/`**, no new NLP. This plan deliberately
chases structured events, not another free-text classifier.

## 1. Two pipelines — do not conflate

- **Backtest / training** → needs an *archive of events with a precise
  actionable timestamp*. This is the hard, tractable part and what we build
  first.
- **Live** → needs a low-latency feed + a processing budget (user infra:
  ~1-3 ms RTT to Binance, sub-second TA→order). Built only after a cell
  shows a large reaction surface.

**Decided: backtest-first.** Live feeds = cell EV-E, deferred.

## 2. Primary deliverable (CLAUDE.md rule 1 — surface, not verdict)

Per `(event-class, horizon h)` cell, the **event-reaction surface**, reported
as the headline:
- `dir_ret` = mean of `sign(event) · r_h` (signed forward log-return),
- win-rate = `mean(sign(event)·r_h > 0)`,
- `med|r_h|` (move magnitude — how much there is to clear cost),
- `rank_IC(event_strength, r_h)` where event_strength is graded (e.g. listing
  notional / unlock %supply / macro surprise z / mint size),
- block-bootstrap SE on all of the above.

The 0.13% strict round-trip cost floor and **events/day** are SECONDARY
annotations (deploy-distance + throughput), never a per-cell kill criterion
and never the framing.

## 3. Horizon grid, entry, latency realism

- `h ∈ {1, 5, 15, 60} m` forward log-return surface (event-studies report the
  surface; horizon is not collapsed to one point here).
- **Entry** = first 1m close STRICTLY after `event_ts + latency_budget`.
  No look-ahead: `searchsorted(event_ts, side="right")` on the kline grid.
- **Latency budget**, reported per cell: user infra realistically enters at
  the event-minute close (sub-second ingest+process); also report
  next-minute-close entry as the conservative bound. Scheduled events (EV-C)
  have detection-latency = 0 (timestamp known in advance).

## 4. Cells (sequence A→B→C→D, then deferred E)

### EV-A — Perp-onboarding drift  *(zero new data — runnable now)*
- **Source:** Binance USDⓈ-M perp first-candle ts ≈ `onboardDate`, from
  `free_v1/klines_1m` (521 syms). Provenance: data.binance.vision dumps.
- **Event:** a new perpetual goes live. n ≈ symbols onboarded *within* the
  2yr window = those with <2yr history ≈ **321** (150 with 1-2yr + 171 <1yr;
  HDATA_RESULTS.md §1).
- **Measure:** the listed symbol's own post-listing drift over {1,5,15,60}m,
  and whether first-N-min return predicts next-N-min (momentum vs reversal);
  optionally broad-alt-market reaction around the event.
- **Caveat (honest):** a perp listing of an already-spot-traded asset pumps
  less than a fresh spot listing, and the symbol has no pre-listing history —
  EV-A is the free *probe*; EV-B is the stronger proper listing test.
- **Runnable immediately** on downloaded klines, no pull.

### EV-B — Exchange-listing announcements
- **Sources (free):**
  - **Upbit** `api-manager.upbit.com/api/v1/announcements` — KRW listings,
    the documented "Upbit effect" (one of the strongest instant pumps);
    historical, precise ts.
  - **Binance** announcement archive (spot listing = pump, delisting = dump)
    — undocumented CMS endpoint / RSS.
  - **Coinbase** listing roadmap / asset announcements ("Coinbase effect").
- **Event:** announcement ts → match listed asset → Binance perp (if exists)
  → reaction surface {1,5,15,60}m.
- **First execution step:** verify each endpoint is reachable + returns a
  precise ts; pull the archive to GCS (`free_v1/orthogonal/events/listings/`).
- **Provenance:** endpoint, pull ts, ts-fidelity, n_events.

### EV-C — Scheduled events (macro surprises + token unlocks)
- **Macro** FOMC / CPI / NFP / PCE: public precise release ts; **actual**
  from FRED (free), **consensus** is the hard free piece (scrape
  forexfactory/investing economic calendar, or a free historical surprise
  set). Reaction on BTC/ETH. detection-latency = 0.
- **Token unlocks:** vesting schedules known ahead; on-chain via Etherscan
  free API, or token.unlocks / CryptoRank free tier. Cliff unlock →
  sell-pressure (directional). Reaction on the unlocked token's perp.
- **n:** macro ≈ 50-100 prints / 2yr; unlocks several / week across universe.

### EV-D — On-chain flows  *(higher frequency)*
- **Sources (free):** stablecoin mints/burns (USDT/USDC treasury via
  Etherscan / Tron RPC), exchange in/outflows, whale transfers (Whale Alert
  free tier), ETF net flows (Farside scrape, daily).
- **Event:** large mint / transfer / flow ts → reaction surface.
- Lifts the throughput vs EV-B/C.

### EV-E — Live fast-news feeds  *(DEFERRED, backtest-first)*
- Tree of Alpha / BWEnews websocket, CryptoPanic API. Thin/no usable
  historical archive → these give *forward* history once we log them.
- Not built until a backtest cell (A-D) shows a large reaction surface; when
  we go live we start the logger to accumulate the dataset EV-E needs.

## 5. Provenance contract (HDATA's only hard constraint + CLAUDE.md)

Every run records: `data_source` + endpoint, pull ts, `event_ts` fidelity,
`n_events`, `n_eff`, `split_method`, latency budget, cost basis (0.13%
strict), `repro_cmd`. A frozen per-cell pre-reg is appended to
`hypotheses.jsonl` *before* the run; the result row goes to
`experiments.jsonl`. Rigor lives in provenance, not in restricting axes.

## 6. Frequency reality (a characteristic, not a kill)

Structured events ≈ 1-3 / day across the ~500-symbol universe (EV-B/C
low-frequency, EV-D higher). This is a **high-conviction low-frequency
overlay**, not a ≥10/day replacement for the TA stack. Recorded explicitly
so the surface is read with throughput in view.

## 7. Status / execution order

1. **EV-A** — runnable now on `free_v1/klines_1m`, no pull. *(next)*
2. **EV-B** — verify Upbit/Binance/Coinbase endpoints → pull archive to GCS →
   event-study.
3. **EV-C** — macro calendar (FRED actuals + consensus) + unlock schedules.
4. **EV-D** — on-chain flow pulls.
5. **EV-E** — deferred (live), gated on a positive backtest surface.
