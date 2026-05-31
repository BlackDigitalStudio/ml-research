# HANDOFF — sub-60s 2×2 cascade: features, realistic execution data, maker labels

> For the next agent. Three things not obvious from the ledger/log: (1) exactly
> what features feed heads A and B, (2) where the realistic adverse-selection
> execution data is built, (3) what the "true maker" labels are. Authoritative
> code paths are cited so you can verify, not trust. Date: 2026-05-29.

## 0. The models (recap, one line)
2×2 cascade (`scripts/mamba2_cascade.py::Cascade2Stream`): **Model A** = vol-gate
(FLAT vs NON-FLAT, P(|rH60|≥13bp), per-symbol) and **Model B** = direction
(UP/DN, pooled top-3 on non-flat windows). Chosen cell = **sized GRU**
(`cell="stub"`); Mamba2 overfit at this n_eff (RESEARCH_LOG §13/§14). Deployable
weights: `gs://market-data-0998ac51/gru_models/{A_DOGE,A_ETH,A_LINK,B_pool}_gru.best.pt`
(+ `B2_ETH_hold.best.pt` = Stage-2 fine-tune).

## 1. Features — what A and B actually consume
**Both heads use the SAME two input streams** (only the head/objective differ;
B additionally has a symbol-embedding because it is pooled). Built by
`scripts/subs60_cache_build.py` → `gs://…/hd2_sub60_cache/{SYM}/{DATE}.npz`.

- **stream-1 `lob` (n_ticks, 80) f16** — raw 20-level L2 order book ticks, from
  the Rust `feature_builder --lob-out` (`rust_ingest/src/features.rs::lob_stream_80`).
  Layout per tick = `[bid_p(20) | bid_s(20) | ask_p(20) | ask_s(20)]`; prices
  encoded `(p − mid)/mid`, sizes `sign·log1p(|s|)`. This is the *fast tick* stream;
  the model gathers the LOB hidden state at the decision tick `t0`.
- **stream-2 `feat` (n_dp, 71) f32** — curated features at the 1-second decision
  grid. Composition (`subs60_cache_build.py:85`, `feat = concat([X, btc_lead, ToD])`):
  - **cols 0–63 = `feats_sub60` X(64)** — the corrected sub-60s feature set built
    from RAW by the Rust `feature_builder` (`rust_ingest/src/features.rs`,
    `NUM_FEATURES = 64`, line 923; exact column names = the FEATURE_KEYS order in
    that file). Includes the OBI ladder (L1/L5/L10/L20), trade-flow/CVD, funding/OI,
    liquidation features, and the repurposed point-to-point `eth_ret_{1,2,5}s`.
    (History: §12 of RESEARCH_LOG — the old book-only `features_v1` was replaced;
    do NOT reuse features_v1.)
  - **cols 64–66 = signed BTC-lead {5, 30, 60}s** — `btc_lead()` in
    `subs60_cache_build.py`: log-return of BTC mid over the trailing 5/30/60 s,
    ×1e4 (bp), aligned causally to the decision ts. Source = BTC mid from
    `gs://…/feats_sub60/BTC-USDT-PERP/*.npz` (keys `td`,`mid`). **Verified REAL,
    not placeholder-zero** (std ≈ 1.9 / 5.0 / 7.2 bp; `subs60_verify_btc.py`) and
    already an active input of the trained models (`in2.weight` is (·,71)). There
    is nothing to "re-add" — BTC lead-lag is in.
  - **cols 67–70 = time-of-day** — sin/cos of hour-of-day and sin/cos of the 8h
    funding cycle (`time_of_day()`).
- Labels in the same npz: `rH60` (signed 60s fwd book-mid logret, bp),
  `y60 = |rH60|≥13bp` (A target), `updn = rH60>0` (B target), `v60` (valid).
- Standardization (`ft_mu/ft_sd`, `lob_mu/lob_sd`) is saved INSIDE each
  `.best.pt`; load it from there — do not recompute.

If you add/remove a feature you MUST rebuild `hd2_sub60_cache` (the 71-dim
stream-2 Linear is shape-locked) and retrain — there is no runtime feature toggle.

## 2. Realistic execution data with adverse selection (the maker-sim)
The grid/fine-tune economics above use two fidelities — know which is which:

- **Optimistic MID-entry** (`scripts/subs60_gru_gridsim.py`): entry = book mid,
  guaranteed fill, no spread, no adverse selection. Good for ranking signal /
  discovering R:R, but its maker-maker numbers are an UPPER BOUND.
- **Realistic MAKER-entry (use this for execution truth)** — the **HUSDC** tooling
  on branch `claude/husdc-rev1` (worktree `C:/Dev/sb-husdc`; binaries built on the
  GCE VM at `/tmp/husdc/rust_ingest/target/release/`):
  - `rust_ingest/src/bin/build_samples.rs` reads **raw book + raw trades** and emits
    per-sample arrays incl. `book_paths (N,H,2)=[best_bid,best_ask]/tick`,
    `flow_paths (N,H,2)=[taker_buy_vol,taker_sell_vol]/tick`, `entry_q (N,2)=top-1
    queue`, `entry_book`, `mid_paths`, `sample_ts` (ms). Raw layout:
    `gs://market-data-0998ac51/raw/{book,trades}/exchange=BINANCE_FUTURES/symbol={SYM}/dt={DATE}/*.parquet`
    (book ~106 ms cadence ⇒ ~563 ticks = 60 s; trades flat schema `side/amount/price/timestamp`).
  - `rust_ingest/src/bin/grid_sim.rs` **maker mode** (flags `--flow-paths --entry-q
    --book-paths --entry-book`, knobs `--queue-mult`, `--entry-window-ticks`,
    `--maker-offset-frac`): a resting limit fills ONLY when realized taker flow
    reaches our level (touch / queue-clear) and **MISSES when price runs away** →
    adverse selection emerges from the path, not a parameter. Unfilled = NaN pnl +
    `*_filled_{long,short}` mask. Physics: `live_sim.rs::simulate_maker_entry`.
    Full operator guide: `research/MAKER_SIM.md` (on the husdc branch). Empirics
    (their ledger): touch fill ~0.99, queue ~0.56, adverse haircut ~0.2bp @45-60s.
  - **Our driver** `scripts/subs60_maker_grid.py`: per test day, `build_samples` →
    GRU inference (A=gate, fine-tuned B2=side) → match `sample_ts(ms)` to cache
    `dtd(ns)` → keep gated samples' maker arrays → `grid_sim` maker, queue-mult sweep.
    Persists `gs://…/research_runs/gru_makergrid/ETH_gated_arrays.npz` (book/flow/
    entry_q/entry_book/mid + side/Alog/Blog) so the entry-POLICY sweep
    (`scripts/subs60_entry_policy_sweep.py`, 18 policies) runs offline with NO
    re-extraction. KEY RESULT: passive maker entry FLIPS the directional edge
    negative (ETH HOLD60 gross −1.97bp touch, WR 0.45) — adverse selection.

## 3. "True maker" labels (vs the optimistic ones)
- The historical project default cache `pl/ps` were **TAKER** (entry=ask/bid) — a
  recurring false-positive source (RESEARCH_LOG §4). Don't trust raw cache pl/ps
  as maker.
- The sub-60s grid `rH60` is **mark-to-book-mid** — neither maker nor taker, just a
  signal mark. Fine for IC/capture; NOT an executable maker P&L.
- The ONLY honest maker label here is the **maker-sim output**: per-sample
  `pnl_{long,short}` (NaN on MISS) + `filled_{long,short}` from `grid_sim` maker
  mode (§2). EV = mean over FILLED by predicted side. That is the number to train
  Stage-2 on if you want "learn what you actually execute" with real maker fills
  (the B2 objective in `mamba2_cascade.py` already accepts per-window `pl/ps`
  sidecars; feed it maker-sim pl/ps instead of the hold proxy).

## 4. Open / pending (so you don't redo it)
- Positive fine-tune/grid/policy magnitudes carry **test=val inflation** (best-val
  early-stop + A-gate threshold both chosen on the same test set). Pre-deploy:
  **val ≠ test** (60-20-20 or CPCV/walk-forward), plus daily rolling-retrain
  (the static far-split decay is largely train-staleness).
- Stage-2 per-symbol fine-tune is done only for ETH (hold payoff). LINK/DOGE need
  bracket-payoff sidecars from grid_sim on TRAIN windows before their B2.
- Base `B_pool_gru` trained on 150/361 days (RAM cap) — undertrained on data; a
  full-data retrain is a known cheap lever.
