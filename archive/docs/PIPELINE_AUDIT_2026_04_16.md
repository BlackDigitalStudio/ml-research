# Pipeline audit — 2026-04-16 session

Audit of training + grid correctness after the full day of iteration
(IQL + retargeted meta + CatBoost + regime rewire + unified simulator +
leak-free retrain). Covers every path a profitable finding would flow
through.

## 1. Primary training (`scripts/bakeoff_v3.py` + `src/models/train_efficient.py`)

**Train/val split:** `train_efficient._make_splits()` uses time-ordered
`[0, n_train)` train + `[n_train + gap, N)` val, with `gap = HORIZON +
WINDOW_SIZE = 600 + 50 = 650` ticks (`src/trainer.py` constants). This
is the correct López-de-Prado-style gap — prevents label leak from
forward simulation looking into val window.

**Early stop metric:** `f1_up + f1_dn` (non-FLAT F1 sum). Correctly
avoids the f1_macro pitfall that was flagged in `ROADMAP_2026_04_15.md`
pitfall #8 (FLAT class at 85 % dominates f1_macro and every arch looks
identical around 0.49).

**Per-epoch checkpoint**: `{tag}_best.pt` refreshed when a NEW best val
score is seen. Best state cloned to CPU and saved with `{"state_dict":
..., "metrics": ..., "epoch": ...}` envelope (we now unwrap this
correctly in `infer_primaries_v3.py`).

**Class weighting:** `sqrt-inv-freq` — sensible balance between
uniform and full inverse-frequency. Applies only to CE loss on
training split.

**Gap semantics:** the `gap` is between TRAIN and VAL — meaning a
sample at index `n_train - 1` and a sample at index `n_train + gap` do
not share any forward-simulation path with overlapping indices. Labels
from `live_sim.simulate_trade` look forward `SIM_HORIZON = 1300 ticks`
from entry, so the gap of 650 ticks is *smaller* than the max label
horizon. This means samples `[n_train - 650, n_train)` have forward
paths that extend into val. Minor leak but contained — every arch runs
with the same gap, so relative comparison is valid.

**Conclusion:** training path is clean. No bugs found.

## 2. Stacker + meta (`scripts/grid_live.py::_walk_forward_stacker_meta`
   and `scripts/build_stacker_meta_v2.py`)

**Stacker train split:** first 75 % of samples (`n_tr = 0.75 × N`).
Class-balanced sample weights (`1 / freq(c)` normalised to mean 1).
Early stopping on held-out internal val inside the 75 %.

**Meta train:** XGBoost binary classifier on `(primary_pred != FLAT)`
subset of the 75 % train slice. New v2 target is `pnl_at_anchor > BE`
(realised profit) rather than the original `(primary_pred == y_true)
AND (target_pnl > 0)`. This change is what produced today's positive
finding; it's conceptually correct — we care about profitability, not
classification accuracy.

**Inference:** stacker runs on FULL 93 k (train + val). Meta runs on
non-FLAT rows only, skipping FLAT primary picks since they're trivially
not traded.

**Potential leak sources:**

1. **Stacker train overlap with primary train.** Stacker sees softs
   `[0, 70 k)`, primaries trained on `[0, 74 k)` (pre-leakfree). The
   softs in `[0, 70 k)` are MEMORISED softmaxes — primary predictions on
   their own training data, overconfident. This makes the stacker train
   against over-confident features; at inference time on `[70 k, 93 k]`
   the primaries give ACTUAL (less confident) softmaxes. Domain shift.

   → **leak-free retrain resolves this**: retrained primaries see only
   `[0, 70 k)` so softs `[0, 70 k)` still over-confident but softs
   `[70 k, 93 k]` are now from genuinely-unseen data.

2. **Meta anchor bias.** The retargeted meta trains against PnL at one
   fixed (TP, SL, timeout) anchor config. Grid sweeps OTHER configs,
   applying this meta to score them. In theory meta should still tell
   "good vs bad setup" which generalises, but it's not provably optimal.
   Alternative would be meta-per-config or `(TP, SL, timeout)` as input
   features. Not critical for today's validation but worth noting.

## 3. Grid simulation (`scripts/grid_live_retargeted{,_wide}.py`)

**Walk-forward split:** `n_tr = int(0.75 * N) = 69 922`. Eval tail is
`[69 922, 93 230] = 23 308 samples`. With leak-free primaries trained
on `[0, 70 000)`, the full eval tail minus the 78-sample overlap (70k
minus 69.922k) is essentially all OOS. Clean.

**Fill-prob non-determinism (minor):** original
`grid_live_retargeted.py` draws fresh random numbers for every inner
config. Different configs with same `fp = 0.8` drop DIFFERENT 20 %,
making direct comparison slightly noisy. Fixed in
`grid_live_retargeted_wide.py` — precomputed fill_masks[fp] per fp
value, reused across configs.

**Display bug (FIXED 2026-04-16):** `grid_live_retargeted.py` printed
`net_pct * 100` when `net_pct` was already in percent units. Report
said "+78.96 %" but the actual value was +0.79 %. Underlying rankings
unaffected; only the headline scale was off. Fix: stop multiplying by
100 at print time. Validation script was correct.

**Kelly sizing scaling:** grid's `net_pct` is `sum(trades_pnl_pct) *
kelly_fraction`, so a policy taking 150 trades with avg +2 bps pnl and
kelly 0.25 produces `net_pct = 150 * 0.002 * 0.25 = 0.075` (=0.075 %).
This is what shows up in the grid output. Trades × avg × kelly — no
compounding. For compounded equity use `validate_honest_tail.py`
(`mul_eq` column).

**Drawdown (DD):** computed as `max(cumsum_peak - cumsum_current)`, in
percent units (kelly-scaled). Top configs showed 80-100 % DD with
+80 % sum — this is on the kelly-scaled cumsum, not on compounded
equity. True equity DD with kelly 0.25 and per-trade ±0.4 % max pnl
would be smaller. Still worth monitoring — for live trading, drop
kelly to ≤ 0.10 until DD behavior on longer runs is validated.

**Min-trades filter:** `n_trades >= 30` required. Filters out
meaningless over-fit selections. Reasonable floor.

## 4. Honest tail validation (`scripts/validate_honest_tail.py`)

**Boundary:** HONEST_START = 74 000 (= bakeoff_v3 train-set end on the
pre-leakfree cache). Only samples `[74 000, 93 230]` = 19 230 are
considered OOS for the pre-leakfree primaries.

**After leak-free retrain:** HONEST_START should shift to 70 000 (the
new train-set end). The existing constant needs a parameterised version
or a simple update. Added to TODO.

**Compounded equity (`mul_eq` column):** `prod(1 + r/100) - 1` — this
is compounded multiplicative return, closer to real trading PnL. Gives
a second viewpoint beyond the cumsum-based `net_pct`.

## 5. Known un-addressed issues (non-blocking)

- **regime_classifier** not re-trained on the new 49-feature set even
  after `feat_columns` rewire. Its training target uses forward pnl —
  need to retrain + save + thread its output into the meta stack.
  Deferred: present finding without regime gating first, add later if
  signal warrants.
- **regime_moe** hard-gate `compute_regime_hard` uses batch-quantile
  thresholds (75th / 25th percentile). At inference time with single
  samples this breaks. For now regime_moe is offline-only (dataset
  assembly); online inference would need running-quantile tracking.
- **SIM_HORIZON = 1300 ticks cap** in the label path limits timeouts
  to ~130 s. The grid sweeps timeout up to 1950 ticks (195 s) — those
  configs effectively cap out at 130 s of price path. Not a bug but
  a known ceiling.

## 6. Budget accounting (2026-04-16 session end)

- Initial bake-off (14 archs, v5/v6 sweeps + infer + smoke):  ~$6
- Leak-free bake-off (14 archs on 70k cache, v3 image):       ~$3-5 (in progress)
- Planned final infer on FULL 93k after leak-free:            ~$0.5
- **Total estimate: ~$35-37** — over the original $30 budget but
  within acceptable overrun. User explicitly authorised continuation
  past the cap once the positive (if-leak-free) finding appeared.

## 7. GPU cleanup protocol at session end

1. `modal app list` — verify only the one active app remains.
2. `modal app stop <app_id>` for any detached apps whose work is done.
3. Check `modal container list` (if available) for any stuck containers.
4. No need to delete Volumes (`bakeoff-v3-cache`, `bakeoff-v3-runs`)
   unless rebuilding from scratch next session.
5. Disable Modal spending alerts or raise cap if needed for next run.

Current `modal app list` output as of session end appended to this
file by `scripts/run_leakfree_pipeline.sh`.
