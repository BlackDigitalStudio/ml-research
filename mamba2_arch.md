# mamba2_arch.md — sub-60s 2×2 cascade (HD2 #3) architecture design

> Plan-first spec. Sync BEFORE building. Supersedes the naive "add stream-2 + set
> H=60s" approach: the old HD2Mamba2 was tuned for 3–30 min and several choices do
> NOT transfer to sub-60s. Operational truth: `mamba2plan.md`. Decisions below are
> user-confirmed (2026-05-28): B-objective = IC base + TP/SL fine-tune; stream-1
> context = SHORT (~minutes); horizon H=60s; first scope = top-3 DOGE/ETH/LINK.

## 0. Goal & cascade
Directionally-tradeable sub-60s edge, H=60s. Two SEPARATE models, each 2-stream:
- **Model A — FLAT / NON-FLAT** (vol-gate): P(|rH60| ≥ 13bp). Trains on ALL windows.
  This is the binding lever (Stage-0): push deployable precision (50–73%) toward the
  oracle so gross rises 2bp→~6.7bp → maker-positive. **PER-SYMBOL** (each symbol has
  ~0.9 M n_eff — plenty).
- **Model B — UP / DN** (direction): P(up). Trains ONLY on NON-FLAT windows
  (|rH60|≥13bp) → full capacity to direction, undiluted by flat. **POOLED across
  symbols** (+ symbol-embedding; per-coin BTC-lead already in stream-2) — B's n_eff
  is the binding constraint (§3), pooling is how we feed it.
- **Inference**: A-conviction selects top-q (~10 trades/day, read off curve, no gate)
  → B gives side → enter with TP/SL (RR 6:1). Deploy verdict = grid_sim TP/SL net
  @maker, NOT mark-to-mid.

## 1. Streams / cache (H=60s, decision grid = feats_sub60 dense 1s)
- **stream-1** `lob (n_ticks,80) f16`: raw 20-level L2 from the **extended Rust
  feature_builder** (one pass over the book it already reads), encoding == hd2:
  `[bid_p|bid_s|ask_p|ask_s]`, prices `(p-mid)/mid`, sizes `sign·log1p(|s|)`.
- **stream-2** `feat (n_dp,F≈71) f32`: curated, at 1s decisions —
  feats_sub60 `X(64)` + **signed BTC-lead {5,30,60}s** (per-symbol coupling) +
  time-of-day {sin/cos h, sin/cos 8h-funding}. (BTC-lead/ToD = cheap Python post-step.)
- `t0 (n_dp,) i64` = last LOB tick ≤ decision ts; `dtd` decision ts (ns).
- **labels**: `rH60` (signed bp), `y60 = |rH60|≥13` (A), `updn = rH60>0` (B), `v60`.
- Cache stays DENSE 1s (flexible); training strides/purges (see §3).

## 2. Context — SHORT (the big change vs HD2)
Sub-60s predictive memory lives in seconds–minutes, not tens of thousands of ticks.
- **stream-1 Mamba**: reset-period `L ≈ 2000–3000 ticks` (~3–10 min at 100–400 ms),
  warmup floor ~tens of seconds. (vs old L=6000–216000.) Gradient-ckpt unnecessary.
- **stream-2 encoder**: short temporal over the 1s decision sequence — small Mamba2
  (or TCN), receptive field ~few minutes (~60–180 steps).
- Rationale: oversized context wastes compute AND worsens overfit. (`L` left as a
  small optional sweep axis {600,3000,15000} if a later check wants the surface.)

## 3. n_eff / capacity budget — CRITICAL (rev24 lesson, now quantified)
Dense 1s decisions × 60s forward ⇒ adjacent labels overlap 59/60 ⇒ n_eff ≪ n_rows
⇒ overfit ("capacity-up CLOSED" was exactly this). n_eff ≈ time/H = days × 1440
non-overlapping 60s windows (clustering makes the EFFECTIVE count even smaller).

**Measured (top-3, full period, train ≈ 65%):**
| | non-flat base | n_eff A (all) | n_eff B (non-flat) |
|---|---|---|---|
| DOGE 362d | 13.4% | 521k | 70k |
| ETH 361d | 10.8% | 520k | 56k |
| LINK 244d | 16.9% | 351k | 59k |
| **top-3** | | **~1.39M (train ~0.9M)** | **~185k (train ~120k; ~30–60k w/ clustering)** |

**Capacity rule: params ≲ 0.1–1 × n_eff (noisy fin-data).** The naive d256/N4
(~2.26M params) is 7–75× too big ⇒ guaranteed overfit (esp. B).
- **TRAIN decision stride ≈ 20–30 s** (≈ H/2) cuts overlap; cache stays dense 1s.
- **Split**: purged + **embargo ≥ H** by day; train ~65%, embargo, test ~32%.
- **Capacity ≤ n_eff** is the primary defense (dropout/wd secondary). Do NOT out-scale.
- **Pool B** across symbols to grow its n_eff (top-3 → ~120k; all-8 → ~250k train).

## 4. Objectives
- **A (non-flat)**: class-weighted BCE on `y60` (imbalance ~4–14%; weight ≈ inverse
  freq or focal). Report vol_AUC + precision@top-q (Stage-0 surface). Calibrated.
- **B (direction)** — two stages (user 2026-05-29: Stage 2 trains on what it EXECUTES;
  R:R is NOT fixed a-priori — discovered per symbol):
  - **Stage 1 (base, pooled, rev10 lever)**: IC/capture `L = −corr(tanh(logit), rH60)`
    (≈ maximize E[tanh(logit)·rH60]) on non-flat only, pooled across symbols + sym-embed.
  - **Stage 2 (per-symbol fine-tune on the executed bracket payoff)**:
    - **2a — discover best R:R per symbol**: run `grid_sim` (Rust) tp×sl×timeout over the
      base model's directed windows → pick R:R maximizing net@maker (EV-vs-conviction curve).
      (Per-coin optimum drifts; universal argmax was tp0.30/sl0.05 but don't assume.)
    - **2b — fine-tune to that R:R** with a DIFFERENTIABLE bracket-payoff loss (TP/SL itself
      is non-diff, but grid_sim already emits per-window `pnl_long`/`pnl_short` at the chosen
      R:R): `L = −mean[ p·pnl_long + (1−p)·pnl_short − commission ]`, `p = sigmoid(logit)`.
      Gradient → model picks the side maximizing the ACTUAL net@maker payoff it will execute.
    - grid_sim used twice: (2a) find R:R, (2b) supply per-window payoffs as the fine-tune
      target. Reuse, not new infra.
  - **Calibrated conviction** (OOS) — recorded lesson: saturated logits kill selection.

## 5. Architecture (per model; A and B separate weights) — SIZED TO n_eff (§3)
- stream-1: `Linear(80→d1)` → `N1×Mamba2` (bounded L) → LayerNorm → hidden@t0.
- stream-2: `Linear(F→d2)` → `N2×Mamba2/TCN` (1s seq) → hidden@decision.
- fuse: `concat(h1,h2)` → small MLP → 1 logit.
- **Model A (per-symbol, n_eff ~0.9M) → target ~150–300k params**:
  `d1=128, N1=2, d2=64, N2=1` (~0.3–0.5 params/sample). ✅
- **Model B (POOLED, n_eff ~120k → 30–60k eff) → target ~30–80k params**:
  `d1=48–64, N1=1–2, d2=32, N2=1` + symbol-embedding (≤16-dim). Crochet-small.
- dropout 0.1–0.2, wd 1e-3. Pluggable cell (`stub` GRU for CPU smoke, `mamba2` GPU).
- NOTE: Mamba2/causal_conv1d may impose a d_model multiple-of-8/headdim constraint
  (old code kernel-locked d=256). If small d fails the CUDA kernel, fall back to a
  GRU/TCN cell for the small models — capacity-fit beats kernel preference.

## 6. Eval (deploy verdict)
- A → top-q conviction (~10/day) → B side → `grid_sim` TP/SL net @maker on the
  A-gated, B-directed windows.
- Benchmarks: oracle 6.7bp / deployable 2bp / OBI 1.6bp / maker floor ~6bp.
- Headline = conditional alpha surface (capture vs conviction percentile), not a gate.

## 7. Build sequence
1. **Extend Rust `feature_builder`** → emit 80-ch LOB tick stream (+ keep features).
   Build top-3 → `gs://…/hd2_sub60_cache/` (or stream dir).
2. **Python glue**: assemble cache = LOB stream + feats_sub60 + BTC-lead + ToD + labels.
3. **Model rewrite** (`hd2_mamba_stream.py` → 2×2): stream-2 encoder, A/B, IC objective,
   bounded L, decision-stride sampler, purged split.
4. **CPU smoke** (stub cells) — orchestration + shapes + objective sanity.
5. **Modal**: re-mint `hd1-gcp` (token TTL!), create `hf-token`, hydrate top-3 cache.
6. **GPU train** A then B (2 phases); cascade eval via grid_sim.

## 8. Risks
- n_eff (central) — decision stride + purge + capacity cap.
- TP/SL surrogate differentiability — start IC, add TP/SL carefully; keep IC fallback.
- stream-2 encoder choice — default small Mamba; ablate vs TCN/MLP if needed.
- GCP token TTL — re-mint `hd1-gcp` immediately before hydration, not earlier.
