# TCN ALPHA CHECKPOINT (HD1) — conditions & architecture for best alpha

**Scope:** every TCN run in HD1 to date — the original 288-run event-clock
sweep (rev25-28), the rev29-30 capacity diagnostic, the Tier-1 reg/head
locks (rev39-44), the Tier-2 RF-matched context surface (rev45-47), and
the rev48-58 raw-LOB / densification / reg / context session. Frame is
CLAUDE.md exploratory: **lead with the conditional alpha surface and its
argmax**; the deploy-gate verdict is a clearly-labelled secondary
annotation, never the headline.

Metric throughout: OOS **rank_IC** (= AUC − 0.5) on the honest time split,
BCE-with-R1 objective, first-passage ±13bp label at horizon H. "best_test"
= max test rank_IC over the (no-early-stop) epoch curve; seed-sd from
3-seed cells where available (1 seed for the rev48+ dense probes unless
noted). block-bootstrap SE ≈ 0.005–0.007 per single-seed cell.

---

## 0. ECONOMIC REALITY (read first) — rank_IC is a PROXY, not money

**rank_IC is a rank-correlation search screen, NOT a tradeable edge.**
Translate it before celebrating any number:

- rank_IC → **AUC = rank_IC + 0.5**. The best TCN cell (BTC-H180,
  +0.0389) is **AUC ≈ 0.539**: given a random up-move and a random
  down-move, the model ranks the up-move higher only ~53.9% of the time
  (vs 50% coin-flip). The LTC-H300 dense argmax (+0.0347) is AUC ≈ 0.535.
  Most cells are AUC 0.51–0.53. This is a **1–4 percentage-point** rank
  edge over chance.
- **The label barrier (±0.13%) equals the round-trip cost floor (0.13%).**
  A correct directional call earns ~13bp at the barrier but pays ~13bp in
  costs → net ≈ 0; a wrong call loses ~13bp + ~13bp. To profit you need a
  win-rate well above ~60%. AUC 0.54 implies ~52–54% → **deeply
  unprofitable**.
- **Economic gate result (from the ledger, NOT estimated):**
  - TCN (HD1): **0 of 12 cells** pass `economic_pass_strict` OR
    `economic_pass_loose`. top-decile abs-move = 0.13% = exactly the cost
    floor (the model's most-confident decile does not separate from cost).
  - HM6 snapshot XGBoost baseline: **0 of 24 cells** pass either gate
    (AUC 0.51–0.55). The baseline is ALSO sub-economic on this label/cost
    regime → "TCN ≈ snapshot" means *both are sub-economic*, not "both
    are good".
  - Whole ledger: 13 of 100 experiments pass — ALL in the HA1/HA2/HA5/H3
    lines (LINK/SOL), and the HM2 rev1 audit flagged HA1/HA5/H3 as
    **magnitude/volatility-conflated objective artifacts** (XGBRegressor
    reg:squarederror on signed return rewards |move| fit, not direction;
    HA5's 0.70 AUC is tautological vs a volatility target). I.e. the
    "passes" are objective mis-specification, not clean directional alpha.

**Bottom line:** across the entire TCN line (and its snapshot baseline)
there is a **faint, statistically-nonzero directional signal (AUC
0.51–0.54) that does NOT clear the 13bp cost floor by a wide margin.** No
TCN config is tradeable as a standalone directional predictor at this
(label, cost) regime. The rank_IC surface below is the *conditional map of
that sub-economic signal* — useful for knowing where the signal is
strongest, but it is not money until something changes the (objective,
label, cost, or combination) regime.

**Caveat on the dense session (rev48-58):** we logged rank_IC only — the
decile-EV / win-rate / `economic_pass` fields were NOT computed for the
dense cells. By AUC analogy (dense +0.0347 = AUC 0.535 ≈ BTC-H180 sparse
0.537, which failed the gate), the dense cells almost certainly also fail,
but the exact dense economic numbers are an open eval (cheap re-run that
emits decile-EV/win-rate, ~$0.5).

---

## 1. The axes, and which ones move alpha

| Axis | Levels tested | Effect on alpha | Where shown |
|---|---|---|---|
| **Horizon H** | 180 / 300 / 600 s | **STRONG.** Monotone: H180 > H300 > H600 for EVERY symbol & representation | rev28 (12 cells), rev46-47 |
| **Symbol** | BTC / ETH / LTC / SOL | **STRONG.** BTC & LTC highest at short H; SOL lowest | rev28, rev46-47 |
| **Data density (STRIDE)** | 4 (sparse) / 1 (dense) | **MODERATE, then saturates.** STRIDE=4→1 lifts LTC-H300 ~+0.008–0.018; n_eff saturates at STRIDE=1 (ACF lag-1 0.16, n_eff/n 0.75) | rev50→rev52, density analysis |
| **Context length L** | 512 / 1024 / 1536 / 2048 | **WEAK / flat.** argmax-L is cell-dependent but the surface is nearly flat; >512 does not help at H300 (raw L2048 < L512) | rev46-47 (eng), rev56 (raw) |
| **Input representation** | 46-ch engineered-per-tick vs 80-ch raw-LOB + 6 globals | **~NEUTRAL.** Equal alpha at matched density+reg+cell (Δ ~0.002, in noise) | rev54/55 C1 |
| **Capacity W** | 16 / 64 / 128 | **~NEUTRAL when regularized + fed data.** W=128 advantage illusory under heavy reg | rev51, rev53, rev55 |
| **Regularization (dropout,wd)** | 3×3 grid on dense | **WEAK.** 9-cell spread 0.004 ≈ seed-sd | rev53 |
| **Head** | last-step vs mean-pool | not cleanly isolated; both work, comparable | rev45-47 (last) vs rev48+ (mean-pool) |
| **Streams** | single vs 2-stream(+globals) | not cleanly isolated; comparable | rev28 (single) vs rev48+ (2-stream) |
| **Depth D** | RF-matched to L | inert as independent axis (rev29 lock D=4 for Tier-0/1; RF-match for Tier-2) | rev28, rev29 |

**One-line summary:** alpha is set by **(horizon, symbol, density)**; the
**architecture/representation/capacity/context knobs are approximately
neutral** once data is sufficient and reg is sane.

---

## 2. The conditional alpha surface (best_test rank_IC by condition)

### 2A. Engineered-46ch per-tick input, single-stream, last-step
Original 288-run event-clock sweep (rev28; W=64, D∈{4,6}, dropout=0.1,
wd=1e-4, STRIDE=4, best-val config per cell, 3 seeds). rank_IC | Δ vs
HM6 snapshot baseline:

| Symbol | H180 | H300 | H600 |
|---|---|---|---|
| **BTC** | **+0.0373** −0.0046 | +0.0272 +0.0018 | +0.0172 +0.0003 |
| **LTC** | +0.0301 −0.0077 | +0.0272 −0.0132 | +0.0171 −0.0086 |
| **ETH** | +0.0230 −0.0005 | +0.0151 −0.0004 | +0.0107 +0.0008 |
| **SOL** | +0.0179 +0.0021 | +0.0104 −0.0059 | +0.0086 −0.0081 |

### 2B. Engineered-46ch, Tier-2 RF-matched (rev46-47; W=16, D per RF,
dropout=0.5, wd=1e-3, last-step, STRIDE=4, L∈{512,1024,1536}, 3 seeds).
argmax-L | full L-surface:

| Cell | argmax | rank_IC (±seed-sd) | L-surface |
|---|---|---|---|
| **BTC-H180** | L512 (D8) | **+0.0389 ±0.0015** | 512:+0.0389, 1024:+0.0375, 1536:+0.0377 |
| **BTC-H300** | L1536 (D9) | +0.0299 ±0.0005 | 512:+0.0297, 1024:+0.0288, 1536:+0.0299 |
| **ETH-H180** | L512 (D8) | +0.0244 ±0.0005 | 512:+0.0244, 1024:+0.0233, 1536:+0.0237 |
| **LTC-H300** [weak] | L1024/1536 | +0.0245 ±0.0085 | 512:+0.0240±0.0069, 1024:+0.0245, 1536:+0.0245±0.0100 |

### 2C. Raw-80ch LOB + 6 globals, 2-stream, mean-pool (rev48-58 session).
LTC-H300-L512 unless noted. 1 seed except rev49.

| Config | density (n_fit) | best_test rank_IC |
|---|---|---|
| rev49 W=128, drop0.5/wd1e-3 | STRIDE=4 (16,524) | +0.0055 ±0.006 (3 seeds; underfit, see rev51) |
| rev50 W=128, drop0.5/wd1e-3 | STRIDE=4 (16,524) | +0.0161 (1 seed, 24ep no-ES ceiling) |
| **rev52 argmax** W=128 drop0.1/wd1e-3 | STRIDE=1 (65,727) | **+0.0347** |
| rev52 #7 W=128 drop0.5/wd1e-3 | STRIDE=1 (65,727) | +0.0337 |
| rev54 C1 **W=16 ENGINEERED** drop0.5/wd1e-3 | STRIDE=1 (65,727) | +0.0318 |
| rev56 L=2048 W=128 (D10) drop0.5/wd1e-3 | STRIDE=1 (65,727) | +0.0299 |

---

## 3. ARGMAX — where TCN's alpha is highest

1. **Global argmax (seed-stable):** **BTC-H180, engineered-46ch, W=16
   RF-matched, L=512 (D8), STRIDE=4 → rank_IC +0.0389 ±0.0015** (rev47).
   The most alpha any TCN config has produced in HD1.
2. **High-vol short-horizon cells dominate:** BTC-H180 (+0.039), LTC-H180
   (+0.030), LTC-H300 (+0.027 eng / +0.035 raw-dense), BTC-H300 (+0.030).
3. **Best on our deep-dive cell (LTC-H300):** raw-80ch W=128 2-stream
   mean-pool, dense (STRIDE=1), dropout=0.1/wd=1e-3 → **+0.0347** (1 seed).
   Matches engineered W=16 dense (+0.0318) within noise.

**Conditions that maximize TCN alpha:** short horizon (H=180s) ≫ long;
high-vol liquid symbol (BTC/LTC) > ETH > SOL; dense sampling (STRIDE=1);
context L≥512 (more does not help). Representation/capacity/head/streams
are interchangeable at the alpha level.

**But in economic terms (see §0):** even the argmax (BTC-H180, AUC 0.539)
fails the economic gate (0/12 TCN cells pass; baseline 0/24). "Best alpha"
here means "least sub-economic rank_IC", not "tradeable edge". The
argmax tells us WHERE the faint directional signal concentrates (short-H,
high-vol symbols) — useful for siting any future objective/cost change —
not that BTC-H180 is deployable.

---

## 4. SETTLED — do NOT re-run these (already characterized)

- **Under-training is NOT the issue.** rev30: removing early-stop and
  training 120 ep does not rescue OOS; with light reg the model overfits
  by ep≈3 (train→0, val→∞). With heavy reg it plateaus. Either way OOS is
  capped. → don't re-test "train longer".
- **Capacity W is ~neutral.** rev51/53/55: W=128 (780k params) gives the
  same OOS as W=16 (15k) when reg is sane and data sufficient. train_loss
  stays ≈ ln(2) at BOTH (intrinsic to BCE-R1 on this binary task — OOS
  skill comes from small-weight majority direction, not train memorization).
  → don't sweep W expecting alpha.
- **Regularization is a weak axis on dense data.** rev53 3×3 grid spread
  0.004 ≈ seed-sd. → don't fine-tune dropout/wd for alpha; any sane point
  (dropout 0.1–0.5, wd 1e-4–1e-2) is within noise.
- **Raw-L2 input ≈ engineered input.** rev54/55 C1: +0.0318 eng vs +0.0337
  raw at matched density/reg/cell, Δ ~0.002 (~0.5 SE). → the rev48 raw-LOB
  /2-stream/mean-pool architecture is NOT a representation win; don't
  re-pitch "raw L2 will unlock more".
- **Context L is flat-to-declining beyond ~512 at H300.** rev56 raw L512
  +0.0337 > L2048 +0.0299; rev46-47 eng LTC-H300 flat 512–1536. → don't
  extend L for H300 expecting gains. (Short-H cells slightly prefer L512;
  long-H slightly prefer longer, but all within ~1 SE.)
- **Density saturates at STRIDE=1** for H300 (ACF lag-1 0.16, ~0 beyond;
  n_eff/n 0.75; asymptote ≈2× n_eff for ≥10× nominal cost). → don't push
  STRIDE<1 expecting material lift on a single (sym,H) cell.
- **TCN ≈ snapshot baseline (deploy annotation, secondary).** rev28
  refuted, rev44 "TCN ≈ snapshot fundamentally": vs the HM6 XGBoost
  snapshot baseline the per-cell deltas are near-zero or negative on
  sparse data. The dense-data deploy comparison is NOT yet done (no dense
  baseline) — see §5.

---

## 5. OPEN — not yet checked (candidate next steps)

- **Dense baseline comparison.** Our +0.034 dense LTC-H300 has no dense
  HM6 snapshot baseline to compare against (the baseline was sparse). To
  answer "does dense TCN beat dense snapshot" we'd need the snapshot model
  re-run on STRIDE=1. Until then the dense deploy delta is unknown.
- **3-seed validation of the dense cells.** rev52/54/56 are 1 seed each;
  the +0.002 arch gap and the +0.0347 argmax could be seed noise
  (seed-sd ~0.006). A 3-seed run on the top 2 cells would firm this.
- **Other (symbol, H) cells on dense data.** All dense work is LTC-H300.
  BTC-H180 (the global argmax at +0.039 sparse) on dense is untested and
  is the highest-prior cell for absolute alpha.
- **Multi-symbol / multi-horizon pooling.** The only identified lever for
  a real n_eff multiplier (labels ~independent across symbols/horizons),
  ~3-4× — untested.
- **Different objective/labeling (HM2/HM5 line).** A directional objective
  rather than representation tweaks; flagged earlier as high-leverage,
  untested in this TCN series.
- **block_size on dense.** placebo_ric +0.012 on STRIDE=1 (vs +0.002
  sparse) → inherited block_size(H)=4 too small for dense; bump ~4-8× for
  any significance test on dense (OOS point estimates unaffected).

---

## 6. Cost ledger (this session, rev48-58)

Total **$27.53** (workspace ksagachev). Per rev in
research/hardware_ledger.jsonl. Largest: rev52 dense build+sweep $10.51,
rev56 L-sweep $13.07 (incl. a build preemption double-run — LESSON: add
row-level build resumability for >45-min builds).

**Canonical sources:** research/hypotheses.jsonl (HD1 revs, append-only),
research/experiments.jsonl (confirmatory HM1 store), per-rev result JSONs
(rev50/52/54/56), research/hardware_ledger.jsonl. Unified browse copy:
research/full_ledger.db (regen: python3 research/build_full_ledger_db.py).
