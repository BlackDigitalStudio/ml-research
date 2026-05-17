# Research Log — scalper-bot

**Structured source of truth is now `research/` (JSONL ledger + SQLite).**
This file is the human narrative; the queryable asset is
`research/experiments.jsonl` + `research/hypotheses.jsonl`, contract in
`research/schema.sql`, plan in `research/PLAN.md`. §3 below is regenerable
via `python3 research/ledger.py frontier` — do not hand-maintain it once
new results land. The ledger *refuses* a result without its fee regime /
cache / split provenance (the chaos that cost us 3 false positives).

> **Infra state 2026-05-16 (critical — do not lose):** Contabo
> `root@84.247.154.229` is **LOST**. Every "LIVE on Contabo" cache (§8) and
> the entire `/root/.claude/projects/-root/memory/*.md` archive are **gone**
> — the §10 memory pointers and STRATEGY.md host references are historical.
> New topology: this repo's container = planning node (stdlib only); GCP
> `blackdigital.kz` 96 vCPU VM = compute node; `gs://blackdigital-scalper-data`
> (Cryptolake, 287.9 GB, persistent) + `gs://scalper-bot-research-data` =
> the only durable data. See `research/README.md` → *Infra reality*.

> **GCP recon verified 2026-05-16** (ADC as `blackdigital.kz@gmail.com`,
> project `project-26a24ad0-1059-4f73-93b` "My First Project"):
> - `gs://blackdigital-scalper-data` — **ALIVE**, `EUROPE-WEST1`, owned by
>   this project. Layout `features_v1/symbol=<SYM>/dt=<YYYY-MM-DD>/{features.npy,indices.npy}`;
>   symbols are Tardis-style (`BNB-USDT-PERP`, not `BNBUSDT`). The Cryptolake
>   feature asset **survived the Contabo loss**.
> - `gs://scalper-bot-research-data` — **403 / no access** for this account
>   (volaware checkpoints/oof; volaware was refuted — not blocking).
> - Compute europe-west1: `CPUS=200` (usage 0), `N2_CPUS=200`,
>   `DISKS_TOTAL_GB=2458`, **`PREEMPTIBLE_CPUS=0` → no spot** (on-demand
>   only), `C2_CPUS=8` (use **N2** for the 96 vCPU box).
> - Connection: ADC user creds in the ephemeral container's
>   `/root/.config/gcloud` — works this session, **not durable** across
>   container death (re-auth or move to SA-secret to persist).

> **Phase B first end-to-end run 2026-05-17 (`phaseb-20260517-003320`):**
> Lost Cryptolake pipeline **reconstructed and run on GCP** (cargo build /
> GCS / rust sim / XGB / grid / ledger all working). **H5 trust gate
> LANDED**: MAKER_FIRST entry integrated, `parity_ok=True` for LINK & SOL
> on 90 d of real data — every number is now MAKER-first honest. First
> numbers are **not a strategy**: the XGB gate is degenerate (~100 %
> take-rate) → LINK EV/tr −0.001 %, 1267 tr/day, net −23.6 %; SOL −0.001 %,
> 645 tr/day, net −15.8 % (`exploratory` in the ledger). H2 inconclusive
> (all PT/TS configs identical → never engaged on a trade-everything
> baseline). **Next bottleneck = model selectivity / trade selection
> (logged as H12, $0 eval-only).** Over-trading, not PT/TS, is the wall.

> **HA1 alpha screen 2026-05-17 (`phaseb-20260517-123148`, 8 alpha rows
> in `v_alpha`):** First execution-neutral signal map. `features_v1` is
> **leak-free** (placebo rank-IC ≈ 0 everywhere). Signal is **real but
> ultra-short-lived**: OOS rank-IC ≈ **0.087/0.073 @30 s** (LINK/SOL),
> decaying monotonically to ≈0.02 by 120-180 s; CI excludes 0 for 7/8.
> **Economically dead as a 60-180 s point prediction:** top-decile
> |move| = 3-10 bp, below even the loose 8 bp maker floor (7/8) and far
> below the 13 bp strict floor (8/8); `decile_monotonic = 0` everywhere.
> RL cannot manufacture 13 bp from a 3-4 bp edge → HA1 **refuted as
> posed**. Decisive redirect: the edge lives **faster than the 24 s
> sampling** — promote **HA4 (sub-24 s cadence)** + HA2 (target form);
> NOT execution/RL. (Run salvaged from an empty-id harness bug, fixed.)

> **The symmetry wall 2026-05-17 (MFE/MAE study, $0, LINK+SOL 5d).**
> Decisive structural result. Median max-favorable excursion: 60 s = 3 bp,
> 180 s = 6 bp, **600 s = 13 bp** (≈ strict floor only at 10 min). At
> 60 s only ~5-6 % of windows ever reach ±13 bp. AND it is ~symmetric:
> `P(MFE≥+13bp) ≈ P(MAE≤−13bp)` at every H (180 s: .235/.220; 600 s:
> .506/.462). → Wider TP/SL + longer timeout (why the old grid always
> "won" wider, and why the feature set is volatility-heavy) **scales the
> win and loss tails equally — it creates no edge**; that is why every
> wide-grid config netted ≈0 − costs. The bind: where moves clear cost
> (≥300-600 s) **direction is unpredictable** (HA1 IC→~0 by 180 s);
> where direction is weakly predictable (≤60 s) **moves are 2-4× below
> cost**. The two never overlap. No TP/SL/timeout/execution/RL fixes a
> symmetric-diffusion-vs-fixed-cost gap. Reconciles HA1 (short-horizon
> IC) with old research (TB-barrier favoured long/wide): different
> targets, both true. **Only escape consistent with the data:
> conditional asymmetry — a rare event/regime that breaks the MFE/MAE
> symmetry on the ≥cost subset (→ HA5).** HA4 (faster cadence) CLOSED:
> √t-trap (shorter window ⇒ smaller move ⇒ worse vs fixed cost).
> Cryptolake event data confirmed available at full fidelity for HA5:
> liquidations (side+qty+price, ~246/d), open_interest (~15k/d), funding
> (rate+mark+index, 1/s), trades (~242k/d) — see
> `research/CRYPTOLAKE_SCHEMA.md`.

> **HA5/HA6 — decisive negative 2026-05-17 (`phaseb-20260517-132629`,
> 6 alpha rows).** On the ≥cost subset (first-passage to ±0.13% within
> H∈{180,300,600}s), LINK+SOL: base `P(up|≥cost) ≈ 0.51-0.52` (symmetry
> confirmed). **head2 directional AUC 0.496-0.522 ≈ placebo 0.48-0.52 —
> indistinguishable from chance** for every conditioner (raw liquidation
> side/qty/count, ΔOI, funding rate/basis, all 59 microstructure feats)
> at every H. (`economic_pass_strict=1` on some rows is a NOISE ARTIFACT
> — cap-sign at AUC≈placebo is meaningless; status forced `refuted`.)
> **head1 ≥cost-feasibility AUC ≈ 0.68-0.71 — strong: volatility/regime
> (WHEN a big move comes) IS predictable, but DIRECTIONLESS (WHICH WAY
> is not).** → HA5 refuted; HA6 refuted (cascade head-2 has nothing to
> predict). **Triple-confirmed (HA1 sub-cost direction · MFE symmetry ·
> HA5 no conditional asymmetry): LINK/SOL LOB+event data contains
> predictable volatility but NO predictable direction at any
> horizon/conditioner.** A directional scalp here is structurally
> non-viable — not fixable by model/features/RL/execution. The cheap
> LOB-directional search space on these alts is **mapped and empty**.
> Open decision **HZ1** (strategy-class pivot, priority 1): non-
> directional vol-harvest (needs options / both-sided MM = different
> instrument), different signal source (cross-asset lead-lag / longer
> timeframe / higher-fidelity events), different asset class, or accept
> no directional alpha here. **Needs a human decision before more compute.**

> **SCOPE CORRECTION 2026-05-17 (over-claim retracted, user challenge).**
> The HA5/HA6 block above is correct about *what was measured* but the
> phrase "directional scalp non-viable / no directional alpha / pivot
> strategy class" **outran the evidence**. Everything run for direction
> (HA1, HA5/HA6, phaseb-003320) shares ONE unvaried slice: **GBT (XGB)
> on a single-tick flattened `features_v1` snapshot** (lost-provenance,
> unverified) + a few hand event aggs; no sequence/temporal model, no
> target-form/feature/cross-asset/ensemble variation, no HP search. The
> negatives bound **only that slice**, not achievable directional
> predictability. Critically, the only prior *honest* best result was a
> **sequence model (LINK TCN −0.040), never reproduced MAKER-first** —
> snapshot-GBT discards the order-flow *dynamics* where short-horizon
> direction lives. **HZ1 (pivot) RETRACTED → refuted.** Real open
> surface = **HD1** (priority 1): direction-improvement cluster —
> HA2 target-form + HA3 feature work ($0 on built cache), then a
> **temporal/sequence model screen** (the conspicuous untested gap).
> Not a strategy-class pivot; a model/representation pivot, still cheap.

> **FEATURES DECODED 2026-05-17 (user challenge; "opaque/unrecoverable"
> RETRACTED — was unverified laziness).** `features_v1` = this repo's
> raw-56 layout (NO DROP applied) + 3 Cryptolake ext = 59; cols 0-55
> mapped to exact FEATURE_KEYS names, empirically verified by value
> signatures (`research/CRYPTOLAKE_SCHEMA.md`). **Material new finding:
> ~10+ columns are DEAD in this build** — every cross-asset/ETH feature
> is 100% zero (eth_*, cross_exch_mom, bybit/okx/bitget/gateio_net_flow)
> and several constant (large_order≡1, spoof≡1, sweep≡0,
> long_short_ratio≡0, liquidation_proximity≡0.015). Models in HA1/HA5
> saw **~46 live features; the entire cross-asset dimension was
> literally zeros.** This narrows every prior negative further AND makes
> **H3 concretely actionable** (not "rebuild cache" — the BTC-lead slots
> physically exist and are empty; BTC raw is in the same bucket to fill
> them). HA5 caveat: its hand-built liq/OI/funding conditioners largely
> DUPLICATED already-live cols (cvd, ofi_*, funding_*) — the genuinely
> absent axis was cross-asset, untested. Decode reopens H3/HA3/HD1 with
> real names.

> **H3 BTC-lead — clean test, refuted 2026-05-17 (`phaseb-20260517-142815`,
> 8 alpha rows).** Filled the empty cross-asset dimension with 8 causal
> BTC-lead aggs (ret 5/30/60/120s, signed-flow 30/60s, rv60s, cumsgn60s);
> `btc_cols_live=1.00` (valid test, isolated base vs +BTC). Δ(rank-IC)
> nil: only LINK h30 +0.009 (< pre-reg 0.01 bar, not on SOL +0.0009),
> 4/8 cells negative, `economic_pass_strict=0` every cell (btc top|move|
> 0.038-0.102% < 0.13% floor; eL=1 only at h180 where IC≈noise). Old
> "eth +6.68%" does not carry to MAKER-first LINK/SOL here. **Now all
> three snapshot-GBT direction axes are negative: intra-asset
> microstructure (HA1) · event/regime (HA5) · cross-asset (H3).** The
> single never-varied axis = **representation: temporal/sequence model**
> (snapshot discards LOB dynamics; the only prior HONEST best was a TCN,
> never reproduced MAKER-first) → **HD1 rev3 priority 1**. Do NOT
> re-claim "no directional alpha" until the temporal axis is tested.

> **METHODOLOGY CORRECTION 2026-05-17 (user challenge — HM1 canon).**
> Root cause of 3 false negatives (HZ1, HA5-scope, H3): I used the
> discrete `economic_pass` gate as a per-search keep/kill. Wrong. In the
> search phase every block is sub-cost alone until stacked; selection is
> by **robust marginal `delta_ic` vs a declared `baseline_ref`**, not the
> economic gate (now canon in schema/README/PLAN/ledger.py + new fields
> `baseline_ref,delta_ic`). Re-classified: **HA1 is NOT dead — it is the
> leak-free directional signal baseline (~0.08 rank-IC @30s) to stack
> on**; **H3 BTC-lead is a weak symbol-inconsistent marginal contributor
> (LINK h30 +10 % rel IC), RETAINED for stacking, not refuted**; HA5 ≈ 0
> marginal over already-live cols. `economic_pass_*` = recorded
> distance-to-deploy + deploy gate for a FINAL candidate only;
> `refuted(alpha)` := Δ within noise/placebo. Prior "all axes
> negative / dead" framing superseded by HM1.

> **OBJECTIVE AUDIT 2026-05-17 (user diagnosis — HM2 canon).** Why is
> volatility strongly predicted but direction a coin-flip? Verified in
> code: HA1/H3 use `XGBRegressor(reg:squarederror)` on signed
> fwd-return → the reward is **magnitude/volatility fit, not
> direction** (squared loss dominated by large |move|; small-move sign
> ≈ free). Headline "success" = rank-IC (magnitude-conflated). HA5
> head1 was trained ON a volatility target (reached ≥ cost) → its 0.70
> AUC is **tautological**, not a separate signal. HA5 head2 used a
> directional objective but only on the degraded ≥cost subset with
> duplicate conditioners. ⇒ **the directional ceiling of these
> features was never cleanly measured with an objective that rewards
> direction**; "direction = coin-flip" is partly an objective artifact
> (rank-IC > 0 proves a small real directional component MSE
> under-extracts). Fix the OBJECTIVE before the representation — a
> sequence model on MSE-return inherits the same bias. **HA2 sharpened
> → priority 1** (directional-objective screen: sign-classifier /
> vol-normalised target vs the HA1 MSE baseline, judged by directional
> AUC + Δ per HM1); temporal demoted to "only if HA2 still ~0.5".

> **HA2 directional two-head — REFUTED as posed; HM2 self-corrected
> 2026-05-17 (`phaseb-20260517-154705`, 6 rows).** Fixed objective
> (directional logloss) + per-head scope (head2 trains on ≥cost subset
> only) + real BTC+ETH cross-asset inputs — all at once, never before.
> Result: head2 **base** (features_v1, correct objective+scope) AUC
> **0.505–0.512 ≈ coin-flip on BOTH symbols**. ⇒ **HM2 partially
> REFUTED**: the objective *was* magnitude-rewarding (still canon for
> future agents) but fixing it did **not** reveal hidden directional
> alpha — the ~0.51 snapshot ceiling is **robust across
> feature-dimension (HA1·H3·HA5) AND objective×scope (HA2)**. +BTC+ETH:
> LINK Δ≈0 (flat), SOL Δ +0.009→+0.016 AUC ~0.52–0.525 placebo-clean =
> a **faint SOL-only sub-cost whisper** (mirrors H3, not a lever).
> Pre-registered bar (AUC>0.52 BOTH symbols) failed on LINK → not
> confirmed. The runner auto-`confirmed` SOL300/600 via an
> economic-cap-sign-at-chance-AUC artifact (HM1 violation) — caught,
> forced `exploratory`, runner patched, `v_alpha_audit=0`. The single
> never-varied axis is now unambiguous: **REPRESENTATION
> (temporal/sequence)** — HD1 priority 1; the only prior honest best
> was a TCN, never reproduced MAKER-first. If a sequence model also
> ≈0.51, directional alpha is genuinely absent at scalp horizons →
> deliberate instrument/cost pivot (not before).

> **HM3 ACCEPTED PRIOR + queue convergence 2026-05-17 (user).** Decision:
> we care only THAT sequence adds directional lift (a known prior — lit.
> + this project's own historical Mamba/TCN ≫ XGB), not how much. The
> temporal/sequence screen is therefore **descoped (not data-refuted)**;
> HD1/H1/H8/H4-seq removed from the active queue, HA6 aligned refuted.
> **Corollary (must not be lost):** the snapshot-bound ≈0.51 directional
> negatives (HA1·H3·HA5·HA2) are limited by the snapshot representation;
> with sequence-superiority an accepted-but-unquantified prior they are
> **NOT** a proof of "no directional alpha". State of play: the cheap
> *snapshot-directional* search on LINK/SOL is **mapped & exhausted**;
> head1 volatility ≈0.70 (real, directionless); the symmetry/cost wall
> (≤14 bp moves vs ~13 bp cost) stands. No remaining cheap directional
> lever in queue. Next is a STRATEGIC decision (well-evidenced now, not
> premature HZ1): adopt-sequence-and-build vs class/instrument pivot vs
> re-scope testbed — a human call, not another screen.

> **HA7 SCOPE SWEEP — pre-registered 2026-05-17 (user challenge).** The
> "snapshot-directional search **exhausted**" framing in the HM3 block
> above was an **over-claim on the SCOPE sub-axis** (my recurring error
> pattern; user caught it). Scope is a *direct* lever on **conditional**
> predictability (≠ objective tuning, ≠ unconditional AUC): HA1/HA5/HA2
> tested essentially **one** scope point — the pooled ≥cost head2 (~0.51).
> One hard constraint reshapes "sweep more" into "sweep the *uncovered*":
> for a GBT **feature-inclusion ≥ hard-subset** (the tree carves the
> conditional itself), and HA5/HA2 already fed liq/OI/funding/features_v1
> as *features* at ≈chance → broad-conditioner scope is covered. The
> genuinely uncovered, pre-registered axes (`scripts/ha7_screen.py`,
> FROZEN, no post-hoc DOF): **(A)** regime-bucket head2 — heterogeneity /
> rare-regime loss-dilution (11 cells); **(B)** alt barrier/target
> definitions (T0±0.13 control / T1±0.25 / T2 asym / T3 signed-deadband);
> **(C)** head1-gated cascade (realistic deploy scope). **Strict bar:**
> block-bootstrap |AUC−0.5|/SE > Bonferroni z\*(α0.05/M) ∧ placebo≈0.5 ∧
> AUC>0.5 ∧ **same cell on BOTH symbols**; no auto-`confirmed` (HM1).
> Orthogonal to **HM3** — HM3 descopes the *sequence/representation*
> prior; HA7 completes the under-tested *scope* axis **within snapshot**.
> HM3's corollary (≈0.51 ≠ "no alpha", representation-bound) **unchanged**.
> If HA7's strict both-symbol bar is not met → snapshot-scope is then
> genuinely closed (still not "no alpha"). Pre-registered & committed
> **before** the run (HA7 rev1, HM2 rev3, HD1 rev9).

> **HA7 RESULT — REFUTED as posed 2026-05-17 (`phaseb-20260517-173640`,
> 6 alpha rows, well-powered: LINK n_oos in thousands).** The strict
> pre-registered both-symbol bar (same axis·cell on LINK&SOL, Bonferroni
> z\*≈2.94–2.96, placebo≈0.5, AUC>0.5) is met by **0 of ~90 cells**.
> **(A) regime buckets:** none cross-symbol; best are single-symbol
> sub-Bonferroni (SOL `oishock=0` z≈2.4–2.76); the genuine rare-regime
> `liqburst=1` was **underpowered every symbol×H** (n_oos 19–315) — too
> rare to train at this cadence, itself informative. **(B):** T1±0.25
> SOL-H180 looked huge (z=3.40) but **placebo 0.538 → guard rejected**
> (the sentinel worked); **T2 asym +0.13/−0.20** is a *weak
> cross-symbol-consistent* sub-threshold marginal (LINK&SOL H300–600
> z 2.07–2.87, clean placebo, AUC≈0.522–0.531, ΔIC≈+0.02–0.03 vs T0) —
> **retained for stacking per HM1, NOT a lever**. **(C) cascade:** lone
> SOL-H600 pass (z=3.36, placebo clean) with **no LINK mirror at any H**
> → isolated, within false-positive expectation for the family; the bar
> correctly blocks it. **Conclusion (matches HD1-rev9 contingency
> verbatim):** the pooled ≈0.51 is **not** a pooling artifact; scope
> conditioning does not recover cross-symbol directional alpha on
> LINK/SOL snapshot+events. The "exhausted" framing is now **earned**
> (systematic, not the 1-point over-claim). HM3 corollary **intact**:
> representation-bound, NOT "no alpha". VM auto-deleted (hard cap).
> **[rev10's "no cheap screen remains → strategic fork" is RETRACTED —
> see HM4 / HA7→HM4 block below; HA7 closed scope-as-subset/label/gating,
> NOT the reward/loss-structure axis.]**

> **HM4 — REWARD/LOSS-STRUCTURE axis is OPEN 2026-05-17 (user challenge,
> 2nd same-pattern catch).** "Representation is not an axis of how the
> model is rewarded/punished — did you really exhaust *that*?" Correct
> answer: **no.** The training reward/loss structure was sampled at
> **~3 points** total — MSE(signed-return) [HA1], logloss(sign) [HA2a],
> MSE(vol-norm) [HA2b]; **HA7 added zero** (every cell = plain logloss
> `_xgbc`, only subset/label/gating varied). I folded that 2–3-point
> sample into "objective exhausted" (HD1 rev10) and **deflected the
> remaining search onto representation/HM3** — the exact HA7 over-claim
> pattern, repeated. **Genuinely never tested** under MAKER-first /
> honest-OOS / Δ-AUC-vs-HA1 judging: (i) error-weighting by economic
> |move| or signed-PnL, (ii) ranking / IC objective (`rank:pairwise`;
> HM2-rev1 named "sign-weighted/IC loss", never run), (iii) asymmetric
> up/down misclassification cost (motivated by T2_asym — the lone
> non-null HA7 thread). This axis is **distinct from representation**
> (HM3) and from **scope-as-subset** (HA7), is **OPEN**, and is **cheap**
> (same harness). HD1 rev11: priority-1 cheap screen = the reward/loss
> sweep, **pre-register then run on user go** (not launched mid-challenge).
> The strategic fork is **NOT** yet reached.

> **HR1 reward/loss sweep — pre-registered & launched 2026-05-17 (user
> go via AskUserQuestion: all R1–R4, HM1-standard bar).** Closes the
> HM4 gap. Same testbed/scope as HA2/HA7 R0 (LINK&SOL, features_v1+conds,
> ≥cost subset, up-first, H{180,300,600}); the **only** thing varied is
> how error is scored/weighted: **R0** plain logloss (anchor == HA2a/HA7
> ~0.51) · **R1** weight ∝ |r_H| (clip p99) · **R2** weight ∝
> max(|r_H|−cost,0) (economic — sub-cost moves get ≈0 weight) · **R3**
> `rank:pairwise` on up (the IC/AUC-surrogate HM2-rev1 named, never run)
> · **R4** asymmetric up:down cost = 0.20:0.13 (T2_asym → into the loss).
> **Bar (HM1-standard, frozen):** a reward point is a robust marginal
> iff paired block-bootstrap (AUC_R−AUC_R0) > 2·SE ∧ >0 ∧ placebo≈0.5 ∧
> beats R0 by >noise on **both** symbols; not economic-gated; no
> auto-`confirmed`. Distinct from HM3 (representation) and HA7 (scope) —
> both unchanged. Pre-registered & committed **before** the run (HR1
> rev1, HD1 rev12).

**Last updated:** 2026-05-17 (HR1 pre-registered & launched — reward/loss-structure sweep R0–R4 [|move|-wt / econ-wt / rank:pairwise / asym-cost], HM1-standard both-symbol bar, the HM4-open axis; distinct from HM3-representation & HA7-scope [both unchanged]; verdict pending; prior: HM4/HD1-rev11, HA7-result, HM3, HA2/HM2, HM1).

---

## 1. Glossary — fixed definitions (do not redefine ad-hoc)

| Term | Definition |
|---|---|
| **base rate** | `P(pl_long > 0)` under correct TAKER fees, per-symbol. BTC canonical ≈ 16% (UP+DN); alts (SOL/LINK/etc.) 10-13%. |
| **WR (win rate)** | Fraction of TAKEN trades with **direction-aware realized net PnL > 0**, after TAKER commissions. Not label-WR, not `prec_NF`. |
| **prec_NF** | Classification precision on non-FL labels. **On canonical TB labels** (`y=UP iff pl>0 AND pl>ps AND not fill_miss`) `prec_NF ≡ WR` by construction (Bug B, 2026-05-09). Both metrics are valid; "lift" must be cited with the base it's measured against. |
| **EV/tr%** | Mean realized net PnL per trade, after commissions, % of notional. **Primary frontier metric.** |
| **tr/day** | Trades per calendar day on holdout. Cited alongside EV/tr to anchor the operating point. |
| **net%** | `EV/tr% × n_trades × kelly_fraction`, % of capital over holdout window. Sensitive to Kelly; **never compare nets at different `k`**. |
| **lift** | `WR / base_rate`. Specify the base (canonical-label vs `P(pl_long>0)` — different numbers). |
| **honest val→test** | Threshold picked on val, applied on test. Anything else is post-hoc bias. |
| **CPCV** | Combinatorial Purged Cross-Validation (López de Prado), N=6, k=2 → 15 combos, embargo=0.5%, purge=label_horizon. Yields PBO. |

## 2. Canonical constants

| Constant | Value | Source |
|---|---|---|
| TP_PCT | 0.20% | `CLAUDE.md` strategy spec |
| SL_PCT | 0.10% | `CLAUDE.md` strategy spec |
| SIM_HORIZON | 1300 ticks (130 s) | `src/trainer.py:56` (env-overridable) |
| R:R | 2:1 | TP/SL ratio |
| TAKER commission, win-side | 0.07% round-trip | `rust_ingest/src/live_sim.rs:66` |
| TAKER commission, loss-side | 0.10% round-trip | `rust_ingest/src/live_sim.rs:67` |
| Break-even WR (no commissions) | 33.3% | `1/(1+R)` |
| **Break-even WR (TAKER, full TP/SL outcomes)** | **~40%** | TP+1.0bp − SL−0.85bp commission drag |
| **Break-even WR (TAKER + timeout asymmetry)** | **~42-44%** | timeouts skew loss-heavy in practice |
| FEATURE_KEYS | 49 (old) / 55 (cryptolake) | `src/features.py::FEATURE_KEYS` |
| Holding zone | 60-180 s, **hard floor 60s** | `strategy_timeframe_constraint.md` |

## 3. Frontier — EV/tr at fixed tr/day, by epoch

**The single comparison table.** Each cell = best honest `EV/tr%` at that operating point.

| Date | Setup | Symbols | EV/tr @ best | EV/tr @ ~2 tr/d | EV/tr @ ~10 tr/d | EV/tr @ ~30 tr/d |
|---|---|---|---:|---:|---:|---:|
| 2026-04-29 | xgb solo (49 feat) | BTC | −0.054% | n/a | n/a | −0.039% |
| 2026-05-02 | 8-model vol-scaled + hybrid maker/taker | BTC | **−0.080%** (n=102, ~5/d) | n/a | n/a | n/a |
| 2026-05-09 | cascade_180s canonical 952K | BTC | −0.061% (n=21) | −0.22% | −0.30% | −0.30% |
| 2026-05-09 | per-symbol cascade XGB (Cryptolake, **TAKER labels**) | 8 syms | −0.027% (DOGE) | varies | varies | varies |
| 2026-05-10 | per-symbol XGB regression grid (Cryptolake, **MAKER-first labels**) | 8 syms | **+0.036%** (ETH n=27) | **+0.030%** (DOGE n=8) | n/a | n/a |
| 2026-05-10 | per-symbol XGB grid (Cryptolake, **MAKER-first** revalidation, DOGE step=5.5s) | DOGE | **−0.047%** (best thr +0.06) | n/a | n/a | n/a |
| 2026-05-10 | LINK TCN lookback=1000 (Cryptolake, **TAKER labels**) | LINK | **−0.040%** | −0.040% | n/a | n/a |
| 2026-05-10 | SOL TCN lookback=1000 (Cryptolake, **TAKER labels**) | SOL | −0.077% | n/a | **−0.077%** | n/a |
| 2026-05-10 | SOL Mamba lookback=3000 (Cryptolake, **TAKER labels**) | SOL | −0.065% | −0.065% | n/a | n/a |

**Reading the frontier:**

- BTC-only era → best EV/tr ~ −0.06% to −0.08% on operating points with n_trades > 100.
- Cryptolake (alts, sequence models, **TAKER labels**) → best EV/tr **−0.040%** (LINK TCN). At matched coverage, ~3-4× improvement vs old.
- Cryptolake (alts, **MAKER-first labels**) → best EV/tr +0.036% ETH was found in one session, **but revalidation with maker-first labels integrated into pipeline showed DOGE = −0.047%/tr** (the +0.036% was likely TAKER-label artifact baked into the build script default).
- **No setup has confirmed positive EV/tr under realistic MAKER-first labels** as of 2026-05-12.

## 4. Resolved confusions (high-cost-to-relearn)

| Confusion | Resolution |
|---|---|
| "WR was 76-85% in old runs" | **Label-artifact** (2026-04-14): `target_pnl > 0 ⟺ y != FLAT`. Was measuring "fraction of taken samples whose TB label is non-FLAT", not realized direction-aware PnL. After fix, honest WR ≈ 20%. |
| "WR ≡ prec_NF on canonical labels" | **By construction** (2026-05-09 Bug B): `y=UP iff pl>0 AND pl>ps AND not fill_miss` → `pred==y ⟺ realized>0` for non-FL. Both are valid metrics but it's the same number on canonical labels. |
| "DOGE +3.9%/month, +19.6%/month TAKER" | **Wrong fees** (COMM_WIN=0.04, COMM_LOSS=0.07 are MAKER round-trip). Correct TAKER no-VIP = 0.07/0.10. Adjusted: +3.9% → −1.5%/month under correct fees. |
| "CPCV best_total = sum across 15 combos" | **5× overlap inflation**: each unique trade appears in 5 of 15 combos at N=6/k=2. Correct: `sum_per_30d = (best_total / 5) × 30 / days_total`. |
| "phase56 +1.30%/30d aggregate positive" | **Labels were TAKER** despite intending MAKER-first. Build script default `entry_long=ask, entry_short=bid` = taker entry; maker-first relabel was applied separately but never copied back to cache `pl/ps`. Real maker-first revalidation: DOGE −1.4%/month. |
| "Vol-scaled grid WR = 0.6-3%" | **Kelly multiplier bug** (2026-05-02): `cfg.tp/cfg.sl` were multipliers but Kelly formula treated them as percentages → `kelly_size = 0` for all → false WR. Fixed via per-sample Kelly in `compute_metrics`. |
| "Maker fill check missing" | **Fixed 2026-05-02**: added `entry_fill_window_ticks` to `LiveSimConfig`. At fill_window=10 (1 s @ 100 ms), **77.6% of samples don't get maker fill** — adverse selection is brutal. Real edge was 7× worse than optimistic backtest showed. |
| "n_folds=1/2 in v8 skips folds" | **Documented, not fixed**. For `n_folds=K`, last fold has `te_end=n=va_end` → skip. Workaround: use `n_folds≥3`. |
| "v3-v8 sequence training used wrong early stop" | **Critical bug, not fixed in v8**: unweighted BCE for early stopping on class-imbalanced binary. Old methodology used `f1_up+f1_dn` (NN) or `prec_NF × sqrt(coverage)` (Optuna). Must fix before next training run. |

## 5. What works structurally

- **CPCV proper** (N=6, k=2, 15 combos, embargo, purge) — reliable validation; PBO calc works.
- **Direct PnL regression** (XGBRegressor on `pnl_long`, `pnl_short`) — best ML baseline. Beats cascade variants, MLP, pooled.
- **Liquidation features** — confirmed `10.7% combined gain` on s2 (UP/DN). Rank 11/15/17 of feat importance.
- **Per-symbol training** — beats pooled XGB (delta ~0) and pooled MLP (loss plateaus epoch 1).
- **Cascade XGB** (FL/NON-FL + UP/DN) — `+3.5/+4.4/+4.1/+3.4 pp` prec_NF vs single 3-class per horizon, but pairwise correlation single↔cascade = 0.977-0.980 → diversity benefit marginal.

## 6. What does NOT work (don't re-try without new reason)

- Pooled XGB / MLP cross-symbol — washes out symbol-specific patterns.
- Isotonic calibration on OOF subset — adds variance more than fixes bias.
- L2 stacker xgb-on-softmax over 4 correlated archs — stacker can't beat AVG when correlation > 0.97.
- Cost-aware loss variants (B: CE×|pnl_diff|; A: y_net relabel) — −4 to −7 pp prec_NF. Re-labels boost non-FL coverage at the cost of precision.
- LdP abstention meta on 23 regime features — OOF lift 5 pp; zero holdout transfer.
- Derivable directional features (oi_velocity, mark_basis) — zero lift, in-noise.
- Winsorize @ p99.9 — 0 effect on prec_NF; XGB hist-binning robust to outliers.
- Binary `P(profit > 0)` classifier — too coarse; ≈ base rate WR.

## 7. Active hypotheses (ordered by expected lift)

| # | Hypothesis | Expected lift on EV/tr | Cost |
|---|---|---|---|
| 1 | **Mamba/sequence models on lookback=10K-100K** | unknown; SSM strength emerges at long-context, untested | $50-100 |
| 2 | **Inner PT/TS params via fused grid_sim** (partial_tp_progress, trailing_step{1,2}_progress/_sl_ratio, trailing_step1_sl_floor_pct). Structurally addresses the main 2026-05-09 bottleneck — full-SL losses (-0.14% net) dominate timeout-wins (+0.005-0.06%). Partial TP locks gain on winning side, trailing SL closes earlier on losing side → asymmetric tail compresses. **Not tested on Cryptolake-era data or under MAKER-first labels.** Fused `grid_sim` binary already supports the 11-param sweep (~30s per 100K configs on Contabo). Wrapper: `src/rust_bridge.py::simulate_labels_grid`. | likely 0.02-0.05% per trade if winning-side avg moves from timeout-drift (~0.04%) toward TP-hit (~0.16%) on subset of trades | model already trained → eval only, ~1 hr Contabo |
| 3 | **Cross-symbol BTC-lead features for alts** (BTC depth/aggTrade as feature for SOL/LINK/etc. models) | OLD: eth_features 6.68% combined gain on s2 → similar order for BTC-lead | cache rebuild |
| 4 | **Multi-axis ensemble**: Mamba + TCN + Transformer + XGB → L2 stacker | low ensemble diversity historically, but archs are different families | model training |
| 5 | **MAKER-first labels integrated in build pipeline** (currently a P0 blocker: cache `pl/ps` are TAKER by default) | aligns backtest to live execution — likely reveals losses we currently hide | code change ~30 min + cache rebuild |
| 6 | **Wider TP/SL with longer timeout** (e.g. 0.30/0.15/600s) — reduces timeout-asymmetry bias | unknown; tested only narrowly | cache rebuild |
| 7 | **SSL pretraining on raw LOB** → fine-tune triple-barrier | unknown, novel for this dataset | $150-250 (8× H100) |
| 8 | **Cross-pair attention** (Mamba on alt + BTC LOB simultaneously) | unknown | $50+ |
| 9 | **Liquidation data — higher-fidelity** (current is frequency-only, not maker/taker-side breakdown) | likely 1-2 pp WR | data procurement |
| 10 | **Dynamic TP/SL per regime** (wide TP when liq-imbalance high) | structural fix for asymmetric loss | medium |
| 11 | **VIP fee tier** (0.04 maker / 0.07 taker) | +0.03% per trade (mechanically) — moves break-even WR down 2-3 pp | requires $1B+/month volume = downstream |

## 8. Cache inventory

| Cache | Status | Notes |
|---|---|---|
| `samples_v3_BTCUSDT_canon_60000h_1778274003` | **LIVE on Contabo** (`/home/scalper/scalper-bot/data/_cache/`) | 952K samples × 1800-tick mid_paths × 49 feat. Canonical era. All sidecars present. |
| `samples_v3_60000h_1777593610` | **DELETED** | 1.85M × 52 feat (49 + 3 derivable). Lost in OOM rebuild. |
| `samples_v3_BTCUSDT_swing30m_unfilt_60000h_1777734022` | LIVE on Contabo | 192K × 18000-tick. 30min swing experiment cache. |
| Cryptolake 8-sym caches | **LOST** (Vast.ai + Runpod terminated) | Rebuildable from `gs://blackdigital-scalper-data` in 30-45 min/symbol with workers=32. |
| Cryptolake source data on GCS | **PERSISTENT** | `gs://blackdigital-scalper-data` (europe-west1). 287.9 GB raw, 1.3 GB features cache. 8 symbols, BINANCE_FUTURES. |

## 9. Code state (as of 2026-05-12)

| Component | State |
|---|---|
| `rust_ingest/src/live_sim.rs` | TAKER fees 0.07/0.10 default. `NotFilled` exit reason. `simulate_trade_hybrid` with taker fallback. Per-sample Kelly fix in `grid_sim.rs`. |
| `rust_ingest/src/bin/sim_labels.rs` | Has `--entry-taker-long/short` for maker-first hybrid. 150ms fill latency, 1s entry fill window canonical. |
| `rust_ingest/src/features.rs` | NUM_FEATURES = 67 (after Cryptolake +8 liquidation cols). |
| `src/features.py` | `_RAW_NUM_FEATURES = 67`, `NUM_FEATURES = 55` after extended `DROP_RAW_INDICES`. |
| `scripts/build_cryptolake_cache.py` | TAKER labels by default. Maker-first relabel **not integrated** (P0 blocker). `--save-mid-paths`, `SCALPER_SAVE_DAY_TICKS`, vol-scaled TP/SL, `--eth-leading` flag. |
| `scripts/train_seq_v8.py` | TCN + Mamba sequence trainer. **Wrong early-stop metric** (unweighted BCE). Must fix before next run. |
| `scripts/cpcv_proper.py` | N=6, k=2, embargo, purge, PBO. Working. |
| Live bot models in `models/` | **EMPTY**. Bot runs in data-collection-only mode. Last weights drained 2026-04-14 during methodology overhaul. |
| Train↔live gap | Research pipeline writes `.pt` + XGB `.json` that **don't match** `HybridModel`'s load format. New inference module needed before paper-trade. |

## 10. Memory pointers (for deep-dive only)

These contain raw session notes. The frontier table above subsumes their decision-relevant content.

- `methodology_bugs_2026_04_14.md` — original 2 bugs (label-WR artifact, full-val stacker fit).
- `experiments_2026_05_01_signal_exhaustion.md` — 5 levers session, BTC era exhaustion.
- `experiments_2026_05_02_methodology_overhaul.md` — maker fill check + Kelly fix.
- `experiments_2026_05_02_swing_attempt.md` — 30 min swing attempt, holdout net=-0.004% at n=9.
- `experiments_2026_05_09_cascade_canonical.md` — cascade vs single, TP/SL grid on cascade_180s.
- `cryptolake_phase0_2026_05_09.md` — 8-symbol cache build, vol-scaled TB, liq features.
- `cryptolake_phase23_2026_05_09.md` — cascade XGB on 8 symbols, pooled training.
- `cryptolake_phase56_2026_05_10.md` — maker-first first POSITIVE EV (**but later found to be TAKER-label artifact**).
- `cryptolake_state_2026_05_10_v2.md` — TAKER vs MAKER reset, DOGE step=5.5s = −1.4%/month real.
- `cryptolake_experiments_2026_05_10_final.md` — 2-day sequence model summary, best −0.040% LINK TCN.

## 11. Update protocol (for Claude)

**Every research session that produces a number:**

1. Append/update row to **Frontier table** (§3) — keep one row per experimentally-distinct setup.
2. If a methodology bug is found/fixed: add row to **Resolved confusions** (§4).
3. If a hypothesis is tested: move it from **Active hypotheses** (§7) to **Frontier** (§3) or **Doesn't work** (§6), with result.
4. If new cache built / old cache deleted: update **Cache inventory** (§8).
5. If code constants change: update **Canonical constants** (§2) and **Code state** (§9).
6. Memory files in `/root/.claude/projects/-root/memory/` still get written per usual auto-memory rules. **Do not duplicate** their full content here — only the decision-relevant rolling state.

**Never:**

- Cite an EV/tr% without `tr/day` and fee regime (TAKER/MAKER).
- Compare nets at different Kelly fractions without renormalizing.
- Report "WR > X%" without specifying base rate AND that it is direction-aware realized.
- Quote a result from this file without verifying against the cited memory or live code (per global Law #2 — facts > theories).
