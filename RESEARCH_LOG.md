# Research Log — scalper-bot

**Structured source of truth is now `research/` (JSONL ledger + SQLite).**
This file is the human narrative; the queryable asset is
`research/experiments.jsonl` + `research/hypotheses.jsonl`, contract in
`research/schema.sql`, plan in `research/PLAN.md`. §3 below is regenerable
via `python3 research/ledger.py frontier` — do not hand-maintain it once
new results land. The ledger *refuses* a result without its fee regime /
cache / split provenance (the chaos that cost us 3 false positives).

> ## ⚠️ DATA REALITY — READ THIS BEFORE EVER WRITING "no data" / "data is missing"
>
> Recurring failure: an agent opens `features_v1/.../features.npy`, sees that
> ~13 of the 59 columns are all-zero, and concludes "we don't have ETH /
> trades / funding / no signal." **That is FALSE. The raw data EXISTS.**
> `features_v1` is a **BOOK-ONLY precompute** — the zero columns were simply
> never computed, not absent. Before any data claim, LOOK AT `raw/`.
>
> **What we ACTUALLY have** — `gs://market-data-0998ac51/raw/` (Binance
> Futures, 8 symbols `*-USDT-PERP`, 2022-12-08 → 2026-05-08, ~585 GB):
> - `raw/book/`   — 20-level LOB (ns timestamps)
> - `raw/trades/` — aggtrades (→ trade-flow, cvd, vpin, kyle, intensity)
> - `raw/funding/` — funding/markprice (→ funding_rate, basis, time-to-next)
> - `raw/liquidations/`, `raw/open_interest/`
> - **ETH** is a full symbol here → `eth_momentum/eth_ofi/eth_leading_signal`
>   are recomputable (they are 0 in features_v1 only because not computed).
> - **Only genuinely absent:** cross-exchange (bybit/okx/bitget/gateio) — raw
>   is BINANCE_FUTURES only. So `*_net_flow` cols stay 0 — that one IS no-data.
>
> **To get the FULL feature set:** run the Rust `feature_builder`
> (`rust_ingest/src/bin/feature_builder.rs`: `--depth --trades --funding --eth
> --indices --out`) on raw → all 46 real + the trade/funding/ETH cols
> populated. It's fast and built for volume (run on the 96-vCPU VM,
> `scripts/hd2_feats_vm.py`). `features_v1` decision-point `indices` == the HD2
> stream-cache decision points (alignment is built-in).
>
> ## Infra state (current — 2026-05-26, supersedes the 2026-05-16 note below)
>
> - **GCP account: `virgin.ship03@gmail.com`**, project
>   **`project-0998ac51-36ba-445c-bc7`** ("My First Project"), billing ENABLED.
> - **Data bucket: `gs://market-data-0998ac51`** (`EUROPE-WEST1`) — full copy
>   (585 GB, verified) of the old bucket. Layout under `raw/` (above) +
>   `features_v1/symbol=<SYM>/dt=<DAY>/{features.npy(N×59), indices.npy}` +
>   `hd2_cache_v1/{streams,midts}/` + `feats_v2/` (full Rust-recomputed
>   features, when built) + `research_runs/`.
> - **OLD account `blackdigital.kz@gmail.com` / project
>   `project-26a24ad0-1059-4f73-93b` / `gs://blackdigital-scalper-data` —
>   MIGRATED FROM (GCP balance low). Old bucket still exists as source; can be
>   deleted once migration is trusted.** `gs://scalper-bot-research-data` =
>   403 (volaware ckpts, refuted — not needed).
> - Compute: same-region VM (europe-west1) ↔ bucket → **egress free**. New
>   project: verify Compute Engine API + N2 quota before the next VM build.
> - Modal account (`virginship08`) + Volume `hd2-cache` are SEPARATE and
>   intact (Modal was never the balance issue). The Modal secret `hd1-gcp`
>   (GCP_ACCESS_TOKEN) must be re-minted for `virgin.ship03` before Modal
>   reads the new bucket.
>
> ---
> _Historical (2026-05-16, superseded above):_ Contabo `root@84.247.154.229`
> **LOST** (every "LIVE on Contabo" §8 cache + `/root/.claude/.../memory/*.md`
> archive gone — §10 pointers historical). Then-topology used GCP
> `blackdigital.kz` 96-vCPU VM + `gs://blackdigital-scalper-data` (287.9 GB at
> the time; grew to 585 GB by 2026-05). `PREEMPTIBLE_CPUS=0` → no spot (use N2
> on-demand). Cryptolake feature asset survived the Contabo loss.

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

> **HR1 RESULT — REFUTED as posed 2026-05-17 (`phaseb-20260517-180221`,
> 6 alpha rows, well-powered n_oos 1763–6104).** No reward point beats
> R0 by >2·SE ∧ >0 ∧ placebo-clean on **both** symbols (only LINK-H600
> R2 locally robust z=2.63; SOL zero → isolated, bar blocks). **R1**
> |move|-weight = the lone *consistent* effect: 6/6 cells both symbols
> ΔR0>0 (+0.002…+0.020) but **none clears 2·SE** → weak marginal,
> **retained** for stacking (HM1 class of T2_asym/H3), not a lever.
> **R2** economic-weight = robust LINK-H600 only; **SOL degenerate**
> (AUC exactly 0.500 — zeroing sub-cost moves collapsed SOL's sample;
> structural re-confirmation of the cost-wall). **R3** rank:pairwise =
> symbol-inconsistent, **significantly negative on SOL** (z≈−2.4/−2.5)
> → the IC-surrogate HM2-rev1 named is refuted. **R4** asym up:dn cost
> ≈0 both → the T2_asym whisper does **not** survive as an objective;
> thread closed. **Bounded conclusion (NOT the over-claim pattern):**
> the decision-relevant reward families (plain/magnitude/economic/
> ranking/asymmetric) + MSE-return/volnorm ≈ **7 pts** are tested-
> negative; residual untested points (recency/rarity reweight, focal,
> bespoke-IC gradient) are **low-EV sub-variants of already-negative
> families**, explicitly flagged as not literally tested — not a new
> axis. The cheap high-EV snapshot search is now **earned-closed across
> three independent pre-registered axes** (feature HA1/H3/HA5 · scope
> HA7 · reward HR1), uniformly ≈0.51 with only retained weak marginals
> (R1·T2_asym·H3).
>
> **[The "4-way strategic fork" framing of HD1 rev13/14 is STRUCK —
> HD1 rev15, user 2026-05-17.]** It re-opened a question HM3 already
> closed and contradicted HM3's own corollary. HM3 (confirmed, user)
> = sequence/temporal representation adds lift, *accepted prior, do
> NOT test*. Therefore "accept-no-cheap-alpha" is **not** an option
> (HM3 corollary: snapshot ≈0.51 ≠ alpha-absent), and "adopt sequence"
> is **not** an option among equals — with the cheap snapshot search
> earned-closed and HM3 confirmed, the next representation is
> **DECIDED: build the sequence/temporal candidate, under R1 (HM5)**.
> The only genuinely open items are (i) *scope of that build* —
> architecture family / symbols·H / compute budget (a narrow scoping
> call), and (ii) one level up, whether the project's GOAL stays
> "directional alpha on this testbed" at all vs. shelve/pivot the
> whole objective (a goal-level call the user owns — distinct from,
> not a branch of, the struck intra-goal fork). (Only remaining cheap
> experiment = stack the 3 retained weak marginals — explicitly
> dominated by the sequence path per HM3; optional side-check,
> offered, not assumed.) VM auto-deleted (hard cap).

> **HM5 — default objective standardized 2026-05-17 (user decision).**
> User reframed the fork: *"which reward point do we keep for ALL
> subsequent research?"* — correctly forcing the objective standard
> BEFORE the fork, else every future path (incl. sequence) inherits the
> R0/MSE wrong-objective defect (mild HM2). **Decision: R1** — logloss
> with `sample_weight ∝ |forward move|` (clip train-p99) — becomes the
> default directional objective and `baseline_ref` (paired with the HA1
> rank-IC anchor). Justification: R1 is the **only** reward uniformly
> ≥ R0 (HR1: **6/6** cells both-sym×H, ΔR0 +0.002…+0.020, placebo
> clean); the **sign consistency** is significant (one-sided sign-test
> **p≈0.016**) though no single cell clears 2·SE — a *real-but-small*
> effect detectable by consistency, not per-cell magnitude. Principled:
> concentrates loss on economically-decisive moves; R0/MSE squander it
> on untradeable sub-cost wiggles. **Bounded:** R1 is a *baseline
> upgrade, NOT a lever/alpha* (won't alone make a strategy); reversible.
> R2/R3/R4 rejected as defaults (R2 SOL-degenerate, R3 SOL-negative,
> R4 null). [The "fork remains open" line here is **superseded by
> HD1 rev15** — see the STRUCK-fork note above: with HM3 confirmed +
> snapshot search earned-closed, the next representation is DECIDED
> (sequence/temporal, under R1); not a 4-way deliberation.] No
> compute launched.

> **Sub-60s hold feasibility (HH rev1) — CONDITIONAL CLOSE 2026-05-27** (independent run; GCE europe-west1, raw Cryptolake 8 sym, 30 d 2026-04-06..05-05, 250 ms grid, exchange-ts; artifacts in `C:\Dev\sub60s-hold-feasibility`, exp `2026-05-27T1631Z_hh_obi_screen`). Q: profitable trading with hold <60 s at standard fees (maker 2 bp/side, taker 5 bp/side)? **A: YES, conditional on (a) sufficient directional predictivity AND (b) a reaction-latency budget — both now bounded.** (1) **Volatility sufficient (NOT the bind):** a selective perfect-foresight oracle nets **4.6–6.6 bp/trade at taker** (10 bp+spread) on all 8 symbols @60 s (DOGE/ETH/LINK lead), 2.9–4.8 bp/trade at maker — ample tail move over cost; mean |move| understates this (oracle takes the tail, not the average). (2) **Latency budget relaxed:** OBI edge half-life secs→tens-of-secs (ETH ~12 s … LTC >60 s); hundreds-of-ms delay erodes single-digit % of edge → budget **~≤1–2 s**, met by a near-Binance box (1–3 ms RTT). `receipt_timestamp` = Cryptolake vendor ingest, excluded from the budget. (3) **Predictivity = the binding gap:** best deployable no-look-ahead OBI-class signal (L0/L5/L20 imbalance, OBI+TFI sign agreement, conviction top-1%/0.1%, trailing-vol regime) caps at **~1.6 bp gross directional capture/trade < 4 bp maker floor** → sub-cost-alone, a **baseline to stack on, NOT refuted** (selection policy). **Limit:** descriptive microstructure screen (no model fit, no train/test split); bounds the OBI-class rule space only, not achievable predictivity from richer signals (cross-asset lead-lag, flow toxicity, OI/funding/liquidation) — the open direction. Detail: `C:\Dev\sub60s-hold-feasibility\sub60s-hold-feasibility.md` (+ `results*.json`). status=`informative`.

**Last updated:** 2026-05-27 — HH rev1: sub-60s hold feasibility conditional close (YES iff sufficient directional predictivity + reaction-latency budget; budget+volatility resolved, binding gap = predictivity — deployable OBI-class capture ~1.6 bp < 4 bp maker floor, sub-cost-alone = baseline to stack, status=informative, exp 2026-05-27T1631Z_hh_obi_screen). Prior update: 2026-05-17 (HM6 rev4 — CANONICAL baseline_ref ESTABLISHED. Run phaseb-20260517-203822, {SOL,BTC,ETH,LTC} aligned 2025-05-13..2026-05-07, N=SOL360/BTC359/ETH357/LTC360, 12/12 ingested, ledger PASSED. R0~.509-.536, R1~.507-.542, rank_ic POSITIVE all 12 (mean .024, max .042 BTC-H180, decays w/ H = HA1 shape); weak/sub-economic, program-consistent. HONESTY CORRECTION HM5 rev2: R1>R0 only 8/12 (mean +0.0017, sub-noise, NOT sign-consistent) vs provisional-243's inflated 11/12 and HR1's 6/6 — HM5 NOT refuted (R1 net>=R0, zero-cost, stays default+baseline_ref) but DOWNGRADED to within-noise default, the 'sign-consistent extraction' claim does NOT replicate; future deltas vs R1 treat the R0-gap as noise. LINK dropped (genuine 119d outage)->LTC (verified clean + deepest history). HD1 rev23: rev16(i) groundwork COMPLETE; open = (i) sequence build-scope, (ii) user goal-level call; nothing auto-launched. prior: HM6 rev3/rev2, HD1 rev19, HM5 rev1).

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

## 12. 2026-05-28 — sub-60s corrected-feature substrate + grid_sim economics

**Context**: new GCP (virgin.ship03 / market-data-0998ac51). Built a CORRECTED sub-60s
feature substrate from RAW and characterized the alpha continuously (no discrete gates,
per CLAUDE.md frame). Full recovery/ops detail in `mamba2plan.md`; scripts `scripts/subs60_*.py`.

**Pipeline fix (was blocking feats_v2)**: the Rust `feature_builder` expected a converted
nested schema (`bid_prices` FSL, `quantity`/`is_buyer_maker`, `funding_rate`) that does NOT
exist in the bucket — raw is FLAT (`bid_0_price..`, `side`+`amount`, `rate`), ts in **ns** not ms.
Patched the 3 readers to read flat raw + convert ns→ms; repurposed the weak ETH cols
(1s-VWAP-diff, IC~0.03) to clean point-to-point `eth_ret_{1,2,5}s` (IC~0.05); added
liquidation/OI features + an OBI ladder (L1/L5/L10/L20). `NUM_FEATURES=64`. Built
`gs://…/feats_sub60/` (2776 sym-days, dense 1s grid, X/td/mid/rH_{15,30,45,60}/valid).

**Two-axis decomposition (OOS, purged split):**
- VOLATILITY predictable: `vol_AUC` 0.78–0.89 (predict |move|≥13bp). But the ≥13bp event is
  RARE (≥13bp@15s base ~0.5-1%; @60s ~3-19%), so operating-point PRECISION is only ~21–40%.
  Opportunity is strongly regime-dependent (vol clustering: DOGE 2.7%@360d vs 0.8%@recent-30d).
- DIRECTION = the harder axis (not a hard wall): selective-tail dir_acc 0.55–0.65.

**grid_sim TP/SL economics** (Rust, fixed-unit EV/trade, NET of commission; universal argmax
**tp=0.30/sl=0.05 (RR 6:1), hold 45–60s**, WR 42–50%):
- Oracle (realized ≥13bp gate = perfect vol-head, CEILING): VIP0 **+5.4…7.2 bp/tr**, maker
  **+0.7…1.7 bp** (7/8 syms +, ETH ~0), taker −2.8…4.6. tr/day(gated) ~1k-2.8k.
- Deployable (vol-MODEL gate): VIP0 **+0.9…2.0 bp/tr**, maker **−4.1…5.0 bp**, taker −8…9.
- Fine 100ms path recovers ~1.3–1.85× the gross of the coarse 1s path (1s undercounts TP touches
  by ~30-45%); our live recorder is sub-ms so prod path is cleaner than the cryptolake backtest.
- Fine path + extreme selectivity (top-0.2% combined vol+dir conviction, ~26–43 tr/day, overlap
  negligible): VIP0 +1…3.6 bp (DOGE/LTC/XRP best), **maker STILL −2…6 bp**.

**BINDING CONSTRAINT = deployable vol-model PRECISION** (not direction-wall, not path coarseness,
not selectivity — all tested). The gross edge (+1-3.6 bp/tr) is below the maker round-trip (~4 bp RT),
so **net-negative at the only fees we actually have**.

> **⚠️ FEE-REALITY CORRECTION (user, 2026-05-31).** Earlier entries call positive-gross
> numbers "VIP0 / zero-fee / show-the-fund". **That framing is RETRACTED.** There is NO
> VIP0 tier on Binance and the user has NO fund / preferential-fee access — it was a
> one-off joke that agents wrongly turned into a target. Only **standard Binance fees**
> apply: maker 0.02%/side = **0.04% RT**, taker 0.05%/side = **0.10% RT**. Read every
> "VIP0"/"zero-fee" in §12 and the 2026-05-28 ledger rows as **gross-before-cost only**
> (a diagnostic upper bound), never as a deployable target. Deploy criterion = net at
> standard fees; maker-maker 0.04% RT is the best real tariff.

**Note (overlap bug, found & scoped)**: earlier "1000–4500 tr/day" were RAW signal firings
(60s holds can't overlap that much, ≤1440/day non-overlap). Per-trade EV is unaffected; only
throughput was inflated. At the selective ~30/day operating point overlap is negligible.

**Next levers** (user-ordered): #4 fine-path DONE; #2 selectivity DONE (doesn't flip maker alone);
#1 improve vol-head precision (bigger TF + fundamental data) = top remaining lever; #3 Mamba-2
stream-2 (curated feats_sub60) + raw stream-1 (nonlinearity to lift gross above maker).


## 13. 2026-05-29 — Mamba2 vs sized-GRU head-to-head (sub-60s 2×2 cascade)

**Context**: lever #3 from §12 — does a heavier nonlinear sequence cell (Mamba2)
on the 2-stream cascade beat a small GRU? Clean head-to-head: **A** FLAT/NON-FLAT
vol-gate (per-symbol) and **B** UP/DN direction (pooled top-3, non-flat windows,
IC/capture objective), each 2-stream (raw LOB80 + curated feat71), training cell
swapped between `cell=mamba2` and `cell=stub` (GRU). Modal L4, per-epoch OOS +
best-val. Scripts `scripts/mamba2_cascade.py`, `scripts/mamba2_sub60_modal.py`.
Ledger `hd2-20260529_mamba2_vs_gru` (HD2, exploratory).

**Data / training**: `hd2_sub60_cache/{DOGE,ETH,LINK}-USDT-PERP` (cryptolake
2025-05-09→2026-05-08). stream-1 = raw 20-level LOB ticks (80-ch); stream-2 = 71
feats (feats_sub60 + signed BTC-lead{5,30,60}s + ToD). dec-stride 25 s, bounded
context L=3000 ticks, warmup 300. A all windows (max_days 300/sym, 6 ep); B
non-flat pooled (max_days 150/sym, 8 ep). **Params**: Mamba2 d1=d2=256, n1=n2=1
(mamba_ssm 2.2.2 kernel locks d_model=256 → shrink via n_layers, not d) →
**A 1,036,337 / B 1,038,409**; sized GRU → **A 263,297** (d128/n2/d64), **B 70,745**
(d64/n2/d32) — ~4× (A) / ~15× (B) smaller.

**Test**: purged day-split — train earliest ~65 % of days, embargo ≥60 s, test
newest ~32 % (per-symbol). n_test: A DOGE 172,756 / ETH 289,205 / LINK 212,702;
B pool 41,048 decisions. Best-val selection (never last-epoch).

**Result (surface)**:
- **A (vol-gate)** — AUC is ~**tied** (Mamba2 0.780/0.814/0.722 vs GRU
  0.782/0.818/0.742, DOGE/ETH/LINK), but **GRU has clearly better
  operating-point precision** (prec@0.2 % 0.623/0.690/0.675 vs Mamba2
  0.571/0.609/0.574) and **Mamba2 overfits early** (best_ep
  0/1/2 of 5 vs GRU 4/4/5).
- **B (direction, pooled)** — GRU wins decisively: cap@10 % **+4.68 bp** vs
  Mamba2 +3.03; cap@20 % **+4.12** vs +2.73; dir-acc@10 %
  **0.612** vs 0.564 — at ~15× fewer params.

**Argmax / takeaway**: best cell for this tier = the **sized GRU**, not Mamba2.
Mamba2 ties on vol-AUC but loses on deployable precision and on direction, while
overfitting from near epoch 0. **Decision: drop Mamba2 for sub-60s** — capacity
must match n_eff (overlapping 60 s labels ⇒ n_eff ≪ rows); Mamba2's long-range
memory + parallel-scan strengths are wasted at short (sec–min) context + small
n_eff, and the broken small-d kernel forced an oversized d=256. *Complexity is an
arena, not an edge in itself.* Artifacts: `gru_models/*_m2.best.pt` (+ `_gru`),
modal vol `hd2-cache:/results/sub60/*_m2.json`.


## 14. 2026-05-29 — GRU (sized-to-n_eff) = the chosen sub-60s cell, + downstream economics

**Context**: the §13 head-to-head picked the **GRU** over Mamba2. This section
records the GRU itself (the deployable model) and everything we ran on its
signal: Stage-2 executed-payoff fine-tune, grid_sim R:R per symbol, the HUSDC
realistic maker-fill sim, and an 18-policy entry sweep. Ledger
`hd2-20260529_gru_cascade` (HD2, exploratory). Same data/cache/split as §13.

**Model (best-val, purged day-split, test newest ~32 %)** — 2×2 cascade, GRU
cells sized to n_eff (**A 263 k**: d128/n2 + d64/n1; **B 70.7 k**: d64/n2 + d32/n1):
- **A (vol-gate, per-symbol)** vol-AUC **0.782 / 0.818 / 0.742**
  (DOGE/ETH/LINK), prec@0.2 % **0.623 / 0.690 / 0.675**, best_ep
  4/4/5 of 5 (still improving — no overfit, unlike Mamba2's ep0 peak).
- **B (direction, pooled top-3, non-flat, IC objective)** cap@10 % **+4.68 bp**,
  cap@20 % **+4.12 bp**, dir-acc@10 % **0.612**, best_ep 6 (n_test 41,048).
  GRU beats model-free + OBI on direction; this is the binding head.

**Downstream economics on the GRU signal (all sub-60s, MAKER-first fees):**

1. **grid_sim TP/SL @MID-entry** (optimistic; per-symbol R:R discovery), top-0.2 %
   gate, net@maker-maker: **DOGE** RR 22.8 (rails to grid corner) −1.13 bp WR 0.16;
   **ETH** RR 1.78 −1.39 WR 0.55; **LINK** RR 6.45 −1.02 WR 0.55. → optimal R:R is
   per-symbol (ETH≈hold-like wide, LINK≈5–13, DOGE corner) — confirms Stage-2 must
   be per-symbol. All net-negative @maker-maker on the broad gate.

2. **Stage-2 executed-payoff fine-tune (B2, ETH, hold)** — differentiable
   `L=−E[σ(z)·PL+(1−σ(z))·PS]`, early-stop best_ep 17/22: exec_gross **+4.98 bp**,
   net@mm **+0.98**, WR 0.615; by |logit| conviction net@mm **+6.23/+7.60/+9.00/
   +9.45 bp** at top-20/10/5/2 %. The payoff objective + σ(z) recalibration fixed
   the direction-selector (base |B| was a dead selector).

3. **Realistic maker entry (HUSDC maker-sim, ETH 116 d, 96,996 samples)** — resting
   limit filled against realized taker flow (touch/queue/MISS; adverse selection
   emerges from the path). HOLD-60s gross **−1.97 bp** (touch, fill 0.95) / **−2.44**
   (queue, fill 0.91), WR **0.45**. Passive entry **FLIPS the directional edge
   negative** — you fill on adverse moves and miss the favorable runaways. The
   +6…+9 bp "maker-maker" of (2) assumed the fill and ignored adverse selection.

4. **Entry-policy sweep (18 policies)** — taker > maker everywhere. At top-0.05 %
   conviction: **taker** gross **+5.64 bp** (WR 0.52), maker_off0 +4.42,
   cancel-on-toxic_p90 +2.73 (no help — adverse is smeared, not in rare bursts, so
   chasing it via cancellation/colocation is not worth it). Broad gate all negative.

**Argmax / takeaway**: the **GRU is the right sub-60s model** (beats model-free and
Mamba2 at 4–15× fewer params; A no-overfit, B the directional lever). On the
**deployable surface**, net>0 appears only under **high conviction + TAKER entry**
(at STANDARD Binance fees — there is NO VIP/zero-fee tier; maker-maker 0.04% RT is
the best real tariff) — **passive maker entry structurally loses to adverse
selection**, and that is robust across 116 days and all 18 policies.

**⚠️ CAVEAT (do not deploy on these magnitudes)**: the positive fine-tune / grid /
policy numbers carry **test=val inflation** — best-val/early-stop AND the A-gate
threshold were both chosen on the *same* test set (day-clustered t≈12–21 is
implausibly high). They are **upper bounds**. Pre-deploy requires the agreed
**60-20-20 (or CPCV / walk-forward) with val ≠ test**, plus the user's daily
rolling-retrain (the static far-split is a pessimistic floor; the §12/§13 half-split
decay is largely train-staleness, removed by daily retrain). Artifacts:
`gru_models/*.best.pt`, `research_runs/{gru_gridsim,gru_finetune,gru_makergrid}/`;
scripts `scripts/subs60_{gru_gridsim,maker_grid,entry_policy_sweep}.py`,
`scripts/mamba2_cascade.py` (B2 objective).


## 15. 2026-05-31 — XGBoost A/B on MAKER-REALISTIC (adverse-selection) labels — conditional surface

**Context**: exploratory tier "under what conditions is XGBoost strongest?" (HD3 rev1,
exp `xgb-20260531_makerlabels_AB`, GCE n2-standard-8). Two deliberate departures from §13/§14:
(1) **labels carry adverse selection** — built a per-symbol dataset (all 8 syms, ~2826 sym-days,
`research_runs/maker_labels/{SYM}.npz`) where each feats decision point gets a REALISTIC maker
P&L from the HUSDC maker-sim (resting limit fills on realized taker-flow touch/queue, MISSES on
runaway → adverse selection from the path), for ALL 3 cfgs (hold-60s/RR6/RR2) × 2 queue-mults
(touch/queue). (2) **Model A vol-gate is per-symbol VOL-ADAPTIVE** (user insight): a fixed
`|rH60|≥13bp` is 2.37σ/3.4 %-of-windows for BTC but 1.25σ/15.6 % for LINK — incomparable gates;
replaced by per-symbol TRAIN-p95(|rH60|) → uniform ~5 % non-flat target. Same 71 feats as the NN.
Optuna tunes HPs only (target/threshold/config are explicit swept CONDITIONS, not auto-collapsed).
Fully instrumented (all trials/importances/val-curves + per-sample test preds with pA+pB+all-cfg×qm
payoffs saved → any operating-point/config surface recomputable offline w/o retraining;
`scripts/subs60_xgb_surface.py`).

**SURFACE (headline):**
- **(A) Vol-gate** — AUC robust **0.818–0.859 across ALL 8 symbols**; deployable prec@0.2 %
  argmax **BTC 0.62** (low-vol symbol's rare big moves predict cleanest) > ETH .57 > XRP .56 >
  LINK/SOL .54 > BNB/LTC .51 > DOGE .45. Volatility is strongly + uniformly predictable.
- **(B) Direction skill** (oracle-gated on realized non-flat) — dir-acc@conv-top10 % **0.658–0.767**
  (BTC .767, BNB .746 best); REAL side-selection skill. But executed maker EV is −8…−14 bp because
  on realized big-move windows BOTH passive-maker sides lose to adverse selection (qm=1).
- **(C) Honest cascade** (gate by Model-A PREDICTION top-g % × B side, net after 4 bp maker RT) —
  **argmax cell per symbol**: **LTC +2.01 bp** and **LINK +1.23 bp** (both **hold / TOUCH, A-top-0.2 %**,
  dir-acc .54–.56, ~12–18 trades/day) go **net-POSITIVE**; XRP −0.78, DOGE −1.10 ≈ breakeven;
  BNB −2.86, ETH −4.28, BTC −4.71 (RR6/touch top-5 %), SOL −5.02 stay negative.

**What maximizes XGBoost's maker alpha (surface shape):** `touch ≫ queue` (first-touch passive fill
far less adverse than queue-clear) · **extreme selectivity** (A-top-0.2 %) · **hold-60s > RR** (except
BTC favours wider RR6). The **binding constraint on net is adverse selection on passive maker fills**,
not direction skill (which is real) nor vol predictability (strong everywhere).

**Caveat (do not deploy on these magnitudes):** val≠test (Optuna on val) and the A-gate threshold are
train-only/honest, BUT the per-symbol argmax is a MAX over ~30 cells (cfg × qm × selectivity) **on the
test surface** → selection-over-conditions optimism. LTC/LINK net-positive cells are **CANDIDATE
CONDITIONS for OOS (walk-forward) confirmation, not a deploy verdict** (§5 gate is a separate question).
Artifacts: `research_runs/xgb_maker/{A_{SYM},B_pool}.{json,xgb.json}`, `preds_{SYM}.npz`, `MANIFEST.json`,
`SURFACE.json`; scripts `scripts/subs60_{makerlabel_build,xgb_makerlabel,xgb_surface,vol_inventory}.py`.

**OOS WALK-FORWARD CONFIRMATION (exp `xgb-20260531_makerlabels_walkforward`, 4 folds, op-point
cfg/qm chosen on VAL@A-top1 % then measured on a later disjoint TEST; HPs reused).** The single-split
argmax **does NOT survive**. Mean test-EV by A-top 5/2/1/0.5/0.2 %: BTC −4.9/−5.0/−5.1/−5.3/−5.4 ·
ETH −5.1/../−5.7 · BNB −5.2/../−6.3 · SOL −5.4/../−5.8 · DOGE −5.9/../−3.1 · XRP −5.5/../−3.4 ·
LTC −5.3/../−2.7 — **7/8 symbols stable NET-NEGATIVE every fold** (dir-acc ≈ 0.50–0.51). **LINK** appears
positive (+6.8 @top1 %, +23.9 @top0.2 %) but it is **driven entirely by fold 0** (earliest OOS window:
+38.9/+95.3 bp, n 1896/375) while folds 1/2/3 are ≈0/negative → a single-fold **regime spike, not a
stable edge** (LINK = least data, 244 d incl. 119 d outage). **Confirmatory verdict:** the maker-realistic
XGBoost cascade does **not** confirm a deployable positive maker edge OOS; the single-split +2.0/+1.2 bp
was selection-over-conditions optimism + small-sample top-0.2 % noise. Binding loss = **adverse selection
on PASSIVE MAKER fills**, not model quality (vol-AUC 0.82–0.86 / dir-acc 0.66–0.77 are real). Corroborates
§14 (passive maker flips the edge negative; positive only with TAKER entry, +5.6 bp). **Next lever is
EXECUTION (taker / alt mechanics), not more boosting.** Result `research_runs/xgb_maker/WALKFORWARD.json`,
script `scripts/subs60_xgb_walkforward.py`.

**RECENCY / regime-drift diagnostic (exp `xgb-20260531_makerlabels_recency`).** Tested the hypothesis
that the val→test degradation is **market drift** (training recency), not HP overfit: HP + #rounds +
per-symbol vol-threshold held **FIXED** (reused from the main run), only the training WINDOW varied,
measured on the SAME clean test [0.68,1.0). Mean over 8 syms — old10[0,0.0975] / val10[0.5525,0.65]
(equal size, isolates recency) / train[0,0.5525] / trainval[0,0.65]: A-AUC **0.835 / 0.838 / 0.846 /
0.847**, prec@0.2 % 0.463/0.453/0.536/0.534, cascEV@1 % −6.17/−6.54/−5.12/−5.10. **Hypothesis NOT
supported:** at equal size the most-recent slice (val10, adjacent to test) ≈ the oldest (old10, ~0.58
history away) — mean ΔAUC +0.003, within noise, sign-inconsistent per symbol (LINK +0.012 .. DOGE −0.006);
no consistent maker-EV recency gain. **DATA VOLUME is what helps** (train/trainval beat both 10 % windows
on every symbol; AUC +0.01–0.03, prec 0.53 vs 0.46); adding recent val to train (trainval vs train) ≈ 0.
⇒ the model trained on year-old data generalizes to the distant test ≈ as well as on recent data; the
val>test gap is **operating-point selection optimism (confirmed via walk-forward) + volume/coverage, NOT
non-stationarity**. Practical: **daily rolling-retrain (the §14 staleness fix) likely helps THIS model
little — volume > freshness.** Caveat: maker-EV is net-negative everywhere (adverse-selection floor) so
AUC/prec are the cleaner read; single seed. Result `research_runs/xgb_maker/RECENCY.json`, script
`scripts/subs60_xgb_recency.py`.

**RECENCY v2 — CORRECTS the diagnostic above (exp `xgb-20260531_makerlabels_recency2`, supersedes v1).**
v1 was confounded (user catch): its large windows were START-anchored (so "volume" ≡ "old data" — it
never gave the model a large RECENT window; the only recency-isolating compare was at 10 %, too small),
and it measured B's dir-acc CROSS-CONFIG (trained qm=1, evaluated touch → ~0.49 noise, masking B's skill).
v2 fixes both: three **EQUAL-SIZE 45 % windows** sliding old→recent (`old_half`[0,0.45] / `mid_half`[0.10,0.55]
/ `recent_half`[0.20,0.65]), same fixed HP/rounds/threshold, same clean test; B measured on its **own qm=1
config** (rank-IC + dir-acc). Mean over 8 syms old→recent: A-AUC **0.845/0.846/0.847 (FLAT)**, A-prec@.2 %
0.527/0.535/0.531 (flat); B-rankIC −0.058/−0.053/−0.051 (improving), **B-dir-makerside@10 % 0.694/0.700/0.703
(improving)**, cascEV@1 % −7.04/−5.80/−5.64 (improving). **Per-model verdict: recency is MODEL-SPECIFIC —
the VOL gate (A) is time-STATIONARY (no recency, +0.001–0.004/sym), the DIRECTION head (B) is mildly
NON-stationary (recency helps: dir-on-target up on 6/8, mean +0.009; rank-IC up 7/8; cascEV less-negative
on 6/8, +1.5–3.4 bp, LINK −6.1→−2.7) — though all maker-EV stays net-negative (adverse floor).** So the
v1 line "volume not recency" is **revised**: a real but modest drift component exists, on B only ⇒
rolling-retrain helps the direction head a little, not the vol gate, and won't flip maker-EV positive.
Side-finding: B's side vs RAW price sign is <0.5 / rank-IC<0 → the maker-profitable side anti-correlates
with the price move (crystallized adverse selection). Result `research_runs/xgb_maker/RECENCY2.json`,
script `scripts/subs60_xgb_recency2.py`.

**RECENT-90d + Optuna-FROM-SCRATCH vs FULL-HISTORY (exp `xgb-20260531_makerlabels_3mo`).** Both A+B
retrained from scratch with fresh Optuna (40 trials) on the **last 90 day-indices** before val (vol-thr
re-derived on that window), same val + clean test [0.68,1.0); head-to-head vs the full-history main run
(~200 d train). **Mean Δ (3mo − full) over 8 syms: A-AUC +0.0028, B oracle dir-acc@10 % +0.0016,
cascEV@1 % −0.34 bp — ALL within noise.** Per-sym A-AUC: BNB .850→.868, BTC .818→.827 (3mo slightly
better), DOGE .856→.849 / XRP .859→.855 (slightly worse), rest ≈ equal. ⇒ **a model trained on the last
90 days ≈ the full-history model on the clean test** — neither recency+retuning nor extra volume moves
test performance. So: vol-gate A is time-STATIONARY (90 d == full, corroborates recency2), direction B
≈ stationary at this scale, **training volume SATURATES by ~90 d** (older data ≈ inert), and **maker-EV
stays net-negative in both** (adverse floor unbroken by recency or fresh tuning). Reconciles recency2
(recent-45 % mildly > old-45 % on B): full-history's predictive content is carried by its recent portion
+ volume saturation. Practical: **no need to retrain on full history — ~90 d suffices; the maker-edge
problem is EXECUTION (adverse selection), not training-window or tuning.** Single seed; A operating-point
(prec) not perfectly apples-to-apples (thresholds re-derived per window), AUC is. Result
`research_runs/xgb_maker_3mo/SURFACE.json`, scripts `scripts/subs60_xgb_{makerlabel(--train-days),surface(--sub)}.py`.

**DOUBLE-FILTER (A∧B confidence) + 1-trade/symbol/day budget (exp `xgb-20260531_makerlabels_dailybudget`,
OFFLINE on the main-run test preds).** Combined confidence = `pct_rank(pA)·pct_rank(|pB−0.5|)` (both
A vol-gate AND B direction must be confident); budget = per (symbol,day) trade the single max-combined
window, B's side, executed net maker EV (hold-60s). **Two levels:** (1) **DEPLOY-HONEST — 1 trade EVERY
day** (per-day pick uses only model outputs, no test-selection): pooled **−0.77 bp** (touch) / −1.27 (queue)
— **near break-even and a big lift over the A-only top-1 % cascade (~−5 bp)**; per-symbol(touch,1/day):
LTC +4.7, LINK +3.1, SOL +0.9 positive, ETH −0.4, BTC −1.4, BNB −3.9, DOGE −4.0, XRP −5.1 (WR 0.34–0.48,
few big hold-60 s tail wins). The A-and-B filter + one-window-per-day selection lifts per-trade quality
far above the A-only cascade. (2) **Cross-day selectivity sweep** (keep top f % of days by combined score)
is strongly positive + monotone: pooled **+4.0 (top50 %) / +14.1 (top25 %) / +30.9 bp (top10 %)** touch,
6/8 syms rising steeply (LTC +71.7, DOGE +54, LINK +37.9 @top10 %) — **BUT this selects the best days ON
TEST → selection-over-conditions optimism** (same failure mode the walk-forward demolished for the
argmax; n@top10 % ≈ 11 trades/sym). So (2) is **PROMISING-BUT-UNCONFIRMED** — needs OOS (day-selectivity
threshold chosen on val, measured on disjoint test). Net: the double-filter + daily budget materially
improves the deployable maker EV to ~break-even, and the confidence-ranked-days gradient hints at real
signal worth an OOS test — **no positive maker edge is CONFIRMED until that OOS check passes.** Result
`research_runs/xgb_maker/DAILYBUDGET.json`, script `scripts/subs60_xgb_dailybudget.py`.

**EXPLICIT 2D confidence-window grid (exp `xgb-20260531_makerlabels_dailygrid`).** Strict `pA in top-qA %
AND |pB−0.5| in top-qB %`, 1 trade/sym/day, on test. Pooled mean EV(bp) rises as BOTH tighten, and
**A-tightness is the BIGGER lever than B-tightness**: A1 %×B10 % = **+20.0** > A10 %×B1 % = **+13.0**;
corner A1 %×B1 % = +23.3 bp but only ~0.11 trd/day (~13 trades/sym). touch ≈ queue (within ~0.5 bp →
robust). Reading: tightening the predictable+stationary VOL gate (A) beats tightening the weaker,
adverse-selection-bound DIRECTION (B). **Same TEST-selection caveat — the grid is MAP coordinates, not a
confirmed edge** (honest deploy = a fixed (qA,qB) cell chosen on val, measured on disjoint test; OOS
pending). Result `research_runs/xgb_maker/DAILYGRID.json`, script `scripts/subs60_xgb_dailygrid.py`.

**PER-SYMBOL R:R CONFIG-AXIS + per-symbol Optuna-tuned B (exp `xgb-20260601_makerlabels_b2grid_optb`).**
Question raised by the user: does a per-symbol-tuned direction head B let some per-symbol R:R (TP/SL)
config beat hold-60s OOS? Per symbol: train A + Stage-1 B(hold) → select `c*` = argmax VAL daily-budget
(10/day) net maker-EV over a **fine 94966-config TP/SL grid** (`grid_sim` on-demand from saved maker
paths) → retrain Stage-2 B on `c*` nets → eval hold vs `c*` on disjoint TEST. B HPs Optuna-tuned
per-symbol (`b_trials=25`, inner-val AUC nested in gate_train; main val/test untouched).
**CONFIG-AXIS surface — net maker-EV(bp) argmax over the config axis is HOLD-60s:** pooled TEST hold
**−1.80** vs `c*` **−4.07**. Per-symbol (test 1bud, hold→`c*` [config]): BNB −5.31→−4.22 [RR1.7],
BTC −2.31→−3.77 [RR0.5], DOGE +0.11→**−9.11** [RR1.9], ETH −1.18→−0.49 [c*=hold], LINK −0.28→−3.55
[c*=hold], LTC +1.17→**−5.84** [RR0.5], SOL −1.78→−1.12 [c*=hold], XRP −4.80→−4.50 [RR2.7]. **Shape:**
on 3/8 (ETH/LINK/SOL) the val-grid itself picks `c*`=hold; where it commits to a real R:R the OOS is
**asymmetric** — the rare R:R "win" is marginal and still deep-negative (BNB +1.1, XRP +0.3) while R:R
losses are large (DOGE −9.2, LTC −7.0; wide-SL `c*` carries a fat left tail the val-selection
underestimates). So per-symbol-tuned B does **not** lift R:R above hold-60s OOS — consistent with the
walk-forward row and the (un-finished) reused-HP baseline grid (BNB/BTC/DOGE `c*`=hold or `c*`<hold on
test). Binding constraint stays **adverse selection on passive maker fills**; next lever = **EXECUTION**
(taker entry §14 +5.6 bp; realistic resting-maker exit), not R:R/HP tuning. **Method note:** this grid
models entry-MISS + adverse SL-knockout absent in the §14 GRU grid, legitimately inverting the old
big-TP/small-SL preference toward **wide SL** (`c*` SL pins near the 0.40 grid ceiling, e.g. BNB 0.391);
the residual optimism is the **touch-modeled TP exit** (tight TP fills too easily) — so neither grid's
argmax R:R is ground truth, and the robust read is OOS = hold. Optuna is **not** at fault (it maximises
inner-val AUC as designed; its window-overfit propagates into the downstream `c*` selection, e.g. DOGE
val RR1.9 edge +0.26 bp → test −9.2 bp — expected optimizer behaviour). Result
`research_runs/maker_labels_rr/B2GRID_RESULT_optb.json`, script `scripts/subs60_xgb_b2_grid.py`.

**2026-06-02/03 — EXECUTION-LEVER session (B-universe, abstention, fill-model C, taker-direction).**
Tier-frame (CLAUDE.md): map *under what conditions* a tradeable edge appears; the maker **apred +3.00 bp,
7/8 net-positive** (`B2_RESULT_apred`, durably re-saved this session as `…_apred_8sym.{log,json}`) remains
the best result of the tier — UNCONFIRMED (one split, not walk-forward). Conditions probed (all on
XGB-snapshot 71-feat, `maker_labels_rr`, single `honest_val_test` split unless noted):
(a) **B training-universe** (`xgb-20260602_…b_universe`) — pure-maker EV on a FIXED A-top-5% test pool is
maximised at the NARROWEST B-training gate (apred 5%) at every budget, both symbols; wider (25/100%)
lowers it monotonically. B's GLOBAL dir-AUC ≈ 0.51 for all → the gap is **train/deploy distribution match**,
not model quality.
(b) **Toxicity-gated abstention** (`…_abstain`) — on the fixed A-pool, ranking trades by an adverse-fill
predictor did not beat B-confidence ranking; the actionable target (`1{filled trade loses}`) was ≈random
here (abstAUC 0.51–0.54). KEY: the predictable part of adverse selection (MISS-on-runaway, tox-AUC ~0.75)
is non-actionable for a maker (a MISS pays 0); the actionable part is ≈the unpredictable 60s sign.
(c) **Model C, fill predictor** (`…_fillmodel_C`) — fill IS strongly predictable (test fill-AUC BTC
0.73/0.72, LINK 0.64/0.62 ≫ the ~0.52 of 60s direction), but the fill-asymmetry carries ~no 60s direction
(rank-IC −0.017), and as a 3rd A∧B selection factor it did not raise +3.00 on these conditions (fill-rate
on B's side already ~0.96 → 'will it fill' near-constant; adverse selection sits in the post-fill outcome).
(d) **Adverse selection quantified** — on A-non-flat windows, fwd-60s | long-limit FILLED = −0.6 bp (BTC)
vs | MISSED = +13.4 (BTC) / +33 (LINK) / +29 (SOL): you fill small-adverse moves, miss big favorable
runaways. (e) **Taker-entry direction pipeline** (`xgb-20260603_…b_taker`, fee_regime=TAKER) — built taker
labels (8 syms, `research_runs/taker_labels`); Stage-1 predict-the-MOVE (XGB regressor on rH; a true
global-corr IC obj underfits XGBoost greedy trees) → grid_sim-TAKER c* (captured-alpha on top-3000 val
conviction) → Stage-2 executed-payoff. Taker entry **captures positive GROSS** (+2.6…+4.6 bp at conviction —
the runaways are catchable gross), but under standard **7 bp taker RT** and one split, taker NET did not
exceed +3.00 this session (net −3…−8 bp pooled BTC+LINK). Predicting the move makes the grid pick a
HIGH-TP/SL c* (BTC RR2.9→RR6.8) vs hold for a binary target, but the val-chosen high-TP/SL c* did not
transfer to test here (same val→test gap as the maker-R:R argmax). Raw-60s-dir-acc rises with B-confidence
(BTC 0.511→0.542) but stayed ~0.54; the strongly-predictable §15 conviction signal (0.66–0.77) is the
maker-better-SIDE (fill microstructure), distinct from raw direction. **Framing (CLAUDE.md, no verdict):**
this session did not achieve taker-positive economics beating +3.00 *on these conditions* — NOT a general
impossibility. Untested levers: walk-forward c*; RR/SL-floor-constrained configs; sequence/longer-context
direction (GRU §14 got dir-acc 0.612 but those economics carried test=val inflation + mid-entry optimism);
lower fee tiers; maker-placement (offset/queue/patience); other horizons. Infra added: `grid_sim
--absolute-timeout` (exit at t0+60 vs fill+60 → ≈identical, corr 0.999; fills are fast). Scripts
`scripts/subs60_{xgb_b_universe,xgb_abstain,xgb_fill,xgb_abc,build_taker_labels,xgb_b_taker}.py`; husdc
Rust source (maker-entry + flag) captured to `research_runs/husdc_src/`. See `HANDOFF_taker_hd3.md`.


## 16. 2026-06-03 — DOGE actualization + OOS walk-forward of the maker apred cascade (regime-adaptive vol-gate)

**Context**: take the tier's best single-split result (maker apred cascade, DOGE **+5.73** /
pooled +3.00 net maker EV/trade) toward paper trading by (a) reproducing it, (b) actualizing the
data to the freshest edge, (c) the honest OOS test the handoff named #1. DOGE-only (paper trading +
validation are DOGE-only per user). Ledger `xgb-20260603_doge_actualize_walkforward` (HD3 rev2,
exploratory). VM `hd2-feats-003`.

**Reproduction (sanity)**: `subs60_xgb_b2.py --gate apred` reproduces DOGE **+5.73 bp** exactly (109
trades) on the original split — pipeline deterministic & intact.

**Data actualized to 2026-06-02**: cryptolake is the only history (the live recorder started
2026-06-01, ~1 day — not a backfill source). Backfilled cryptolake raw 2026-05-09→06-02 to GCS
(`backfill_cryptolake_to_gcs.py`; cryptolake S3 parquet schema is a verified DIRECT MIRROR of GCS
`raw/` → ingestion = plain copy), 2.27 GB: DOGE book/trades 23/25 d, BTC book 16 d, ETH trades 15 d
(vendor coverage is intermittent in recent weeks; BTC/ETH worst). Rebuilt `feats_sub60/DOGE` 21
fresh days (`subs60_orch.py` — the feats builder, **recovered from the VM and committed; it was not
in the repo**) + minimal `feats_sub60/BTC` mid 16 d (`subs60_btcmid_backfill.py`, for the btc_ret
lead) + `maker_labels_rr_freshtail/DOGE` N=124180.

**Feature-dependency correction**: the DOGE feat71 base-64 (`feats_sub60`, Rust feature_builder) DOES
include 3 ETH cols (eth_logret lags 14/16/54) + liq(56-58) + OI(59-60) + OBI(61-63); btc_ret(64-66) is
added at maker-build. External-stream importance for DOGE ≈ **8 % (A) / 17 % (B)**. DOGE trains
independently of BTC/ETH *models* (A per-symbol; B per-symbol, HP-shared) → not blocked by their gaps,
but its *feats* need ETH + BTC mid, dead on vendor-gap days.

**Fresh-tail held-out validation** (train on clean ≤05-08, test 2026-05-09→06-01 never trained;
PESSIMISTIC — eth dead 48 % of rows / btc 35 % on gap days, the live feed will carry them): the exact
+5.73 model → **+10.08 ± 8.4 bp** (median +6.10, 11/20 pos). At n≈20 the per-trade mean SE spans +5.73
and 0 → sign/median confirmed, magnitude too thin to pin. Regime shift visible: DOGE p95|rH60| = 22.94
(May–Nov) / 20.54 (all-train) / **13.01 (fresh tail)** — vol ~halved.

**Full-year WALK-FORWARD — the tight estimate** (`subs60_xgb_walkforward_adaptive.py`, rolling
W=200/T=30, 6 folds, ~150 OOS trades, maker 4 bp, 1/day, hold-60s). **SURFACE** = OOS net maker EV vs
(fold/regime × vol-gate threshold-adaptivity); **ARGMAX = regime-ADAPTIVE threshold**:

| branch | pooled OOS net | median | win | folds + |
|---|---|---|---|---|
| **adaptive** (p95 per fold) | **+13.58 ± 3.8 bp** | +7.09 | 97/155 (63 %) | **6/6** |
| fixed (frozen 22.95 bp) | +6.38 ± 4.4 | +3.99 | 90/151 | 5/6 |

Adaptive > fixed by **+7 bp pooled**; the gap concentrates in low-vol late folds where the frozen
high-vol threshold mis-calibrates A — `thr_ad` tracks **22.9→18.8 bp** as DOGE vol declines, and fold 5
swings **adaptive +13.25 vs fixed −13.92**. Fold 3 is a counterexample (fixed +12.76 vs adaptive +0.96),
so adaptive is not fold-wise monotone, but pooled it is clearly higher and positive every fold. The
single-split +5.73 is **not single-split luck on these conditions**; the lever surfaced here (user
insight) = the "5 % most-volatile deviation" hardcoded into Model A is **period-dependent**, so the
threshold should be regime-adaptive (rolling p95) in walk-forward folds and live.

**Caveats / next**: fills are maker-SIM (touch/queue/MISS), NOT live execution — **paper trading DOGE**
validates execution next; one symbol, one year; A/B HP reused (no per-fold re-tune). The §5 deploy-gate
(clears the 4 bp maker floor robustly OOS) is a secondary annotation, not a deploy verdict. Artifacts
`research_runs/maker_labels_rr/{WALKFORWARD_ADAPTIVE.json, freshtail→maker_labels_rr_freshtail/DOGE.npz}`;
scripts `subs60_{orch,btcmid_backfill,makerlabel_build,xgb_freshtail_eval,xgb_walkforward_adaptive}.py`,
`backfill_cryptolake_to_gcs.py`. Ledger `xgb-20260603_doge_actualize_walkforward`.


## 17. 2026-06-04 — ZERO-fee sub-minute directional-maker pivot + the f2 covariate-shift fix (DOGE)

**Context**: user pivot — trade **USDC** (maker fee **0**; net = gross) and **sub-minute holds**
("from now compute without maker fee"). DOGE-only. Ledger `xgb-20260604_doge_zerofee_volnorm_baseline`
(HD3 rev3, exploratory). VM `hd2-feats-003`.

**Data**: multi-hold maker labels (`subs60_makerlabel_build --holds-sec 15 30 60` →
`maker_labels_h/DOGE.npz`, N=2.15M; holds 15/30/60s × touch/queue + rH15/30/60). Per-fold prediction
caches (`wf_cache/DOGE_h{15,30,60}s_preds.npz`) → deploy/metric sweeps now instant.

**Horizon (15/30/60s, zero fee, causal rolling)** — Sharpe is the lens (bp/trade structurally favours
longer horizons = bigger moves). **30s is the risk-adjusted argmax**; **15s collapses CAUSALLY**
(Sharpe ≈ 0 — its post-hoc edge was look-ahead); 60s lower Sharpe. Model-A vol-AUC ~0.84–0.85 flat
across horizons (shorter ≠ stronger A-signal).

**Vol-gate A — needed?** At matched CAUSAL frequency **noA (B-full direction, no gate) ≥ AxB** and is
the deploy argmax (the gate's rolling threshold drift hurts the rank-product). Post-hoc AxB > noA, but
post-hoc is look-ahead.

**Causal ≠ post-hoc.** Post-hoc top-N (within-day ranking) is look-ahead, inflates ~3–4×; the
deployable rolling/frozen threshold is what counts. A learned trade-EV **selectivity model
underperformed** the pA×pB heuristic (per-trade EV too noisy to regress; the A(vol)×B(dir)
decomposition is better-conditioned).

**The f2 puzzle — a VOLATILITY COVARIATE SHIFT** (`subs60_xgb_fold_traindata`). The per-fold edge
concentrated in fold-2 (test 2026-01-26→02-25) because **f2 is the only fold whose TEST vol matches
its TRAIN**: DOGE vol declined monotonically over the year, so the trailing-200d train is always more
volatile than the recent test — except f2's Jan-Feb spike. test/train p95|rH30|: f0 0.67 · f1 0.73 ·
**f2 1.01** · f3 0.91 · f4 0.64 · f5 0.85. Microstructure features (OFI/returns/liq) **scale with
volatility** → compress in calm regimes → the model (**A AND B**) under-captures — a covariate shift
on the **shared features** (so noA, with no A, concentrated in f2 too). Drift ≈ 0, no trend, no
leakage (2-day embargo) — purely vol-scale.

**Fix = vol-NORMALIZE the features** (`subs60_xgb_volnorm`): causal trailing per-feature z-score
(K=20 prior days). **Works**: f2 collapses **+23.2 → +1.9**, calm folds lift (f0 −0.3 → +3.1),
per-fold becomes **broad** [3.1, 4.5, 1.9, −1.1, 4.0, −1.0].

**BASELINE (recorded)** — vol-norm noA 30s, causal rolling, target 5/day (actual 1.7/day), **zero
fee**: **annualized Sharpe +2.42**, EV **+4.26 bp/trade**, hit 51.1 %, **+7.14 bp/day**, broad
per-fold. The non-normalized annS **+5.39** was **f2-spike-inflated** (regime-dependent, unreliable);
the broad **+2.42** is the **honest regime-robust** baseline. *AUC/dir-acc is size-blind* — hit ~51 %
yet +EV because wins are bigger (momentum on big moves; size-weighted EV is the right metric).

**Caveats / next**: maker-SIM fills, NOT live; one symbol/year; **HP frozen** (Optuna-tuned on the
60s/non-normalized features → suboptimal here; per-fold Optuna on normalized feats queued); blanket
71-feat normalization (targeted norm untested); zero-fee assumes USDC maker 0 % (venue-confirm; Binance
testnet keys verified, DOGEUSDT TRADING, 5000 USDT demo). Iterate from this baseline. Scripts
`subs60_{makerlabel_build,xgb_horizon_wf,xgb_noA_test,xgb_causal_deploy,xgb_fold_traindata,xgb_volnorm,
xgb_baseline_v2}.py`. Ledger `xgb-20260604_doge_zerofee_volnorm_baseline`.

## 18. 2026-06-06 — size-aware (IC) per-fold B-tuning of the zero-fee directional-maker (DOGE)

**Question:** is XGBoost-B's hyperparameter / tuning-**metric** a lever for the thin direction
signal, or is the §17 frozen-HP baseline (annS +2.42) already at the ceiling?

**Surface — the B size-aware signal is real and stable.** Score B by **IC = corr(pB−0.5, netl−nets)**
on the last-30d sub-val (size-weighted, unlike AUC). Per-fold IC(val) =
**[+0.014, +0.021, +0.018, +0.011, +0.027, +0.024]** — positive on **all 6 folds**, mean **~+0.019**,
no sign flip. So B genuinely predicts the better maker **side of large moves** above chance every
fold; the AUC≈0.52 was **size-blind** and hid it (which is why tuning on AUC failed last run, +0.30).

**Argmax (policy × budget).** Per-fold Optuna (25 trials; A on sub-val AUC, B on sub-val IC),
blanket vol-norm, causal-rolling deploy, zero fee:

| pol | tgt | trd/d | EV/trd (bp) | annS | hit% | per-fold (%) |
|-----|-----|-------|-------------|------|------|--------------|
| **noA** | **10** | 9.8 | **+1.40** | **+2.45** | 53.0 | [12.1, 9.8, −3.1, 1.9, 7.4, −5.9] |
| noA | 5 | 4.7 | +1.76 | +2.18 | 55.7 | [6.7, 5.6, −0.9, 2.4, 4.7, −5.2] |
| AxB | 5 | 4.6 | +0.19 | +0.15 | 52.2 | [6.4, 14.1, −8.4, −0.9, −0.5, −9.3] |
| AxB | 10 | 10.6 | −1.56 | −2.11 | 48.5 | [6.9, 13.0, −16.5, 0.9, −16.2, −14.5] |

**noA dominates AxB** at both budgets — in zero-fee market-making the vol-gate A only sheds alpha.
EV/trade is **higher at t5** (+1.76 vs +1.40); annS is higher at t10 only because it scales with
√(trd/day) — frequency, not signal quality.

**The lever was the tuning metric, not HP/norm.** Three independent ways of turning the B knob around
the **same single signal** converge to annS ~+2.4: frozen HP **+2.42**, Optuna-on-AUC **+0.30**
(size-blind metric → val-overfit on a ~0.52 coin-flip → broke the deploy), Optuna-on-IC **+2.45**
(restores frozen parity). A size-aware objective is the correct knob — but it does **not lift the
ceiling**, which is set by the **thickness** of the single direction signal (IC≈0.019). HP / objective /
normalization cannot thicken it.

**Baseline (declared, working):** **noA t10 IC-tuned, annS +2.45**, EV +1.40 bp/trade, hit 53.0 %,
+13.83 bp/day, per-fold [12.1, 9.8, −3.1, 1.9, 7.4, −5.9] (supersedes §17 frozen-HP +2.42).

**Secondary (deploy annotation):** maker-SIM fills, not live; one symbol/year; zero-fee assumes USDC
maker 0 % (venue-confirm; Binance testnet ready, DOGEUSDT, 5000 USDT demo). **Next:** raise the
ceiling via **more symbols** (portfolio Sharpe on the same 30s zero-fee noA), then a **new uncorrelated
signal**. Scripts `subs60_xgb_optuna_ic.py` (+ `subs60_xgb_optuna_volnorm.py` AUC variant). Ledger
`xgb-20260606_doge_zerofee_ic_tuned_baseline`.

## 19. 2026-06-06 — realistic maker EXIT (pegged, always-last) + symmetric-fill conditioning (DOGE/ETH/BTC)

**Scope of validity (read before citing any number here):** all results below are for **DOGE/ETH/BTC
USDT-perps, Binance Futures, 2025-05-09…2026-05-08, 30s hold, walk-forward W=200/T=30 (6 folds),
ZERO maker fee + NO taker**, under a **specific execution model** (below). This is a measured
**response surface**, not a universal property — a different market / model / window / execution must
be re-measured before any claim here is reused.

**Two-leg realism upgrade.** Prior maker labels had a realistic ENTRY but an optimistic EXIT. A probe
of the OLD exit (hold-to-timeout → 2 s maker-at-mid → else taker-cross, commissions hardcoded 0)
found **59–83 % of BTC/ETH exits were TAKER** (TimeoutMarket), charged 0 fee; at the user-confirmed
**5 bp** taker fee that exit cost alone ≈ 10× the per-trade EV. So we built a **pegged maker-only
exit** (`simulate_pegged_exit` in `grid_sim_exitdbg.rs`): rest a close-limit pegged to the touch,
**always last in queue**, re-quote on adverse move (queue resets to back), held until opposing taker
flow fills it, mark at last touch on ran-out — **no taker anywhere**. Then set the ENTRY symmetric
**always-last** (`queue_mult=1`) to match.

**Conditional surface (per-trade annS / EV bp/trade; ENTRY+EXIT BOTH always-last):**

| sym | AxB t5 | AxB t10 | noA t5 | noA t10 |
|-----|--------|---------|--------|---------|
| **DOGE** | **+3.40 / +4.17** (hit 56, 5/6+) | +1.65 / +1.31 | −5.81 / −1.72 | −7.81 / −1.52 |
| **ETH** | **+3.24 / +3.50** (hit 57) | +2.54 / +1.59 (6/6+) | −21.5 / −0.96 | −28.8 / −1.05 |
| BTC | −0.23 / −0.14 (~flat) | −4.84 / −1.76 | −28.8 / −1.16 | −35.4 / −1.14 |

**Argmax under these conditions = AxB t5 for DOGE (annS +3.40, EV +4.17 bp) and ETH (+3.24 / +3.50)**,
both broad (5–6/6 folds positive). BTC ≈ flat — its entry fills only **52–57 %** when always-last
(deep book → large queue → most entries missed), vs DOGE 84–89 %, ETH 75–81 %.

**The conditional actually measured (do NOT generalize past it):** whether the **vol-gate A is
needed** depends on the **entry-fill assumption**, measured directly on this data —
- **front-of-queue entry (`queue_mult=0`, optimistic):** noA is **positive and ≥ AxB** (e.g. DOGE
  noA-t10 pegged-exit EV +2.26 / daily-annS +3.62);
- **always-last entry (`queue_mult=1`, this rev):** noA goes **negative**, **AxB dominates**;
- entry effect isolated (DOGE, OLD exit, B-only noA): qm0 EV **+1.40** → qm1 EV **−1.18**; entry fill
  rate **0.978 → 0.82**.

So for **this** symbol-set / year / horizon / execution, "is the vol-gate needed?" answers **no under
front-of-queue, yes under always-last** — a surface, not a verdict on noA vs AxB in general.

**Conditioning of earlier results:** every prior maker number in this log (the +5.73 apred cascade,
§16–18 baselines, the portfolio) was computed under **`queue_mult=0` front-of-queue entry** → their
absolute levels are conditioned on that optimistic-entry assumption. The direction signal (IC) is
stable across the change: DOGE ≈ 0.04, BTC ≈ 0.056, ETH ≈ 0.066.

**Caveats (bound the numbers):** maker-SIM fills (touch/queue/flow), not live; `entry_q` (entry-time
touch depth) used as the exit-queue-depth proxy — true per-re-quote depth needs a `build_samples`
L1-qty output; no re-quote latency; the pegged exit keeps queue priority on favorable up-ticks (mild
winner-optimism); per-trade annS overstates daily by ~1.3–1.5× (DOGE AxB-t5 daily-annS ≈ +2.7); one
symbol-set, one year. **Reproduce:** per symbol `subs60_makerlabel_build.py … --grid-bin
/tmp/edbg_target/release/grid_sim_exitdbg --exit-queue-mult 1 --queue-mults 1.0`, then
`subs60_xgb_optuna_ic.py {sym} maker_labels_pegexit_qm1 0`. Exit source `scripts/grid_sim_exitdbg.rs`
(`simulate_pegged_exit`), built on VM `/tmp/husdc` against `live_sim` (has `simulate_maker_entry`).
Ledger `xgb-20260606_pegexit_alwayslast_axb_noA_doge_eth_btc`.

## 20. 2026-07-05 — HONEST time-based windows + the label-matching lookahead (HD3 rev6)

**Scope:** DOGE-USDT-PERP CL year 2025-05-09…2026-06-02, same walk-forward protocol as §18-19
(`subs60_xgb_optuna_ic`, W=200/T=30/EMB=2, per-fold Optuna A-AUC+B-IC, causal-rolling, zero fee).

**Motivation (audit):** CL books are event-driven (1.45-2.32 snaps/s), so §19's tick windows were NOT
wall-clock: "282t hold" = 2-3 min, "120t entry" = 51-83 s. Live executes 12.8 s entry / 30 s hold.
Rebuilt the year with TIME-BASED windows (grid_sim time-mode, verified vs python replication:
1000 samples, 0 mismatches; hold median 30.76 s) at a 3 s time-uniform decision grid (live cadence).

**Honest surface (live-parity inputs btc=0; gross, 0 fee):** AxB EV/tr **t5 −3.77 / t10 −3.20 /
t20 −3.30 / t40 −3.16**; all 7 folds ≤ +0.1 at t5. Real-btc variant +0.3-0.6 bp better, sign unchanged.
B IC(val) +0.049…+0.090 on every fold — the direction signal is alive; selection does not convert it
at 30 s (top-tail = A's vol bursts, where the 30 s maker cycle is most adverse; selected −3.77 vs
unconditional filled −2.8).

**Root cause of the historical positive year (cell isolation):** `makerlabel_build` matched each
decision point to the NEAREST build sample within **±2.5 s** → in ~half the rows the maker entry was
priced/queued at a book tick up to 2.5 s in the PAST relative to feature time = **label-matching
lookahead**. Rebuilding labels at the EXACT feature tick (robust2 F + grid + SAME 2-3-min tick
windows, only matching removed): **+3.77 → −1.74** (t5; per-fold −7.5, +14.9, −7.2, −3.5, +0.1, −11.0),
+1.95 → −0.72 (t10). **≈5.5 bp/tr of the recorded edge was the artifact.** §19's qm1 baseline
(+4.17/annS +3.40) used the same matching and carries the same inflation.

**Cell matrix (EV/tr t5):** qm1 matched +4.17 | robust2 matched +3.77 | robust2 EXACT-labels **−1.74** |
my-grid exact (legacy repro, 388 d) +0.91 | honest 30 s 3 s-grid **−3.77** | **live-config**
(2-3-min-trained B → honest 30 s execution, ts-join) **−2.35** (per-fold −2.0, −4.6, −9.3, −2.6,
**+1.2, +2.7**, −0.0 — the two most recent folds positive, consistent with the matching-clean 7-day
recorder-EV +10.3 = regime wobble, not persistent edge).

**Pipeline validation:** recomputing FB-full features at robust2's exact ticks + robust2's own labels
reproduced the recorded +3.77 **bit-exactly** → the new build path is exonerated, and robust2's F ≡
FB-full (so qm1 vs robust2 differ ONLY in vol cols 34-39 — which alone reshuffled the fold structure
[9.6, 11.3, 4.9, 2.5, 4.0, −0.6] → [−1.1, 23.2, 16.0, −1.9, −3.4, −3.9]).

**Selectivity mechanics (tested on saved per-fold scores, no retrain):** take-clustering at the 3 s
grid is real (47-50 % of takes < 60 s apart) but bounded: 12 s/24 s subsampling → −3.67/−2.40;
cooldown 30/60/120 s → best −1.11; equal-quantile control clean. ≤ 1-2.5 bp, does not flip the sign.

**Standing conclusion (this data/model/execution):** under exact label alignment the AxB maker family
shows ~0 (±1.5 fold-noise) EV at the old 2-3-min semantics and −2.4…−3.8 bp/tr at honest 30 s live
semantics, all budgets. Only regime-conditional positives remain (recent folds, 7-d recorder-EV).
**Rule going forward: labels must be built at the exact decision tick — never nearest-matched.**
Artifacts: `research_runs/maker_labels_tb3s*`, `maker_labels_robust2_{fbfull,exactlab}`; ledger rev6.

### §20a. 2026-07-05 — CORRECTION: the "label-matching lookahead" attribution is WITHDRAWN

A model-free test refuted the mechanism claimed in §20: the build grid is dense (40k/day →
matched offset |dt| median ~0.3 s, overlap move p90 = 0.91 bp) and the matched labels do NOT pay
the overlap move (corr(r_ov, netl): matched −0.031 vs exact −0.042; follow-overlap-side tail EV
negative on both). Row-level matched-vs-exact across the year: **corr 0.99, mean diff −0.001 bp,
fill agreement 97.4 %, yB target agreement 95.0 %** — no leak, no systematic label change.

**Revised interpretation of +3.77 → −1.74:** a ~5 % zero-mean perturbation of B's training targets
swings the year walk-forward total by ~5.5 bp ⇒ the protocol's year-EV estimate has a **structural
variance of several bp** (top-5/day tail selection × deep trees × per-fold Optuna). The
old-semantics measurements (+4.17, +3.77, +0.91, −1.74) are draws from one wide distribution
(mean ≈ +1 ± 3): **the year never statistically identified an edge at the old semantics** — which
retroactively explains the fold reshuffle from the 6-column feature swap and DOGE-only positivity.
The honest-30 s cells are consistently negative across all draws (−3.8…−2.4, every budget, both
training horizons) — that conclusion is stable.

**Process rule:** any year-EV from this protocol family requires a **perturbation gate** — retrain
on label-jittered / target-flipped copies; the spread across copies is the error bar. §20's window
semantics facts, pipeline validation, and selectivity bounds stand; only the lookahead attribution
is withdrawn.

### §20b. 2026-07-06 — seed-variance proof (closes the "high Sharpe = significant" challenge)

Byte-identical robust2 data, same code, ONLY the RNG seed varied: AxB t5 year EV = **+3.77 (seed 0,
recorded) / −1.87 / +2.81 / +2.07** (seeds 1-3); t10 +1.95/−0.42/+0.39/+1.76. **5.6 bp peak-to-peak
from RNG alone** — the per-trade Sharpe CI is conditional on one training realization. Combined
equivalent-realization ensemble at the old tick semantics: {+4.17, +3.77, +2.81, +2.07, +0.91, −1.74,
−1.87} = **+1.4 ± 2.4 bp/tr**. The regime PATTERN reproduces across seeds (winter folds usually
positive, last fold consistently negative) but fold amplitudes swing tens of bp. The deployed live
weights are literally the seed-0 member (train_deploy SEED=0). Seed/perturbation gate mandatory.

## 21. 2026-07-07 — H150 honest cell: cross-symbol seed ensembles (HD3 rev6 close-out)

**Cell:** honest time-based maker cycle — entry 60 s from decision, hold **150 s FROM FILL**
(cfgs 90/150/240 s), pegged never-taker chase 300 s, always-last both legs, 0 fee, 3 s grid,
**FULL features** (funding/liq/OI/ETH/btc real), exact labels, CL-year ~371 d, **4 seeds/symbol**.

| sym | t5 EV (seeds) | t5 min | annS mean/min | t10 EV / min | gate |
|-----|---------------|--------|---------------|--------------|------|
| DOGE | **+6.27 ± 0.90** [7.2 5.0 5.9 7.0] | +4.97 | +3.71 / +2.54 | +3.53 / +2.49 | **PASS** |
| ETH | +4.29 ± 4.24 [11.5 0.4 2.9 2.4] | +0.42 | +2.19 / +0.30 | +3.23 / **−1.52** | FAIL (sd ≈ mean) |
| BTC | **+9.69 ± 1.82** [8.3 11.8 11.1 7.5] | +7.48 | **+6.95 / +5.65** | +4.63 / +3.43 | **PASS** (winter-heavy folds, avg worst −9.8) |

**Robustness argmax:** DOGE **t10** (folds>0 79 %, avg worst fold **+0.4** — no losing month on
average; annS min +2.59 ≥ t5's). **Hold sweep** (DOGE seed-0 scores, no retrain): 90 s +3.20 /
**150 s +7.19** / 240 s +0.57 → interior max: the reversion pays in ~2-2.5 min and decays by 4 min.
**Surprising conditional:** BTC — flat at the old 30 s tick semantics (§19 −0.14) — is the
STRONGEST at 150 s honest with full features despite always-last fill ~0.5: deep-book symbols need
the longer horizon + funding/OI/liq information. First cells in the program where edge > estimator
noise under honest semantics (contrast §20a/b).

**Deploy package (pending user):** DOGE t10 (+ optional BTC), 4-seed ENSEMBLE scoring (mean
pA/pBg — harvests the measured seed noise), USDC pairs. Prereqs: recorder-EV cross-check of this
config on live recorder days; live engine wiring (btc bookTicker, funding, OI, ETH aggTrade;
DECIDE_S 3 s, ENTRY_WIN_S 60, HOLD_S 150). Ops note: systemd-run expands `${VAR}` in payloads →
use bare `$VAR`/script files; 5 seed aggregates were overwritten and deterministically rerun.

## 22. 2026-07-10 — BASELINE: anchored h150 ensemble (DEPLOYED) — year-validated, bit-exact live engine

**Status: NEW PROJECT BASELINE.** First configuration in this project with (a) a year-scale
walk-forward measurement of the exact traded policy, (b) a live engine proven bit-identical
to the validation pipeline, and (c) live trading running on it (axb-engine-doge, DOGEUSDC,
since 2026-07-09 09:13 UTC; first live trade 2026-07-08 verified inside the sim envelope).

### 22.1 The strategy (what actually trades)

- **Signal universe / venue split**: signal computed on DOGEUSDT perp streams (book diffs,
  aggTrades, forceOrders, markPrice, OI poll, ETH-lead trades, BTC-lead L1 mid); execution
  on DOGEUSDC (0% maker fee venue).
- **Decision cadence**: every 3s on the exchange-timestamp grid anchored at calendar UTC
  midnight; decision tick = last book tick <= grid point (np.unique dedupe semantics).
- **Model**: two-head XGBoost per seed — A (activity: |rH| top-5% classifier, AUC-tuned) x
  Bg (direction: gated fill-weighted classifier, size-aware-IC-tuned); per-seed score =
  rankCDF(pA) * rankCDF(|pBg-0.5|); **deployed score = mean over 4 seeds** (ensemble is
  load-bearing — see 22.2); side = mean pBg >= 0.5.
- **Selectivity**: causal day-level tau (np.quantile of a 30-day rolling score buffer,
  budget t5 = 5 trades/day nominal; realized ~3/day); tau frozen within a UTC day.
- **Execution policy**: maker-only. GTX entry at touch, 60s entry window from decision;
  hold 150s FROM FILL; pegged maker-only reduce-only exit chased until filled (taker
  backstop only at catastrophic guards). Notional = 1x deposit (SIZE_FRAC=1.0, lev 2 =
  margin headroom only).
- **Feature semantics (ANCHORED — intentional, validated)**: col13 (funding rate) frozen at
  the day's first mark_price value; col44 (funding basis) = 0. Found via the 2026-07-08
  funding ns/ms bug forensics (ledger, fix 079fa29): the "broken" day-anchored variant
  measured ROBUST (+8.6bp, LOO 0/10 neg, jitter P>0=100%) while the "correct" true-funding
  semantics measured -2.1bp (noise) on the same recorder days — freezing a hypersensitive
  input (±0.17 score per 1e-4 raw) is variance reduction that tail selection rewards.
  Adopted as the policy definition, not a bug.

### 22.2 The result (all cells preregistered / frozen-protocol)

| Cell | EV/tr t5 | Robustness |
|---|---|---|
| **YEAR x ENSEMBLE (= deployed scoring)** — 371d CL, 6 walk-forward folds (W200/T30/EMB2), 563 trades, 3.1/day, hit 65.2% | **+13.35bp** | ALL 6 folds positive (+2.1..+31.0 %/fold-month); LOFO +10.85..+15.19; score-jitter sd=0.02 -> p50 +7.49, sd=0.05 -> p50 +3.13, **P(EV>0)=100% both** |
| YEAR x per-seed (4 seeds) | +8.14±2.55 [7.0/12.5/6.0/7.1] | 4/4 seeds positive; fold2 carries ~60% (LOFO-fold2 +3.50); jitter-fragile (sd0.05 -> +0.24, P>0=74%) |
| 10 recorder days x ensemble (live venue view) | +8.61bp (66tr) | LOO 0/10 negative; jitter P>0 = 100/100/98% at sd .02/.05/.10 |
| t10 (per-seed year) | +4.64±0.93 | budget surface monotone t5>t10>t20>t40~0 |

Consistency triangle: year-ensemble +13.4 / year-per-seed +8.1 / recorder-10d-ensemble +8.6
— one alpha class across two venues and two horizons. **Conservative stressed floor for the
traded config ~ +3bp/tr** (year ensemble under sd=0.05 selection noise, never negative in
100 reps). ROI translation at current size (~$10 notional): ~40bp/day base (~+12%/mo, all
fold-months positive), ~10bp/day stressed floor (~+3%/mo). CAPACITY CAVEAT: these are edge
densities at $10-1k notional (always-last queue model); not scalable ROI claims.

Key structural finding: **ensemble averaging of rank-scores is load-bearing** — it removes
both the fold concentration (fold4: negative in all 4 seeds per-seed -> +2.5bp/tr ensemble)
and the selection-noise fragility (P>0 at sd0.05: 74% per-seed -> 100% ensemble). The
multiplicative rank-score is structurally fragile near pBg~0.5 (measured live: ~5 sim-only
takes/day from independent-WS jitter); averaging 4 seeds is what stabilizes the tail.

### 22.3 Architecture (as deployed)

- **axb_engine (Rust, ~70us decision path, live 1.0-1.2ms incl. logging)** on the recorder
  VM: MirrorBook full-book reconstruction from @depth@100ms diffs (port of the recorder's
  OrderBookV2: REST seed limit=100, cap 100 levels, skip u<=last, no pu-chain, reconcile
  900s / reseed>=2 findings) for DOGEUSDT + BTCUSDT; features_incr day-anchored append-only
  prefix state (midnight reset == per-day sim files); gbt bit-exact XGBoost predictor;
  causal tau port; decision JSONL; orders via Unix socket.
- **axb_exec (Python sidecar)**: verbatim battle-tested maker trade lifecycle + own
  DOGEUSDC bookTicker WS + hourly GCS decision upload.
- **axb_boot (Python, ExecStartPre)**: GCS bundle -> npys; **empirically solved xgboost
  base-margin bits** (one-tree equation; the float ProbToMargin formula is 1 ulp off on 1
  of 8 models); tau seed from the anchored recorder score distribution; funding day-anchor
  (recorder local file -> recorder GCS bucket -> REST fallback).
- **Bit-exactness keys** (all measured, not assumed): day-anchored prefixes reproduce batch
  float summation order; xgboost f32 margin accumulates base-FIRST; sigmoid = glibc expf
  (numpy SIMD exp does NOT match); leaf values live in split_conditions.

### 22.4 How it was validated (the part that makes this a baseline)

1. **Golden parity harnesses** (fb_incr_harness + score_harness, day 20260707, 28546
   samples): features 0/2.03M cells mismatched vs frozen feature_builder; F71 (incl. libm
   ToD/btc_lead) 0/2.03M; 8-model predictions 0/228k vs Python xgboost; ensemble score
   0/28546. The engine equals the validation pipeline BYTE-FOR-BYTE by construction; any
   model/feature change must re-pass both harnesses before deploy.
2. **Year cell, frozen protocol, preregistered** (ledger tb3s-20260709_h150anch_year_4seeds
   BEFORE running): the original subs60_xgb_optuna_ic.py byte-unchanged; intervention =
   dataset only (col13 := day-first, col44 := 0 on the tb3s h150 combined npz). Per-fold
   Optuna (25 trials, A-AUC/B-IC), 4 seeds sequential, full artifact capture (per-fold
   scores, OPTUNA jsons, run log -> GCS).
3. **Robustness battery on every cell**: leave-one-fold/day-out + score perturbation at the
   measured live-jitter scale (live-vs-sim same-tick score |d| p50 0.007-0.026) — the
   perturbation gate is now REQUIRED for any tail-selection EV claim (extends the §20 rule).
4. **EV(latency) sweep** (entry-delay-patched grid_sim, selection held fixed): EV flat
   +8.6..+10.4bp across 0-3000ms entry delay -> the policy is latency-insensitive; the
   engine's value is parity-by-construction + CPU, NOT latency alpha (ledger
   latsweep-20260709).
5. **Sim-live execution parity** (first live trade, 2026-07-08): long 139 DOGE @0.07311 ->
   0.07291, -27.4bp live vs sim adjacent-tick envelope -24.6/-35.5bp; ROI@2x -0.55%
   reproduced. Decision-layer live parity: take5 agreement 74/74 vs the Python engine in
   shadow; tau matched to 6dp.

### 22.5 Known limitations / next

- Majority-vote side approximation in the year-ensemble cell (15.9% raw 2-2 ties; deployed
  uses mean pBg) — exact-side rerun is cheap if it ever matters.
- Regime structure: the edge is carried by strong windows (fold2-type months up to +30%);
  months near zero are NORMAL; one mild negative regime exists per-seed (fold4) though the
  ensemble held it positive.
- Capacity curve unmeasured beyond ~$1k notional — scale-up must be data-driven (live
  fill-rate / queue degradation), NO realized-EV-based scaling before ~100-200 live trades
  (se ~ 2bp at 3/day ~ 1-2 months).
- Next: WS capture layer (engine records its own consumed stream; daily replay must equal
  the decision log bit-for-bit) — removes the last measurement/live gap; twin-engine
  session-jitter quantification; per-trade live-vs-sim execution ledger as trades accrue.

## 23. 2026-07-11 — HD3 rev8: cross-symbol year surface of the anchored h150 policy (7 symbols)

**Question (exploratory):** on which symbols does the DEPLOYED anchored-h150 policy class
(60s entry / 150s hold from fill / pegged never-taker chase / anchored funding / causal t5 /
4-seed ensemble) yield year-scale alpha, how large, and how seed/selection-stable?
Protocol byte-frozen = rev7; new datasets (BNB/LTC/SOL/XRP/LINK) built at the recovered
cross-symbol parameter **H_TICKS=1800** (FULLFEAT; bit-exact pipeline reproduction proven
before launch); BTC/ETH label files reused from s21 builds; anchored intervention
col13:=day-first, col44:=0. Ledger `tb3s-20260710_h150anch_year_xsym` (+ 2 preregistered
amendments).

**The surface** (t5, bp/trade; per-seed = 4 seeds mean±sd; ENS = deployed mean-rank scoring,
majority-vote side; jitter = P(EV>0) under score perturbation, 100 reps):

| sym | per-seed EV [seeds] | ENS EV (n, hit) | LOFO min | jitter sd.02 / sd.05 |
|---|---|---|---|---|
| DOGE (ref, s22) | +8.14±2.55 [7.0/12.5/6.0/7.1] 4/4 | **+13.35** (563, 65.2%) | +10.85 | P100% / **P100%** |
| **XRP** | **+10.97±3.50** [10.6/11.4/6.0/15.9] 4/4 | **+12.68** (504, 58.5%) | +8.36 | P100% / P41% |
| BTC | +6.20±2.76 [4.5/6.9/10.4/3.0] 4/4 | +3.83 (574, 54.9%) | +1.11 | P56% / P0% |
| ETH | +4.86±3.83 [-0.7/7.1/3.5/9.5] 3/4 | +7.36 (419, 55.6%) | +3.34 | P23% / P4% |
| SOL | -2.59±4.37 | -5.05 (773) | — | P0% / P0% |
| BNB | -4.24±0.57 0/4 | -3.00 (775) | — | P0% / P0% |
| LTC | -8.77±4.29 0/4 | -10.77 (611) | — | P0% / P0% |
| LINK | degenerate: 246d/2 folds, causal t5 selects 0-3 trades/seed | — | — | — |

**Conditions that drive the surface (the deliverable):**
1. The policy class carries year alpha on a **minority of symbols**; argmax = XRP at
   DOGE-magnitude. Positive cells are thin-book alts (DOGE 1.4-2.3/s, XRP ~1.9/s).
2. **Ensemble stabilization is NOT universal**: DOGE amplifies (+8.1→+13.4) and is
   jitter-proof at sd.05; XRP amplifies (+11.0→+12.7) but survives only sd.02; BTC
   ensemble DEGRADES the per-seed mean (+6.2→+3.8) and is selection-fragile. The
   load-bearing DOGE finding (s22) does not transfer for free.
3. Sign does not reduce to book density: LTC/BNB (~1.3-1.4/s) and SOL (~2.3/s) are
   negative at densities similar to the positives. Symbol identity matters beyond density.
4. BTC/ETH cells carry the **H_TICKS=1800 caveat** (~200s forward path at ~9 ticks/s →
   chase run-out marked at touch more often) — not execution-comparable to thin-book cells.
5. Fold structure: XRP concentrated in folds 1-3, BTC in fold3; LTC negative broadly
   (5/6 folds); months near zero remain normal for this class (s22).

**Secondary deploy-gate annotation** (confirmatory question only): XRP = conditional
candidate — passes seed-gate (mean−sd = +7.47) and ens jitter at sd.02 (P100), fails sd.05
(P41); next step per prereg = rev6-style recorder-EV cross-check with that flag. BTC fails
the required perturbation gate despite seed-gate pass. ETH marginal. Others fail. No
capital action taken.

**Ops (research-throughput line):** campaign ran on 2 VMs (n2-highmem-8 → n2-highmem-32 in
the 32-vCPU delmiron27 project), 28 seed-runs ≈ 200 core-h ≈ $15, ~19h wall first-launch →
last-ensemble incl. three debugged incidents (OOM cgroup kill; /tmp binary wipe → partial
datasets caught+guarded; GCE default read-only scopes). Seed-parallel orchestrator v2
(`research/runtime/orchestrate2.py`, per-seed jsons recomputed from PERFOLD — <1e-7bp vs
direct) did 15 seed-runs in 5.7h. Next-run projection with baked image + full parallelism:
~3h end-to-end. All run knowledge captured in `research/runtime/KNOWN_PITFALLS.md`.

## 24. 2026-07-12 — HD4 rev1: 10s-направление предсказуемо model-free на всех 4 символах; BTC — сильнейший (dir10 screen)

**Program**: BTC/ETH expansion, user hypothesis 2 (продление h150-холда 10s-квантами по
сигналу направления в t=fill+150s). Rev1 = stage-1 exploratory screen: predictability of
the NEXT-10s mid direction from the deployed algorithm's own feature signals, **no ML, no
fitting** (prereg `dir10-20260712_cl_screen`; script `scripts/subs60_dir10_screen.py`).
Cell: CL year (2025-05-09..2026-06-02), tb3s 3s decision grid (maker_labels_tb3s_h150
dailies, F71), forward mid log-ret at H∈{5,10,15,20,30,60}s (realized horizon ∈[H−2,H],
no future-side slack), daily rank-IC / dir-hit / signed capture at |signal| cuts
q∈{0,.5,.9,.99}; COMP = preregistered fixed-sign rank composite
[0,1,12,26,27,28,62,63,64,65,66].

**The surface (headline, H=10s, mean daily rank-IC over the year):**

| sym | COMP ric@10s | argmax feature ric@10s | hit@q90 | capture @q90 / @q99 (bp/10s) | months ric>0 |
|---|---|---|---|---|---|
| **BTC** | **+0.239** | OBI_L1/microprice **+0.266** | **0.666** | +0.73 / +0.85 | 14/14 (+0.13..+0.33) |
| ETH | +0.146 | OBI_L1/imb_L5 +0.169 | 0.62 | +0.76 / +0.87 | 13/13 |
| XRP | +0.143 | microprice +0.189 | 0.576 | +0.80 / +0.93 | 13/13 |
| DOGE | +0.119 | microprice/OBI_L1 +0.171 | 0.561 | +0.86 / +0.95 | 13/13 |

Conditions that drive the surface:
1. **Argmax = мгновенная форма книги** (OBI_L1/microprice ≈ rank-дубликаты, imb_L5,
   OBI_L10/L20). OFI-семейство (1–5s суммы) — самое стабильное по дням (BTC imb_d5
   t=+141), но capture ниже (~+0.3–0.4bp). Кросс-АКТИВНЫЙ суб-минутный лид жив, но
   второго порядка (eth_r5→BTC ric +0.106, t+72; btc_r* нигде не в топ-12).
2. **Порядок символов ИНВЕРТИРОВАН относительно h150 maker-EV**: BTC (слабейший в year
   maker-EV) — сильнейший по 10s-направлению; плотная книга делает book-shape
   информативнее. Сигнал для продления холда есть именно там, где базовая политика
   слабее всего.
3. **Decay**: rank-IC ~половинится 5s→15s (BTC COMP +0.30→+0.20); кумулятивный capture
   слабо растёт с H при падающем hit — 10s-квант около колена кривой.
4. **Монотонность по силе сигнала** на всех символах/горизонтах (COMP@10s q0→q99:
   +0.28→+0.77..0.84bp) — «уверенность» для гейта продления существует.

**Что это НЕ измеряет (stage 2, отдельный prereg):** capture = валовый mid-ход за
горизонт, не EV продления (нет стоимости перестановки pegged-exit / adverse selection);
ячейки безусловные — не conditioned на «150s внутри заполненной h150-позиции»;
кросс-БИРЖЕВЫЕ колонки в CL структурно мертвы → rev на данных рекордера следом.
Coverage r10: BTC/ETH ~0.99, DOGE/XRP 0.77–0.81 (свойство ячейки: разреженная книга +
правило [H−2,H]). Artifacts: `research_runs/h2_dir10/` (daily {ts,mid,bid0,ask0,R,RV} —
переиспользуемы для stage-2 сима; per-day stat tensors; code). VM dir10-1, ~35 мин, <$1.

**s24 addendum (tail-selectivity, user question)**: top-K/день cut (K=5 ≈ деплойные
0.017%): захват НАСЫЩАЕТСЯ в хвосте — 1%→0.02% добавляет лишь ~10–25% (BTC imb_L5@10s
+0.84→+1.02bp hit 0.72; DOGE OBI_L1 +0.93→+1.14bp; ETH/XRP COMP ~+1.0/+1.35bp), в
отличие от обученного скора, где хвост несёт всё. Осторожно: f32 top3_asym в экстремальном
хвосте DOGE/XRP ИНВЕРТИРУЕТСЯ (реверсия) — монотонность хвоста проверять посигнально;
book-shape и COMP монотонны на всех 4. Ledger `dir10-20260712_cl_screen_tailamend`.

**s24 addendum-2 (chained 10s holds, user rule)**: цепочечное правило работает model-free,
но его рабочая форма — **порог продолжения мягче порога входа** (вход q90/q99, продолжение
|S|>q50 того же знака): длина цепи 1.3–1.75 окон, P(≥2) 0.20–0.42, продление добавляет
+0.16..+0.28bp/эпизод к +0.65..+0.97bp первого окна (~+20–35%), added>0 в 95–100% дней
(q90-вход). Строгое продолжение (q_cont=q_entry) убивает цепи (len→1.0, added→0).
Пошаговый захват положителен на шагах 2–4 на всех символах (BTC +0.76/+0.48/+0.44/+0.37);
выходы: ~60% flip / ~35% weak. BTC/ETH добавляют больше DOGE/XRP — согласуется с порядком
предиктивности. Ledger `dir10-20260712_cl_chain`.

**s24 addendum-3 (квант удержания 30s/60s vs 10s)**: тотал bp/эпизод ПЛОСКИЙ по квантам
(~+1.0–1.26 при 10/30/60s), длительность растёт 3–6×, bp/сек падает 0.07–0.09 → 0.012–
0.017, стабильность added по дням деградирует (BTC d+ 0.97→0.78, ETH 0.86→0.64). Добыча
сигнала ~1bp реализуется в первые ~10–30 секунд; 10s-квант — argmax по времени и
стабильности (согласуется с decay rank-IC 5→15s). Мягкое продолжение работает на всех
квантах; строгое убивает цепи везде. Ledger `dir10-20260712_cl_chain_h3060`.

## 25. 2026-07-12 — HD5 rev1: статичный порог селективности (гипотеза 1) — поверхность

**Вопрос**: захардкоженный порог уверенности вместо динамического causal tau (правило
юзера: среднее пост-фактум топ-K скоров по дням валидационного окна). Ре-анализ PERFOLD
(anchored h150, 4 seeds × 6 folds, ens-скоринг), базлайн = динамический tau на тех же
артефактах (воспроизведён точно). Ledger `h1fixedtau-20260712_cl_year` (+FIXQ-амендмент).

**Буквальная форма правила сломана шкалой скора**: тестовые AxB-скоры клипаются в 1.0
(rank-CDF относительно train-окна), порог «среднее топ-K» ≈0.998 пропускает 32–60
сделок/день; EV/tr → BTC −0.67 / ETH −1.72 / XRP −1.83 / DOGE +1.33.

**FIXQ-форма (замороженный квантиль валидации под K/день, без дневной адаптации) — честный
тест «статичный vs адаптивный»**. Главный механизм: статичный порог НЕ уменьшает число
сделок, а **перераспределяет их во времени** — 69–86% дней без сделок, торговля
концентрируется в ~15–30% дней по ~15–20/день (высокоскорные режимы):

| cell | FIXQ EV/tr (bpd) | DYN EV/tr (bpd) | FIXQ jitter .02/.05 |
|---|---|---|---|
| DOGE t5 | **+17.07** (+55.7) | +13.35 (+44.5) | p50 +7.46 **P100** / +2.92 **P100** |
| DOGE t10 | **+10.51** (+74.9) | +8.42 (+61.5) | P100 / P100 |
| XRP t5 | +21.31 (+52.4) | +12.68 (+39.2) | p50 +3.64 P100 / **P7** (DYN P41) |
| BTC t5 | +2.25 (+5.1) | +3.83 (+13.1) | P66 / P0 |
| ETH t5 | +5.08 (+7.7) | +7.36 (+19.4) | P5 / P1 |

Чтение: на тонких книгах (DOGE/XRP) уровень скора несёт режимную информацию — концентрация
в высокоскорные режимы и есть alpha (вспоминаемый юзером эффект «EV/tr сильно растёт»
подтверждён); на BTC/ETH уровень скора не переносится между режимами, статичная планка
просто пропускает хорошие месяцы. **Гипотеза 1 не чинит BTC/ETH** (цели расширения).
Secondary deploy-примечание: DOGE FIXQ t5/t10 — единственная ячейка, бьющая деплой-политику
по базовому EV при полном прохождении jitter-батареи; кандидат на recorder cross-check,
капитальных действий нет. Caveats: FIXQ рекалибруется на границе фолда (месячно);
occupancy-профиль (86% пустых дней) для портфеля не оценён.
