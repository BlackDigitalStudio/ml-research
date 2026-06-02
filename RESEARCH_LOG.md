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
