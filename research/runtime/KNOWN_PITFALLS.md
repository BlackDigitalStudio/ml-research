# Known pitfalls — each entry cost real hours once. Check BEFORE debugging "new" issues.

## GCP / VM
- **/tmp is wiped on every VM stop/start.** Binaries built into /tmp die on resize/
  restart, and the dataset builder then "succeeds" with per-day EXC lines and rc=0
  (it catches exceptions per day) -> partial garbage datasets got combined (2026-07-10).
  Fix: binaries in HOME (`bins.sh`), + completeness floor before combine (orchestrate v1
  `MIN_DAYS`), + train guard on daily count.
- **GCE default access scopes are storage READ-ONLY** regardless of IAM role. A VM
  created without `--scopes=cloud-platform` gets 403 `Provided scope(s) are not
  authorized` on upload. Scope change requires stop -> set-service-account -> start,
  and gsutil may cache the OLD token (`rm -rf ~/.gsutil`).
- **systemd kills the whole unit cgroup on OOM of one child** (default OOMPolicy=stop):
  one over-RAM training killed orchestrator + all builds (2026-07-10). Launch runners
  with `-p OOMPolicy=continue` and let the runner retry the killed job.
- **systemd-run expands `${VAR}` (braced) inside payloads** against unit env -> empty
  substitutions (2026-07-07 incident: seed aggregates overwritten). Use script files
  or bare `$VAR`; `--setenv` is safe.
- **CPU quota**: project-0998ac51 global 12 vCPU; gen-lang-client-0075410383 global cap
  `CPUS_ALL_REGIONS=32` (region N2 limit is higher — the GLOBAL cap binds).
- Debian 12 pip needs `--break-system-packages` (PEP 668).
- **Training RAM is 13-14GB/job** (371d dataset, float64 F + Fn + DMatrix). n2-standard-8
  (31GB) fits TWO jobs, not three-plus-builds. highmem for parallel campaigns.

## Process management
- `pgrep -f 'pattern'` matches YOUR OWN ssh/bash wrapper carrying the pattern string —
  `kill -STOP $(pgrep -f ...)` froze the ssh session itself (2026-07-10). Match exactly:
  `ps -eo pid,args | awk '$2=="/usr/bin/python3" && $3=="<script>" {print $1}'`.
- SIGSTOP on an orchestrator is a safe freeze: child trainings keep running (they
  upload their own artifacts at process end); finished children become zombies with
  exit codes preserved. Used to migrate runs between VMs without killing healthy work.
- Windows local gsutil may be broken (`python3.14: command not found`) — use
  `gcloud storage` locally; gsutil works on the VMs.

## Protocol / data
- **H_TICKS IS PER-SYMBOL AND MUST BE MEASURED, NEVER ASSUMED** (2026-07-30). Verified by
  exact ts match against the parent dailies on two sampled days each:
  **DOGE 1500** (the DOGE-dedicated builder), **XRP 1800** (cross-symbol h150),
  **BTC 5100, ETH 5100** (the h150d dense rebuild). Getting it wrong shifts the decision
  grid, so the rebuilt rows do not line up with the dataset and nothing joins — the
  strict-fill runner's ts gate caught exactly this on its first DOGE smoke. To recover it
  for a dataset with unknown provenance: rebuild the 3s grid from the raw book and find
  the H whose `ends` filter reproduces the stored `ts` array element-for-element.
- **H_TICKS=1800 is the cross-symbol tb3s/h150 build parameter** (recovered from
  journald unit tb3sym). The DOGE-dedicated `subs60_tb3s_h150_build.py` uses H=1500 —
  NOT the cross-symbol protocol. At H=1800, dense books (~9 ticks/s: BTC/ETH) get only
  ~200s forward path -> exit chase run-out marked at touch more often than on ~2/s
  books (~900s). Property of the existing cells; keep for comparability.
- **FULLFEAT=1 or funding/liq/OI/eth cols are silently zero** (the historical
  qm1-rebuild bug; CLAUDE.md interpreting-records rule 2a).
- The shared `OPTUNA_IC_{SYM}_qm0.json` is write-racy under parallel seeds — dead
  artifact; per-seed jsons come from `perseed_from_pf.py` (PERFOLD-derived, <1e-7bp
  vs direct — residual is float32 netl storage in PERFOLD).
- PERFOLD does NOT store `noa_tr` -> noA metrics cannot be recomputed from artifacts
  (noA is out of deploy scope anyway).
- LINK has a genuine 119-day raw outage -> 246 days in the year window -> ~2
  walk-forward folds (reduced-power cell, not a bug).
- CL trades on recent days can be triplicated by id — builder dedups (`dup=` in build
  log; 1.00 on clean days).
- Combined-npz existence in GCS is the build-resume marker: a PARTIAL combined npz
  poisons resume. Only combine after the completeness floor passes; delete bad
  combined npz (dailies are the durable unit).
- **Recorder streams are NOT uniform over time** (three data regimes): CL year =
  full features; recorder early days = book/trades(+OI) only; full recorder streams
  later. Per-symbol onsets (XRPUSDT): depth/trades/OI 2026-06-01, mark_price
  (funding) 2026-06-15, liquidation 2026-06-28 -> full-feature recorder window
  starts 06-28 (why the DOGE deploy cell starts there). ALWAYS inventory per-stream
  day ranges before a recorder-EV window; recorder_ev auto-skips days without
  funding, but a "N days" request silently spans regime boundaries.
- **Raw recorder-EV script output is NOT the measurement cell**: its causal() has an
  empty day-1 tau buffer -> day-1 selects ~everything (thousands of trades). The
  cell is computed by the separate causal analysis (subs60_recev_causal.py
  methodology: day-0 warmup, q vs deploy WPD). True for the DOGE +8.61 cell too.
- Recorder-day tail cells are tau-warmup sensitive: with a 1-day warmup buffer a
  10-day window gave XRP +0.11bp; the same days under the full 13-day window
  (3-day warmup, deployment-like) gave +14.78bp all-LOO-positive. Use the maximum
  stream-complete window, never a bare "last N days".

## Parallel day-runners
- **A `w{i % NWORK}` workdir scheme is NOT worker-isolation.** The index is the TASK
  index, not the executing thread, so as soon as one day runs long, a later task reuses a
  directory that is still in use and the two race on the downloaded parquet and the
  build_samples outputs (observed 2026-07-30: `BS-fail: Parquet argument error: end of
  file` and a missing `entry_q.npy`). Use a PER-ITEM workdir and delete it at the end.
  The parity gates caught every affected day, which is the reason to have gates that
  compare against a stored reference rather than only checking for a non-zero exit.

## Maker fill model
- **The default entry-fill model OVER-FILLS: every maker cell dated before 2026-07-26 is
  an UPPER BOUND, not an estimate.** `live_sim::simulate_maker_entry` fills unconditionally
  the moment the touch gaps past our level (`if b.bid < level_px - eps { return FILLED }`)
  — no flow, no queue. Measured on the live DOGEUSDC anchor: 3/3 phantom entry fills came
  through that branch, and on a full day the model fills 0.72 vs 0.42 for the correct rule.
  Fix is opt-in: `build_samples --emit-level-flow` + `grid_sim_exitdbg --strict-entry-fill
  --level-flow-paths ...` (README §Entry-fill model). Defaults unchanged, so old artifacts
  stay byte-reproducible — which also means **you get the broken model unless you pass the
  flags**.
- `flow_paths.npy` is PRICE-AGNOSTIC (total taker volume per tick). Any fill logic that
  needs "volume that traded through OUR level" must use `flow_lvl_paths.npy`; a patch that
  only requires *some* flow in the gap tick was measured and rejected (OPS-EXEC rev15).
- Fills happen on the USDC venue, features/scores on USDT — do NOT "fix" the signal side
  to USDC. Only the fill layer is venue-wrong. Recorder carries DOGEUSDC depth+aggTrade
  (both needed) since 2026-06-01 / -06-28 respectively, so this is measurable without
  buying data.

## xgboost / determinism
- Results depend on nthread (hist accumulation order): same seed + same nthread =
  reproducible; changing nthread changes low bits -> a (symbol,seed) job must run
  with ONE nthread end-to-end. Never let two instances of the same (symbol,seed)
  run concurrently (artifact interleaving).
- Optuna TPE is sequential: `n_jobs>1` changes the search trajectory = different
  hyperparameters = a different cell. Parallelize across jobs, never within a study.
- Bit-exactness keys for engine parity: day-anchored prefixes, base-FIRST f32
  accumulation, glibc expf (see RESEARCH_LOG s22.3).
