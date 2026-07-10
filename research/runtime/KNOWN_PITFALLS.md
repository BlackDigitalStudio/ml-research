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

## xgboost / determinism
- Results depend on nthread (hist accumulation order): same seed + same nthread =
  reproducible; changing nthread changes low bits -> a (symbol,seed) job must run
  with ONE nthread end-to-end. Never let two instances of the same (symbol,seed)
  run concurrently (artifact interleaving).
- Optuna TPE is sequential: `n_jobs>1` changes the search trajectory = different
  hyperparameters = a different cell. Parallelize across jobs, never within a study.
- Bit-exactness keys for engine parity: day-anchored prefixes, base-FIRST f32
  accumulation, glibc expf (see RESEARCH_LOG s22.3).
