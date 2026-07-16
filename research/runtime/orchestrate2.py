#!/usr/bin/env python3
"""HD3 rev8 orchestrator v2 — SEED-PARALLEL. Jobs are independent (symbol, seed) pairs;
the shared OPTUNA_IC_{SYM}_qm0.json is no longer consumed (write-racy under parallel
seeds) — per-seed jsons are recomputed deterministically from the seed-tagged PERFOLD
artifacts by perseed_from_pf.py right after each job. Training script BYTE-UNCHANGED.

Env: XSYM_JOBS="BNB:1,BNB:2,XRP:1,..."  XSYM_NTHREAD (default 2). Skip marker =
OPTUNA_IC_{SYM}_qm0_SEED{s}.json exists. After all jobs: ens_sym.py for every symbol
with all 4 SEED jsons present."""
import os
import subprocess
import threading
import time

HOME = "/home/delmi"; XD = os.environ.get("XSYM_XD", f"{HOME}/xsym")
GB = "gs://market-data-0998ac51/research_runs"
SUB_A = os.environ.get("XSYM_SUB", "maker_labels_tb3s_h150anch")  # children read XSYM_SUB too
JOBS = [(p.split(":")[0], int(p.split(":")[1]))
        for p in os.environ.get("XSYM_JOBS", "").split(",") if p]
NTHREAD = os.environ.get("XSYM_NTHREAD", "2")
# XSYM_TRAINER: which frozen trainer script to run (must exist in XD). Default = v1
# for backward compatibility; pass subs60_xgb_sobol_v2.py for protocol-v2 campaigns.
TRAINER = os.environ.get("XSYM_TRAINER", "subs60_xgb_optuna_ic.py")
run_lock = threading.Lock()
n_running = 0

# RAM budget clamp (2026-07-15 pitfall: 96%-RAM job packing killed the guest network).
# Whatever SLOTS asks for, never exceed 75% of physical RAM at XSYM_JOB_GB per job.
JOB_GB = float(os.environ.get("XSYM_JOB_GB", "14"))
try:
    _TOT_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    RAM_SLOTS = max(1, int(_TOT_GB * 0.75 / JOB_GB))
except (ValueError, OSError):
    _TOT_GB, RAM_SLOTS = 0.0, 15


def log(s):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {s}", flush=True)


def slots():
    try:
        return max(1, min(int(open(f"{XD}/SLOTS").read().strip()), RAM_SLOTS))
    except Exception:
        return min(15, RAM_SLOTS)


def run(cmd, env, logf):
    e = dict(os.environ); e.update(env)
    with open(logf, "ab") as lf:
        lf.write(f"\n===== {time.strftime('%m-%d %H:%M:%S')} {' '.join(cmd)}\n".encode())
        lf.flush()
        return subprocess.run(cmd, env=e, stdout=lf, stderr=subprocess.STDOUT).returncode


def gcs_exists(path):
    return subprocess.run(["gsutil", "-q", "stat", path],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def job(sym, s):
    global n_running
    if gcs_exists(f"{GB}/{SUB_A}/OPTUNA_IC_{sym}_qm0_SEED{s}.json"):
        log(f"{sym} seed{s}: done marker exists, skip"); return
    while True:
        with run_lock:
            if n_running < slots():
                n_running += 1; break
        time.sleep(30)
    try:
        log(f"{sym} seed{s}: TRAIN start")
        env = {"SEED": str(s), "CFGIDX": "1", "BUDGETS": "5", "SAVE_PF": "1", "PFTAG": f"_S{s}"}
        rc = run(["/usr/bin/python3", f"{XD}/{TRAINER}", sym, SUB_A, "0", NTHREAD],
                 env, f"{XD}/train_{sym}_s{s}.log")
        if rc != 0:
            log(f"{sym} seed{s}: rc={rc}, RETRY")
            rc = run(["/usr/bin/python3", f"{XD}/{TRAINER}", sym, SUB_A, "0", NTHREAD],
                     env, f"{XD}/train_{sym}_s{s}.log")
        if rc != 0:
            log(f"{sym} seed{s}: FAILED rc={rc}"); return
        rc = run(["/usr/bin/python3", f"{XD}/perseed_from_pf.py", sym, str(s)], {},
                 f"{XD}/train_{sym}_s{s}.log")
        log(f"{sym} seed{s}: DONE (perseed rc={rc})")
    finally:
        with run_lock:
            n_running -= 1


def main():
    log(f"xsym v2 (seed-parallel) start | jobs={JOBS} nthread={NTHREAD} slots={slots()} "
        f"(RAM {_TOT_GB:.0f}GB -> clamp {RAM_SLOTS} at {JOB_GB}GB/job)")
    if "swapfile" not in open("/proc/swaps").read():
        log("WARNING: no swap active — system daemons have no OOM lifeline (KNOWN_PITFALLS)")
    ths = []
    for sym, s in JOBS:
        t = threading.Thread(target=job, args=(sym, s), name=f"{sym}s{s}"); t.start(); ths.append(t)
        time.sleep(45)   # stagger the load phase (GCS + float64 conversion spikes)
    for t in ths:
        t.join()
    for sym in sorted(set(s for s, _ in JOBS)):
        if all(gcs_exists(f"{GB}/{SUB_A}/OPTUNA_IC_{sym}_qm0_SEED{k}.json") for k in range(4)):
            rc = run(["/usr/bin/python3", f"{XD}/ens_sym.py", sym], {}, f"{XD}/ens_{sym}.log")
            log(f"{sym}: ENS done rc={rc}")
        else:
            log(f"{sym}: ENS skipped (not all 4 seeds present)")
    log("xsym v2 ALL DONE")
    subprocess.run(["gsutil", "-q", "cp", f"{XD}/orchestrator2.log",
                    f"{GB}/{SUB_A}/xsym32_orchestrator2_log.txt"])


if __name__ == "__main__":
    main()
