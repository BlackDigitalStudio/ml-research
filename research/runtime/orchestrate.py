#!/usr/bin/env python3
"""HD3 rev8 orchestrator — cross-symbol year run of the anchored h150 policy
(prereg tb3s-20260710_h150anch_year_xsym_PREREG). Everything resumable via GCS artifacts.

Per symbol: [build dailies (5 new syms) -> combine -> anch transform] -> train chain
(seeds 0-3 SEQUENTIAL, subs60_xgb_optuna_ic.py BYTE-UNCHANGED, CFGIDX=1 BUDGETS=5
SAVE_PF=1 PFTAG=_S{s}) -> ens_sym.py. BTC/ETH skip the build (combined npz exist).
Builds parallel (RAYON_NUM_THREADS=2 each); memory-heavy steps (combine/anch) serialized;
max concurrent training jobs read from ~/xsym/SLOTS each acquisition (31GB RAM bound)."""
import os
import subprocess
import threading
import time

HOME = "/home/delmi"; XD = f"{HOME}/xsym"
GB = "gs://market-data-0998ac51/research_runs"
SUB_H = "maker_labels_tb3s_h150"; SUB_A = "maker_labels_tb3s_h150anch"
BUILD_SYMS = [s for s in os.environ.get("XSYM_BUILD", "BNB,LTC,SOL,XRP,LINK").split(",") if s]
READY_SYMS = [s for s in os.environ.get("XSYM_READY", "BTC,ETH").split(",") if s]
BUILD_ENV = {"FULLFEAT": "1", "H_TICKS": "1800", "ENTRY_MS": "60000",
             "HOLDS_S": "90,150,240", "CHASE_MS": "300000", "STEP_S": "3",
             "START": "2025-05-09", "END": "2026-06-02",
             "OUTSUB": f"research_runs/{SUB_H}", "RAYON_NUM_THREADS": "2",
             # persistent binary paths — /tmp is wiped on every VM stop/start
             "FB_BIN": f"{XD}/bins/fb_target/release/feature_builder",
             "BS_BIN": f"{XD}/bins/husdc_target/release/build_samples",
             "GRID_BIN": f"{XD}/bins/husdc_target/release/grid_sim_exitdbg"}
NTHREAD = os.environ.get("XSYM_NTHREAD", "3")
# completeness floor before combine/train: LINK has a genuine 119d raw outage
MIN_DAYS = {"LINK": 230}
MIN_DAYS_DEFAULT = 350
heavy_lock = threading.Lock()          # combine + anch prep (memory-heavy) serialized
run_lock = threading.Lock()
n_training = 0


def log(s):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {s}", flush=True)


def slots():
    try:
        return max(1, int(open(f"{XD}/SLOTS").read().strip()))
    except Exception:
        return 2


def run(cmd, env, logf):
    e = dict(os.environ); e.update(env)
    with open(logf, "ab") as lf:
        lf.write(f"\n===== {time.strftime('%m-%d %H:%M:%S')} {' '.join(cmd)}\n".encode())
        lf.flush()
        return subprocess.run(cmd, env=e, stdout=lf, stderr=subprocess.STDOUT).returncode


def gcs_exists(path):
    return subprocess.run(["gsutil", "-q", "stat", path],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def acquire_slot(sym, s):
    global n_training
    while True:
        with run_lock:
            if n_training < slots():
                n_training += 1
                return
        time.sleep(60)


def release_slot():
    global n_training
    with run_lock:
        n_training -= 1


def train_chain(sym):
    # guard: never train on a short/partial dataset (walk-forward needs >202 days)
    nd = daily_count(sym)
    if sym not in ("BTC", "ETH") and nd < MIN_DAYS.get(sym, MIN_DAYS_DEFAULT):
        log(f"{sym}: TRAIN REFUSED — only {nd} dailies")
        return
    for s in range(4):
        if gcs_exists(f"{GB}/{SUB_A}/OPTUNA_IC_{sym}_qm0_SEED{s}.json"):
            log(f"{sym} seed{s}: already done, skip")
            continue
        acquire_slot(sym, s)
        try:
            log(f"{sym} seed{s}: TRAIN start")
            env = {"SEED": str(s), "CFGIDX": "1", "BUDGETS": "5", "SAVE_PF": "1",
                   "PFTAG": f"_S{s}"}
            rc = run(["/usr/bin/python3", f"{XD}/subs60_xgb_optuna_ic.py", sym, SUB_A,
                      "0", NTHREAD], env, f"{XD}/train_{sym}.log")
            if rc != 0:   # one retry (transient GCS/OOM)
                log(f"{sym} seed{s}: rc={rc}, RETRY")
                rc = run(["/usr/bin/python3", f"{XD}/subs60_xgb_optuna_ic.py", sym, SUB_A,
                          "0", NTHREAD], env, f"{XD}/train_{sym}.log")
            if rc != 0:
                log(f"{sym} seed{s}: FAILED rc={rc} — chain aborted")
                return
            subprocess.run(["gsutil", "-q", "cp", f"{GB}/{SUB_A}/OPTUNA_IC_{sym}_qm0.json",
                            f"{GB}/{SUB_A}/OPTUNA_IC_{sym}_qm0_SEED{s}.json"], check=True)
            log(f"{sym} seed{s}: TRAIN done")
        finally:
            release_slot()
    rc = run(["/usr/bin/python3", f"{XD}/ens_sym.py", sym], {}, f"{XD}/ens_{sym}.log")
    log(f"{sym}: CHAIN DONE (ens rc={rc})")


def prep_anch(sym):
    if gcs_exists(f"{GB}/{SUB_A}/{sym}.npz"):
        log(f"{sym}: anch npz exists, skip prep")
        return True
    with heavy_lock:
        log(f"{sym}: anch prep start")
        rc = run(["/usr/bin/python3", f"{XD}/prep_anch_sym.py", sym], {}, f"{XD}/prep_{sym}.log")
    log(f"{sym}: anch prep rc={rc}")
    return rc == 0


def daily_count(sym):
    r = subprocess.run(["gsutil", "ls", f"{GB}/{SUB_H}/daily/{sym}_*.npz"],
                       capture_output=True, text=True)
    return len([l for l in r.stdout.splitlines() if l.endswith(".npz")])


def build_symbol(sym):
    env = dict(BUILD_ENV, SYMF=f"{sym}-USDT-PERP", WORKDIR=f"{XD}/wk_{sym}")
    need = MIN_DAYS.get(sym, MIN_DAYS_DEFAULT)
    if not gcs_exists(f"{GB}/{SUB_H}/{sym}.npz"):
        log(f"{sym}: BUILD dailies start")
        rc = run(["/usr/bin/python3", f"{XD}/subs60_build_tb3s_labels.py"], env,
                 f"{XD}/build_{sym}.log")
        if rc != 0:
            log(f"{sym}: BUILD FAILED rc={rc} — aborted"); return
        nd = daily_count(sym)
        if nd < need:
            log(f"{sym}: BUILD INCOMPLETE ({nd} dailies < {need}) — aborted, NOT combining")
            return
        log(f"{sym}: build complete ({nd} dailies)")
        with heavy_lock:
            log(f"{sym}: COMBINE start")
            rc = run(["/usr/bin/python3", f"{XD}/subs60_build_tb3s_labels.py"],
                     dict(env, COMBINE="1"), f"{XD}/build_{sym}.log")
        if rc != 0:
            log(f"{sym}: COMBINE FAILED rc={rc} — aborted"); return
    else:
        log(f"{sym}: combined h150 npz exists, skip build")
    if prep_anch(sym):
        train_chain(sym)


def ready_symbol(sym):
    if prep_anch(sym):
        train_chain(sym)


def main():
    log(f"xsym orchestrator start | ready={READY_SYMS} build={BUILD_SYMS} slots={slots()}")
    ths = []
    for sym in READY_SYMS:
        t = threading.Thread(target=ready_symbol, args=(sym,), name=sym); t.start(); ths.append(t)
        time.sleep(5)
    for sym in BUILD_SYMS:
        t = threading.Thread(target=build_symbol, args=(sym,), name=sym); t.start(); ths.append(t)
        time.sleep(5)
    for t in ths:
        t.join()
    log("xsym orchestrator ALL DONE")
    subprocess.run(["gsutil", "-q", "cp", f"{XD}/orchestrator.log",
                    f"{GB}/{SUB_A}/xsym_orchestrator_log.txt"])


if __name__ == "__main__":
    main()
