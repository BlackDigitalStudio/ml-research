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

HOME = "/home/delmi"; XD = f"{HOME}/xsym"
GB = "gs://market-data-0998ac51/research_runs"
SUB_A = "maker_labels_tb3s_h150anch"
NTHREAD = os.environ.get("XSYM_NTHREAD", "2")

# --- ARM MODE (OBJSEL rev2), additive: the legacy (sym,seed) path below is unchanged.
# Jobs become ARM:SYM:SEED triples; each runs the HP_FIX-patched trainer with a fixed
# per-fold hyperparameter spec instead of searching. perseed/ens are NOT run - the arm
# cells are computed offline from the PERFOLD artifacts, as strictfill_cells.py does.
ARM_MODE = os.environ.get("XSYM_ARM_MODE", "") == "1"
TRAINER = os.environ.get("XSYM_TRAINER", "subs60_xgb_optuna_ic.py")
HPSPEC = os.environ.get("XSYM_HPSPEC", "research_runs/objsel_refit")
# symbol -> (labelsub, deployed budget, DROP_COLS, trainer, artifact out_sub)
# NOTE the INC arm is NOT run: it reproduces the stored PERFOLD_S{s}_{SYM} bit-exactly
# (OBJSEL rev2 gate, both protocols), so the published artifacts ARE that arm.
SYMCFG = {
    "DOGE": ("maker_labels_tb3s_h150anch", "10", "", "subs60_xgb_optuna_ic_v1cap.py", ""),
    "XRP":  ("maker_labels_tb3s_h150anch", "5", "", "subs60_xgb_optuna_ic_v1cap.py", ""),
    "BTC":  ("maker_labels_tb3s_h150d", "5", "", "subs60_xgb_optuna_ic_v1cap.py", ""),
    "ETH":  ("maker_labels_tb3s_h150danch", "5", "67,68,69,70", "subs60_xgb_sobol_v2.py",
             "maker_labels_tb3s_h150danch_v2notod"),
}
if ARM_MODE:
    JOBS = [tuple(p.split(":")) for p in os.environ.get("XSYM_JOBS", "").split(",") if p]
else:
    JOBS = [(p.split(":")[0], int(p.split(":")[1]))
            for p in os.environ.get("XSYM_JOBS", "").split(",") if p]
run_lock = threading.Lock()
n_running = 0


def log(s):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {s}", flush=True)


def slots():
    try:
        return max(1, int(open(f"{XD}/SLOTS").read().strip()))
    except Exception:
        return 15


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
        rc = run(["/usr/bin/python3", f"{XD}/subs60_xgb_optuna_ic.py", sym, SUB_A, "0", NTHREAD],
                 env, f"{XD}/train_{sym}_s{s}.log")
        if rc != 0:
            log(f"{sym} seed{s}: rc={rc}, RETRY")
            rc = run(["/usr/bin/python3", f"{XD}/subs60_xgb_optuna_ic.py", sym, SUB_A, "0", NTHREAD],
                     env, f"{XD}/train_{sym}_s{s}.log")
        if rc != 0:
            log(f"{sym} seed{s}: FAILED rc={rc}"); return
        rc = run(["/usr/bin/python3", f"{XD}/perseed_from_pf.py", sym, str(s)], {},
                 f"{XD}/train_{sym}_s{s}.log")
        log(f"{sym} seed{s}: DONE (perseed rc={rc})")
    finally:
        with run_lock:
            n_running -= 1


def arm_job(arm, sym, s):
    """One (arm, symbol, seed): fixed-HP refit to deployed semantics. OBJSEL rev2."""
    global n_running
    sub, budget, drop, trainer, out_sub = SYMCFG[sym]
    out = out_sub or sub
    tag = f"_{arm}_S{s}"
    if gcs_exists(f"{GB}/{out}/PERFOLD{tag}_{sym}_qm0_f5.npz"):
        log(f"{arm} {sym} seed{s}: last-fold artifact exists, skip"); return
    while True:
        with run_lock:
            if n_running < slots():
                n_running += 1; break
        time.sleep(30)
    try:
        log(f"{arm} {sym} seed{s}: TRAIN start (sub={sub} t{budget})")
        env = {"SEED": str(s), "CFGIDX": "1", "BUDGETS": budget, "SAVE_PF": "1", "PFTAG": tag,
               "TRIAL_DUMP": "0",   # no search runs under HP_FIX; final models still dumped
               "HP_FIX": f"{HPSPEC}/HPSPEC_{arm}_{sym}_S{s}.json"}
        if drop:
            env["DROP_COLS"] = drop
        if out_sub:
            env["OUT_SUB"] = out_sub
        logf = f"{XD}/arm_{arm}_{sym}_s{s}.log"
        cmd = ["/usr/bin/python3", f"{XD}/{trainer}", sym, sub, "0", NTHREAD]
        rc = run(cmd, env, logf)
        if rc != 0:
            log(f"{arm} {sym} seed{s}: rc={rc}, RETRY")
            rc = run(cmd, env, logf)
        log(f"{arm} {sym} seed{s}: {'DONE' if rc == 0 else f'FAILED rc={rc}'}")
    finally:
        with run_lock:
            n_running -= 1


def main():
    log(f"xsym v2 ({'ARM' if ARM_MODE else 'seed-parallel'}) start | jobs={len(JOBS)} "
        f"nthread={NTHREAD} slots={slots()}")
    ths = []
    if ARM_MODE:
        for arm, sym, s in JOBS:
            t = threading.Thread(target=arm_job, args=(arm, sym, int(s)), name=f"{arm}{sym}s{s}")
            t.start(); ths.append(t)
            time.sleep(45)
        for t in ths:
            t.join()
        log("xsym v2 ARM MODE ALL DONE")
        return
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
