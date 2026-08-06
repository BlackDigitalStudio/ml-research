#!/usr/bin/env python3
"""HBV1: DOGE anchored-h150 replication on free Bybit archives — Modal campaign.

Stages (run each as `modal run modal_app.py::<stage>` from the repo checkout):
  probe      - egress sanity (Bybit archive + REST geo status) — informational
  aux        - one-shot funding/OI venue-proxy inputs (fetch_aux.py)
  days       - fan-out per-day convert + frozen day-build (subs60_build_tb3s_labels.py)
  combine    - frozen COMBINE=1 -> DOGE.npz
  anch       - frozen prep_anch_sym.py (col13 day-first, col44=0)
  train      - 4 seeds x frozen PROTOCOL v2 (subs60_xgb_sobol_v2.py), parallel
  finish     - perseed_from_pf x4 + ens_sym -> prints the surface

All artifact IO goes through the google.cloud.storage local-FS shim onto the
Volume (LOCAL_GCS_ROOT=/vol/gcs) — frozen scripts run byte-identical.
Ledger: hbv1-20260806_bybit_doge_h150anch_repl_PREREG. Budget: $30 virginship07.
"""
import os
import subprocess
import sys

import modal

app = modal.App("hbv1-bybit-repl")
vol = modal.Volume.from_name("bybit-cl", create_if_missing=True)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "curl", "git", "libgomp1")
    .pip_install("numpy", "pyarrow", "orjson", "sortedcontainers", "xgboost==2.1.4", "scipy")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y -q --default-toolchain stable")
    .add_local_dir(os.path.join(REPO, "rust_ingest"), "/build/rust_ingest", copy=True)
    .add_local_dir("/home/user/research_bins/husdc_src/rust_ingest", "/build/husdc/rust_ingest", copy=True)
    .run_commands(
        ". $HOME/.cargo/env && cd /build/rust_ingest && cargo build --release --bin feature_builder"
        " && cp target/release/feature_builder /usr/local/bin/",
        ". $HOME/.cargo/env && cd /build/husdc/rust_ingest && cargo build --release --bin build_samples"
        " --bin grid_sim_exitdbg && cp target/release/build_samples target/release/grid_sim_exitdbg /usr/local/bin/",
    )
    .pip_install("optuna")  # v1 protocol dependency (appended layer — keeps cargo layers cached)
    .add_local_dir(os.path.join(REPO, "scripts"), "/repo/scripts")
    .add_local_dir(os.path.join(REPO, "research", "runtime"), "/repo/runtime")
)

ENV = {
    "LOCAL_GCS_ROOT": "/vol/gcs",
    "PYTHONPATH": "/repo/runtime/bybit_repl/gcs_shim",
}
START, END = "2025-05-09", "2026-06-02"
BUILD_ENV = {
    "SYMF": "DOGE-USDT-PERP", "FULLFEAT": "1", "H_TICKS": "5100",
    "ENTRY_MS": "60000", "HOLDS_S": "90,150,240", "CHASE_MS": "300000", "STEP_S": "3",
    "OUTSUB": "research_runs/maker_labels_tb3s_h150",
    "FB_BIN": "/usr/local/bin/feature_builder",
    "BS_BIN": "/usr/local/bin/build_samples",
    "GRID_BIN": "/usr/local/bin/grid_sim_exitdbg",
}


def _env(extra=None):
    e = dict(os.environ)
    e.update(ENV)
    if extra:
        e.update(extra)
    return e


def _run(cmd, extra=None, cwd=None):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=_env(extra), cwd=cwd, capture_output=True, text=True)
    print(r.stdout[-8000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-4000:], flush=True)
        raise RuntimeError(f"cmd failed rc={r.returncode}")
    return r.stdout


@app.function(image=image, timeout=300)
def probe():
    import urllib.request
    out = {}
    for tag, url in (("archive", "https://public.bybit.com/trading/DOGEUSDT/"),
                     ("rest", "https://api.bybit.com/v5/market/time")):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                out[tag] = f"HTTP {r.status}: {r.read(120)[:120]}"
        except Exception as e:
            out[tag] = f"FAIL {e}"
    return out


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=4096, timeout=3600)
def aux():
    _run([sys.executable, "/repo/runtime/bybit_repl/fetch_aux.py"],
         extra={"START": "2025-05-01", "END": "2026-06-05"})
    os.makedirs("/vol/gcs/market-data-0998ac51", exist_ok=True)
    with open("/vol/gcs/market-data-0998ac51/VENUE_NOTE.md", "w") as f:
        f.write("DATA VENUE = BYBIT LINEAR (free archives), NOT Binance.\n"
                "raw/ path says exchange=BINANCE_FUTURES only because the frozen build\n"
                "script hard-codes that prefix. funding/OI are Binance Vision venue-proxies\n"
                "(ledger hbv1-20260806_bybit_doge_h150anch_repl_PREREG). GCS is dead\n"
                "(2026-08-05); this Volume is the artifact store via the gcs_shim.\n")
    vol.commit()
    return "aux done"


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=9216, timeout=5400,
              retries=modal.Retries(max_retries=2), max_containers=40)
def day_fn(day: str):
    done = f"/vol/gcs/market-data-0998ac51/research_runs/maker_labels_tb3s_h150/daily/DOGE_{day}.npz"
    if os.path.exists(done):
        return f"{day}: already-done"
    sys.path.insert(0, "/repo/runtime/bybit_repl")
    os.environ.update(ENV)
    os.environ["WORK"] = "/tmp/conv"
    import convert_bybit
    conv = convert_bybit.run_day(day)
    vol.commit()
    if "book-missing" in conv or "trades-missing" in conv.split("|")[1]:
        return f"{day}: SKIP no-raw ({conv})"
    out = _run([sys.executable, "/repo/scripts/subs60_build_tb3s_labels.py"],
               extra=dict(BUILD_ENV, START=day, END=day, WORKDIR="/tmp/tb3s"))
    vol.commit()
    tail = [ln for ln in out.splitlines() if ln.strip().startswith(day)]
    return f"{day}: {conv} || {tail[-1].strip() if tail else 'no-day-line'}"


@app.local_entrypoint()
def days():
    import numpy as np
    all_days = [str(d) for d in np.arange(np.datetime64(START), np.datetime64(END) + 1)]
    print(f"{len(all_days)} days {all_days[0]}..{all_days[-1]}")
    ok = skip = 0
    for res in day_fn.map(all_days, return_exceptions=True):
        s = str(res)
        print(s, flush=True)
        if "SKIP" in s or "Exception" in s:
            skip += 1
        else:
            ok += 1
    print(f"[days done] ok={ok} skip/fail={skip}")


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=20480, timeout=5400)
def combine_fn():
    _run([sys.executable, "/repo/scripts/subs60_build_tb3s_labels.py"],
         extra=dict(BUILD_ENV, START=START, END=END, COMBINE="1", WORKDIR="/tmp/tb3s"))
    vol.commit()
    return "combined"


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=12288, timeout=3600)
def anch_fn():
    _run([sys.executable, "/repo/runtime/prep_anch_sym.py", "DOGE"])
    vol.commit()
    return "anch done"


@app.function(image=image, volumes={"/vol": vol}, cpu=6, memory=20480, timeout=4 * 3600)
def train_fn(seed: int):
    _run([sys.executable, "/repo/scripts/subs60_xgb_sobol_v2.py", "DOGE", "maker_labels_tb3s_h150anch", "0", "6"],
         extra={"SEED": str(seed), "CFGIDX": "1", "BUDGETS": "5,10", "SAVE_PF": "1",
                "PFTAG": f"_S{seed}", "MODEL_DUMP": "1", "DATA_CACHE": "/tmp/cache",
                "SOBOL_PAR": "6", "FOLD_PAR": "1"})
    vol.commit()
    return f"seed{seed} trained"


@app.local_entrypoint()
def train():
    for r in train_fn.map([0, 1, 2, 3], return_exceptions=True):
        print(r, flush=True)


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=3600)
def finish_fn():
    for s in (0, 1, 2, 3):
        _run([sys.executable, "/repo/runtime/perseed_from_pf.py", "DOGE", str(s)])
    out = _run([sys.executable, "/repo/runtime/ens_sym.py", "DOGE"])
    out += _run([sys.executable, "/repo/runtime/bybit_repl/bybit_t10.py", "DOGE"], extra={"TGT": "10"})
    out += _run([sys.executable, "/repo/runtime/bybit_repl/bybit_t10.py", "DOGE"], extra={"TGT": "5"})
    vol.commit()
    return out


V1SUB = "maker_labels_tb3s_h150anch_v1"


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=2048, timeout=1800)
def v1_prep():
    import shutil
    src = "/vol/gcs/market-data-0998ac51/research_runs/maker_labels_tb3s_h150anch/DOGE.npz"
    dst_dir = f"/vol/gcs/market-data-0998ac51/research_runs/{V1SUB}"
    os.makedirs(dst_dir, exist_ok=True)
    dst = f"{dst_dir}/DOGE.npz"
    if not os.path.exists(dst):
        shutil.copyfile(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    vol.commit()
    return f"v1 dataset prefix ready ({os.path.getsize(dst)/1e6:.0f} MB)"


@app.function(image=image, volumes={"/vol": vol}, cpu=5, memory=18432, timeout=6 * 3600)
def train_v1_fn(seed: int):
    _run([sys.executable, "/repo/scripts/subs60_xgb_optuna_ic.py", "DOGE", V1SUB, "0", "5"],
         extra={"SEED": str(seed), "CFGIDX": "1", "BUDGETS": "5,10", "SAVE_PF": "1",
                "PFTAG": f"_S{seed}", "FOLD_PAR": "1"})
    vol.commit()
    return f"v1 seed{seed} trained"


NOOI_SUB = "maker_labels_tb3s_h150anch_v2_nooi"


@app.function(image=image, volumes={"/vol": vol}, cpu=6, memory=20480, timeout=4 * 3600)
def train_nooi_fn(seed: int):
    _run([sys.executable, "/repo/scripts/subs60_xgb_sobol_v2.py", "DOGE", "maker_labels_tb3s_h150anch", "0", "6"],
         extra={"SEED": str(seed), "CFGIDX": "1", "BUDGETS": "5,10", "SAVE_PF": "1",
                "PFTAG": f"_S{seed}", "MODEL_DUMP": "1", "DATA_CACHE": "/tmp/cache",
                "SOBOL_PAR": "6", "FOLD_PAR": "1", "DROP_COLS": "59,60", "OUT_SUB": NOOI_SUB})
    vol.commit()
    return f"nooi seed{seed} trained"


@app.local_entrypoint()
def train_nooi():
    for r in train_nooi_fn.map([0, 1, 2, 3], return_exceptions=True):
        print(r, flush=True)


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=3600)
def finish_nooi_fn():
    ex = {"XSYM_SUB": NOOI_SUB}
    for s in (0, 1, 2, 3):
        _run([sys.executable, "/repo/runtime/perseed_from_pf.py", "DOGE", str(s)], extra=ex)
    out = _run([sys.executable, "/repo/runtime/ens_sym.py", "DOGE"], extra=ex)
    for tgt in ("1", "2", "3", "5", "10"):
        out += _run([sys.executable, "/repo/runtime/bybit_repl/bybit_t10.py", "DOGE"],
                    extra=dict(ex, TGT=tgt))
    vol.commit()
    return out


@app.local_entrypoint()
def train_v1():
    print(v1_prep.remote())
    for r in train_v1_fn.map([0, 1, 2, 3], return_exceptions=True):
        print(r, flush=True)


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=3600)
def finish_v1_fn():
    ex = {"XSYM_SUB": V1SUB}
    for s in (0, 1, 2, 3):
        _run([sys.executable, "/repo/runtime/perseed_from_pf.py", "DOGE", str(s)], extra=ex)
    out = _run([sys.executable, "/repo/runtime/ens_sym.py", "DOGE"], extra=ex)
    for tgt in ("1", "2", "3", "5", "10", "20"):
        out += _run([sys.executable, "/repo/runtime/bybit_repl/bybit_t10.py", "DOGE"],
                    extra=dict(ex, TGT=tgt))
    vol.commit()
    return out


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=1800)
def tsurf_fn(tgt: str):
    return _run([sys.executable, "/repo/runtime/bybit_repl/bybit_t10.py", "DOGE"], extra={"TGT": tgt})


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=3600)
def gain_fn(sub: str = "maker_labels_tb3s_h150anch", drop: str = ""):
    return _run([sys.executable, "/repo/runtime/bybit_repl/models_gain.py", "DOGE"],
                extra={"XSYM_SUB": sub, "DROP_COLS": drop})


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=6144, timeout=3600)
def audit_fn(pools: str = "", ens_t: str = "1,2.5,5,10", union_t: str = "0.625,1.25,2.5,5"):
    extra = {"ENS_T": ens_t, "UNION_T": union_t}
    if pools:
        extra["POOLS"] = pools
    return _run([sys.executable, "/repo/runtime/bybit_repl/full_audit.py"], extra=extra)


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=4096, timeout=1800)
def corr_fn(sub: str = "maker_labels_tb3s_h150anch"):
    return _run([sys.executable, "/repo/runtime/bybit_repl/seed_corr.py", "DOGE"],
                extra={"XSYM_SUB": sub})


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=4096, timeout=1800)
def union_fn(sub: str = "maker_labels_tb3s_h150anch", seeds: str = "0,1,2,3", tgts: str = "1.25,2.5,5",
             members: str = "", utag: str = ""):
    extra = {"XSYM_SUB": sub, "SEEDS": seeds, "TGTS": tgts}
    if members:
        extra["MEMBERS"] = members
    if utag:
        extra["UTAG"] = utag
    return _run([sys.executable, "/repo/runtime/bybit_repl/union_policy.py", "DOGE"], extra=extra)


V1NOOI_SUB = "maker_labels_tb3s_h150anch_v1_nooi"


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=2048, timeout=1800)
def v1_nooi_prep():
    import shutil
    src = "/vol/gcs/market-data-0998ac51/research_runs/maker_labels_tb3s_h150anch/DOGE.npz"
    dst_dir = f"/vol/gcs/market-data-0998ac51/research_runs/{V1NOOI_SUB}"
    os.makedirs(dst_dir, exist_ok=True)
    dst = f"{dst_dir}/DOGE.npz"
    if not os.path.exists(dst):
        shutil.copyfile(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    vol.commit()
    return "v1_nooi dataset prefix ready"


@app.function(image=image, volumes={"/vol": vol}, cpu=5, memory=18432, timeout=6 * 3600)
def train_v1_nooi_fn(seed: int):
    _run([sys.executable, "/repo/scripts/subs60_xgb_optuna_ic.py", "DOGE", V1NOOI_SUB, "0", "5"],
         extra={"SEED": str(seed), "CFGIDX": "1", "BUDGETS": "5,10", "SAVE_PF": "1",
                "PFTAG": f"_S{seed}", "FOLD_PAR": "1", "DROP_COLS": "59,60"})
    vol.commit()
    return f"v1_nooi seed{seed} trained"


@app.local_entrypoint()
def train_v1_missing(seeds: str = "0,1,2,4,5,6"):
    print(v1_nooi_prep.remote())
    calls = [train_v1_nooi_fn.spawn(int(s)) for s in seeds.split(",")]
    for c in calls:
        try:
            print(c.get(timeout=6 * 3600), flush=True)
        except Exception as e:
            print("FAIL:", e, flush=True)


@app.local_entrypoint()
def train_8seed():
    print(v1_nooi_prep.remote())
    calls = [train_v1_nooi_fn.spawn(s) for s in range(8)] + [train_nooi_fn.spawn(s) for s in (4, 5, 6, 7)]
    for c in calls:
        try:
            print(c.get(timeout=6 * 3600), flush=True)
        except Exception as e:
            print("FAIL:", e, flush=True)


@app.local_entrypoint()
def tsurf():
    for r in tsurf_fn.map(["1", "2", "3", "20"], return_exceptions=True):
        print(r, flush=True)


@app.local_entrypoint()
def stage(name: str = "probe"):
    if name == "probe":
        print(probe.remote())
    elif name == "aux":
        print(aux.remote())
    elif name == "combine":
        print(combine_fn.remote())
    elif name == "anch":
        print(anch_fn.remote())
    elif name == "finish":
        print(finish_fn.remote())
    else:
        raise SystemExit(f"unknown stage {name}")
