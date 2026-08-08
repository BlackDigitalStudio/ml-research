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
  train_decor - HBV1 rev15: 4 feature-bag + 2 hold-horizon members (v2.1, env-only)
  (audit_fn cons_t/cons_ks - HBV1 rev14 consensus K-of-N grid via full_audit.py)

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


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=6144, timeout=2 * 3600)
def audit_fn(pools: str = "", ens_t: str = "1,2.5,5,10", union_t: str = "0.625,1.25,2.5,5",
             fee_bp: str = "0", cons_t: str = "", cons_ks: str = "2,3,4,5,6,7,8",
             out_tag: str = "", sym: str = "DOGE"):
    extra = {"ENS_T": ens_t, "UNION_T": union_t, "FEE_BP": fee_bp,
             "CONS_T": cons_t, "CONS_KS": cons_ks, "SYM": sym}
    if pools:
        extra["POOLS"] = pools
    if out_tag:
        extra["OUT_TAG"] = out_tag
    return _run([sys.executable, "/repo/runtime/bybit_repl/full_audit.py"], extra=extra)


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=10240, timeout=2 * 3600)
def bexec_fn(mode: str = "validate", members: str = "", tgt: str = "0.3125"):
    return _run([sys.executable, "/repo/runtime/bybit_repl/binance_exec.py"],
                extra={"MODE": mode, "MEMBERS": members, "TGT": tgt})


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=8192, timeout=2 * 3600)
def bootf_fn(target_dd: str = "0.25", fee_bp: str = "4", out_tag: str = ""):
    return _run([sys.executable, "/repo/runtime/bybit_repl/bootstrap_f.py"],
                extra={"TARGET_DD": target_dd, "FEE_BP": fee_bp, "OUT_TAG": out_tag})


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=2 * 3600)
def levnorm_fn(target_dd: str = "0.50", out_tag: str = ""):
    return _run([sys.executable, "/repo/runtime/bybit_repl/leverage_norm.py"],
                extra={"TARGET_DD": target_dd, "OUT_TAG": out_tag})


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=4096, timeout=1800)
def hp_fn():
    return _run([sys.executable, "/repo/runtime/bybit_repl/hp_tendency.py"])


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


# ---- HBV1 rev15: DIVERSITY-AXIS WIDENING (beyond seed) --------------------
# Feature-bag members: v2.1 frozen trainer, env-only — DROP_COLS = no-OI {59,60}
# + a deterministic 25% bag of the 56 live cols (RNG 1000+j over the live set;
# dead-on-Bybit cols 17,18,19,24,30,44,50-53,56-58 excluded from the draw —
# dropping dead cols would be a no-op, not diversity). Bags hardcoded for exact
# reproducibility (regenerate: np.random.default_rng(1000+j).choice(live,14)).
FBAG_DROPS = {
    0: "8,10,15,26,29,32,43,59,60,61,62,64,66,68,69,70",
    1: "0,3,9,10,27,31,40,41,42,43,48,59,60,65,67,69",
    2: "3,10,16,20,22,31,34,39,40,45,48,55,59,60,68,69",
    3: "8,11,12,13,22,27,28,31,38,45,46,48,59,60,68,69",
}
FBAG_SUB = "maker_labels_tb3s_h150anch_v2_nooi_fb{j}"
# Hold-horizon members: CFGIDX indexes HOLDS_S=[90,150,240] of the SAME dataset
# (0=h90, 2=h240; 1=h150 is the deploy cell). Used as SELECTORS only — in any
# mixed pool the exec member (first in MEMBERS/POOLS) stays an h150 artifact.
HOLD_SUBS = {0: "maker_labels_tb3s_h150anch_v2_nooi_h90",
             2: "maker_labels_tb3s_h150anch_v2_nooi_h240"}


@app.function(image=image, volumes={"/vol": vol}, cpu=6, memory=20480, timeout=4 * 3600)
def train_fbag_fn(j: int):
    sub = FBAG_SUB.format(j=j)
    _run([sys.executable, "/repo/scripts/subs60_xgb_sobol_v2.py", "DOGE", "maker_labels_tb3s_h150anch", "0", "6"],
         extra={"SEED": str(j), "CFGIDX": "1", "BUDGETS": "5,10", "SAVE_PF": "1",
                "PFTAG": f"_S{j}", "MODEL_DUMP": "1", "DATA_CACHE": "/tmp/cache",
                "SOBOL_PAR": "6", "FOLD_PAR": "1", "DROP_COLS": FBAG_DROPS[j], "OUT_SUB": sub})
    import json as _json
    os.makedirs(f"/vol/gcs/market-data-0998ac51/research_runs/{sub}", exist_ok=True)
    with open(f"/vol/gcs/market-data-0998ac51/research_runs/{sub}/FBAG_SPEC.json", "w") as f:
        _json.dump({"seed": j, "drop_cols": FBAG_DROPS[j], "rng": f"default_rng({1000 + j})",
                    "frac": 0.25, "base": "no-OI (59,60) + 25% of 56 live cols"}, f)
    vol.commit()
    return f"fbag{j} trained (DROP {FBAG_DROPS[j]})"


@app.function(image=image, volumes={"/vol": vol}, cpu=6, memory=20480, timeout=4 * 3600)
def train_hold_fn(cfgidx: int):
    sub = HOLD_SUBS[cfgidx]
    _run([sys.executable, "/repo/scripts/subs60_xgb_sobol_v2.py", "DOGE", "maker_labels_tb3s_h150anch", "0", "6"],
         extra={"SEED": "0", "CFGIDX": str(cfgidx), "BUDGETS": "5,10", "SAVE_PF": "1",
                "PFTAG": "_S0", "MODEL_DUMP": "1", "DATA_CACHE": "/tmp/cache",
                "SOBOL_PAR": "6", "FOLD_PAR": "1", "DROP_COLS": "59,60", "OUT_SUB": sub})
    vol.commit()
    return f"hold cfgidx={cfgidx} trained -> {sub}"


@app.local_entrypoint()
def train_decor():
    calls = [train_fbag_fn.spawn(j) for j in range(4)] + [train_hold_fn.spawn(c) for c in (0, 2)]
    for c in calls:
        try:
            print(c.get(timeout=4 * 3600), flush=True)
        except Exception as e:
            print("FAIL:", e, flush=True)


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=3600)
def finish_decor_fn():
    out = ""
    for j in range(4):
        out += _run([sys.executable, "/repo/runtime/perseed_from_pf.py", "DOGE", str(j)],
                    extra={"XSYM_SUB": FBAG_SUB.format(j=j)})
    for sub in HOLD_SUBS.values():
        out += _run([sys.executable, "/repo/runtime/perseed_from_pf.py", "DOGE", "0"],
                    extra={"XSYM_SUB": sub})
    vol.commit()
    return out


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=6144, timeout=3600)
def forensics_fn(members: str, sym: str = "DOGE", tgt: str = "5", ftag: str = ""):
    return _run([sys.executable, "/repo/runtime/bybit_repl/member_forensics.py"],
                extra={"MEMBERS": members, "SYM": sym, "TGT": tgt, "FTAG": ftag})


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=14336, timeout=3600)
def overlap_fn(forms: str, sym: str = "DOGE", f: str = "1"):
    return _run([sys.executable, "/repo/runtime/bybit_repl/overlap_probe.py"],
                extra={"FORMS": forms, "SYM": sym, "F": f})


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=16384, timeout=7200)
def overlap_capped_fn(forms: str, out_tag: str, sym: str = "DOGE",
                      caps: str = "0.5,1,2,3,burst1", span_s: str = "210",
                      per_form: str = ""):
    return _run([sys.executable, "/repo/runtime/bybit_repl/overlap_capped.py"],
                extra={"FORMS": forms, "SYM": sym, "CAPS": caps, "SPAN_S": span_s,
                       "OUT_TAG": out_tag, "PER_FORM": per_form})


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=16384, timeout=7200)
def sized_burst_fn(forms: str, out_tag: str, sym: str = "DOGE", gammas: str = "1,2",
                   hard_cap: str = "", f_grid: str = "3,6,12,24,48"):
    return _run([sys.executable, "/repo/runtime/bybit_repl/sized_burst.py"],
                extra={"FORMS": forms, "SYM": sym, "GAMMAS": gammas, "OUT_TAG": out_tag,
                       "HARD_CAP": hard_cap, "F_GRID": f_grid})


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=16384, timeout=3600)
def corr_members_fn(members: str, corrtag: str, tgts: str = "5,10", sym: str = "DOGE"):
    return _run([sys.executable, "/repo/runtime/bybit_repl/seed_corr.py", sym],
                extra={"MEMBERS": members, "CORRTAG": corrtag, "TGTS": tgts})


# ---- HBV1 rev16: full RF recipe over members = feature subspace (same bags as
# fb0-3) + day-level train bagging (BAG_FRAC=0.632 ~ bootstrap-equivalent
# subsample, no replacement) + seed. rf{j} vs fb{j} isolates the bag axis.
RF_SUB = "maker_labels_tb3s_h150anch_v2_nooi_rf{j}"


@app.function(image=image, volumes={"/vol": vol}, cpu=6, memory=20480, timeout=4 * 3600)
def train_rf_fn(j: int):
    sub = RF_SUB.format(j=j)
    _run([sys.executable, "/repo/scripts/subs60_xgb_sobol_v2.py", "DOGE", "maker_labels_tb3s_h150anch", "0", "6"],
         extra={"SEED": str(j), "CFGIDX": "1", "BUDGETS": "5,10", "SAVE_PF": "1",
                "PFTAG": f"_S{j}", "MODEL_DUMP": "1", "DATA_CACHE": "/tmp/cache",
                "SOBOL_PAR": "6", "FOLD_PAR": "1", "DROP_COLS": FBAG_DROPS[j],
                "BAG_FRAC": "0.632", "BAG_SEED": str(j), "OUT_SUB": sub})
    vol.commit()
    return f"rf{j} trained (bag 0.632 + DROP {FBAG_DROPS[j]})"


@app.local_entrypoint()
def train_rf():
    calls = [train_rf_fn.spawn(j) for j in range(4)]
    for c in calls:
        try:
            print(c.get(timeout=4 * 3600), flush=True)
        except Exception as e:
            print("FAIL:", e, flush=True)


# ---- HBV1 rev18 wave-1 analyses ----
@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=8192, timeout=2 * 3600)
def sizedu_fn(pools: str = "", tgts: str = "2.5,5,10", gammas: str = "0,0.5,1,2,3", fee_bp: str = "4",
              sym: str = "DOGE"):
    extra = {"TGTS": tgts, "GAMMAS": gammas, "FEE_BP": fee_bp, "SYM": sym}
    if pools:
        extra["POOLS"] = pools
    return _run([sys.executable, "/repo/runtime/bybit_repl/sized_union.py"], extra=extra)


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=8192, timeout=2 * 3600)
def portfolio_fn(fee_bp: str = "4", target_dd: str = "0.25"):
    return _run([sys.executable, "/repo/runtime/bybit_repl/form_portfolio.py"],
                extra={"FEE_BP": fee_bp, "TARGET_DD": target_dd})


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=14336, timeout=2 * 3600)
def anatomy_fn(fee_bp: str = "4"):
    return _run([sys.executable, "/repo/runtime/bybit_repl/regime_anatomy.py"], extra={"FEE_BP": fee_bp})


# ---- HBV1 rev19: forest scaling — bag generator for arbitrary j (reproduces
# FBAG_DROPS exactly on j=0..3; asserted at call time).
def _bag(j: int) -> str:
    import numpy as np
    dead = {17, 18, 19, 24, 30, 44, 50, 51, 52, 53, 56, 57, 58}
    live = [i for i in range(71) if i not in dead and i not in (59, 60)]
    extra = np.random.default_rng(1000 + j).choice(live, size=14, replace=False)
    return ",".join(map(str, sorted({59, 60} | {int(x) for x in extra})))


@app.function(image=image, volumes={"/vol": vol}, cpu=6, memory=20480, timeout=4 * 3600)
def train_rf_any_fn(j: int):
    drop = _bag(j)
    if j in FBAG_DROPS:
        assert drop == FBAG_DROPS[j], f"bag generator mismatch at j={j}"
    sub = RF_SUB.format(j=j)
    done = f"/vol/gcs/market-data-0998ac51/research_runs/{sub}/PERFOLD_S{j}_DOGE_qm0_f6.npz"
    if os.path.exists(done):
        return f"rf{j} already-done"
    _run([sys.executable, "/repo/scripts/subs60_xgb_sobol_v2.py", "DOGE", "maker_labels_tb3s_h150anch", "0", "6"],
         extra={"SEED": str(j), "CFGIDX": "1", "BUDGETS": "5,10", "SAVE_PF": "1",
                "PFTAG": f"_S{j}", "MODEL_DUMP": "1", "DATA_CACHE": "/tmp/cache",
                "SOBOL_PAR": "6", "FOLD_PAR": "1", "DROP_COLS": drop,
                "BAG_FRAC": "0.632", "BAG_SEED": str(j), "OUT_SUB": sub})
    vol.commit()
    return f"rf{j} trained (bag 0.632 + DROP {drop})"


@app.local_entrypoint()
def train_forest(js: str = "4-31"):
    a, b = js.split("-")
    calls = [train_rf_any_fn.spawn(j) for j in range(int(a), int(b) + 1)]
    for c in calls:
        try:
            print(c.get(timeout=4 * 3600), flush=True)
        except Exception as e:
            print("FAIL:", e, flush=True)


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=2 * 3600)
def finish_forest_fn(js: str = "0-31"):
    a, b = js.split("-")
    out = ""
    for j in range(int(a), int(b) + 1):
        out += _run([sys.executable, "/repo/runtime/perseed_from_pf.py", "DOGE", str(j)],
                    extra={"XSYM_SUB": RF_SUB.format(j=j)})
    vol.commit()
    return out


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=3600)
def finish_rf_fn():
    out = ""
    for j in range(4):
        out += _run([sys.executable, "/repo/runtime/perseed_from_pf.py", "DOGE", str(j)],
                    extra={"XSYM_SUB": RF_SUB.format(j=j)})
    vol.commit()
    return out


# ---- HBV2: symbol-parameterized build/train stages (DOGE stages above stay
# byte-identical; new symbols run on their own accounts/volumes). spec format
# for day fan-out: "day|SYM|SYMF" (e.g. "2025-05-09|1000PEPEUSDT|1000PEPE-USDT-PERP").
@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=4096, timeout=3600)
def aux_sym_fn(sym: str):
    _run([sys.executable, "/repo/runtime/bybit_repl/fetch_aux.py"],
         extra={"START": "2025-05-01", "END": "2026-06-05", "SYM": sym})
    vol.commit()
    return f"aux {sym} done"


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=9216, timeout=5400,
              retries=modal.Retries(max_retries=2), max_containers=40)
def day_sym_fn(spec: str):
    day, sym, symf = spec.split("|")
    base = symf.split("-")[0]
    done = f"/vol/gcs/market-data-0998ac51/research_runs/maker_labels_tb3s_h150/daily/{base}_{day}.npz"
    if os.path.exists(done):
        return f"{day}: already-done"
    sys.path.insert(0, "/repo/runtime/bybit_repl")
    os.environ.update(ENV)
    os.environ["WORK"] = "/tmp/conv"
    os.environ["SYM"] = sym
    os.environ["SYMF_CONV"] = symf
    import convert_bybit
    conv = convert_bybit.run_day(day)
    vol.commit()
    if "book-missing" in conv or "trades-missing" in conv.split("|")[1]:
        return f"{day}: SKIP no-raw ({conv})"
    out = _run([sys.executable, "/repo/scripts/subs60_build_tb3s_labels.py"],
               extra=dict(BUILD_ENV, SYMF=symf, START=day, END=day, WORKDIR="/tmp/tb3s"))
    vol.commit()
    tail = [ln for ln in out.splitlines() if ln.strip().startswith(day)]
    return f"{day}: {conv} || {tail[-1].strip() if tail else 'no-day-line'}"


@app.local_entrypoint()
def days_sym(sym: str, symf: str, start: str = START, end: str = END):
    import numpy as np
    all_days = [str(d) for d in np.arange(np.datetime64(start), np.datetime64(end) + 1)]
    print(f"{sym}: {len(all_days)} days {all_days[0]}..{all_days[-1]}")
    specs = [f"{d}|{sym}|{symf}" for d in all_days]
    ok = skip = 0
    for res in day_sym_fn.map(specs, return_exceptions=True):
        s = str(res)
        print(s, flush=True)
        if "SKIP" in s or "Exception" in s:
            skip += 1
        else:
            ok += 1
    print(f"[days done] ok={ok} skip/fail={skip}")


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=20480, timeout=5400)
def combine_sym_fn(symf: str):
    _run([sys.executable, "/repo/scripts/subs60_build_tb3s_labels.py"],
         extra=dict(BUILD_ENV, SYMF=symf, START=START, END=END, COMBINE="1", WORKDIR="/tmp/tb3s"))
    vol.commit()
    return f"combined {symf}"


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=12288, timeout=3600)
def anch_sym_fn(base: str):
    _run([sys.executable, "/repo/runtime/prep_anch_sym.py", base])
    vol.commit()
    return f"anch {base} done"


@app.function(image=image, volumes={"/vol": vol}, cpu=6, memory=20480, timeout=4 * 3600)
def train_sym_rf_fn(spec: str):
    base, j_s = spec.split("|")
    j = int(j_s)
    drop = _bag(j)
    sub = f"maker_labels_tb3s_h150anch_v2_nooi_rf{j}"
    done = f"/vol/gcs/market-data-0998ac51/research_runs/{sub}/PERFOLD_S{j}_{base}_qm0_f6.npz"
    if os.path.exists(done):
        return f"{base} rf{j} already-done"
    _run([sys.executable, "/repo/scripts/subs60_xgb_sobol_v2.py", base, "maker_labels_tb3s_h150anch", "0", "6"],
         extra={"SEED": str(j), "CFGIDX": "1", "BUDGETS": "5,10", "SAVE_PF": "1",
                "PFTAG": f"_S{j}", "MODEL_DUMP": "1", "DATA_CACHE": "/tmp/cache",
                "SOBOL_PAR": "6", "FOLD_PAR": "1", "DROP_COLS": drop,
                "BAG_FRAC": "0.632", "BAG_SEED": str(j), "OUT_SUB": sub})
    vol.commit()
    return f"{base} rf{j} trained"


# HBV2 rev3 (signal-first canon): SINGLE-model condition cells — SEED=0, no
# data-bag, TARGETED (non-random) drop sets from member forensics / HD3 axes.
# spec: "BASE|tag|drop_cols|cfgidx"
@app.function(image=image, volumes={"/vol": vol}, cpu=6, memory=20480, timeout=4 * 3600)
def train_solo_fn(spec: str):
    base, tag, drop, cfg = spec.split("|")
    sub = f"maker_labels_tb3s_h150anch_solo_{tag}"
    done = f"/vol/gcs/market-data-0998ac51/research_runs/{sub}/PERFOLD_S0_{base}_qm0_f6.npz"
    if os.path.exists(done):
        return f"{base} solo_{tag} already-done"
    _run([sys.executable, "/repo/scripts/subs60_xgb_sobol_v2.py", base, "maker_labels_tb3s_h150anch", "0", "6"],
         extra={"SEED": "0", "CFGIDX": cfg, "BUDGETS": "5,10", "SAVE_PF": "1",
                "PFTAG": "_S0", "MODEL_DUMP": "1", "DATA_CACHE": "/tmp/cache",
                "SOBOL_PAR": "6", "FOLD_PAR": "1", "DROP_COLS": drop, "OUT_SUB": sub})
    vol.commit()
    return f"{base} solo_{tag} trained (DROP {drop} cfg {cfg})"


@app.local_entrypoint()
def train_solo(base: str, specs: str):
    """specs: semicolon-separated tag:drop:cfgidx triples."""
    calls = []
    for spec in specs.split(";"):
        tag, drop, cfg = spec.split(":")
        calls.append(train_solo_fn.spawn(f"{base}|{tag}|{drop}|{cfg}"))
    for c in calls:
        try:
            print(c.get(timeout=4 * 3600), flush=True)
        except Exception as e:
            print("FAIL:", e, flush=True)


# HBV2 rev4: targeted-axis ensemble members — the WINNING solo spec (fixed
# drop set + cfgidx) x seed/data-bag diversity (NO random feature bags: the
# symbol's axes stay fixed; decorrelation comes from seed + BAG_FRAC only).
@app.function(image=image, volumes={"/vol": vol}, cpu=6, memory=20480, timeout=4 * 3600)
def train_axis_fn(spec: str):
    base, tag, drop, cfg, j_s = spec.split("|")
    j = int(j_s)
    sub = f"maker_labels_tb3s_h150anch_ax_{tag}_m{j}"
    done = f"/vol/gcs/market-data-0998ac51/research_runs/{sub}/PERFOLD_S{j}_{base}_qm0_f6.npz"
    if os.path.exists(done):
        return f"{base} ax_{tag} m{j} already-done"
    _run([sys.executable, "/repo/scripts/subs60_xgb_sobol_v2.py", base, "maker_labels_tb3s_h150anch", "0", "6"],
         extra={"SEED": str(j), "CFGIDX": cfg, "BUDGETS": "5,10", "SAVE_PF": "1",
                "PFTAG": f"_S{j}", "MODEL_DUMP": "1", "DATA_CACHE": "/tmp/cache",
                "SOBOL_PAR": "6", "FOLD_PAR": "1", "DROP_COLS": drop,
                "BAG_FRAC": "0.632", "BAG_SEED": str(j), "OUT_SUB": sub})
    vol.commit()
    return f"{base} ax_{tag} m{j} trained"


@app.local_entrypoint()
def train_axis(base: str, tag: str, drop: str, cfg: str = "1", js: str = "0-5"):
    a, b = js.split("-")
    calls = [train_axis_fn.spawn(f"{base}|{tag}|{drop}|{cfg}|{j}") for j in range(int(a), int(b) + 1)]
    for c in calls:
        try:
            print(c.get(timeout=4 * 3600), flush=True)
        except Exception as e:
            print("FAIL:", e, flush=True)


@app.local_entrypoint()
def train_sym_forest(base: str, js: str = "0-7"):
    a, b = js.split("-")
    calls = [train_sym_rf_fn.spawn(f"{base}|{j}") for j in range(int(a), int(b) + 1)]
    for c in calls:
        try:
            print(c.get(timeout=4 * 3600), flush=True)
        except Exception as e:
            print("FAIL:", e, flush=True)


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
