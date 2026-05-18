#!/usr/bin/env python3
"""HD1-seq runner on Modal — raw-L2 per-tick TCN vs HM6 baseline_ref.

Frozen design: HD1 rev25 (research/hypotheses.jsonl, freeze commit
d56344f). This module is the OPTIMIZED, parity-gated implementation:
column-projected streaming L2 cache build (Modal CPU fan-out, persist
only packed fp16 windows), in-memory AMP GPU training grouped per
(symbol, L). Numeric label/scope/split/R1/AUC/bootstrap come from
scripts.hd1_seq_core (bit-exact vs frozen hr1/ha5; gated by
tests/test_hd1_parity.py before any sweep).

Pipeline (local entrypoint orchestrates Modal):
  parity-gate -> plan_window -> measure_egress (ABORT if proj > $58)
  -> build_symbol_day.map -> reduce_symbol -> train_cell.map
  (cost-guard degradation) -> ingest -> experiments.jsonl + build-db.

Run:  modal run scripts/hd1_seq_modal.py            # full pipeline
      modal run scripts/hd1_seq_modal.py --dry 1    # measure+plan only
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

# ---- frozen constants (HD1 rev25 / HM6 testbed) -------------------------
BUCKET = "blackdigital-scalper-data"
GCP_PROJECT = "project-26a24ad0-1059-4f73-93b"
SYMS = ["SOL-USDT-PERP", "BTC-USDT-PERP", "ETH-USDT-PERP", "LTC-USDT-PERP"]
FLOOR = "2025-05-09"
N_DAYS = 360
FREEZE_COMMIT = "d56344f"
HS = (180, 300, 600)

# Frozen HM6 rev4 baseline_ref (run phaseb-20260517-203822), rank_ic_oos
# per (symbol, H). delta_ic = HD1-seq rank_ic_oos - this. DO NOT EDIT
# (would break the comparability invariant; pre-registered).
BASELINE_REF_RIC = {
    ("SOL-USDT-PERP", 180): 0.0158, ("SOL-USDT-PERP", 300): 0.0163,
    ("SOL-USDT-PERP", 600): 0.0167,
    ("BTC-USDT-PERP", 180): 0.0419, ("BTC-USDT-PERP", 300): 0.0254,
    ("BTC-USDT-PERP", 600): 0.0169,
    ("ETH-USDT-PERP", 180): 0.0234, ("ETH-USDT-PERP", 300): 0.0155,
    ("ETH-USDT-PERP", 600): 0.0099,
    ("LTC-USDT-PERP", 180): 0.0378, ("LTC-USDT-PERP", 300): 0.0404,
    ("LTC-USDT-PERP", 600): 0.0257,
}
BASELINE_RUN = "phaseb-20260517-203822"

# Modal pricing ($/s) for the cost-guard (modal.com/pricing, 2026-05).
PRICE = {"L4": 0.000222, "T4": 0.000164, "CPU_CORE": 0.0000131,
         "MEM_GIB": 0.00000222}
GCS_EGRESS_PER_GB = 0.12          # conservative cross-cloud egress
BUDGET = 58.0
COST_GUARD = 54.0                 # pre-registered degradation trigger

# ---- Modal app / image / volume / secret --------------------------------
app = modal.App("hd1-seq")

CPU_IMG = (modal.Image.debian_slim(python_version="3.11")
           .pip_install("numpy==2.2.4", "pyarrow==19.0.1",
                        "scikit-learn", "google-cloud-storage")
           .add_local_dir(str(REPO), "/root/proj",
                           ignore=["**/.git/**", "**/__pycache__/**",
                                   "data/**", "models/**", "**/*.db",
                                   "research_runs/**"]))
GPU_IMG = (modal.Image.debian_slim(python_version="3.11")
           .pip_install("numpy==2.2.4", "scikit-learn", "torch")
           .add_local_dir(str(REPO), "/root/proj",
                          ignore=["**/.git/**", "**/__pycache__/**",
                                  "data/**", "models/**", "**/*.db",
                                  "research_runs/**"]))
VOL = modal.Volume.from_name("hd1-seq-cache", create_if_missing=True)
MNT = "/cache"


def _gcs():
    """storage bucket from whatever GCS credential the env carries.

    Mirrors phase_b_vm._bearer_token/_creds: the Claude-web/Modal env
    supplies a bare ~1h OAuth token (ya29.*) in GCP_ACCESS_TOKEN /
    GCP_SA_KEY_B64 / GCP_SA_KEY, OR a long-lived service-account JSON.
    A bare token has no refresh fields, so google-auth refresh is
    neutralised (a bare token is valid for its own lifetime only — the
    build must finish inside that TTL or use an SA-JSON credential)."""
    import base64
    from google.cloud import storage
    raw = None
    for name in ("GCP_ACCESS_TOKEN", "GCP_SA_KEY_B64", "GCP_SA_KEY"):
        v = os.environ.get(name)
        if not v:
            continue
        if name == "GCP_SA_KEY_B64":
            try:
                v = base64.b64decode(v).decode()
            except Exception:
                continue
        raw = v
        s = "".join(v.split())
        if s.startswith("ya29.") or s.startswith("ya29_"):
            import google.oauth2.credentials

            class _Static(google.oauth2.credentials.Credentials):
                def refresh(self, request):      # bare token: no refresh
                    return
            cl = storage.Client(project=GCP_PROJECT,
                                 credentials=_Static(token=s))
            return cl.bucket(BUCKET)
        if s[:1] == "{":
            info = json.loads(v)
            if info.get("type") == "authorized_user":
                import google.oauth2.credentials as goc
                cr = goc.Credentials.from_authorized_user_info(info)
                return storage.Client(project=GCP_PROJECT,
                                      credentials=cr).bucket(BUCKET)
            return storage.Client.from_service_account_info(
                info, project=GCP_PROJECT).bucket(BUCKET)
    if raw is None:
        raise RuntimeError("no GCS credential in env (GCP_ACCESS_TOKEN/"
                           "GCP_SA_KEY[_B64])")
    raise RuntimeError("GCS credential present but unrecognised format")


def _list_days(bk, sym):
    pref = f"features_v1/symbol={sym}/"
    it = bk.client.list_blobs(bk, prefix=pref, delimiter="/")
    for _ in it:
        pass
    return sorted(p.split("dt=")[1].rstrip("/") for p in it.prefixes)


def _window(per):
    """baseline_360._window rule, verbatim (FROZEN comparability)."""
    import datetime as dt
    if not all(per.values()):
        return None, None, {}
    start = max([FLOOR] + [d[0] for d in per.values()])
    end = min(d[-1] for d in per.values())
    lo_cal = (dt.date.fromisoformat(end)
              - dt.timedelta(days=N_DAYS - 1)).isoformat()
    winlo = max(start, lo_cal)
    psd = {s: [d for d in per[s] if winlo <= d <= end] for s in SYMS}
    return winlo, end, psd


# =========================================================================
# plan_window — list GCS days, apply frozen window rule
# =========================================================================
@app.function(image=CPU_IMG, timeout=900,
              secrets=[modal.Secret.from_name("hd1-gcp")])
def plan_window():
    bk = _gcs()
    per = {s: _list_days(bk, s) for s in SYMS}
    winlo, winhi, psd = _window(per)
    return {"winlo": winlo, "winhi": winhi,
            "psd": psd, "full_counts": {s: len(per[s]) for s in SYMS}}


# =========================================================================
# measure_egress — pre-flight: true raw/book bytes for one symbol-day
# =========================================================================
@app.function(image=CPU_IMG, timeout=600,
              secrets=[modal.Secret.from_name("hd1-gcp")])
def measure_egress(sym: str, day: str):
    bk = _gcs()
    pref = f"raw/book/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/"
    blobs = [b for b in bk.client.list_blobs(bk, prefix=pref)
             if b.name.endswith(".parquet")]
    nbytes = 0
    for b in blobs:
        b.reload()
        nbytes += b.size or 0
    return {"sym": sym, "day": day, "n_files": len(blobs),
            "bytes": int(nbytes)}


# =========================================================================
# build_symbol_day — column-projected L2 -> packed fp16 windows shard
# =========================================================================
@app.function(image=CPU_IMG, cpu=2.0, timeout=3600,
              volumes={MNT: VOL},
              secrets=[modal.Secret.from_name("hd1-gcp")])
def build_symbol_day(sym: str, day: str, day_ord: int):
    import numpy as np
    import pyarrow.parquet as pq
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    out = f"{MNT}/shards/{sym}"
    os.makedirs(out, exist_ok=True)
    shard = f"{out}/{day_ord:04d}_{day}.npz"
    if os.path.exists(shard):
        return {"sym": sym, "day": day, "cached": True}

    bk = _gcs()
    # indices.npy = decision points (same grid as HM6/features_v1)
    idx = np.load(io.BytesIO(bk.blob(
        f"features_v1/symbol={sym}/dt={day}/indices.npy"
    ).download_as_bytes()), allow_pickle=False).astype(np.int64)

    cols = (["timestamp"]
            + [f"bid_{k}_price" for k in range(C.N_LEVELS)]
            + [f"bid_{k}_size" for k in range(C.N_LEVELS)]
            + [f"ask_{k}_price" for k in range(C.N_LEVELS)]
            + [f"ask_{k}_size" for k in range(C.N_LEVELS)])
    pref = f"raw/book/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/"
    blobs = sorted((b for b in bk.client.list_blobs(bk, prefix=pref)
                    if b.name.endswith(".parquet")), key=lambda b: b.name)
    if not blobs:
        return {"sym": sym, "day": day, "error": "no book parquet"}
    parts = []
    for b in blobs:
        t = pq.read_table(io.BytesIO(b.download_as_bytes()), columns=cols)
        parts.append(t)
    import pyarrow as pa
    T = pa.concat_tables(parts)
    ts = T.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)

    bid_p = np.column_stack([T.column(f"bid_{k}_price").to_numpy(
        zero_copy_only=False) for k in range(C.N_LEVELS)]).astype(np.float64)
    bid_s = np.column_stack([T.column(f"bid_{k}_size").to_numpy(
        zero_copy_only=False) for k in range(C.N_LEVELS)]).astype(np.float64)
    ask_p = np.column_stack([T.column(f"ask_{k}_price").to_numpy(
        zero_copy_only=False) for k in range(C.N_LEVELS)]).astype(np.float64)
    ask_s = np.column_stack([T.column(f"ask_{k}_size").to_numpy(
        zero_copy_only=False) for k in range(C.N_LEVELS)]).astype(np.float64)

    mid = 0.5 * (bid_p[:, 0] + ask_p[:, 0])
    n_ticks = ts.shape[0]
    sel, i = C.select_decision_idx(idx, n_ticks)
    if i.size == 0:
        np.savez_compressed(shard, empty=True)
        VOL.commit()
        return {"sym": sym, "day": day, "n_dp": 0}

    tf = C.tick_features(bid_p, bid_s, ask_p, ask_s)      # (n_ticks, 46)
    win, _ = C.gather_windows(tf, i, L=C.MAX_L)            # (n_dp,512,46) f16
    t0 = ts[i].astype(np.int64)

    lab = {}
    for H in HS:
        y0, rH, _, _ = C.labels_for_H(ts, mid, i, t0, H)
        lab[f"y0_{H}"] = y0
        lab[f"rH_{H}"] = rH.astype(np.float32)

    np.savez_compressed(
        shard, X=win, i=i.astype(np.int64), t0=t0,
        day_ord=np.int32(day_ord),
        n_dp=np.int64(i.size), **lab)
    VOL.commit()
    return {"sym": sym, "day": day, "n_dp": int(i.size),
            "shard_bytes": os.path.getsize(shard)}


# =========================================================================
# reduce_symbol — concat shards in (day asc, sel asc) -> packed.npz
# =========================================================================
@app.function(image=CPU_IMG, cpu=2.0, timeout=3600, volumes={MNT: VOL})
def reduce_symbol(sym: str):
    import numpy as np
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    VOL.reload()
    sd = f"{MNT}/shards/{sym}"
    files = sorted(f for f in os.listdir(sd) if f.endswith(".npz"))
    Xs, t0s, lab = [], [], {f"y0_{H}": [] for H in HS}
    for H in HS:
        lab[f"rH_{H}"] = []
    for f in files:
        d = np.load(f"{sd}/{f}")
        if "empty" in d.files:
            continue
        Xs.append(d["X"])
        t0s.append(d["t0"])
        for H in HS:
            lab[f"y0_{H}"].append(d[f"y0_{H}"])
            lab[f"rH_{H}"].append(d[f"rH_{H}"])
    X = np.concatenate(Xs)                       # (n, 512, 46) f16
    t0 = np.concatenate(t0s)
    n = X.shape[0]
    tr, te, n_tr = C.honest_split(n)
    packed = {"X": X, "t0": t0, "n": np.int64(n),
              "n_tr": np.int64(n_tr)}
    for H in HS:
        packed[f"y0_{H}"] = np.concatenate(lab[f"y0_{H}"]).astype(np.int8)
        packed[f"rH_{H}"] = np.concatenate(lab[f"rH_{H}"]).astype(np.float32)
    os.makedirs(f"{MNT}/packed", exist_ok=True)
    np.savez(f"{MNT}/packed/{sym}.npz", **packed)
    for f in files:                              # drop shards
        os.remove(f"{sd}/{f}")
    VOL.commit()
    return {"sym": sym, "n": int(n), "n_tr": int(n_tr),
            "packed_gib": round(X.nbytes / 2**30, 3)}


# =========================================================================
# TCN  (causal, dilated residual; frozen non-swept hyperparams)
# =========================================================================
def _build_tcn(F, W, D, dropout=0.1):
    import torch.nn as nn

    class Chomp(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c

        def forward(self, x):
            return x[:, :, :-self.c].contiguous() if self.c else x

    class Block(nn.Module):
        def __init__(self, ci, co, dil):
            super().__init__()
            pad = (3 - 1) * dil
            self.net = nn.Sequential(
                nn.Conv1d(ci, co, 3, padding=pad, dilation=dil),
                Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(co, co, 3, padding=pad, dilation=dil),
                Chomp(pad), nn.ReLU(), nn.Dropout(dropout))
            self.down = nn.Conv1d(ci, co, 1) if ci != co else None
            self.relu = nn.ReLU()

        def forward(self, x):
            r = x if self.down is None else self.down(x)
            return self.relu(self.net(x) + r)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers, ci = [], F
            for b in range(D):
                layers.append(Block(ci, W, 2 ** b))
                ci = W
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(W, 1)

        def forward(self, x):                    # x: (B, L, F)
            h = self.tcn(x.transpose(1, 2))      # (B, W, L)
            return self.head(h[:, :, -1]).squeeze(-1)   # last step

    return TCN()


@app.function(image=GPU_IMG, gpu=["L4", "T4"], timeout=5400,
              volumes={MNT: VOL}, retries=1)
def train_cell(sym: str, L: int, grid: list):
    """One container per (sym, L): trains all (H, D, seed) in `grid`
    (amortizes the packed-cache load). Returns per-H best-val metrics."""
    import numpy as np
    import torch
    import torch.nn.functional as Fnn
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    t_start = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    VOL.reload()
    P = np.load(f"{MNT}/packed/{sym}.npz")
    Xfull = P["X"]                                # (n,512,46) f16
    n = int(P["n"])
    XL = np.ascontiguousarray(Xfull[:, -L:, :])   # causal slice
    tr, te, _ = C.honest_split(n)

    res = {}
    for H in HS:
        y0 = P[f"y0_{H}"]
        rH = P[f"rH_{H}"].astype(np.float64)
        reached = (y0 != 0) & np.isfinite(rH)
        up = (y0 == 1).astype(np.float32)
        s_tr_all = tr & reached
        s_te = te & reached
        ntr, nte = int(s_tr_all.sum()), int(s_te.sum())
        if ntr < C.N_TR_FLOOR or nte < C.N_OOS_FLOOR:
            res[H] = {"error": f"underpowered n_tr={ntr} n_oos={nte}"}
            continue
        fit_m, val_m = C.train_val_split(s_tr_all)
        w1 = C.r1_weights(rH, s_tr_all).astype(np.float32)

        # standardize on FIT rows only (train-split stats, no leakage)
        fr = XL[fit_m].reshape(-1, C.N_TICK_FEAT)
        mu = fr.mean(0).astype(np.float32)
        sd = fr.std(0).astype(np.float32) + 1e-6

        def _T(mask):
            x = (XL[mask].astype(np.float32) - mu) / sd
            return torch.from_numpy(x)

        Xfit, Xval, Xte = _T(fit_m), _T(val_m), _T(s_te)
        yfit = torch.from_numpy(up[fit_m]); yval = torch.from_numpy(up[val_m])
        wfit = torch.from_numpy(w1[fit_m]); wval = torch.from_numpy(w1[val_m])
        yte = up[s_te].astype(int)
        block = C.block_size(H)

        grid_out, best = [], None
        for (HH, D, seed) in grid:
            if HH != H:
                continue
            torch.manual_seed(seed); np.random.seed(seed)
            net = _build_tcn(C.N_TICK_FEAT, C.W_FIXED, D).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3,
                                   weight_decay=C.WD_FIXED)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30)
            scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
            bs, best_val, patience, best_state = 1024, 1e9, 0, None
            idx_fit = np.arange(Xfit.shape[0])
            for ep in range(30):
                net.train()
                np.random.shuffle(idx_fit)
                for s in range(0, len(idx_fit), bs):
                    j = idx_fit[s:s + bs]
                    xb = Xfit[j].to(dev, non_blocking=True)
                    yb = yfit[j].to(dev); wb = wfit[j].to(dev)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=dev == "cuda"):
                        lo = net(xb)
                        loss = (Fnn.binary_cross_entropy_with_logits(
                            lo, yb, reduction="none") * wb).sum() / (
                            wb.sum() + 1e-9)
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
                sch.step()
                net.eval()
                with torch.no_grad(), torch.amp.autocast(
                        "cuda", enabled=dev == "cuda"):
                    vl = []
                    for s in range(0, Xval.shape[0], 4096):
                        xb = Xval[s:s + 4096].to(dev)
                        vl.append(net(xb).float().cpu())
                    vp = torch.cat(vl)
                    vloss = float((Fnn.binary_cross_entropy_with_logits(
                        vp, yval, reduction="none") * wval).sum()
                        / (wval.sum() + 1e-9))
                if vloss < best_val - 1e-5:
                    best_val, patience = vloss, 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in net.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 5:
                        break
            if best_state:
                net.load_state_dict(best_state)
            net.eval()
            with torch.no_grad(), torch.amp.autocast(
                    "cuda", enabled=dev == "cuda"):
                pl = []
                for s in range(0, Xte.shape[0], 4096):
                    pl.append(net(Xte[s:s + 4096].to(dev)).float().cpu())
                p = torch.sigmoid(torch.cat(pl)).numpy()
            a = C.auc(yte, p)
            g = {"H": H, "D": D, "seed": seed, "val_logloss": round(best_val, 6),
                 "auc": round(a, 6), "ric": round(a - 0.5, 6)}
            grid_out.append(g)
            if best is None or best_val < best["val_logloss"]:
                best = {**g, "_p": p}
        if best is None:
            res[H] = {"error": "no config trained"}
            continue
        p = best.pop("_p")
        plac = C.placebo_auc(yte, p)
        se = C.block_bootstrap_auc_se(yte, p, block)
        res[H] = {**best, "placebo_ric": round(plac - 0.5, 6),
                  "boot_se": None if not np.isfinite(se) else round(se, 6),
                  "n_tr": ntr, "n_oos": nte, "block": block,
                  "grid": grid_out}
    res["_gpu_seconds"] = round(time.time() - t_start, 1)
    return {"sym": sym, "L": L, "res": res}


# =========================================================================
# local entrypoint — orchestrate, cost-guard, ingest
# =========================================================================
def _parity_gate():
    import subprocess
    r = subprocess.run([sys.executable, "tests/test_hd1_parity.py"],
                       cwd=str(REPO), capture_output=True, text=True)
    print(r.stdout[-800:])
    if r.returncode != 0:
        print(r.stderr[-1500:])
        raise SystemExit("PARITY GATE FAILED — sweep aborted (frozen "
                         "numeric contract violated).")
    print("[parity] PASS — optimized core == frozen contract.")


@app.local_entrypoint()
def main(dry: int = 0):
    _parity_gate()

    plan = plan_window.remote()
    psd = plan["psd"]
    syms_days = [(s, d) for s in SYMS for d in psd[s]]
    n_sd = len(syms_days)
    print(f"[plan] window {plan['winlo']}..{plan['winhi']} "
          f"days/sym={ {s: len(psd[s]) for s in SYMS} } total_sd={n_sd}")

    # --- pre-flight egress: sample 1 day per symbol, extrapolate -------
    probe = [(s, psd[s][len(psd[s]) // 2]) for s in SYMS if psd[s]]
    meas = list(measure_egress.starmap(probe))
    by_sym_bytes = {m["sym"]: m["bytes"] for m in meas}
    proj_bytes = sum(by_sym_bytes[s] * len(psd[s]) for s in SYMS
                     if s in by_sym_bytes)
    egress_gb = proj_bytes / 2**30
    egress_usd = egress_gb * GCS_EGRESS_PER_GB
    # compute estimate: 288 train ~8min L4 + 16 build·reduce CPU + probes
    train_usd = 288 * 8 * 60 * PRICE["L4"]
    cpu_usd = (n_sd + len(SYMS)) * 120 * PRICE["CPU_CORE"] * 2
    proj_total = egress_usd + train_usd + cpu_usd
    print(f"[egress] sample {[ (m['sym'], m['bytes']) for m in meas ]}")
    print(f"[egress] projected raw/book egress = {egress_gb:.1f} GiB "
          f"= ${egress_usd:.2f}; +train≈${train_usd:.2f} "
          f"+cpu≈${cpu_usd:.2f}  => PROJECTED TOTAL ${proj_total:.2f} "
          f"(budget ${BUDGET})")
    if proj_total > BUDGET:
        raise SystemExit(
            f"ABORT: projected ${proj_total:.2f} > budget ${BUDGET}. "
            f"Egress dominates (${egress_usd:.2f}). Not transferring. "
            f"Options: (1) raise budget; (2) re-freeze §3 with fewer LOB "
            f"levels/days. Reporting to user — no spend incurred.")
    if dry:
        print("[dry] measure+plan only; stopping before build.")
        return

    # --- build (fan-out) + reduce -------------------------------------
    args = [(s, d, di) for s in SYMS
            for di, d in enumerate(psd[s])]
    built = list(build_symbol_day.starmap(args))
    n_dp = sum(b.get("n_dp", 0) for b in built)
    print(f"[build] {len(built)} shards, total decision points={n_dp}")
    red = list(reduce_symbol.map(SYMS))
    for r in red:
        print(f"[reduce] {r}")

    # --- sweep with pre-registered cost-guard degradation -------------
    sys.path.insert(0, str(REPO))
    from scripts import hd1_seq_core as C

    def make_grid(degrade):
        seeds = (0, 1) if degrade >= 1 else C.SEED_GRID
        Ds = (6,) if degrade >= 2 else C.D_GRID
        return seeds, Ds

    spent, degrade, results = 0.0, 0, {}
    L_GRID = C.L_GRID
    for si, sym in enumerate(SYMS):
        seeds, Ds = make_grid(degrade)
        calls = []
        for L in L_GRID:
            grid = [(H, D, sd) for H in HS for D in Ds for sd in seeds]
            calls.append((sym, L, grid))
        for out in train_cell.starmap(calls):
            results[(out["sym"], out["L"])] = out["res"]
            spent += out["res"].get("_gpu_seconds", 0) * PRICE["L4"]
        done_frac = (si + 1) / len(SYMS)
        proj = egress_usd + cpu_usd + spent / max(done_frac, 1e-6)
        print(f"[cost] after {sym}: spent≈${spent:.2f} "
              f"proj_total≈${proj:.2f} degrade={degrade}")
        if proj > COST_GUARD and degrade < 2:
            degrade += 1
            print(f"[cost-guard] proj>${COST_GUARD} -> degrade={degrade} "
                  f"(pre-registered; primary axis L untouched)")

    # --- ingest: 12 cells, frozen §5 gate -----------------------------
    _ingest(results, plan, egress_usd + cpu_usd + spent)
    print("HD1_SEQ_DONE")


def _ingest(results, plan, total_cost):
    import numpy as np
    sys.path.insert(0, str(REPO))
    from scripts import hd1_seq_core as C
    from research import ledger as L

    run_id = f"hd1seq-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    psd = plan["psd"]
    # best config per (sym,H) = the L with lowest val_logloss
    cells = {}
    for sym in SYMS:
        for H in HS:
            best = None
            for Lv in C.L_GRID:
                r = results.get((sym, Lv), {}).get(H)
                if not r or "error" in r:
                    continue
                if best is None or r["val_logloss"] < best["val_logloss"]:
                    best = {**r, "L": Lv}
            cells[(sym, H)] = best

    # cross-symbol consistency at matched H (frozen §5(c): >=3/4 sign)
    recs = []
    for H in HS:
        signs = []
        for sym in SYMS:
            b = cells.get((sym, H))
            if not b:
                continue
            d = C.cell_delta_ic(b["ric"], BASELINE_REF_RIC[(sym, H)])
            signs.append(1 if d > 0 else 0)
        cross_ok = sum(signs) >= 3
        for sym in SYMS:
            b = cells.get((sym, H))
            base = BASELINE_REF_RIC[(sym, H)]
            if not b:
                recs.append({"symbol": sym, "horizon_sec": H,
                             "error": "no result"})
                continue
            d_ic = C.cell_delta_ic(b["ric"], base)
            ab, why = C.gate_cell(d_ic, b["placebo_ric"], b["boot_se"])
            status = C.status_for_cell(ab, cross_ok)
            days = psd[sym]
            recs.append({
                "experiment_id": f"{run_id}_HD1SEQ_{sym}_H{H}",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_commit": FREEZE_COMMIT, "author": "claude(modal)",
                "hypothesis_id": "HD1", "status": status, "kind": "alpha",
                "setup": (f"HD1-seq raw-L2 20lvl per-tick causal TCN "
                          f"({sym}, H={H}s, >=cost scope; HM5 rev3 R1; "
                          f"vs HM6 baseline_ref {BASELINE_RUN})"),
                "model_family": "tcn",
                "params": {"input_repr": "raw_l2_20lvl_eventclock",
                           "L_best": b["L"], "D_best": b["D"],
                           "W": C.W_FIXED, "wd": C.WD_FIXED,
                           "seed_best": b["seed"], "block": b["block"],
                           "val_logloss": b["val_logloss"],
                           "auc_oos": b["auc"], "placebo_ric": b["placebo_ric"],
                           "boot_se": b["boot_se"], "grid": b["grid"],
                           "gate": why, "cross_symbol_ok": cross_ok,
                           "freeze": "HD1 rev25 d56344f"},
                "data_source": "cryptolake",
                "cache_id": (f"cryptolake_{sym}_raw_l2_20lvl_HD1SEQ_"
                             f"{plan['winlo']}_{plan['winhi']}"),
                "symbols": [sym], "date_range_start": days[0],
                "date_range_end": days[-1], "n_samples": int(b["n_tr"]
                                                             + b["n_oos"]),
                "fee_regime": "MAKER_FIRST", "commission_win_pct": 0.04,
                "commission_loss_pct": 0.07,
                "split_method": "honest_val_test", "embargo": str(C.EMB),
                "label_def": ("first-passage up-first on >=cost subset "
                              "(f=0.0013), byte-identical to HM6; input = "
                              "raw 20-level LOB per-tick event-clock seq"),
                "alpha_target": "updown_first_on_ge_cost_subset",
                "horizon_sec": H,
                "rank_ic_oos": round(b["ric"], 5),
                "auc_oos": round(b["auc"], 4),
                "baseline_ref": (f"HM6 rev4 {BASELINE_RUN} {sym} H{H} "
                                 f"rank_ic={base}"),
                "delta_ic": d_ic,
                "top_decile_absmove_pct": C.F_T0 * 100,
                "bot_decile_absmove_pct": C.F_T0 * 100,
                "cost_floor_pct": C.F_T0 * 100, "decile_monotonic": None,
                "economic_pass_loose": 0, "economic_pass_strict": 0,
                "n_eff": int(b["n_oos"]),
                "repro_cmd": "modal run scripts/hd1_seq_modal.py",
                "artifact_path": f"{MNT}/results/{run_id}.json",
                "note": (f"HD1-seq (frozen HD1 rev25). delta_ic={d_ic} vs "
                         f"HM6 baseline_ref {base}. gate(a&b)={ab} "
                         f"cross_sym>=3/4={cross_ok} -> {status}. HM1: "
                         f"never auto-confirmed (economic deploy-gate). "
                         f"approx run cost ${total_cost:.2f}."),
            })

    res_doc = {"run_id": run_id, "model_family": "tcn",
               "input_repr": "raw_l2_20lvl_eventclock",
               "symbol_set": SYMS,
               "window": [plan["winlo"], plan["winhi"]],
               "n_days_per_symbol": {s: len(psd[s]) for s in SYMS},
               "freeze": "HD1 rev25 d56344f",
               "approx_cost_usd": round(total_cost, 2),
               "records": recs,
               "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime())}
    art = REPO / "research_runs" / run_id
    art.mkdir(parents=True, exist_ok=True)
    (art / "results.json").write_text(json.dumps(res_doc, indent=2,
                                                 default=str))

    ok = 0
    expp = REPO / "research" / "experiments.jsonl"
    with open(expp, "a") as fh:
        for r in recs:
            if "error" in r and "experiment_id" not in r:
                continue
            try:
                L.validate_experiment(r)
            except L.LedgerError as e:
                print(f"[ingest] SKIP invalid: {e}")
                continue
            fh.write(json.dumps(r, default=str) + "\n")
            ok += 1
    print(f"[ingest] appended {ok}/{len(recs)} -> experiments.jsonl")
    import argparse
    L.cmd_build_db(argparse.Namespace(db=str(REPO / "research" /
                                             "research.db")))
    print(f"[ingest] results.json -> {art/'results.json'}")
