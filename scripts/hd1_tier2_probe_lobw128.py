#!/usr/bin/env python3
"""HD1 rev48 EXPLORATORY PROBE: architecture-on-raw-L2 at LTC-H300-L=512.

Sub-tier of rev45/47 (weak-control LTC cell only). Builds a raw-L2 pack
for LTC inside Modal (streaming from GCS, no GCP VM, no cross-cloud
transfer), then probes a 2-stream TCN (raw 20-lvl LOB at W=128
mean-pool + 6 engineered last-tick globals at Linear(6,32)) at 3 seeds.

Pre-reg: research/hypotheses.jsonl HD1 rev48.

Decision rule per CLAUDE.md exploratory frame: report the surface vs
the rev47 LTC-H300-L=512 anchor (+0.0240 +- 0.0069), NO sec5 verdict.

Usage:
  modal run scripts/hd1_tier2_probe_lobw128.py --stage build
  modal run scripts/hd1_tier2_probe_lobw128.py --stage probe
  modal run scripts/hd1_tier2_probe_lobw128.py --stage all
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ---- rev48 frozen probe spec ----------------------------------------------
SYM = "LTC-USDT-PERP"
H = 300
L = 512
D = 8                  # RF-matched for L=512 (rev45 map)
W_LOB = 128            # NEW vs rev45/rev47 (W=16)
F_LOB = 80             # 20 levels x 4 channels (bid_p, bid_s, ask_p, ask_s)
F_GLOB = 6             # last-tick engineered globals [40:46]
SEEDS = (0, 1, 2)
DROPOUT = 0.5
WD = 1e-3
EP_CAP = 12
PATIENCE = 3
T_MAX = 8
MIN_DELTA = 0.0
TR_BATCH = 1024
EV_CHUNK = 4096
L4_USD_PER_S = 0.000222

BUCKET = "blackdigital-scalper-data"
GCP_PROJECT = "project-26a24ad0-1059-4f73-93b"

# ---- Modal app / image / volume -------------------------------------------
_IGN = ["**/.git/**", "**/__pycache__/**", "data/**", "models/**",
        "**/*.db", "research_runs/**", "**/target/**"]

app = modal.App("hd1-tier2-probe-lobw128")
IMG = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("numpy==2.2.4", "scikit-learn", "torch",
                    "pyarrow", "google-cloud-storage")
       .add_local_dir(REPO, "/root/proj", ignore=_IGN))
VOL = modal.Volume.from_name("hd1-seq-cache", create_if_missing=True)
MNT = "/cache"
PACK_DIR = f"{MNT}/packed_l1536"
RAW_PACK = f"{PACK_DIR}/{SYM}_raw_L{L}.npy"
RESULT_PATH = f"{MNT}/tier2/rev48_probe_lobw128_result.json"


def _gcs_bucket():
    """== hd1_seq_modal._gcs (ADC + env-token fallbacks)."""
    from google.cloud import storage
    try:
        return storage.Client(project=GCP_PROJECT).bucket(BUCKET)
    except Exception:
        pass
    for name in ("GCP_ACCESS_TOKEN", "GCP_SA_KEY_B64", "GCP_SA_KEY"):
        v = os.environ.get(name)
        if not v:
            continue
        if name == "GCP_SA_KEY_B64":
            import base64
            v = base64.b64decode(v).decode()
        s = "".join(v.split())
        if s.startswith("ya29.") or s.startswith("ya29_"):
            import google.oauth2.credentials as goc

            class _Static(goc.Credentials):
                def refresh(self, request):
                    return

            return storage.Client(project=GCP_PROJECT,
                                  credentials=_Static(token=s)).bucket(BUCKET)
        if s[:1] == "{":
            info = json.loads(v)
            if info.get("type") == "authorized_user":
                import google.oauth2.credentials as goc
                return storage.Client(
                    project=GCP_PROJECT,
                    credentials=goc.Credentials.from_authorized_user_info(
                        info)).bucket(BUCKET)
            return storage.Client.from_service_account_info(
                info, project=GCP_PROJECT).bucket(BUCKET)
    raise RuntimeError("no GCS credential in env")


# ---- BUILD: stream raw-L2 parquets, gather windows at LTC_meta.t0 ---------
@app.function(image=IMG, cpu=4.0, memory=16384, timeout=10800,
              volumes={MNT: VOL},
              secrets=[modal.Secret.from_name("hd1-gcp")])
def build_raw_l2_ltc():
    """Stream LTC raw-book parquets day-by-day from GCS, gather L=512
    right-causal windows at the SAME LTC_meta.t0 decision points
    already in the rev45 pack, normalize per channel (prices as
    (p-mid)/mid, sizes as sign*log1p(|s|)), write a (n,L,80) f32 .npy
    to the Volume.

    Cross-day windows: a rolling tail buffer keeps the last L ticks of
    the previous day so windows straddling midnight gather correctly.

    Skip-if-exists: if RAW_PACK already exists with the expected shape,
    return immediately (idempotent rerun)."""
    import tempfile
    import numpy as np
    import pyarrow.parquet as pq

    VOL.reload()
    os.makedirs(PACK_DIR, exist_ok=True)
    os.makedirs(f"{MNT}/tier2", exist_ok=True)

    meta = np.load(f"{PACK_DIR}/{SYM}_meta.npz")
    n_exp = int(meta["n"])
    t0 = meta["t0"].astype(np.int64)   # ns since epoch (UTC)
    assert t0.size == n_exp

    if os.path.exists(RAW_PACK):
        try:
            arr = np.load(RAW_PACK, mmap_mode="r")
            if arr.shape == (n_exp, L, F_LOB):
                return {"status": "skip_existing",
                        "shape": list(arr.shape)}
        except Exception:
            pass

    NS_PER_DAY = 86400 * 1_000_000_000
    EPOCH = dt.date(1970, 1, 1)

    def d2s(days_since_epoch: int) -> str:
        return (EPOCH + dt.timedelta(days=int(days_since_epoch))).isoformat()

    # decision-point days
    t0_day = (t0 // NS_PER_DAY).astype(np.int64)
    unique_days = np.unique(t0_day)
    # also fetch the day BEFORE the first dp-day so cross-midnight windows
    # at the very beginning have prior-day ticks available
    earliest_day = int(unique_days.min())
    days_to_fetch = [earliest_day - 1] + list(unique_days)

    bk = _gcs_bucket()

    # pre-allocate output .npy memmap
    out = np.lib.format.open_memmap(
        RAW_PACK, mode="w+", dtype=np.float32, shape=(n_exp, L, F_LOB))

    # channel order: bid_p[0..19], bid_s[0..19], ask_p[0..19], ask_s[0..19]
    chan_cols = []
    for k in range(20):
        chan_cols.append(f"bid_{k}_price")
    for k in range(20):
        chan_cols.append(f"bid_{k}_size")
    for k in range(20):
        chan_cols.append(f"ask_{k}_price")
    for k in range(20):
        chan_cols.append(f"ask_{k}_size")
    assert len(chan_cols) == F_LOB

    tail_ts = np.empty((0,), np.int64)
    tail_X = np.empty((0, F_LOB), np.float32)
    total_filled = 0
    t_start = time.time()
    log_lines = []

    for d_int in days_to_fetch:
        day_str = d2s(d_int)
        day_start_ns = d_int * NS_PER_DAY
        day_end_ns = day_start_ns + NS_PER_DAY
        prefix = (f"raw/book/exchange=BINANCE_FUTURES/"
                  f"symbol={SYM}/dt={day_str}/")
        blobs = sorted([b for b in bk.client.list_blobs(bk, prefix=prefix)
                        if b.name.endswith(".parquet")],
                       key=lambda b: b.name)
        if not blobs:
            log_lines.append(f"{day_str}: no parquets (skipping)")
            continue

        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, b in enumerate(blobs):
                p = f"{td}/p{i:04d}.parquet"
                b.download_to_filename(p)
                paths.append(p)
            ts_parts, X_parts = [], []
            for p in paths:
                t = pq.read_table(p, columns=["timestamp"] + chan_cols)
                ts_parts.append(t["timestamp"].to_numpy().astype(np.int64))
                X_parts.append(np.column_stack(
                    [t[c].to_numpy().astype(np.float32) for c in chan_cols]))
            ts_day = np.concatenate(ts_parts)
            X_day = np.concatenate(X_parts, axis=0)
            order = np.argsort(ts_day, kind="stable")
            ts_day = ts_day[order]
            X_day = X_day[order]

            # normalize per channel
            bid_0_p = X_day[:, 0]
            ask_0_p = X_day[:, 40]
            mid = 0.5 * (bid_0_p + ask_0_p)
            mid_safe = np.where(mid > 0, mid, 1.0).astype(np.float32)
            # price channels
            for k in range(20):
                X_day[:, k] = (X_day[:, k] - mid) / mid_safe
                X_day[:, 40 + k] = (X_day[:, 40 + k] - mid) / mid_safe
            # size channels: sign*log1p(|s|)
            for k in range(20):
                s = X_day[:, 20 + k]
                X_day[:, 20 + k] = np.sign(s) * np.log1p(np.abs(s))
                s = X_day[:, 60 + k]
                X_day[:, 60 + k] = np.sign(s) * np.log1p(np.abs(s))

            ts_full = np.concatenate([tail_ts, ts_day])
            X_full = np.concatenate([tail_X, X_day], axis=0)

            in_day = (t0 >= day_start_ns) & (t0 < day_end_ns)
            dp_indices = np.where(in_day)[0]
            for dp_i in dp_indices:
                t = int(t0[dp_i])
                j = int(np.searchsorted(ts_full, t, side="right")) - 1
                if j < 0:
                    out[dp_i] = 0.0
                    continue
                lo = j - L + 1
                if lo >= 0:
                    out[dp_i] = X_full[lo:j + 1]
                else:
                    pad = -lo
                    win = np.zeros((L, F_LOB), np.float32)
                    win[pad:] = X_full[:j + 1]
                    out[dp_i] = win
            total_filled += dp_indices.size

            # tail = last L ticks for next day's cross-midnight windows
            keep = min(L, ts_full.size)
            tail_ts = np.ascontiguousarray(ts_full[-keep:])
            tail_X = np.ascontiguousarray(X_full[-keep:])

        log_lines.append(
            f"{day_str}: ticks={ts_day.size} dp_in_day="
            f"{int(in_day.sum())} total_filled={total_filled}")
        if len(log_lines) % 20 == 0:
            print(" ".join(log_lines[-3:]))
            sys.stdout.flush()
            VOL.commit()

    out.flush()
    del out
    VOL.commit()
    elapsed = round(time.time() - t_start, 1)
    return {"status": "built", "n_expected": n_exp,
            "total_filled": total_filled,
            "shape": [n_exp, L, F_LOB],
            "path": RAW_PACK, "elapsed_s": elapsed,
            "days_processed": len(days_to_fetch),
            "tail_log": log_lines[-5:]}


# ---- MODEL: 2-stream TCN (raw LOB W=128 mean-pool + 6-globals Linear) -----
def _build_two_stream_tcn():
    import torch
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
                Chomp(pad), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Conv1d(co, co, 3, padding=pad, dilation=dil),
                Chomp(pad), nn.ReLU(), nn.Dropout(DROPOUT))
            self.down = nn.Conv1d(ci, co, 1) if ci != co else None
            self.relu = nn.ReLU()

        def forward(self, x):
            r = x if self.down is None else self.down(x)
            return self.relu(self.net(x) + r)

    class TwoStream(nn.Module):
        def __init__(self):
            super().__init__()
            layers, ci = [], F_LOB
            for b in range(D):
                layers.append(Block(ci, W_LOB, 2 ** b))
                ci = W_LOB
            self.lob = nn.Sequential(*layers)
            self.glob = nn.Sequential(
                nn.Linear(F_GLOB, 32),
                nn.GELU(),
                nn.Dropout(DROPOUT))
            self.head = nn.Linear(W_LOB + 32, 1)

        def forward(self, x_lob, x_glob):
            # x_lob (B, L, 80) -> (B, 80, L)
            h = self.lob(x_lob.transpose(1, 2))  # (B, W_LOB, L)
            h = h.mean(dim=2)                    # mean-pool
            g = self.glob(x_glob)                # (B, 32)
            z = torch.cat([h, g], dim=1)         # (B, W_LOB+32)
            return self.head(z).squeeze(-1)

    return TwoStream()


# ---- PROBE: one seed of LTC-H300-L=512 W=128 2-stream BCE ----------------
@app.function(image=IMG, gpu="L4", timeout=3600, memory=32768,
              volumes={MNT: VOL})
def probe_lobw128(seed: int):
    """One seed of rev48: LTC-H300-L=512 D=8 W_lob=128 2-stream TCN
    (raw 20-lvl LOB mean-pool + 6 last-tick engineered globals)
    + BCE-with-r1 + rev45-locked schedule."""
    import numpy as np
    import torch
    import torch.nn.functional as Fnn

    VOL.reload()
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X_eng = np.load(f"{PACK_DIR}/{SYM}_X.npy", mmap_mode="r")   # (n,1536,46)
    X_raw = np.load(RAW_PACK, mmap_mode="r")                    # (n,L,80)
    meta = np.load(f"{PACK_DIR}/{SYM}_meta.npz")
    n = int(meta["n"])
    y0 = meta[f"y0_{H}"]
    rH = meta[f"rH_{H}"].astype(np.float64)

    # MASKS: same rev45/rev47 contract
    tr, te, _ = C.honest_split(n)
    reached = (y0 != 0) & np.isfinite(rH)
    up = (y0 == 1).astype(np.float32)
    s_tr_all = tr & reached
    s_te = te & reached
    fit_m, val_m = C.train_val_split(s_tr_all)
    w1 = C.r1_weights(rH, s_tr_all).astype(np.float32)

    fit_idx = np.where(fit_m)[0]
    val_idx = np.where(val_m)[0]
    te_idx = np.where(s_te)[0]

    # GLOBALS stream: last-tick cols [40:46] from engineered pack
    G_all = np.ascontiguousarray(
        X_eng[:, -1, 40:46]).astype(np.float32)  # (n, 6)
    g_mu = G_all[fit_m].mean(axis=0).astype(np.float32)
    g_sd = G_all[fit_m].std(axis=0).astype(np.float32) + 1e-6
    G_all = (G_all - g_mu) / g_sd

    # LOB stream: per-channel z-score on fit rows only (streamed, f64)
    s_acc = np.zeros(F_LOB, np.float64)
    ss_acc = np.zeros(F_LOB, np.float64)
    cnt = 0
    SCHUNK = 1024
    for c in range(0, fit_idx.size, SCHUNK):
        blk = np.ascontiguousarray(
            X_raw[fit_idx[c:c + SCHUNK]]).reshape(-1, F_LOB).astype(np.float64)
        s_acc += blk.sum(axis=0)
        ss_acc += np.square(blk).sum(axis=0)
        cnt += blk.shape[0]
    x_mu = (s_acc / cnt).astype(np.float32)
    x_sd = (np.sqrt(np.maximum(ss_acc / cnt - (s_acc / cnt) ** 2, 0.0))
            .astype(np.float32) + 1e-6)

    def _gather(idx):
        x = np.ascontiguousarray(X_raw[idx]).astype(np.float32)
        x = (x - x_mu) / x_sd
        return x, G_all[idx]

    torch.manual_seed(seed)
    np.random.seed(seed)
    net = _build_two_stream_tcn().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_MAX)
    scaler = torch.amp.GradScaler("cuda", enabled=dev != "cpu")

    y_dev = torch.from_numpy(up).to(dev)
    w_dev = torch.from_numpy(w1).to(dev)

    def _logits(indices):
        out = []
        with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=dev != "cpu"):
            for s in range(0, indices.size, EV_CHUNK):
                ii = indices[s:s + EV_CHUNK]
                xb, gb = _gather(ii)
                xb = torch.from_numpy(xb).to(dev, non_blocking=True)
                gb = torch.from_numpy(gb).to(dev, non_blocking=True)
                out.append(net(xb, gb).float().cpu())
                del xb, gb
        return torch.cat(out)

    best_ric, pat, best_state = -1e9, 0, None
    for ep in range(EP_CAP):
        net.train()
        perm = np.random.permutation(fit_idx)
        for s in range(0, perm.size, TR_BATCH):
            ii = perm[s:s + TR_BATCH]
            xb, gb = _gather(ii)
            xb = torch.from_numpy(xb).to(dev, non_blocking=True)
            gb = torch.from_numpy(gb).to(dev, non_blocking=True)
            jt = torch.as_tensor(ii, device=dev, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=dev != "cpu"):
                lo = net(xb, gb)
                bce = Fnn.binary_cross_entropy_with_logits(
                    lo, y_dev[jt], reduction="none")
                loss = (bce * w_dev[jt]).sum() / (w_dev[jt].sum() + 1e-9)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            del xb, gb
        if ep < T_MAX:
            sch.step()
        net.eval()
        vp = _logits(val_idx)
        y_val = up[val_idx].astype(int)
        v_ric = C.auc(y_val, torch.sigmoid(vp).numpy()) - 0.5
        if v_ric > best_ric + MIN_DELTA:
            best_ric, pat = v_ric, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    p = torch.sigmoid(_logits(te_idx)).numpy()
    y_te = up[te_idx].astype(int)
    ric = float(C.auc(y_te, p) - 0.5)
    plac = float(C.placebo_auc(y_te, p) - 0.5)
    block = int(C.block_size(H))
    se = C.block_bootstrap_auc_se(y_te, p, block)
    import numpy as _np
    se_f = None if not _np.isfinite(se) else round(float(se), 6)

    return {"seed": int(seed), "ric": round(ric, 6),
            "placebo_ric": round(plac, 6), "boot_se": se_f,
            "n_fit": int(fit_m.sum()), "n_val": int(val_m.sum()),
            "n_te": int(s_te.sum()), "block": block,
            "gpu_s": round(time.time() - t0, 2),
            "best_val_ric": round(float(best_ric), 6),
            "ep_used": int(ep + 1)}


@app.function(image=IMG, cpu=1.0, memory=4096, timeout=600,
              volumes={MNT: VOL})
def save_result(payload: dict):
    """Persist the rev48 collected result to the Volume for later
    rev49 result-append."""
    VOL.reload()
    os.makedirs(f"{MNT}/tier2", exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    VOL.commit()
    return {"saved_to": RESULT_PATH,
            "size_bytes": os.path.getsize(RESULT_PATH)}


@app.local_entrypoint()
def main(stage: str = "all"):
    """Stage: build | probe | all."""
    import statistics as st
    t_main = time.time()
    if stage in ("build", "all"):
        print(f"[stage={stage}] launching build_raw_l2_ltc on Modal CPU...")
        r_build = build_raw_l2_ltc.remote()
        print(f"[build] result:\n{json.dumps(r_build, indent=2, default=str)}")

    if stage in ("probe", "all"):
        print(f"[stage={stage}] launching probe across seeds={list(SEEDS)}...")
        results = list(probe_lobw128.map(SEEDS))
        rics = [r["ric"] for r in results]
        mean = st.mean(rics)
        sd = st.pstdev(rics) if len(rics) > 1 else 0.0
        gpu_s = sum(r["gpu_s"] for r in results)
        spend = round(gpu_s * L4_USD_PER_S, 2)
        ANCHOR_MEAN, ANCHOR_SD = 0.0240, 0.0069
        delta = mean - ANCHOR_MEAN
        within_anchor_band = abs(delta) <= (ANCHOR_SD + sd)
        payload = {
            "rev": 48,
            "cell": f"{SYM}-H{H}",
            "L": L, "D": D, "W_lob": W_LOB, "F_lob": F_LOB,
            "F_glob": F_GLOB, "seeds": list(SEEDS),
            "ric_mean": round(mean, 6),
            "ric_seed_sd": round(sd, 6),
            "ric_per_seed": sorted([round(x, 6) for x in rics]),
            "anchor_rev47_L512": {"ric_mean": ANCHOR_MEAN,
                                   "seed_sd": ANCHOR_SD},
            "delta_vs_anchor": round(delta, 6),
            "within_anchor_band": within_anchor_band,
            "placebo_ric_s0": results[0]["placebo_ric"],
            "boot_se_s0": results[0]["boot_se"],
            "n_fit": results[0]["n_fit"],
            "n_val": results[0]["n_val"],
            "n_te": results[0]["n_te"],
            "block": results[0]["block"],
            "per_seed_full": results,
            "gpu_s_total": gpu_s,
            "spend_usd_est": spend,
            "wall_s": round(time.time() - t_main, 1),
        }
        print("\n=== rev48 PROBE RESULT ===")
        print(json.dumps(payload, indent=2, default=str))
        save_result.remote(payload)
    print(f"\ntotal wall: {time.time() - t_main:.1f}s")
