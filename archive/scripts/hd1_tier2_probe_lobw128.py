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

# rev50 diagnostic (lcurve): no early-stop, full per-epoch logging
EP_DIAG = 24
T_MAX_DIAG = 24

# rev52 dense (STRIDE=1) artefacts + 3x3 reg sweep grid
# (use module-level PACK_DIR/SYM/L which are defined below)
REGSWEEP_GRID = [
    (0.5, 1e-3),  # rev48 anchor reg cell (on dense pack -> n_fit x ~4)
    (0.5, 1e-4),
    (0.5, 1e-2),
    (0.3, 1e-3),
    (0.3, 1e-4),
    (0.3, 1e-2),
    (0.1, 1e-3),
    (0.1, 1e-4),
    (0.1, 1e-2),
]
F_T0_LABEL = 0.0013

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
GLOB_PACK = f"{PACK_DIR}/{SYM}_globals_lasttick.npy"
RESULT_PATH = f"{MNT}/tier2/rev48_probe_lobw128_result.json"
# rev52 dense (STRIDE=1) artefacts
META_S1 = f"{PACK_DIR}/{SYM}_meta_stride1.npz"
RAW_PACK_S1 = f"{PACK_DIR}/{SYM}_raw_L{L}_stride1.npy"
GLOB_PACK_S1 = f"{PACK_DIR}/{SYM}_globals_lasttick_stride1.npy"
DONE_S1 = f"{PACK_DIR}/{SYM}_raw_L{L}_stride1.DONE"
# rev54 C1: engineered (46-ch) dense pack at STRIDE=1 -- rev45/47 anchor arch
X_ENG_S1 = f"{PACK_DIR}/{SYM}_X_L{L}_eng_stride1.npy"
DONE_ENG_S1 = f"{PACK_DIR}/{SYM}_X_L{L}_eng_stride1.DONE"
F_ENG = 46  # rev45 tick_features output dim
W_ANCHOR = 16  # rev45/47 anchor
# rev56 L-sweep: one raw pack at L_MAX (f16), slice shorter L at probe time
L_MAX = 2048
RAW_PACK_S1_LMAX = f"{PACK_DIR}/{SYM}_raw_L{L_MAX}_stride1_f16.npy"
DONE_S1_LMAX = f"{PACK_DIR}/{SYM}_raw_L{L_MAX}_stride1_f16.DONE"
# rev45 RF-forced D map (RF = 1 + 4*(2^D - 1) >= L), extended to 2048
D_FOR_L_EXT = {512: 8, 1024: 9, 1536: 9, 2048: 10}


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
@app.function(image=IMG, cpu=64.0, memory=131072, timeout=14400,
              volumes={MNT: VOL},
              secrets=[modal.Secret.from_name("hd1-gcp")])
def build_raw_l2_ltc():
    """Stream LTC raw-book parquets day-by-day from GCS at SAME t0
    decision points the rev45 pack uses. Per-day windowing (NO
    cross-day tail buffer): start-of-day windows are zero-padded on
    the left, matching the rev45 Rust-build contract exactly so the
    LOB-stream input is bit-comparable to a hypothetical rev45-style
    raw-L2 pack. Two outputs:

      RAW_PACK  (n, L, 80) f32: per-tick raw 20-lvl LOB, normalized
                  (prices as (p-mid)/mid; sizes as sign*log1p(|s|))
                  -- the LOB-stream input.

      GLOB_PACK (n, 6) f32: per-decision-point last-tick engineered
                  globals (cols 40-45 of the rev45 tick_features:
                  log-return, spread, L5 depth-imb, L20 depth-imb,
                  Cont-OFI, microprice-mid offset). Computed by
                  calling the SAME hd1_seq_core.tick_features on the
                  same per-day raw L2 -> bit-identical to rev45's
                  globals at the corresponding decision points.

    Skip-if-exists: requires BOTH packs at the expected shape AND a
    sentinel /cache/packed_l1536/<sym>_raw_L512.DONE marker (so a
    partially-written memmap from a killed run is NOT mistaken for a
    completed build)."""
    import tempfile
    import numpy as np
    import pyarrow.parquet as pq

    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    VOL.reload()
    os.makedirs(PACK_DIR, exist_ok=True)
    os.makedirs(f"{MNT}/tier2", exist_ok=True)

    meta = np.load(f"{PACK_DIR}/{SYM}_meta.npz")
    n_exp = int(meta["n"])
    t0 = meta["t0"].astype(np.int64)   # ns since epoch (UTC)
    assert t0.size == n_exp

    done_marker = f"{PACK_DIR}/{SYM}_raw_L{L}.DONE"
    if (os.path.exists(RAW_PACK) and os.path.exists(GLOB_PACK)
            and os.path.exists(done_marker)):
        try:
            a1 = np.load(RAW_PACK, mmap_mode="r")
            a2 = np.load(GLOB_PACK, mmap_mode="r")
            if (a1.shape == (n_exp, L, F_LOB)
                    and a2.shape == (n_exp, F_GLOB)):
                return {"status": "skip_existing",
                        "raw_shape": list(a1.shape),
                        "glob_shape": list(a2.shape)}
        except Exception:
            pass

    NS_PER_DAY = 86400 * 1_000_000_000
    EPOCH = dt.date(1970, 1, 1)

    def d2s(d):
        return (EPOCH + dt.timedelta(days=int(d))).isoformat()

    t0_day = (t0 // NS_PER_DAY).astype(np.int64)
    unique_days = sorted(int(d) for d in np.unique(t0_day))

    bk = _gcs_bucket()

    # pre-allocate output memmaps
    out_raw = np.lib.format.open_memmap(
        RAW_PACK, mode="w+", dtype=np.float32, shape=(n_exp, L, F_LOB))
    out_glob = np.lib.format.open_memmap(
        GLOB_PACK, mode="w+", dtype=np.float32, shape=(n_exp, F_GLOB))

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

    total_filled = 0
    days_with_data = 0
    days_missing = 0
    t_start = time.time()
    log_lines = []

    for d_int in unique_days:
        day_str = d2s(d_int)
        day_start_ns = d_int * NS_PER_DAY
        day_end_ns = day_start_ns + NS_PER_DAY
        prefix = (f"raw/book/exchange=BINANCE_FUTURES/"
                  f"symbol={SYM}/dt={day_str}/")
        blobs = sorted([b for b in bk.client.list_blobs(bk, prefix=prefix)
                        if b.name.endswith(".parquet")],
                       key=lambda b: b.name)
        if not blobs:
            days_missing += 1
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
                ts_parts.append(
                    t["timestamp"].to_numpy().astype(np.int64))
                X_parts.append(np.column_stack(
                    [t[c].to_numpy().astype(np.float64)
                     for c in chan_cols]))
            ts_day = np.concatenate(ts_parts)
            X_day_raw = np.concatenate(X_parts, axis=0)   # f64
            order = np.argsort(ts_day, kind="stable")
            ts_day = ts_day[order]
            X_day_raw = X_day_raw[order]

            # split into 4 sub-arrays for tick_features bit-match
            bid_p = X_day_raw[:, 0:20]
            bid_s = X_day_raw[:, 20:40]
            ask_p = X_day_raw[:, 40:60]
            ask_s = X_day_raw[:, 60:80]

            # bit-identical engineered globals (same fn rev45 uses)
            feat = C.tick_features(bid_p, bid_s, ask_p, ask_s)   # (n_day, 46)
            globals_day = feat[:, 40:46].astype(np.float32)      # (n_day, 6)

            # LOB-stream normalization, written into a fresh f32 array
            mid = 0.5 * (bid_p[:, 0] + ask_p[:, 0])
            mid_safe = np.where(mid > 0, mid, 1.0)
            lob = np.empty((ts_day.size, F_LOB), np.float32)
            # prices: (p - mid) / mid
            lob[:, 0:20] = ((bid_p - mid[:, None]) / mid_safe[:, None]
                            ).astype(np.float32)
            lob[:, 40:60] = ((ask_p - mid[:, None]) / mid_safe[:, None]
                             ).astype(np.float32)
            # sizes: sign*log1p(|s|)
            lob[:, 20:40] = (np.sign(bid_s) * np.log1p(np.abs(bid_s))
                             ).astype(np.float32)
            lob[:, 60:80] = (np.sign(ask_s) * np.log1p(np.abs(ask_s))
                             ).astype(np.float32)

            in_day = (t0 >= day_start_ns) & (t0 < day_end_ns)
            dp_idx = np.where(in_day)[0]
            for dp_i in dp_idx:
                t = int(t0[dp_i])
                j = int(np.searchsorted(ts_day, t, side="right")) - 1
                if j < 0:
                    out_raw[dp_i] = 0.0
                    out_glob[dp_i] = 0.0
                    continue
                lo = j - L + 1
                if lo >= 0:
                    out_raw[dp_i] = lob[lo:j + 1]
                else:
                    pad = -lo
                    win = np.zeros((L, F_LOB), np.float32)
                    win[pad:] = lob[:j + 1]
                    out_raw[dp_i] = win
                out_glob[dp_i] = globals_day[j]
            total_filled += dp_idx.size
            days_with_data += 1

        log_lines.append(
            f"{day_str}: ticks={ts_day.size} "
            f"dp_in_day={int(in_day.sum())} "
            f"total_filled={total_filled}")
        if len(log_lines) % 10 == 0:
            print(" ".join(log_lines[-2:]))
            sys.stdout.flush()
            VOL.commit()

    out_raw.flush(); del out_raw
    out_glob.flush(); del out_glob
    # DONE marker last (so partial runs don't trip skip-if-exists)
    with open(done_marker, "w") as f:
        f.write(f"n={n_exp} filled={total_filled} "
                f"days_with_data={days_with_data} "
                f"days_missing={days_missing}\n")
    VOL.commit()
    elapsed = round(time.time() - t_start, 1)
    return {"status": "built", "n_expected": n_exp,
            "total_filled": total_filled,
            "raw_shape": [n_exp, L, F_LOB],
            "glob_shape": [n_exp, F_GLOB],
            "raw_path": RAW_PACK, "glob_path": GLOB_PACK,
            "elapsed_s": elapsed,
            "days_with_data": days_with_data,
            "days_missing": days_missing,
            "tail_log": log_lines[-5:]}


# ---- MODEL: 2-stream TCN (raw LOB W=128 mean-pool + 6-globals Linear) -----
def _build_two_stream_tcn(dropout=None, d_blocks=None):
    """Build the 2-stream TCN. dropout=None -> uses module-level DROPOUT
    (rev48/rev50 default); pass a float to override (rev52 sweep).
    d_blocks=None -> uses module-level D (rev48/50/52); pass an int to
    override (rev56 L-sweep RF-matched D)."""
    if dropout is None:
        dropout = DROPOUT
    if d_blocks is None:
        d_blocks = D
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
                Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(co, co, 3, padding=pad, dilation=dil),
                Chomp(pad), nn.ReLU(), nn.Dropout(dropout))
            self.down = nn.Conv1d(ci, co, 1) if ci != co else None
            self.relu = nn.ReLU()

        def forward(self, x):
            r = x if self.down is None else self.down(x)
            return self.relu(self.net(x) + r)

    class TwoStream(nn.Module):
        def __init__(self):
            super().__init__()
            layers, ci = [], F_LOB
            for b in range(d_blocks):
                layers.append(Block(ci, W_LOB, 2 ** b))
                ci = W_LOB
            self.lob = nn.Sequential(*layers)
            self.glob = nn.Sequential(
                nn.Linear(F_GLOB, 32),
                nn.GELU(),
                nn.Dropout(dropout))
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
@app.function(image=IMG, gpu="A100-80GB", timeout=7200, memory=65536,
              volumes={MNT: VOL})
def probe_lobw128(seed: int):
    """One seed of rev48: LTC-H300-L=512 D=8 W_lob=128 2-stream TCN
    (raw 20-lvl LOB mean-pool + 6 last-tick engineered globals)
    + BCE-with-r1 + rev45-locked schedule. Inputs read entirely from
    the rev48 build's two packs (RAW_PACK + GLOB_PACK) -- NO
    dependency on the engineered 46-channel X pack."""
    import numpy as np
    import torch
    import torch.nn.functional as Fnn

    VOL.reload()
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X_raw = np.load(RAW_PACK, mmap_mode="r")        # (n, L, 80)
    G_all_raw = np.load(GLOB_PACK, mmap_mode="r")   # (n, 6)
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

    # GLOBALS stream: standardize on fit rows
    G_all = np.ascontiguousarray(G_all_raw).astype(np.float32)
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


# ---- DIAGNOSTIC: rev50 learning-curve (per-epoch val/test ric/logloss) ----
@app.function(image=IMG, gpu="A100-80GB", timeout=3600, memory=65536,
              volumes={MNT: VOL})
def lcurve_lobw128(seed: int = 0):
    """rev50 DIAGNOSTIC: same 2-stream W=128 raw-LOB + 6-globals
    architecture as rev48, ONE seed, NO early-stop, EP_DIAG=24 epochs,
    per-epoch logging of train_loss / val_logloss / val_ric / test_ric
    / lr. Cosine T_MAX_DIAG=24 matched to actual run length (NOT the
    rev48-probe T_MAX=8 which collapses LR before the diagnostic
    window closes).

    Purpose: distinguish (i) capacity-overfit-on-val (val_ric peaks
    very early, test_ric peaks even earlier, both decay -- rev30 motif
    on a bigger model) from (ii) val/test distribution shift (val_ric
    rises smoothly, test_ric stays low throughout, no overfit-decay).

    Pre-reg: research/hypotheses.jsonl HD1 rev50 (this rev). Outcome
    only INFORMS interpretation of rev49; rev49 result unchanged."""
    import numpy as np
    import torch
    import torch.nn.functional as Fnn

    VOL.reload()
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X_raw = np.load(RAW_PACK, mmap_mode="r")
    G_all_raw = np.load(GLOB_PACK, mmap_mode="r")
    meta = np.load(f"{PACK_DIR}/{SYM}_meta.npz")
    n = int(meta["n"])
    y0 = meta[f"y0_{H}"]
    rH = meta[f"rH_{H}"].astype(np.float64)

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

    G_all = np.ascontiguousarray(G_all_raw).astype(np.float32)
    g_mu = G_all[fit_m].mean(axis=0).astype(np.float32)
    g_sd = G_all[fit_m].std(axis=0).astype(np.float32) + 1e-6
    G_all = (G_all - g_mu) / g_sd

    # LOB z-score on fit rows (streamed)
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
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_MAX_DIAG)
    scaler = torch.amp.GradScaler("cuda", enabled=dev != "cpu")

    y_dev = torch.from_numpy(up).to(dev)
    w_dev = torch.from_numpy(w1).to(dev)

    def _eval(indices):
        """Return (logits_np, weighted_logloss)."""
        out = []
        wll_num, wll_den = 0.0, 0.0
        with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=dev != "cpu"):
            for s_ in range(0, indices.size, EV_CHUNK):
                ii = indices[s_:s_ + EV_CHUNK]
                xb, gb = _gather(ii)
                xb = torch.from_numpy(xb).to(dev, non_blocking=True)
                gb = torch.from_numpy(gb).to(dev, non_blocking=True)
                lo = net(xb, gb).float()
                jt = torch.as_tensor(ii, device=dev, dtype=torch.long)
                bce = Fnn.binary_cross_entropy_with_logits(
                    lo, y_dev[jt], reduction="none")
                wll_num += float((bce * w_dev[jt]).sum().item())
                wll_den += float(w_dev[jt].sum().item())
                out.append(lo.cpu())
                del xb, gb
        wll = wll_num / max(wll_den, 1e-9)
        return torch.cat(out).numpy(), wll

    y_val_np = up[val_idx].astype(int)
    y_te_np = up[te_idx].astype(int)

    history = []
    for ep in range(EP_DIAG):
        net.train()
        perm = np.random.permutation(fit_idx)
        tr_num, tr_den = 0.0, 0.0
        for s_ in range(0, perm.size, TR_BATCH):
            ii = perm[s_:s_ + TR_BATCH]
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
            tr_num += float((bce.detach() * w_dev[jt]).sum().item())
            tr_den += float(w_dev[jt].sum().item())
            del xb, gb
        lr_now = float(opt.param_groups[0]["lr"])
        if ep < T_MAX_DIAG:
            sch.step()

        net.eval()
        v_lo, v_wll = _eval(val_idx)
        t_lo, t_wll = _eval(te_idx)
        v_p = 1.0 / (1.0 + np.exp(-v_lo))
        t_p = 1.0 / (1.0 + np.exp(-t_lo))
        v_auc = float(C.auc(y_val_np, v_p))
        t_auc = float(C.auc(y_te_np, t_p))
        rec = {
            "ep": ep + 1,
            "lr": round(lr_now, 7),
            "train_loss": round(tr_num / max(tr_den, 1e-9), 6),
            "val_logloss": round(v_wll, 6),
            "val_ric": round(v_auc - 0.5, 6),
            "val_auc": round(v_auc, 6),
            "test_logloss": round(t_wll, 6),
            "test_ric": round(t_auc - 0.5, 6),
            "test_auc": round(t_auc, 6),
        }
        history.append(rec)
        print(f"ep{ep+1:>2d}  lr={lr_now:.5f}  "
              f"trL={rec['train_loss']:.4f}  "
              f"vL={rec['val_logloss']:.4f}  vR={rec['val_ric']:+.4f}  "
              f"tL={rec['test_logloss']:.4f}  tR={rec['test_ric']:+.4f}")
        sys.stdout.flush()

    # post-hoc summaries at the three "interesting" epochs
    def _stats_at(ep_idx):
        net_state = "live"  # we don't restore; just describe the live state
        # We don't checkpoint per-epoch (would 24x VRAM); summary fields
        # are read from history. boot_se/placebo computed only at end.
        return history[ep_idx]

    last = history[-1]
    by_val = max(history, key=lambda r: r["val_ric"])
    by_test = max(history, key=lambda r: r["test_ric"])

    # end-of-run placebo + boot_se on TEST set (live final-epoch model)
    _, _ = _eval(te_idx)  # warm
    t_lo_final, _ = _eval(te_idx)
    t_p_final = 1.0 / (1.0 + np.exp(-t_lo_final))
    plac = float(C.placebo_auc(y_te_np, t_p_final) - 0.5)
    block = int(C.block_size(H))
    se = C.block_bootstrap_auc_se(y_te_np, t_p_final, block)
    se_f = None if not np.isfinite(se) else round(float(se), 6)

    return {
        "rev": 50,
        "seed": int(seed),
        "cfg": {"sym": SYM, "H": H, "L": L, "D": D, "W_lob": W_LOB,
                "F_lob": F_LOB, "F_glob": F_GLOB,
                "EP_DIAG": EP_DIAG, "T_MAX_DIAG": T_MAX_DIAG,
                "DROPOUT": DROPOUT, "WD": WD, "TR_BATCH": TR_BATCH,
                "n_params": int(n_params)},
        "n_fit": int(fit_m.sum()), "n_val": int(val_m.sum()),
        "n_te": int(s_te.sum()), "block": block,
        "history": history,
        "summary": {
            "last_ep": last,
            "by_best_val_ric": by_val,
            "by_best_test_ric": by_test,
        },
        "final_ep_diagnostics": {
            "placebo_ric": round(plac, 6),
            "boot_se": se_f,
        },
        "gpu_s": round(time.time() - t0, 2),
    }


# ---- BUILD-DENSE: rev52 STRIDE=1 dense pack ------------------------------
@app.function(image=IMG, cpu=64.0, memory=131072, timeout=14400,
              volumes={MNT: VOL},
              secrets=[modal.Secret.from_name("hd1-gcp")])
def build_dense_stride1():
    """rev52 dense build for LTC-USDT-PERP H=300 L=512: STRIDE=1
    over the rev45/ha5 'eligible-tick' set (features_v1/.../indices.npy
    per day), recomputing first-passage labels in Python via the
    FROZEN hd1_seq_core.labels_for_H (bit-identical to ha5_screen
    _first_passage). Outputs:

      META_S1      n, t0, y0_300, rH_300            (~175k dp expected)
      RAW_PACK_S1  (n, L, 80) f32                    (~27 GiB)
      GLOB_PACK_S1 (n, 6) f32                        (~4 MiB)
      DONE_S1      sentinel marker

    PARITY SANITY (must pass before writing dense pack): re-compute
    y0_300/rH_300 on the EXISTING STRIDE=4 t0 set (loaded from
    {SYM}_meta.npz) using the same Python path; assert y0 exact
    agreement >=99.5% and |rH_diff| < 1e-5 max. Mismatch -> abort,
    do NOT clobber the existing dense pack."""
    import tempfile
    import numpy as np
    import pyarrow.parquet as pq

    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    VOL.reload()
    os.makedirs(PACK_DIR, exist_ok=True)
    os.makedirs(f"{MNT}/tier2", exist_ok=True)

    # skip-if-exists with DONE marker
    if (os.path.exists(RAW_PACK_S1) and os.path.exists(GLOB_PACK_S1)
            and os.path.exists(META_S1) and os.path.exists(DONE_S1)):
        try:
            m = np.load(META_S1)
            n_exp = int(m["n"])
            a1 = np.load(RAW_PACK_S1, mmap_mode="r")
            a2 = np.load(GLOB_PACK_S1, mmap_mode="r")
            if a1.shape == (n_exp, L, F_LOB) and a2.shape == (n_exp, F_GLOB):
                return {"status": "skip_existing", "n": n_exp,
                        "raw_shape": list(a1.shape),
                        "glob_shape": list(a2.shape)}
        except Exception:
            pass

    NS_PER_DAY = 86400 * 1_000_000_000
    EPOCH = dt.date(1970, 1, 1)

    def d2s(d):
        return (EPOCH + dt.timedelta(days=int(d))).isoformat()

    bk = _gcs_bucket()

    # day list comes from the existing meta's t0 (so we cover the SAME
    # 360-day window rev45 used)
    ref_meta = np.load(f"{PACK_DIR}/{SYM}_meta.npz")
    ref_t0 = ref_meta["t0"].astype(np.int64)
    ref_y0 = ref_meta[f"y0_{H}"].astype(np.int8)
    ref_rH = ref_meta[f"rH_{H}"].astype(np.float64)
    ref_t0_day = (ref_t0 // NS_PER_DAY).astype(np.int64)
    unique_days = sorted(int(d) for d in np.unique(ref_t0_day))

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

    # Phase A: PARITY check + size estimation (per day, NO pack writes).
    # We hold per-day intermediates in a list to reuse in Phase B.
    # To bound memory, we restream parquets in Phase B if needed.
    print("[build_dense] phase A: PARITY against rev45 meta + size sweep...")
    sys.stdout.flush()
    per_day = []                 # list of dicts {d, t0_dense, j_dense, n_dp}
    parity_y_total, parity_y_match = 0, 0
    parity_rh_maxdiff = 0.0
    days_missing = 0

    for d_int in unique_days:
        day_str = d2s(d_int)
        day_start_ns = d_int * NS_PER_DAY
        day_end_ns = day_start_ns + NS_PER_DAY
        prefix = (f"raw/book/exchange=BINANCE_FUTURES/"
                  f"symbol={SYM}/dt={day_str}/")
        blobs = sorted([b for b in bk.client.list_blobs(bk, prefix=prefix)
                        if b.name.endswith(".parquet")],
                       key=lambda b: b.name)
        if not blobs:
            days_missing += 1
            per_day.append({"d": d_int, "n_dp": 0})
            continue

        # read eligible indices.npy
        idx_blob = bk.blob(
            f"features_v1/symbol={SYM}/dt={day_str}/indices.npy")
        if not idx_blob.exists():
            per_day.append({"d": d_int, "n_dp": 0, "no_idx": True})
            continue
        with tempfile.TemporaryDirectory() as td:
            ip = f"{td}/indices.npy"
            idx_blob.download_to_filename(ip)
            day_idx_all = np.load(ip).astype(np.int64)

            # read book parquets
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
                    [t[c].to_numpy().astype(np.float64) for c in chan_cols]))
            ts_day = np.concatenate(ts_parts)
            X_day = np.concatenate(X_parts, axis=0)
            order = np.argsort(ts_day, kind="stable")
            ts_day = ts_day[order]
            X_day = X_day[order]
            n_tk = ts_day.size

            bid_0 = X_day[:, 0]
            ask_0 = X_day[:, 40]
            mid_day = 0.5 * (bid_0 + ask_0)

            # parity: re-compute y0_300/rH_300 on this day's PORTION
            # of the existing STRIDE=4 t0 set
            in_day = (ref_t0_day == d_int)
            if in_day.any():
                t0_ref_day = ref_t0[in_day]
                # for each ref t0, find tick index i s.t. ts_day[i] == t0
                # (existing meta's t0 should be exact tick timestamps)
                i_ref = np.searchsorted(ts_day, t0_ref_day)
                # safety: clamp + drop any not-found
                valid = ((i_ref < n_tk) &
                         (ts_day[np.clip(i_ref, 0, n_tk - 1)] == t0_ref_day))
                if valid.any():
                    i_ref = i_ref[valid]
                    t0_ref = t0_ref_day[valid]
                    ref_y_day = ref_y0[in_day][valid]
                    ref_rH_day = ref_rH[in_day][valid]
                    y_py, rH_py, _, _ = C.labels_for_H(
                        ts_day, mid_day, i_ref, t0_ref, H)
                    parity_y_total += y_py.size
                    parity_y_match += int(np.sum(y_py == ref_y_day))
                    finite = np.isfinite(rH_py) & np.isfinite(ref_rH_day)
                    if finite.any():
                        d_rh = np.abs(rH_py[finite] - ref_rH_day[finite])
                        parity_rh_maxdiff = max(parity_rh_maxdiff,
                                                float(d_rh.max()))

            # STRIDE=1: ALL eligible indices, no decimation
            ok = (day_idx_all > 0) & (day_idx_all < n_tk - 1)
            j_dense = day_idx_all[ok]
            t0_dense = ts_day[j_dense]
            per_day.append({"d": d_int, "n_dp": int(j_dense.size),
                            "t0": t0_dense, "j": j_dense})
        if len(per_day) % 20 == 0:
            print(f"[build_dense] phase A: processed {len(per_day)} days; "
                  f"parity_y_match={parity_y_match}/{parity_y_total} "
                  f"rh_maxdiff={parity_rh_maxdiff:.2e}")
            sys.stdout.flush()

    parity_y_frac = (parity_y_match / parity_y_total) if parity_y_total else 0.0
    print(f"[build_dense] PARITY: y0_match={parity_y_match}/{parity_y_total}"
          f" ({parity_y_frac:.4f}); rh_maxdiff={parity_rh_maxdiff:.2e};"
          f" days_missing={days_missing}")
    sys.stdout.flush()
    if parity_y_frac < 0.995:
        raise RuntimeError(
            f"PARITY FAIL: y0 agreement {parity_y_frac:.4f} < 0.995; "
            "Python first_passage path diverges from rev45 meta. "
            "Refusing to write dense pack.")
    if parity_rh_maxdiff > 1e-5:
        raise RuntimeError(
            f"PARITY FAIL: rH max abs diff {parity_rh_maxdiff:.2e} > 1e-5; "
            "log-return computation mismatch. Refusing to write dense pack.")

    # Phase B: write packs + labels
    n_total = int(sum(d.get("n_dp", 0) for d in per_day))
    print(f"[build_dense] phase B: write dense pack n_dp={n_total}")
    sys.stdout.flush()
    if n_total == 0:
        raise RuntimeError("no dense decision points after eligibility filter")

    out_raw = np.lib.format.open_memmap(
        RAW_PACK_S1, mode="w+", dtype=np.float32, shape=(n_total, L, F_LOB))
    out_glob = np.lib.format.open_memmap(
        GLOB_PACK_S1, mode="w+", dtype=np.float32, shape=(n_total, F_GLOB))
    t0_out = np.empty(n_total, np.int64)
    y0_out = np.empty(n_total, np.int8)
    rH_out = np.empty(n_total, np.float32)

    # Restream parquets day-by-day for Phase B (we threw the X arrays away
    # in Phase A to bound RAM; download cost is small + bound by GCS).
    write_off = 0
    for entry in per_day:
        if not entry.get("n_dp"):
            continue
        d_int = entry["d"]
        day_str = d2s(d_int)
        j_dense = entry["j"]
        t0_dense = entry["t0"]
        prefix = (f"raw/book/exchange=BINANCE_FUTURES/"
                  f"symbol={SYM}/dt={day_str}/")
        blobs = sorted([b for b in bk.client.list_blobs(bk, prefix=prefix)
                        if b.name.endswith(".parquet")],
                       key=lambda b: b.name)
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
                    [t[c].to_numpy().astype(np.float64) for c in chan_cols]))
            ts_day = np.concatenate(ts_parts)
            X_day = np.concatenate(X_parts, axis=0)
            order = np.argsort(ts_day, kind="stable")
            ts_day = ts_day[order]
            X_day = X_day[order]

            bid_p = X_day[:, 0:20]
            bid_s = X_day[:, 20:40]
            ask_p = X_day[:, 40:60]
            ask_s = X_day[:, 60:80]
            mid_day = 0.5 * (bid_p[:, 0] + ask_p[:, 0])
            mid_safe = np.where(mid_day > 0, mid_day, 1.0)
            feat = C.tick_features(bid_p, bid_s, ask_p, ask_s)
            globals_day = feat[:, 40:46].astype(np.float32)

            lob = np.empty((ts_day.size, F_LOB), np.float32)
            lob[:, 0:20] = ((bid_p - mid_day[:, None]) /
                            mid_safe[:, None]).astype(np.float32)
            lob[:, 40:60] = ((ask_p - mid_day[:, None]) /
                             mid_safe[:, None]).astype(np.float32)
            lob[:, 20:40] = (np.sign(bid_s) *
                             np.log1p(np.abs(bid_s))).astype(np.float32)
            lob[:, 60:80] = (np.sign(ask_s) *
                             np.log1p(np.abs(ask_s))).astype(np.float32)

            # labels via FROZEN first_passage
            y_day, rH_day, _, _ = C.labels_for_H(
                ts_day, mid_day, j_dense, t0_dense, H)

            n_dp_day = j_dense.size
            for k in range(n_dp_day):
                j = int(j_dense[k])
                lo = j - L + 1
                if lo >= 0:
                    out_raw[write_off + k] = lob[lo:j + 1]
                else:
                    pad = -lo
                    win = np.zeros((L, F_LOB), np.float32)
                    win[pad:] = lob[:j + 1]
                    out_raw[write_off + k] = win
                out_glob[write_off + k] = globals_day[j]
            t0_out[write_off:write_off + n_dp_day] = t0_dense
            y0_out[write_off:write_off + n_dp_day] = y_day
            rH_out[write_off:write_off + n_dp_day] = rH_day.astype(np.float32)
            write_off += n_dp_day
        if write_off % 5000 < 600:
            print(f"[build_dense] phase B: written {write_off}/{n_total}")
            sys.stdout.flush()
            VOL.commit()

    assert write_off == n_total
    n_tr = int(n_total * C.TRAIN_FRAC)
    np.savez(META_S1, n=np.int64(n_total), n_tr=np.int64(n_tr),
             t0=t0_out, y0_300=y0_out, rH_300=rH_out)

    out_raw.flush(); del out_raw
    out_glob.flush(); del out_glob
    with open(DONE_S1, "w") as f:
        f.write(f"n={n_total} parity_y_frac={parity_y_frac:.6f} "
                f"rh_maxdiff={parity_rh_maxdiff:.2e}\n")
    VOL.commit()
    return {"status": "built", "n": n_total, "n_tr": n_tr,
            "raw_shape": [n_total, L, F_LOB],
            "glob_shape": [n_total, F_GLOB],
            "parity_y_frac": round(parity_y_frac, 6),
            "parity_rh_maxdiff": parity_rh_maxdiff,
            "days_missing": days_missing,
            "meta": META_S1, "raw": RAW_PACK_S1, "glob": GLOB_PACK_S1}


# ---- REG-SWEEP: parametrized lcurve on the dense pack --------------------
@app.function(image=IMG, gpu="A100-80GB", timeout=3600, memory=65536,
              volumes={MNT: VOL})
def regsweep_lobw128_cell(cell: dict):
    """rev52 reg sweep cell. Runs ONE (dropout, wd) point on the
    STRIDE=1 dense pack, 1 seed, EP_DIAG=24 epochs no early-stop, full
    per-epoch logging (same protocol as rev50 lcurve_lobw128). Loads
    META_S1 / RAW_PACK_S1 / GLOB_PACK_S1 instead of the rev48 packs.

    cell = {'dropout': float, 'wd': float, 'seed': int (default 0)}"""
    import numpy as np
    import torch
    import torch.nn.functional as Fnn

    VOL.reload()
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dropout = float(cell["dropout"])
    wd = float(cell["wd"])
    seed = int(cell.get("seed", 0))
    cell_tag = f"drop{dropout:g}_wd{wd:g}_seed{seed}"

    X_raw = np.load(RAW_PACK_S1, mmap_mode="r")
    G_all_raw = np.load(GLOB_PACK_S1, mmap_mode="r")
    meta = np.load(META_S1)
    n = int(meta["n"])
    y0 = meta["y0_300"]
    rH = meta["rH_300"].astype(np.float64)

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

    G_all = np.ascontiguousarray(G_all_raw).astype(np.float32)
    g_mu = G_all[fit_m].mean(axis=0).astype(np.float32)
    g_sd = G_all[fit_m].std(axis=0).astype(np.float32) + 1e-6
    G_all = (G_all - g_mu) / g_sd

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
    net = _build_two_stream_tcn(dropout=dropout).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_MAX_DIAG)
    scaler = torch.amp.GradScaler("cuda", enabled=dev != "cpu")

    y_dev = torch.from_numpy(up).to(dev)
    w_dev = torch.from_numpy(w1).to(dev)

    def _eval(indices):
        out = []
        wll_num, wll_den = 0.0, 0.0
        with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=dev != "cpu"):
            for s_ in range(0, indices.size, EV_CHUNK):
                ii = indices[s_:s_ + EV_CHUNK]
                xb, gb = _gather(ii)
                xb = torch.from_numpy(xb).to(dev, non_blocking=True)
                gb = torch.from_numpy(gb).to(dev, non_blocking=True)
                lo = net(xb, gb).float()
                jt = torch.as_tensor(ii, device=dev, dtype=torch.long)
                bce = Fnn.binary_cross_entropy_with_logits(
                    lo, y_dev[jt], reduction="none")
                wll_num += float((bce * w_dev[jt]).sum().item())
                wll_den += float(w_dev[jt].sum().item())
                out.append(lo.cpu())
                del xb, gb
        return torch.cat(out).numpy(), wll_num / max(wll_den, 1e-9)

    y_val_np = up[val_idx].astype(int)
    y_te_np = up[te_idx].astype(int)

    history = []
    for ep in range(EP_DIAG):
        net.train()
        perm = np.random.permutation(fit_idx)
        tr_num, tr_den = 0.0, 0.0
        for s_ in range(0, perm.size, TR_BATCH):
            ii = perm[s_:s_ + TR_BATCH]
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
            tr_num += float((bce.detach() * w_dev[jt]).sum().item())
            tr_den += float(w_dev[jt].sum().item())
            del xb, gb
        lr_now = float(opt.param_groups[0]["lr"])
        if ep < T_MAX_DIAG:
            sch.step()
        net.eval()
        v_lo, v_wll = _eval(val_idx)
        t_lo, t_wll = _eval(te_idx)
        v_p = 1.0 / (1.0 + np.exp(-v_lo))
        t_p = 1.0 / (1.0 + np.exp(-t_lo))
        v_auc = float(C.auc(y_val_np, v_p))
        t_auc = float(C.auc(y_te_np, t_p))
        rec = {"ep": ep + 1, "lr": round(lr_now, 7),
               "train_loss": round(tr_num / max(tr_den, 1e-9), 6),
               "val_logloss": round(v_wll, 6),
               "val_ric": round(v_auc - 0.5, 6),
               "test_logloss": round(t_wll, 6),
               "test_ric": round(t_auc - 0.5, 6)}
        history.append(rec)
        print(f"[{cell_tag}] ep{ep+1:>2d} lr={lr_now:.5f} "
              f"trL={rec['train_loss']:.4f} "
              f"vR={rec['val_ric']:+.4f} tR={rec['test_ric']:+.4f}")
        sys.stdout.flush()

    last = history[-1]
    by_val = max(history, key=lambda r: r["val_ric"])
    by_test = max(history, key=lambda r: r["test_ric"])
    t_lo_final, _ = _eval(te_idx)
    t_p_final = 1.0 / (1.0 + np.exp(-t_lo_final))
    plac = float(C.placebo_auc(y_te_np, t_p_final) - 0.5)
    block = int(C.block_size(H))
    se = C.block_bootstrap_auc_se(y_te_np, t_p_final, block)
    se_f = None if not np.isfinite(se) else round(float(se), 6)

    return {"rev": 52, "cell_tag": cell_tag,
            "dropout": dropout, "wd": wd, "seed": seed,
            "n_params": int(n_params),
            "n_fit": int(fit_m.sum()), "n_val": int(val_m.sum()),
            "n_te": int(s_te.sum()), "block": block,
            "history": history,
            "summary": {"last_ep": last, "by_best_val_ric": by_val,
                        "by_best_test_ric": by_test},
            "final_ep_diagnostics": {"placebo_ric": round(plac, 6),
                                     "boot_se": se_f},
            "gpu_s": round(time.time() - t0, 2)}


# ---- BUILD: rev54 C1 engineered dense pack at STRIDE=1 ------------------
@app.function(image=IMG, cpu=64.0, memory=131072, timeout=14400,
              volumes={MNT: VOL},
              secrets=[modal.Secret.from_name("hd1-gcp")])
def build_dense_engineered_stride1():
    """rev54 C1: build a 46-channel engineered dense pack at STRIDE=1
    for LTC-USDT-PERP H=300 L=512 -- the rev45/47 anchor input
    representation, on the SAME decision-point set as the rev52 dense
    raw pack. Reads the existing meta_stride1 to get t0/n_dp (no new
    label computation; just gather windows). For each day: read book
    parquets, run hd1_seq_core.tick_features (the same fn rev45 Rust
    binary's reference Python uses; bit-identical), gather L=512
    windows at each in-day dp (zero-pad at day start, matching the
    rev45 per-day-shard contract). Output:

      X_ENG_S1     (n, L, 46) f32                       (~16.4 GiB)
      DONE_ENG_S1  sentinel marker

    Skip-if-exists. Reuses META_S1 written by rev52 build_dense_stride1."""
    import tempfile
    import numpy as np
    import pyarrow.parquet as pq

    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    VOL.reload()
    if not os.path.exists(META_S1):
        raise RuntimeError(f"META_S1 missing ({META_S1}) -- run build_dense"
                           " (rev52 Part A) first")
    meta = np.load(META_S1)
    n_exp = int(meta["n"])
    t0 = meta["t0"].astype(np.int64)

    if (os.path.exists(X_ENG_S1) and os.path.exists(DONE_ENG_S1)):
        try:
            a = np.load(X_ENG_S1, mmap_mode="r")
            if a.shape == (n_exp, L, F_ENG):
                return {"status": "skip_existing",
                        "shape": list(a.shape), "path": X_ENG_S1}
        except Exception:
            pass

    NS_PER_DAY = 86400 * 1_000_000_000
    EPOCH = dt.date(1970, 1, 1)

    def d2s(d):
        return (EPOCH + dt.timedelta(days=int(d))).isoformat()

    t0_day = (t0 // NS_PER_DAY).astype(np.int64)
    unique_days = sorted(int(d) for d in np.unique(t0_day))

    bk = _gcs_bucket()
    out_eng = np.lib.format.open_memmap(
        X_ENG_S1, mode="w+", dtype=np.float32, shape=(n_exp, L, F_ENG))

    chan_cols = []
    for k in range(20):
        chan_cols.append(f"bid_{k}_price")
    for k in range(20):
        chan_cols.append(f"bid_{k}_size")
    for k in range(20):
        chan_cols.append(f"ask_{k}_price")
    for k in range(20):
        chan_cols.append(f"ask_{k}_size")

    write_total = 0
    days_missing = 0
    t_start = time.time()
    log_lines = []

    for d_int in unique_days:
        day_str = d2s(d_int)
        day_start_ns = d_int * NS_PER_DAY
        day_end_ns = day_start_ns + NS_PER_DAY
        prefix = (f"raw/book/exchange=BINANCE_FUTURES/"
                  f"symbol={SYM}/dt={day_str}/")
        blobs = sorted([b for b in bk.client.list_blobs(bk, prefix=prefix)
                        if b.name.endswith(".parquet")],
                       key=lambda b: b.name)
        if not blobs:
            days_missing += 1
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
                    [t[c].to_numpy().astype(np.float64) for c in chan_cols]))
            ts_day = np.concatenate(ts_parts)
            X_day = np.concatenate(X_parts, axis=0)
            order = np.argsort(ts_day, kind="stable")
            ts_day = ts_day[order]
            X_day = X_day[order]

            bid_p = X_day[:, 0:20]
            bid_s = X_day[:, 20:40]
            ask_p = X_day[:, 40:60]
            ask_s = X_day[:, 60:80]
            feat = C.tick_features(bid_p, bid_s, ask_p, ask_s)  # (n, 46) f32

            in_day = (t0 >= day_start_ns) & (t0 < day_end_ns)
            dp_idx = np.where(in_day)[0]
            for dp_i in dp_idx:
                t = int(t0[dp_i])
                j = int(np.searchsorted(ts_day, t, side="right")) - 1
                if j < 0:
                    out_eng[dp_i] = 0.0
                    continue
                lo = j - L + 1
                if lo >= 0:
                    out_eng[dp_i] = feat[lo:j + 1]
                else:
                    pad = -lo
                    win = np.zeros((L, F_ENG), np.float32)
                    win[pad:] = feat[:j + 1]
                    out_eng[dp_i] = win
            write_total += dp_idx.size
        log_lines.append(
            f"{day_str}: dp_in_day={int(in_day.sum())} cum={write_total}")
        if len(log_lines) % 20 == 0:
            print(" ".join(log_lines[-2:]))
            sys.stdout.flush()
            VOL.commit()

    out_eng.flush(); del out_eng
    with open(DONE_ENG_S1, "w") as f:
        f.write(f"n={n_exp} write_total={write_total} "
                f"days_missing={days_missing}\n")
    VOL.commit()
    return {"status": "built", "n": n_exp,
            "write_total": int(write_total),
            "shape": [n_exp, L, F_ENG],
            "path": X_ENG_S1,
            "elapsed_s": round(time.time() - t_start, 1),
            "days_missing": days_missing}


# ---- C1 PROBE: rev45-arch (W=16 engineered single-stream) on dense pack -
def _build_tcn_single_stream(F_in, W, D_in, dropout):
    """rev45/47 anchor TCN. Block topology + last-step head BYTE-IDENTICAL
    to hd1_seq_tier2._build_tcn_t2 / hd1_seq_modal._build_tcn (head='last');
    parametrized for F_in / W / D / dropout."""
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
            layers, ci = [], F_in
            for b in range(D_in):
                layers.append(Block(ci, W, 2 ** b))
                ci = W
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(W, 1)

        def forward(self, x):                        # x: (B, L, F)
            h = self.tcn(x.transpose(1, 2))          # (B, W, L)
            return self.head(h[:, :, -1]).squeeze(-1)

    return TCN()


@app.function(image=IMG, gpu="A100-80GB", timeout=3600, memory=65536,
              volumes={MNT: VOL})
def probe_engineered_dense_cell(cell: dict):
    """rev54 C1: W=16 single-stream engineered TCN on the STRIDE=1
    dense pack, 1 seed, 24 ep no early-stop, per-epoch log. Directly
    compares to rev52's W=128 raw-LOB cell at the SAME (dropout, wd,
    n_dp) -- isolates the arch/representation lever from densification.

    cell = {'dropout': float, 'wd': float, 'seed': int (default 0)}"""
    import numpy as np
    import torch
    import torch.nn.functional as Fnn

    VOL.reload()
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dropout = float(cell["dropout"])
    wd = float(cell["wd"])
    seed = int(cell.get("seed", 0))
    cell_tag = f"eng_W{W_ANCHOR}_drop{dropout:g}_wd{wd:g}_seed{seed}"

    X_eng = np.load(X_ENG_S1, mmap_mode="r")     # (n, L, 46)
    meta = np.load(META_S1)
    n = int(meta["n"])
    y0 = meta["y0_300"]
    rH = meta["rH_300"].astype(np.float64)

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

    # per-channel z-score on fit rows only (streamed)
    s_acc = np.zeros(F_ENG, np.float64)
    ss_acc = np.zeros(F_ENG, np.float64)
    cnt = 0
    SCHUNK = 1024
    for c in range(0, fit_idx.size, SCHUNK):
        blk = np.ascontiguousarray(
            X_eng[fit_idx[c:c + SCHUNK]]).reshape(-1, F_ENG).astype(np.float64)
        s_acc += blk.sum(axis=0)
        ss_acc += np.square(blk).sum(axis=0)
        cnt += blk.shape[0]
    x_mu = (s_acc / cnt).astype(np.float32)
    x_sd = (np.sqrt(np.maximum(ss_acc / cnt - (s_acc / cnt) ** 2, 0.0))
            .astype(np.float32) + 1e-6)

    def _gather(idx):
        x = np.ascontiguousarray(X_eng[idx]).astype(np.float32)
        x = (x - x_mu) / x_sd
        return x

    torch.manual_seed(seed)
    np.random.seed(seed)
    net = _build_tcn_single_stream(F_in=F_ENG, W=W_ANCHOR, D_in=D,
                                   dropout=dropout).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_MAX_DIAG)
    scaler = torch.amp.GradScaler("cuda", enabled=dev != "cpu")

    y_dev = torch.from_numpy(up).to(dev)
    w_dev = torch.from_numpy(w1).to(dev)

    def _eval(indices):
        out = []
        wll_num, wll_den = 0.0, 0.0
        with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=dev != "cpu"):
            for s_ in range(0, indices.size, EV_CHUNK):
                ii = indices[s_:s_ + EV_CHUNK]
                xb = _gather(ii)
                xb = torch.from_numpy(xb).to(dev, non_blocking=True)
                lo = net(xb).float()
                jt = torch.as_tensor(ii, device=dev, dtype=torch.long)
                bce = Fnn.binary_cross_entropy_with_logits(
                    lo, y_dev[jt], reduction="none")
                wll_num += float((bce * w_dev[jt]).sum().item())
                wll_den += float(w_dev[jt].sum().item())
                out.append(lo.cpu())
                del xb
        return torch.cat(out).numpy(), wll_num / max(wll_den, 1e-9)

    y_val_np = up[val_idx].astype(int)
    y_te_np = up[te_idx].astype(int)

    history = []
    for ep in range(EP_DIAG):
        net.train()
        perm = np.random.permutation(fit_idx)
        tr_num, tr_den = 0.0, 0.0
        for s_ in range(0, perm.size, TR_BATCH):
            ii = perm[s_:s_ + TR_BATCH]
            xb = _gather(ii)
            xb = torch.from_numpy(xb).to(dev, non_blocking=True)
            jt = torch.as_tensor(ii, device=dev, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=dev != "cpu"):
                lo = net(xb)
                bce = Fnn.binary_cross_entropy_with_logits(
                    lo, y_dev[jt], reduction="none")
                loss = (bce * w_dev[jt]).sum() / (w_dev[jt].sum() + 1e-9)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_num += float((bce.detach() * w_dev[jt]).sum().item())
            tr_den += float(w_dev[jt].sum().item())
            del xb
        lr_now = float(opt.param_groups[0]["lr"])
        if ep < T_MAX_DIAG:
            sch.step()
        net.eval()
        v_lo, v_wll = _eval(val_idx)
        t_lo, t_wll = _eval(te_idx)
        v_p = 1.0 / (1.0 + np.exp(-v_lo))
        t_p = 1.0 / (1.0 + np.exp(-t_lo))
        v_auc = float(C.auc(y_val_np, v_p))
        t_auc = float(C.auc(y_te_np, t_p))
        rec = {"ep": ep + 1, "lr": round(lr_now, 7),
               "train_loss": round(tr_num / max(tr_den, 1e-9), 6),
               "val_logloss": round(v_wll, 6),
               "val_ric": round(v_auc - 0.5, 6),
               "test_logloss": round(t_wll, 6),
               "test_ric": round(t_auc - 0.5, 6)}
        history.append(rec)
        print(f"[{cell_tag}] ep{ep+1:>2d} lr={lr_now:.5f} "
              f"trL={rec['train_loss']:.4f} "
              f"vR={rec['val_ric']:+.4f} tR={rec['test_ric']:+.4f}")
        sys.stdout.flush()

    last = history[-1]
    by_val = max(history, key=lambda r: r["val_ric"])
    by_test = max(history, key=lambda r: r["test_ric"])
    t_lo_final, _ = _eval(te_idx)
    t_p_final = 1.0 / (1.0 + np.exp(-t_lo_final))
    plac = float(C.placebo_auc(y_te_np, t_p_final) - 0.5)
    block = int(C.block_size(H))
    se = C.block_bootstrap_auc_se(y_te_np, t_p_final, block)
    se_f = None if not np.isfinite(se) else round(float(se), 6)

    return {"rev": 54, "cell_tag": cell_tag,
            "arch": "engineered_singlestream_W16_laststep",
            "dropout": dropout, "wd": wd, "seed": seed,
            "n_params": int(n_params),
            "n_fit": int(fit_m.sum()), "n_val": int(val_m.sum()),
            "n_te": int(s_te.sum()), "block": block,
            "history": history,
            "summary": {"last_ep": last, "by_best_val_ric": by_val,
                        "by_best_test_ric": by_test},
            "final_ep_diagnostics": {"placebo_ric": round(plac, 6),
                                     "boot_se": se_f},
            "gpu_s": round(time.time() - t0, 2)}


# ---- BUILD: rev56 raw pack at L_MAX=2048 (f16), STRIDE=1 ----------------
@app.function(image=IMG, cpu=64.0, memory=131072, timeout=21600,
              volumes={MNT: VOL},
              secrets=[modal.Secret.from_name("hd1-gcp")])
def build_dense_raw_lmax():
    """rev56 L-sweep build: raw 20-lvl LOB windows at L_MAX=2048,
    STRIDE=1, on the SAME META_S1 dp set (n=174,606, parity-verified).
    Single-phase (labels already in META_S1; no parity recompute).
    Stored as float16 (~55 GiB vs 109 GiB f32): the raw normalized
    channels are f16-safe -- (p-mid)/mid ~ 1e-2, sign*log1p(|size|)
    <= ~14, both << f16 max 65504, with ~1e-3 relative rounding that
    is immaterial vs the alpha signal (NO Cont-OFI channel in the raw
    80-ch pack, unlike the engineered pack where f16 overflowed in
    rev26). Shorter L probes slice the last L ticks of each window.
    Output: RAW_PACK_S1_LMAX (n, 2048, 80) f16 + DONE marker."""
    import tempfile
    import numpy as np
    import pyarrow.parquet as pq

    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C  # noqa: F401 (parity provenance)

    VOL.reload()
    if not os.path.exists(META_S1):
        raise RuntimeError(f"META_S1 missing ({META_S1}); run build_dense first")
    meta = np.load(META_S1)
    n_exp = int(meta["n"])
    t0 = meta["t0"].astype(np.int64)

    if os.path.exists(RAW_PACK_S1_LMAX) and os.path.exists(DONE_S1_LMAX):
        try:
            a = np.load(RAW_PACK_S1_LMAX, mmap_mode="r")
            if a.shape == (n_exp, L_MAX, F_LOB):
                return {"status": "skip_existing", "shape": list(a.shape)}
        except Exception:
            pass

    NS_PER_DAY = 86400 * 1_000_000_000
    EPOCH = dt.date(1970, 1, 1)

    def d2s(d):
        return (EPOCH + dt.timedelta(days=int(d))).isoformat()

    t0_day = (t0 // NS_PER_DAY).astype(np.int64)
    unique_days = sorted(int(d) for d in np.unique(t0_day))
    bk = _gcs_bucket()
    out = np.lib.format.open_memmap(
        RAW_PACK_S1_LMAX, mode="w+", dtype=np.float16,
        shape=(n_exp, L_MAX, F_LOB))

    chan_cols = []
    for k in range(20):
        chan_cols.append(f"bid_{k}_price")
    for k in range(20):
        chan_cols.append(f"bid_{k}_size")
    for k in range(20):
        chan_cols.append(f"ask_{k}_price")
    for k in range(20):
        chan_cols.append(f"ask_{k}_size")

    write_total = 0
    days_missing = 0
    t_start = time.time()
    log = []
    for d_int in unique_days:
        day_str = d2s(d_int)
        day_start_ns = d_int * NS_PER_DAY
        day_end_ns = day_start_ns + NS_PER_DAY
        prefix = (f"raw/book/exchange=BINANCE_FUTURES/"
                  f"symbol={SYM}/dt={day_str}/")
        blobs = sorted([b for b in bk.client.list_blobs(bk, prefix=prefix)
                        if b.name.endswith(".parquet")],
                       key=lambda b: b.name)
        if not blobs:
            days_missing += 1
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
                    [t[c].to_numpy().astype(np.float64) for c in chan_cols]))
            ts_day = np.concatenate(ts_parts)
            X_day = np.concatenate(X_parts, axis=0)
            order = np.argsort(ts_day, kind="stable")
            ts_day = ts_day[order]
            X_day = X_day[order]

            bid_p = X_day[:, 0:20]
            bid_s = X_day[:, 20:40]
            ask_p = X_day[:, 40:60]
            ask_s = X_day[:, 60:80]
            mid = 0.5 * (bid_p[:, 0] + ask_p[:, 0])
            mid_safe = np.where(mid > 0, mid, 1.0)
            lob = np.empty((ts_day.size, F_LOB), np.float32)
            lob[:, 0:20] = ((bid_p - mid[:, None]) / mid_safe[:, None])
            lob[:, 40:60] = ((ask_p - mid[:, None]) / mid_safe[:, None])
            lob[:, 20:40] = (np.sign(bid_s) * np.log1p(np.abs(bid_s)))
            lob[:, 60:80] = (np.sign(ask_s) * np.log1p(np.abs(ask_s)))
            lob = lob.astype(np.float16)

            in_day = (t0 >= day_start_ns) & (t0 < day_end_ns)
            dp_idx = np.where(in_day)[0]
            for dp_i in dp_idx:
                t = int(t0[dp_i])
                j = int(np.searchsorted(ts_day, t, side="right")) - 1
                if j < 0:
                    out[dp_i] = 0
                    continue
                lo = j - L_MAX + 1
                if lo >= 0:
                    out[dp_i] = lob[lo:j + 1]
                else:
                    pad = -lo
                    win = np.zeros((L_MAX, F_LOB), np.float16)
                    win[pad:] = lob[:j + 1]
                    out[dp_i] = win
            write_total += dp_idx.size
        log.append(f"{day_str}: cum={write_total}")
        if len(log) % 20 == 0:
            print(f"[build_lmax] {log[-1]}")
            sys.stdout.flush()
            VOL.commit()

    out.flush(); del out
    with open(DONE_S1_LMAX, "w") as f:
        f.write(f"n={n_exp} write_total={write_total} "
                f"days_missing={days_missing} dtype=f16 L_MAX={L_MAX}\n")
    VOL.commit()
    return {"status": "built", "n": n_exp, "write_total": int(write_total),
            "shape": [n_exp, L_MAX, F_LOB], "dtype": "float16",
            "path": RAW_PACK_S1_LMAX,
            "elapsed_s": round(time.time() - t_start, 1),
            "days_missing": days_missing}


# ---- L-SWEEP PROBE: 2-stream raw-LOB at sliced L, RF-matched D ----------
@app.function(image=IMG, gpu="A100-80GB", timeout=7200, memory=98304,
              volumes={MNT: VOL})
def probe_raw_lsweep_cell(cell: dict):
    """rev56: 2-stream raw-LOB + 6-globals at a chosen context length L
    (sliced from the L_MAX=2048 f16 pack), RF-matched D = D_FOR_L_EXT[L],
    rev45-anchor reg (dropout, wd), 1 seed, 24 ep no early-stop,
    per-epoch logging. Globals come from the existing GLOB_PACK_S1
    (last-tick, L-independent). Isolates the CONTEXT-LENGTH axis for
    the RAW representation (vs rev52 #7 fixed at L=512).

    cell = {'L': int, 'dropout': float, 'wd': float, 'seed': int}"""
    import numpy as np
    import torch
    import torch.nn.functional as Fnn

    VOL.reload()
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    L_use = int(cell["L"])
    d_blocks = D_FOR_L_EXT[L_use]
    dropout = float(cell.get("dropout", 0.5))
    wd = float(cell.get("wd", 1e-3))
    seed = int(cell.get("seed", 0))
    cell_tag = f"raw_L{L_use}_D{d_blocks}_drop{dropout:g}_wd{wd:g}_seed{seed}"

    X_raw = np.load(RAW_PACK_S1_LMAX, mmap_mode="r")   # (n, 2048, 80) f16
    G_all_raw = np.load(GLOB_PACK_S1, mmap_mode="r")    # (n, 6) f32
    meta = np.load(META_S1)
    n = int(meta["n"])
    y0 = meta["y0_300"]
    rH = meta["rH_300"].astype(np.float64)

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

    G_all = np.ascontiguousarray(G_all_raw).astype(np.float32)
    g_mu = G_all[fit_m].mean(axis=0).astype(np.float32)
    g_sd = G_all[fit_m].std(axis=0).astype(np.float32) + 1e-6
    G_all = (G_all - g_mu) / g_sd

    sl = slice(L_MAX - L_use, L_MAX)   # last L_use ticks of each window

    # PRELOAD the sliced windows for all needed rows into RAM in ONE mmap
    # pass (fit u val u te ~= 105k rows). Avoids re-reading the 55 GiB f16
    # Volume pack every epoch (would be ~480 GiB IO over 24 ep at L=2048).
    needed = np.union1d(np.union1d(fit_idx, val_idx), te_idx)
    remap = -np.ones(n, dtype=np.int64)
    remap[needed] = np.arange(needed.size)
    Xmem = np.empty((needed.size, L_use, F_LOB), np.float16)
    CH = 2048
    for c in range(0, needed.size, CH):
        blk = needed[c:c + CH]
        Xmem[c:c + blk.size] = X_raw[blk, sl, :]
    print(f"[{cell_tag}] preloaded {Xmem.shape} f16 "
          f"({Xmem.nbytes/2**30:.1f} GiB) into RAM")
    sys.stdout.flush()

    fit_rows = remap[fit_idx]
    # z-score on fit rows from RAM
    s_acc = np.zeros(F_LOB, np.float64)
    ss_acc = np.zeros(F_LOB, np.float64)
    cnt = 0
    for c in range(0, fit_rows.size, CH):
        blk = np.ascontiguousarray(
            Xmem[fit_rows[c:c + CH]]).reshape(-1, F_LOB).astype(np.float64)
        s_acc += blk.sum(axis=0)
        ss_acc += np.square(blk).sum(axis=0)
        cnt += blk.shape[0]
    x_mu = (s_acc / cnt).astype(np.float32)
    x_sd = (np.sqrt(np.maximum(ss_acc / cnt - (s_acc / cnt) ** 2, 0.0))
            .astype(np.float32) + 1e-6)

    def _gather(idx):
        x = Xmem[remap[idx]].astype(np.float32)
        x = (x - x_mu) / x_sd
        return x, G_all[idx]

    torch.manual_seed(seed)
    np.random.seed(seed)
    net = _build_two_stream_tcn(dropout=dropout, d_blocks=d_blocks).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_MAX_DIAG)
    scaler = torch.amp.GradScaler("cuda", enabled=dev != "cpu")

    y_dev = torch.from_numpy(up).to(dev)
    w_dev = torch.from_numpy(w1).to(dev)

    def _eval(indices):
        out = []
        wll_num, wll_den = 0.0, 0.0
        with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=dev != "cpu"):
            for s_ in range(0, indices.size, EV_CHUNK):
                ii = indices[s_:s_ + EV_CHUNK]
                xb, gb = _gather(ii)
                xb = torch.from_numpy(xb).to(dev, non_blocking=True)
                gb = torch.from_numpy(gb).to(dev, non_blocking=True)
                lo = net(xb, gb).float()
                jt = torch.as_tensor(ii, device=dev, dtype=torch.long)
                bce = Fnn.binary_cross_entropy_with_logits(
                    lo, y_dev[jt], reduction="none")
                wll_num += float((bce * w_dev[jt]).sum().item())
                wll_den += float(w_dev[jt].sum().item())
                out.append(lo.cpu())
                del xb, gb
        return torch.cat(out).numpy(), wll_num / max(wll_den, 1e-9)

    y_val_np = up[val_idx].astype(int)
    y_te_np = up[te_idx].astype(int)

    EV_CHUNK_L = max(512, EV_CHUNK // max(1, L_use // 512))  # smaller chunk at big L
    history = []
    for ep in range(EP_DIAG):
        net.train()
        perm = np.random.permutation(fit_idx)
        tr_num, tr_den = 0.0, 0.0
        for s_ in range(0, perm.size, TR_BATCH):
            ii = perm[s_:s_ + TR_BATCH]
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
            tr_num += float((bce.detach() * w_dev[jt]).sum().item())
            tr_den += float(w_dev[jt].sum().item())
            del xb, gb
        lr_now = float(opt.param_groups[0]["lr"])
        if ep < T_MAX_DIAG:
            sch.step()
        net.eval()
        v_lo, v_wll = _eval(val_idx)
        t_lo, t_wll = _eval(te_idx)
        v_p = 1.0 / (1.0 + np.exp(-v_lo))
        t_p = 1.0 / (1.0 + np.exp(-t_lo))
        v_auc = float(C.auc(y_val_np, v_p))
        t_auc = float(C.auc(y_te_np, t_p))
        rec = {"ep": ep + 1, "lr": round(lr_now, 7),
               "train_loss": round(tr_num / max(tr_den, 1e-9), 6),
               "val_logloss": round(v_wll, 6),
               "val_ric": round(v_auc - 0.5, 6),
               "test_logloss": round(t_wll, 6),
               "test_ric": round(t_auc - 0.5, 6)}
        history.append(rec)
        print(f"[{cell_tag}] ep{ep+1:>2d} lr={lr_now:.5f} "
              f"trL={rec['train_loss']:.4f} "
              f"vR={rec['val_ric']:+.4f} tR={rec['test_ric']:+.4f}")
        sys.stdout.flush()

    last = history[-1]
    by_val = max(history, key=lambda r: r["val_ric"])
    by_test = max(history, key=lambda r: r["test_ric"])
    t_lo_final, _ = _eval(te_idx)
    t_p_final = 1.0 / (1.0 + np.exp(-t_lo_final))
    plac = float(C.placebo_auc(y_te_np, t_p_final) - 0.5)
    block = int(C.block_size(H))
    se = C.block_bootstrap_auc_se(y_te_np, t_p_final, block)
    se_f = None if not np.isfinite(se) else round(float(se), 6)

    return {"rev": 56, "cell_tag": cell_tag, "L": L_use, "D": d_blocks,
            "dropout": dropout, "wd": wd, "seed": seed,
            "n_params": int(n_params),
            "n_fit": int(fit_m.sum()), "n_val": int(val_m.sum()),
            "n_te": int(s_te.sum()), "block": block,
            "history": history,
            "summary": {"last_ep": last, "by_best_val_ric": by_val,
                        "by_best_test_ric": by_test},
            "final_ep_diagnostics": {"placebo_ric": round(plac, 6),
                                     "boot_se": se_f},
            "gpu_s": round(time.time() - t0, 2)}


@app.function(image=IMG, cpu=1.0, memory=4096, timeout=600,
              volumes={MNT: VOL})
def save_result(payload: dict):
    """Persist a result payload to the Volume. If payload['out_path']
    is provided, write there (preferred -- per-rev unique path); else
    fall back to the rev48 RESULT_PATH (legacy)."""
    VOL.reload()
    out = payload.get("out_path") or RESULT_PATH
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    VOL.commit()
    return {"saved_to": out, "size_bytes": os.path.getsize(out)}


@app.local_entrypoint()
def main(stage: str = "all", lcurve_seed: int = 0):
    """Stage: build | probe | lcurve | build_dense | regsweep | all
    | all_dense."""
    import statistics as st
    t_main = time.time()
    if stage == "lcurve":
        print(f"[stage={stage}] launching lcurve_lobw128 seed={lcurve_seed}"
              f" (EP_DIAG={EP_DIAG}, T_MAX_DIAG={T_MAX_DIAG}, NO early-stop)...")
        r = lcurve_lobw128.remote(lcurve_seed)
        out_path = f"{MNT}/tier2/rev50_lcurve_seed{lcurve_seed}.json"
        save_result.remote({"out_path": out_path, **r})
        print("\n=== rev50 LCURVE RESULT ===")
        print(json.dumps(r, indent=2, default=str))
        print(f"\ntotal wall: {time.time() - t_main:.1f}s")
        return
    if stage == "lsweep":
        # rev56: build L_MAX=2048 raw pack (if missing), then probe
        # L in {512, 2048} at rev45-anchor reg (drop=0.5 wd=1e-3) seed=0.
        print(f"[stage={stage}] launching build_dense_raw_lmax (L_MAX="
              f"{L_MAX}, f16)...")
        rb = build_dense_raw_lmax.remote()
        print(f"[build_lmax] result:\n{json.dumps(rb, indent=2, default=str)}")
        if rb.get("status") not in ("built", "skip_existing"):
            print("[lsweep] build failed; aborting.")
            return
        cells = [{"L": 512, "dropout": 0.5, "wd": 1e-3, "seed": 0},
                 {"L": 2048, "dropout": 0.5, "wd": 1e-3, "seed": 0}]
        print(f"[stage={stage}] launching {len(cells)} L-cells in parallel...")
        results = list(probe_raw_lsweep_cell.map(cells))
        rows = []
        for r in results:
            s = r["summary"]
            rows.append({"L": r["L"], "D": r["D"], "n_params": r["n_params"],
                         "best_val_ric": s["by_best_val_ric"]["val_ric"],
                         "test_at_best_val": s["by_best_val_ric"]["test_ric"],
                         "best_val_ep": s["by_best_val_ric"]["ep"],
                         "best_test_ric": s["by_best_test_ric"]["test_ric"],
                         "best_test_ep": s["by_best_test_ric"]["ep"],
                         "last_train_loss": s["last_ep"]["train_loss"],
                         "placebo_ric": r["final_ep_diagnostics"]["placebo_ric"],
                         "boot_se": r["final_ep_diagnostics"]["boot_se"],
                         "gpu_s": r["gpu_s"]})
        rows.sort(key=lambda x: x["L"])
        payload = {"rev": 56, "cell": f"{SYM}-H{H}-W{W_LOB}-raw2stream",
                   "rows": rows, "full_histories": results,
                   "wall_s": round(time.time() - t_main, 1)}
        print("\n=== rev56 L-SWEEP SUMMARY (raw, drop=0.5 wd=1e-3 seed=0) ===")
        for row in rows:
            print(f"  L={row['L']:>4} D={row['D']}  "
                  f"bestVal@ep{row['best_val_ep']:>2d}={row['best_val_ric']:+.4f}"
                  f" ->test={row['test_at_best_val']:+.4f}  | "
                  f"bestTest@ep{row['best_test_ep']:>2d}={row['best_test_ric']:+.4f}"
                  f"  trL={row['last_train_loss']:.4f} params={row['n_params']}"
                  f" gpu_s={row['gpu_s']:.0f}")
        save_result.remote({"out_path": f"{MNT}/tier2/rev56_lsweep.json",
                            **payload})
        print(f"\ntotal wall: {time.time() - t_main:.1f}s")
        return
    if stage == "c1":
        # rev54 C1: builds engineered dense pack (if missing) + runs ONE
        # cell (rev45-anchor reg drop=0.5 wd=1e-3) of W=16 single-stream
        # engineered TCN at STRIDE=1 -- isolates arch from densification.
        print(f"[stage={stage}] launching build_dense_engineered_stride1 ...")
        rb = build_dense_engineered_stride1.remote()
        print(f"[build_dense_engineered] result:\n"
              f"{json.dumps(rb, indent=2, default=str)}")
        if rb.get("status") not in ("built", "skip_existing"):
            print("[c1] build failed; aborting probe.")
            return
        print(f"[stage={stage}] launching probe_engineered_dense_cell"
              " (drop=0.5 wd=1e-3 seed=0)...")
        rp = probe_engineered_dense_cell.remote(
            {"dropout": 0.5, "wd": 1e-3, "seed": 0})
        print("\n=== rev54 C1 PROBE RESULT ===")
        print(json.dumps(rp, indent=2, default=str))
        save_result.remote(
            {"out_path": f"{MNT}/tier2/rev54_c1_engineered_dense.json",
             **rp})
        print(f"\ntotal wall: {time.time() - t_main:.1f}s")
        return
    if stage in ("build_dense", "all_dense"):
        print(f"[stage={stage}] launching build_dense_stride1 on Modal CPU...")
        r = build_dense_stride1.remote()
        print(f"[build_dense] result:\n{json.dumps(r, indent=2, default=str)}")
    if stage in ("regsweep", "all_dense"):
        print(f"[stage={stage}] launching regsweep over "
              f"{len(REGSWEEP_GRID)} (dropout, wd) cells in parallel...")
        cells = [{"dropout": d, "wd": w, "seed": 0}
                 for (d, w) in REGSWEEP_GRID]
        results = list(regsweep_lobw128_cell.map(cells))
        # build a compact summary table
        rows = []
        for r in results:
            s = r["summary"]
            rows.append({"cell_tag": r["cell_tag"],
                         "dropout": r["dropout"], "wd": r["wd"],
                         "n_fit": r["n_fit"], "n_te": r["n_te"],
                         "best_val_ric": s["by_best_val_ric"]["val_ric"],
                         "test_at_best_val": s["by_best_val_ric"]["test_ric"],
                         "best_val_ep": s["by_best_val_ric"]["ep"],
                         "best_test_ric": s["by_best_test_ric"]["test_ric"],
                         "best_test_ep": s["by_best_test_ric"]["ep"],
                         "last_train_loss": s["last_ep"]["train_loss"],
                         "placebo_ric": r["final_ep_diagnostics"][
                             "placebo_ric"],
                         "boot_se": r["final_ep_diagnostics"]["boot_se"],
                         "gpu_s": r["gpu_s"]})
        rows.sort(key=lambda x: -x["best_test_ric"])
        payload = {"rev": 52, "cell": f"{SYM}-H{H}-L{L}-W{W_LOB}",
                   "grid_size": len(REGSWEEP_GRID), "rows": rows,
                   "full_histories": [r for r in results],
                   "wall_s": round(time.time() - t_main, 1)}
        print("\n=== rev52 REGSWEEP SUMMARY (sorted by best_test_ric) ===")
        for row in rows:
            print(f"  drop={row['dropout']:.1f} wd={row['wd']:>.0e}  "
                  f"bestVal@ep{row['best_val_ep']:>2d}={row['best_val_ric']:+.4f} "
                  f"->test={row['test_at_best_val']:+.4f}  | "
                  f"bestTest@ep{row['best_test_ep']:>2d}={row['best_test_ric']:+.4f}"
                  f"  trL={row['last_train_loss']:.4f}  gpu_s={row['gpu_s']:.0f}")
        save_result.remote({"out_path": f"{MNT}/tier2/rev52_regsweep.json",
                            **payload})
        print(f"\ntotal wall: {time.time() - t_main:.1f}s")
        return
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
