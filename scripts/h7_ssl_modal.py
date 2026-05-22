#!/usr/bin/env python3
"""H7 rev3 — SSL-vs-from-scratch alpha test on Modal (PatchTST, 46-ch L256).

Frozen pre-reg: research/hypotheses.jsonl H7 rev3. Fixed-backbone ablation:
ONE PatchTST encoder, two arms differing ONLY in init -- SSL masked-pretrain
vs random -- finetuned on the 4-symbol H180 row. Label/scope/split/metric
are the FROZEN hd1_seq_core (bit-exact vs HM6); only model_family + init vary.

Deliverable = the Delta rank_IC(SSL - scratch) surface by cell + argmax +
seed-band. The frozen section-5 deploy-gate vs BASELINE_REF_RIC is a DEMOTED
secondary annotation (CLAUDE.md rule 2), never the headline.

Data: the GCP-built compact pack gs://<bucket>/hd1seq_ssl_pack/l256/{sym}.npz
(X (n,256,46) f32, t0, n, n_tr, y0_{H}, rH_{H}); fetched once to the Volume
within the access-token TTL, then pretrain/finetune read the Volume (no GCS).

Run:
  # de-risk first (tiny pretrain + 1 finetune; NaN/finite-loss gate):
  modal run scripts/h7_ssl_modal.py --smoke 1
  # full sweep:
  modal run scripts/h7_ssl_modal.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

# ---- frozen constants (== hd1_seq_modal.py / HD1 rev25 testbed) ---------
BUCKET = "blackdigital-scalper-data"
GCP_PROJECT = "project-26a24ad0-1059-4f73-93b"
SYMS = ["SOL-USDT-PERP", "BTC-USDT-PERP", "ETH-USDT-PERP", "LTC-USDT-PERP"]
PACK_PREFIX = "hd1seq_ssl_pack/l256"
H = 180                       # the pre-registered H180 row
L = 256                       # frozen context (== gcpbuild --max-l 256)
F = 46                        # hd1_seq_core.N_TICK_FEAT
SEEDS = (0, 1, 2)             # >=3 finetune seeds/arm (pre-reg)
ARMS = ("ssl", "scratch")
FREEZE = "H7 rev3"

# Frozen HM6 rev4 baseline_ref rank_ic_oos per (symbol, H) -- verbatim from
# hd1_seq_modal.BASELINE_REF_RIC. delta_ic = rank_ic_oos - this. DO NOT EDIT.
BASELINE_REF_RIC = {
    ("SOL-USDT-PERP", 180): 0.0158, ("BTC-USDT-PERP", 180): 0.0419,
    ("ETH-USDT-PERP", 180): 0.0234, ("LTC-USDT-PERP", 180): 0.0378,
}
BASELINE_RUN = "phaseb-20260517-203822"

# PatchTST encoder (== src/ssl/model.BackboneConfig; channel-independent).
PATCH_LEN, STRIDE_P, D_MODEL, N_HEADS, N_LAYERS, FFN, DROP = 16, 16, 192, 8, 4, 512, 0.15
MASK_RATIO = 0.20

app = modal.App("h7-ssl")
_IGN = ["**/.git/**", "**/__pycache__/**", "data/**", "models/**",
        "**/*.db", "research_runs/**", "**/target/**"]
IO_IMG = (modal.Image.debian_slim(python_version="3.11")
          .pip_install("numpy==2.2.4", "google-cloud-storage"))
GPU_IMG = (modal.Image.debian_slim(python_version="3.11")
           .pip_install("numpy==2.2.4", "scikit-learn", "torch")
           .add_local_dir(str(REPO), "/root/proj", ignore=_IGN))
VOL = modal.Volume.from_name("h7-ssl-cache", create_if_missing=True)
MNT = "/cache"


def _gcs():
    """GCS bucket from the env credential (== hd1_seq_modal._gcs).

    The Modal secret h7-gcs carries a fresh ~1h OAuth token in
    GCP_ACCESS_TOKEN; a bare token has no refresh, so the fetch must
    finish inside its TTL (it does -- one bulk download of the compact
    pack, then everything reads the Volume)."""
    import base64
    from google.cloud import storage
    for name in ("GCP_ACCESS_TOKEN", "GCP_SA_KEY_B64", "GCP_SA_KEY"):
        v = os.environ.get(name)
        if not v:
            continue
        if name == "GCP_SA_KEY_B64":
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
    raise RuntimeError("no GCS credential in env (GCP_ACCESS_TOKEN/SA)")


# =========================================================================
# fetch_pack — GCS -> Volume (once, within token TTL)
# =========================================================================
@app.function(image=IO_IMG, timeout=3600, volumes={MNT: VOL},
              secrets=[modal.Secret.from_name("h7-gcs")])
def fetch_pack():
    bk = _gcs()
    os.makedirs(f"{MNT}/packed", exist_ok=True)
    out = {}
    for sym in SYMS:
        dst = f"{MNT}/packed/{sym}.npz"
        if os.path.exists(dst):
            out[sym] = {"cached": True, "bytes": os.path.getsize(dst)}
            continue
        key = f"{PACK_PREFIX}/{sym}.npz"
        bk.blob(key).download_to_filename(dst)
        out[sym] = {"bytes": os.path.getsize(dst)}
    VOL.commit()
    return out


# =========================================================================
# PatchTST encoder + heads
# =========================================================================
def _backbone(num_channels=F, time_dim=L):
    """src/ssl/model.PatchTSTBackbone with our frozen config."""
    sys.path.insert(0, "/root/proj")
    from src.ssl.model import BackboneConfig, PatchTSTBackbone
    cfg = BackboneConfig(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                         ffn_dim=FFN, dropout=DROP, patch_len=PATCH_LEN,
                         stride=STRIDE_P)
    return PatchTSTBackbone(num_channels, time_dim, cfg), cfg


class _Reconstructor:
    pass


def _build_reconstructor():
    sys.path.insert(0, "/root/proj")
    from src.ssl.model import BackboneConfig, PatchTSTReconstructor
    cfg = BackboneConfig(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                         ffn_dim=FFN, dropout=DROP, patch_len=PATCH_LEN,
                         stride=STRIDE_P)
    return PatchTSTReconstructor(F, L, cfg)


def _build_classifier(device):
    """PatchTST encoder + thin binary head: mean-pool over (patches,
    channels) -> LayerNorm -> Linear(d_model -> 1)."""
    import torch.nn as nn
    bb, cfg = _backbone()

    class Clf(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = bb
            self.head = nn.Sequential(nn.LayerNorm(cfg.d_model),
                                      nn.Linear(cfg.d_model, 1))

        def forward(self, x):                  # x: (B, C=F, T=L)
            enc = self.backbone(x)             # (B, C, N, d_model)
            pooled = enc.mean(dim=(1, 2))      # (B, d_model)
            return self.head(pooled).squeeze(-1)

    return Clf().to(device)


# =========================================================================
# pretrain_encoder — masked LOB modeling on pooled TRAIN rows (embargoed)
# =========================================================================
@app.function(image=GPU_IMG, gpu="A10G", timeout=14400, volumes={MNT: VOL},
              retries=0)
def pretrain_encoder(epochs: int = 20, samples_cap: int = 0,
                     smoke: int = 0):
    """One shared encoder, masked-reconstruction on pooled per-symbol
    TRAIN rows only (honest_split tr mask => strictly pre-test => the
    SSL-specific embargo). Saves backbone state_dict to the Volume."""
    import numpy as np
    import torch
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    VOL.reload()
    # pool TRAIN rows across symbols (embargo: never a test row)
    parts = []
    for sym in SYMS:
        P = np.load(f"{MNT}/packed/{sym}.npz")
        n = int(P["n"])
        tr, _, _ = C.honest_split(n)
        parts.append(P["X"][tr].astype(np.float32))   # (ntr, L, F)
    Xtr = np.concatenate(parts)                        # (N, L, F)
    rng = np.random.default_rng(42)
    if smoke:
        epochs = 1
        Xtr = Xtr[rng.choice(Xtr.shape[0], min(4096, Xtr.shape[0]),
                             replace=False)]
    elif samples_cap and Xtr.shape[0] > samples_cap:
        Xtr = Xtr[rng.choice(Xtr.shape[0], samples_cap, replace=False)]
    N = Xtr.shape[0]
    print(f"[pretrain] pooled train windows N={N} dev={dev} epochs={epochs}",
          flush=True)

    model = _build_reconstructor().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    bs = 256
    nb = max(1, N // bs)
    total = epochs * nb
    warm = max(1, int(0.05 * total))

    def lr_at(step):
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + np.cos(np.pi * p))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    idx = np.arange(N)
    hist = []
    for ep in range(epochs):
        model.train()
        rng.shuffle(idx)
        ep_loss, seen = 0.0, 0
        for s in range(0, N, bs):
            j = idx[s:s + bs]
            xb = torch.from_numpy(Xtr[j]).to(dev)        # (b, L, F)
            xb = xb.transpose(1, 2).contiguous()         # (b, F, L)
            T = xb.shape[-1]
            nmask = max(1, int(round(T * MASK_RATIO)))
            m = torch.zeros(xb.shape[0], T, dtype=torch.bool, device=dev)
            for r in range(xb.shape[0]):
                mi = torch.from_numpy(
                    rng.choice(T, nmask, replace=False)).to(dev)
                m[r, mi] = True
            inp = xb.clone()
            inp[:, :, :] = torch.where(m.unsqueeze(1), torch.zeros_like(xb), xb)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=dev == "cuda"):
                recon = model(inp)
                loss = model.loss(recon, xb, m)
            if not torch.isfinite(loss):
                raise RuntimeError(f"NON-FINITE pretrain loss at ep{ep} "
                                   f"step{s} -- abort (H7 rev1 NaN guard)")
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            ep_loss += float(loss) * len(j); seen += len(j)
        ep_loss /= max(seen, 1)
        hist.append(ep_loss)
        print(f"[pretrain] ep{ep+1}/{epochs} loss={ep_loss:.5f} "
              f"{time.time()-t0:.0f}s", flush=True)

    os.makedirs(f"{MNT}/ssl", exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()}
    torch.save(sd, f"{MNT}/ssl/backbone.pt")
    VOL.commit()
    return {"N": int(N), "epochs": epochs, "loss_hist": hist,
            "gpu_seconds": round(time.time() - t0, 1)}


# =========================================================================
# finetune_cell — one (sym): both arms x SEEDS, OOS rank_IC
# =========================================================================
@app.function(image=GPU_IMG, gpu="A10G", timeout=10800, volumes={MNT: VOL},
              retries=1)
def finetune_cell(sym: str, seeds: list, smoke: int = 0):
    import numpy as np
    import torch
    import torch.nn.functional as Fnn
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    VOL.reload()
    P = np.load(f"{MNT}/packed/{sym}.npz")
    X = P["X"].astype(np.float32)                       # (n, L, F)
    n = int(P["n"])
    tr, te, _ = C.honest_split(n)
    y0 = P[f"y0_{H}"]; rH = P[f"rH_{H}"].astype(np.float64)
    reached = (y0 != 0) & np.isfinite(rH)
    up = (y0 == 1).astype(np.float32)
    s_tr = tr & reached; s_te = te & reached
    ntr, nte = int(s_tr.sum()), int(s_te.sum())
    if ntr < C.N_TR_FLOOR or nte < C.N_OOS_FLOOR:
        return {"sym": sym, "error": f"underpowered n_tr={ntr} n_oos={nte}"}
    fit_m, val_m = C.train_val_split(s_tr)
    w1 = C.r1_weights(rH, s_tr).astype(np.float32)
    block = C.block_size(H)
    yte = up[s_te].astype(int)

    def _tens(mask):
        return torch.from_numpy(
            np.ascontiguousarray(X[mask].transpose(0, 2, 1)))  # (b,F,L)
    Xfit, Xval, Xte = _tens(fit_m), _tens(val_m), _tens(s_te)
    yfit = torch.from_numpy(up[fit_m]); yval = torch.from_numpy(up[val_m])
    wfit = torch.from_numpy(w1[fit_m]); wval = torch.from_numpy(w1[val_m])

    bb_path = f"{MNT}/ssl/backbone.pt"
    out = {"sym": sym, "n_tr": ntr, "n_oos": nte, "block": block, "arms": {}}
    epochs = 1 if smoke else 30
    seeds = seeds[:1] if smoke else seeds
    for arm in ARMS:
        per_seed = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            net = _build_classifier(dev)
            if arm == "ssl":
                sd = torch.load(bb_path, map_location=dev)
                rep = net.backbone.load_state_dict(sd, strict=False)
                miss = len(getattr(rep, "missing_keys", []))
                if miss > 4:
                    return {"sym": sym,
                            "error": f"ssl backbone load missing {miss} keys"}
            opt = torch.optim.AdamW(net.parameters(), lr=3e-4,
                                    weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30)
            scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
            bs, best_val, patience, best_state = 512, 1e9, 0, None
            ifit = np.arange(Xfit.shape[0])
            for ep in range(epochs):
                net.train(); np.random.shuffle(ifit)
                for s in range(0, len(ifit), bs):
                    j = ifit[s:s + bs]
                    xb = Xfit[j].to(dev); yb = yfit[j].to(dev)
                    wb = wfit[j].to(dev)
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
                    vp = []
                    for s in range(0, Xval.shape[0], 4096):
                        vp.append(net(Xval[s:s + 4096].to(dev)).float().cpu())
                    vp = torch.cat(vp)
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
            ric = C.auc(yte, p) - 0.5
            per_seed.append({"seed": seed, "ric": round(ric, 6),
                             "val_logloss": round(best_val, 6), "_p": p})
        rics = [d["ric"] for d in per_seed]
        ric_mean = float(np.mean(rics))
        # placebo/boot_se on the best-val seed's predictions
        best = min(per_seed, key=lambda d: d["val_logloss"])
        plac = C.placebo_auc(yte, best["_p"]) - 0.5
        se = C.block_bootstrap_auc_se(yte, best["_p"], block)
        out["arms"][arm] = {
            "ric_mean": round(ric_mean, 6),
            "ric_seeds": [round(r, 6) for r in rics],
            "seed_sd": round(float(np.std(rics)), 6),
            "placebo_ric": round(plac, 6),
            "boot_se": None if not np.isfinite(se) else round(se, 6),
        }
    out["gpu_seconds"] = round(time.time() - t0, 1)
    return out


# =========================================================================
# coordinator — fetch -> pretrain -> finetune.map -> ingest
# =========================================================================
@app.function(image=GPU_IMG, volumes={MNT: VOL}, timeout=21600)
def coordinator(smoke: int = 0, epochs: int = 20):
    import numpy as np  # noqa
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C
    from research import ledger as Lg

    run_id = f"h7ssl-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    fp = fetch_pack.remote()
    print(f"[fetch] {fp}", flush=True)
    pre = pretrain_encoder.remote(epochs=epochs, smoke=smoke)
    print(f"[pretrain] {({k: pre[k] for k in ('N', 'epochs', 'gpu_seconds')})} "
          f"loss[-1]={pre['loss_hist'][-1]:.5f}", flush=True)

    cells = list(finetune_cell.starmap([(s, list(SEEDS), smoke) for s in SYMS]))
    by = {c["sym"]: c for c in cells if "error" not in c}

    # ---- Delta rank_IC surface (HEADLINE) + frozen section-5 (secondary)
    surface, signs = [], []
    for sym in SYMS:
        c = by.get(sym)
        if not c:
            surface.append({"sym": sym, "error": "no result"}); continue
        ssl, scr = c["arms"]["ssl"], c["arms"]["scratch"]
        d = round(ssl["ric_mean"] - scr["ric_mean"], 6)
        se = ssl["boot_se"] or 0.0
        signs.append(1 if d > 0 else 0)
        surface.append({
            "sym": sym, "H": H,
            "ssl_ric_mean": ssl["ric_mean"], "ssl_seed_sd": ssl["seed_sd"],
            "scratch_ric_mean": scr["ric_mean"],
            "scratch_seed_sd": scr["seed_sd"],
            "delta_ric_ssl_minus_scratch": d,
            "delta_gt_1se": bool(abs(d) > (se if se else 1.0)),
        })
    cross_consistent = sum(signs) >= 3
    argmax = max((s for s in surface if "delta_ric_ssl_minus_scratch" in s),
                 key=lambda s: s["delta_ric_ssl_minus_scratch"], default=None)

    recs = []
    for sym in SYMS:
        c = by.get(sym)
        if not c:
            continue
        for arm in ARMS:
            a = c["arms"][arm]
            base = BASELINE_REF_RIC[(sym, H)]
            d_ic = C.cell_delta_ic(a["ric_mean"], base)
            ab, why = C.gate_cell(d_ic, a["placebo_ric"], a["boot_se"])
            status = C.status_for_cell(ab, cross_consistent)
            recs.append({
                "experiment_id": f"{run_id}_H7SSL_{sym}_H{H}_{arm}",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_commit": "h7rev3", "author": "claude(modal)",
                "hypothesis_id": "H7", "status": status, "kind": "alpha",
                "setup": (f"H7 SSL test: PatchTST {arm}-init, raw-L2 46-ch "
                          f"L256 ({sym}, H={H}s); vs HM6 baseline_ref"),
                "model_family": f"patchtst_{arm}",
                "params": {"input_repr": "raw_l2_20lvl_eventclock", "L": L,
                           "init": arm, "d_model": D_MODEL,
                           "n_layers": N_LAYERS, "patch_len": PATCH_LEN,
                           "seeds": list(SEEDS),
                           "ric_seeds": a["ric_seeds"],
                           "seed_sd": a["seed_sd"],
                           "placebo_ric": a["placebo_ric"],
                           "boot_se": a["boot_se"], "gate": why,
                           "cross_symbol_ok": cross_consistent,
                           "freeze": FREEZE},
                "data_source": "cryptolake",
                "cache_id": f"cryptolake_{sym}_raw_l2_20lvl_H7SSL_l256",
                "symbols": [sym], "n_samples": int(c["n_tr"] + c["n_oos"]),
                "fee_regime": "MAKER_FIRST", "commission_win_pct": 0.04,
                "commission_loss_pct": 0.07,
                "split_method": "honest_val_test", "embargo": str(C.EMB),
                "label_def": ("first-passage up-first on >=cost subset "
                              "(f=0.0013), byte-identical to HM6; input = "
                              "raw 20-level LOB per-tick 46-ch seq L256"),
                "alpha_target": "updown_first_on_ge_cost_subset",
                "horizon_sec": H, "rank_ic_oos": round(a["ric_mean"], 5),
                "auc_oos": round(a["ric_mean"] + 0.5, 4),
                "baseline_ref": f"HM6 rev4 {BASELINE_RUN} {sym} H{H} rank_ic={base}",
                "delta_ic": d_ic, "cost_floor_pct": C.F_T0 * 100,
                "top_decile_absmove_pct": C.F_T0 * 100,
                "decile_monotonic": None, "economic_pass_loose": 0,
                "economic_pass_strict": 0, "n_eff": int(c["n_oos"]),
                "repro_cmd": "modal run scripts/h7_ssl_modal.py",
                "note": (f"H7 rev3 SSL test, arm={arm}. delta_ic={d_ic} vs "
                         f"HM6 {base}. SSL-vs-scratch delta_ric surface is "
                         f"the headline (see results.json); section-5 "
                         f"gate(a&b)={ab} cross>=3/4={cross_consistent} -> "
                         f"{status} is the demoted deploy annotation."),
            })

    res_doc = {"run_id": run_id, "hypothesis": "H7 rev3", "H": H, "L": L,
               "HEADLINE_delta_ric_surface": surface,
               "argmax_cell": argmax,
               "cross_symbol_consistent_ge3of4": cross_consistent,
               "falsification_gate_note": (
                   "SSL lift confirmed iff delta_ric > +1 block-boot SE AND "
                   "consistent sign across >=3/4 cells"),
               "secondary_section5_per_arm": "see experiments.jsonl recs",
               "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    outd = f"{MNT}/out/{run_id}"
    os.makedirs(outd, exist_ok=True)
    lines = []
    for r in recs:
        try:
            Lg.validate_experiment(r); lines.append(json.dumps(r, default=str))
        except Lg.LedgerError as e:
            print(f"[ingest] SKIP invalid: {e}", flush=True)
    with open(f"{outd}/results.json", "w") as fh:
        fh.write(json.dumps(res_doc, indent=2, default=str))
    with open(f"{outd}/recs.jsonl", "w") as fh:
        fh.write(("\n".join(lines) + "\n") if lines else "")
    with open(f"{MNT}/out/LATEST", "w") as fh:
        fh.write(run_id)
    with open(f"{outd}/DONE", "w") as fh:
        fh.write(run_id)
    VOL.commit()
    print("H7_SSL_DONE", json.dumps({"run_id": run_id, "surface": surface,
                                     "argmax": argmax}), flush=True)
    return {"run_id": run_id, "surface": surface, "argmax": argmax,
            "n_recs": len(lines)}


@app.local_entrypoint()
def main(smoke: int = 0, epochs: int = 20):
    print(f"[h7-ssl] smoke={smoke} epochs={epochs} -- spawning coordinator")
    h = coordinator.spawn(smoke=smoke, epochs=epochs)
    print(f"[spawn] coordinator id={getattr(h, 'object_id', '?')}; "
          f"results land on Volume /out/<run_id> (collect with the modal "
          f"volume CLI). Server-side; disconnect-immune.")
