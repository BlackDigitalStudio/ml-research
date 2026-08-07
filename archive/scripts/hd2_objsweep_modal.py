#!/usr/bin/env python3
"""HD2 rev5 objective round 1 = BARRIER/TARGET-FORM screen on Modal L4.

Screen: ~11 target-forms x {SOL,LTC} x seed0, L frozen=216000, all model HPs at
current. Ranks forms by mean OOS rank_IC over the 6 (sym,H) cells + an economic
guard. Right-sized GPU = L4 (5GB model). Idempotent + resumable.

  modal run scripts/hd2_objsweep_modal.py            # validate: 1 unit, 2 ep
  modal run scripts/hd2_objsweep_modal.py --full     # 22-unit screen
"""
from pathlib import Path
import modal

REPO = Path(__file__).resolve().parent.parent
_CCV = ("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/"
        "causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
_MAMBA = ("https://github.com/state-spaces/mamba/releases/download/v2.2.2/"
          "mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
IMG = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy==2.2.4", "scipy", "scikit-learn",
                 "einops", "packaging", "transformers==4.43.3")
    .pip_install(_CCV, _MAMBA)
    .add_local_dir(str(REPO / "scripts"), "/root/scripts", copy=True)
)
VOL = modal.Volume.from_name("hd2-cache", create_if_missing=True)
app = modal.App("hd2-objsweep")
MNT = "/cache"
SYMS = ["SOL-USDT-PERP", "LTC-USDT-PERP"]
ALL_SYMS_POOL = ["BNB-USDT-PERP", "BTC-USDT-PERP", "DOGE-USDT-PERP", "ETH-USDT-PERP",
                 "LINK-USDT-PERP", "LTC-USDT-PERP", "SOL-USDT-PERP", "XRP-USDT-PERP"]
L_FROZEN = 216000
HS = (180, 600, 1800)


def _impl(task):
    """Shared train/eval body for the L4 + H100 @app.function wrappers below.
    task may override batch_periods/lr/gpu (speed bench + tuning)."""
    import sys, json, os
    sys.path.insert(0, "/root/scripts")
    import hd2_train_full as T
    import hd2_targets as TGT
    import hd2_losses as LOSS
    import numpy as np

    sym, seed, epochs = task["sym"], task["seed"], task["epochs"]
    tname = task.get("target", "fp_0.30")     # R2: barrier frozen at +-0.30 (rev6)
    loss = task.get("loss")                    # None => barrier round (R1 loss)
    dropout = task.get("dropout")              # None => FullCfg default
    wd = task.get("wd")                        # round 3 sweeps these (IC frozen)
    bp = task.get("batch_periods")             # None => default heuristic
    lr = task.get("lr")                         # None => FullCfg default
    gpu = task.get("gpu", "L4")                 # tag/label only
    ts = task.get("ts")                         # round 4: (dt_min, dt_max, A_lo, A_hi)
    dconv = task.get("dconv")                   # round 4b: local conv width
    ds = task.get("d_state")                    # CAPACITY: SSM recurrent state dim
    nl = task.get("n_layers")                   # CAPACITY: depth
    if task.get("round") == "bench":           # speed/validity bench
        tag = f"BENCH_{gpu}_bp{bp}_lr{lr}__{sym}"; sub = "hd2_bench"
    elif task.get("round") == "r4":            # ROUND 4: SSM timescales (dt, A_init)
        tag = f"R4_dt{ts[0]}-{ts[1]}_A{ts[2]}-{ts[3]}__{sym}_s{seed}"
        sub = "hd2_r4_confirm" if task.get("confirm") else "hd2_r4"
    elif task.get("round") == "r4b":           # ROUND 4b: d_conv (local conv width)
        tag = f"R4B_dconv{dconv}__{sym}_s{seed}"
        sub = "hd2_r4b_confirm" if task.get("confirm") else "hd2_r4b"
    elif task.get("round") == "cap":           # CAPACITY: d_state / n_layers (pooled)
        cap_parts = ([f"ds{ds}"] if ds is not None else []) + \
                    ([f"nl{nl}"] if nl is not None else [])
        tag = f"CAP_{'_'.join(cap_parts) or 'def'}_s{seed}"
        sub = "hd2_cap_confirm" if task.get("confirm") else "hd2_cap"
    elif task.get("round") == "pool":          # DATA-SCALE: pooled all-symbols
        tag = f"POOL_{task.get('tag_extra', 'v')}_s{seed}"
        sub = "hd2_pool_confirm" if task.get("confirm") else "hd2_pool"
    elif task.get("profit3"):                  # LEGACY REPRO: 3-class profitability + prec_NF
        tag = (f"P3_L{task.get('L', L_FROZEN)}_H{task.get('p3_H',130)}_"
               f"tp{task.get('p3_tp',0.20)}_sl{task.get('p3_sl',0.10)}__{sym}_s{seed}")
        sub = "hd2_profit3"
    elif task.get("round") == "r3":            # ROUND 3: reg dropout x wd
        tag = f"R3_d{dropout}_wd{wd}__{sym}_s{seed}"; sub = "hd2_r3"
    elif loss:
        tag = f"R2_{loss}__{tname}__{sym}_s{seed}"
        sub = "hd2_r2_confirm" if task.get("confirm") else "hd2_r2"
    else:
        tag = f"{tname}__{sym}_s{seed}"; sub = "hd2_obj"
    rdir = f"{MNT}/results/{sub}"; os.makedirs(rdir, exist_ok=True)
    eval_only = bool(task.get("eval_only"))
    rpath = f"{rdir}/{tag}.rescore.json" if eval_only else f"{rdir}/{tag}.json"
    VOL.reload()
    if os.path.exists(rpath) and not task.get("force"):
        with open(rpath) as f:
            return json.load(f).get("flat", {"tag": tag, "skip": True})

    cfg_kw = dict(symbol=sym, L=L_FROZEN, seed=seed, epochs=epochs,
                  batch_periods=(bp if bp is not None else max(1, 576000 // L_FROZEN)),
                  target_spec=TGT.VARIANTS[tname], target_name=tname,
                  loss_spec=(LOSS.VARIANTS[loss] if loss else None),
                  loss_name=(loss or "R1"), eval_only=eval_only,
                  ckpt_path=f"{rdir}/{tag}.ckpt")
    if dropout is not None:
        cfg_kw["dropout"] = dropout
    if wd is not None:
        cfg_kw["wd"] = wd
    if lr is not None:
        cfg_kw["lr"] = lr
    if ts is not None:
        cfg_kw["dt_min"], cfg_kw["dt_max"] = ts[0], ts[1]
        cfg_kw["a_init_low"], cfg_kw["a_init_high"] = ts[2], ts[3]
    if dconv is not None:
        cfg_kw["d_conv"] = dconv
    if ds is not None:
        cfg_kw["d_state"] = ds
    if nl is not None:
        cfg_kw["n_layers"] = nl
    if task.get("train_day_stride"):
        cfg_kw["train_day_stride"] = task["train_day_stride"]
    if task.get("dump_rl"):
        cfg_kw["dump_rl"] = True
    if task.get("pooled"):
        cfg_kw["pooled"] = True
        cfg_kw["train_symbols"] = tuple(task.get("train_symbols", ALL_SYMS_POOL))
        cfg_kw["eval_symbol"] = task.get("eval_symbol", "LTC-USDT-PERP")
        cfg_kw["split_date"] = task.get("split_date", "2025-12-10")
    if task.get("profit3"):                    # LEGACY REPRO: 3-class UP/DN/FL profitability
        cfg_kw["profit3"] = True
        cfg_kw["target_spec"] = None           # skip binary relabel (fast cached labels, unused)
        cfg_kw["target_name"] = "profit3"
        cfg_kw["p3_H"] = int(task.get("p3_H", 130))
        cfg_kw["p3_tp"] = float(task.get("p3_tp", 0.20))
        cfg_kw["p3_sl"] = float(task.get("p3_sl", 0.10))
        if task.get("L"):
            cfg_kw["L"] = int(task["L"])       # CONTEXT variable (short-context variant)
    cfg = T.FullCfg(**cfg_kw)
    out, preds = T.train_cell(MNT, cfg, log=lambda s: print(tag, s))
    if cfg.profit3:                            # LEGACY REPRO summary (prec_NF vs base)
        flat = {"tag": tag, "sym": sym, "profit3": True, "gpu": gpu,
                "p3_H": cfg.p3_H, "p3_tp": cfg.p3_tp, "p3_sl": cfg.p3_sl, "L": cfg.L,
                "prec_nf": out.get("prec_nf"), "coverage": out.get("coverage"),
                "dir_acc_all": out.get("dir_acc_all"),
                "dir_acc_committed": out.get("dir_acc_committed"),
                "base_nonfl": out.get("base_nonfl"), "q_commit_nonfl": out.get("q_commit_nonfl"),
                "base_up": out.get("base_up"), "base_dn": out.get("base_dn"),
                "base_fl": out.get("base_fl"), "n_nf": out.get("n_nf"),
                "n_dir": out.get("n_dir"), "n_eval": out.get("n_eval"),
                "elapsed_s": out.get("elapsed_s")}
        rec = {**out, "flat": flat}
        tmp = rpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, default=float)
        os.replace(tmp, rpath)
        VOL.commit()
        print(tag, "PROFIT3_RESULT " + json.dumps(flat, default=float))
        return flat
    # economic guard inputs: top-decile |move| vs cost, per (H) on test
    flat = {"tag": tag, "target": tname, "loss": (loss or "R1"), "sym": sym,
            "dropout": cfg.dropout, "wd": cfg.wd, "lr": cfg.lr,
            "batch_periods": cfg.batch_periods, "gpu": gpu,
            "dt_min": cfg.dt_min, "dt_max": cfg.dt_max,
            "a_init": [cfg.a_init_low, cfg.a_init_high], "d_conv": cfg.d_conv,
            "elapsed_s": out.get("elapsed_s"),
            "by_H": {H: {"rank_ic": out["by_H"][H]["all"].get("rank_ic"),
                         "auc": out["by_H"][H]["all"].get("auc"),
                         "n": out["by_H"][H]["all"].get("n"),
                         "top_absmove": out["by_H"][H]["all"].get("top_decile_absmove"),
                         "econ_pass": out["by_H"][H]["all"].get("econ_pass"),
                         "cap_edge_gross": out["by_H"][H]["all"].get("cap_edge_gross"),
                         "cap_edge_net": out["by_H"][H]["all"].get("cap_edge_net")}
                     for H in HS}}
    rec = {**out, "flat": flat}
    tmp = rpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, default=float)
    os.replace(tmp, rpath)
    VOL.commit()
    print(tag, "OBJ_RESULT " + json.dumps(flat["by_H"], default=float))
    return flat


@app.function(image=IMG, gpu="L4", timeout=5400, volumes={MNT: VOL}, retries=2)
def train_obj(task):
    return _impl(task)


@app.function(image=IMG, gpu="H100", timeout=5400, volumes={MNT: VOL}, retries=2,
              memory=98304)
def train_h100(task):
    return _impl(task)


@app.function(image=IMG, gpu="A100-40GB", timeout=10800, volumes={MNT: VOL},
              retries=2, memory=49152)
def train_a100(task):
    # Fallback GPU when H100 capacity is unavailable (Modal-side scarcity). GPU
    # choice is selection-invariant (cap_edge depends on the trained model, not
    # the device; established in the bench round). Lower memory (48GB vs 96GB)
    # + A100 (sm_80, well-supported by the prebuilt cu122 wheels) schedule far
    # easier; longer timeout for A100's slower per-step at L=216000.
    return _impl(task)


@app.function(image=IMG, gpu=["H100", "A100-80GB", "A100-40GB"], timeout=10800,
              volumes={MNT: VOL}, retries=2, memory=49152)
def train_big(task):
    # Capacity rounds incl. LARGER models (more layers / d_model): PREFER H100
    # (fast for big models), Modal falls back through A100-80/40GB in list order
    # when H100 capacity is unavailable ("H100 if available, else A100"). GPU
    # choice is selection-invariant (bench). 48GB host RAM (real need <16GB) +
    # the fallback list schedule far easier than the old H100/96GB combo.
    return _impl(task)


@app.local_entrypoint()
def main(full: bool = False, confirm: bool = False, r2: bool = False,
         r2validate: bool = False, rescore: bool = False, r2confirm: bool = False,
         reg: bool = False, bench: bool = False, tsweep: bool = False,
         tsvalidate: bool = False, r4confirm: bool = False, dconvsweep: bool = False,
         poolvalidate: bool = False, poolrun: bool = False,
         capvalidate: bool = False, capsweep: bool = False,
         capconfirm: bool = False, nlsweep: bool = False,
         stride5: bool = False, regpass: bool = False,
         regconfirm: bool = False, dumprl: bool = False,
         profit3: bool = False, profit3short: bool = False):
    import sys, json
    sys.path.insert(0, str(REPO / "scripts"))
    import hd2_targets as TGT
    import hd2_losses as LOSS
    variants = list(TGT.VARIANTS.keys())
    if rescore:
        # eval-only re-score of the finished R2 ckpts to add the magnitude-aware
        # cap_edge metric (signed captured return in the top-confidence decile,
        # net of cost) -- the winner-selection criterion (user 2026-05-24). No
        # retraining; loads each ckpt, forwards on OOS test. -> *.rescore.json.
        losses = list(LOSS.VARIANTS.keys())
        tasks = [{"sym": s, "target": "terminal", "loss": l, "seed": 0,
                  "epochs": 10, "eval_only": True} for l in losses for s in SYMS]
        handles = [train_obj.spawn(t) for t in tasks]
        print(f"R2 RESCORE SPAWNED {len(handles)} eval-only units (detached -> "
              f"Volume /cache/results/hd2_r2/*.rescore.json):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  R2_{t['loss']}__terminal__{t['sym']}_s0")
        return
    if r2confirm:
        # seed-stability confirm of the R2 winner (IC) vs incumbent R1, by the
        # magnitude-aware cap_edge_gross (captured signed alpha; cost is a
        # SEPARATE annotation, not the HD2 metric). Fresh training -> native
        # cap_edge on the best-val model. {IC,R1} x {SOL,LTC} x seeds {0,1,2}.
        losses = ["IC", "R1"]
        tasks = [{"sym": s, "target": "terminal", "loss": l, "seed": sd,
                  "epochs": 10, "confirm": True}
                 for l in losses for s in SYMS for sd in (0, 1, 2)]
        handles = [train_obj.spawn(t) for t in tasks]
        print(f"R2 CONFIRM SPAWNED {len(handles)} units (detached -> Volume "
              f"/cache/results/hd2_r2_confirm):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  R2_{t['loss']}__terminal__{t['sym']}_s{t['seed']}")
        return
    if reg:
        # ROUND 3 (rev11): regularization dropout x wd screen. Objective IC frozen
        # (rev10), target r_H, L frozen. Select by mean cap_edge_gross. seed0 screen
        # -> then r3 confirm winner vs default (0.1,1e-3) on 3 seeds.
        drops = [0.0, 0.1, 0.2]; wds = [1e-4, 1e-3, 1e-2]
        tasks = [{"sym": s, "target": "terminal", "loss": "IC", "seed": 0,
                  "epochs": 10, "round": "r3", "dropout": d, "wd": w}
                 for d in drops for w in wds for s in SYMS]
        handles = [train_obj.spawn(t) for t in tasks]
        print(f"R3 REG SCREEN SPAWNED {len(handles)} units (detached -> "
              f"Volume /cache/results/hd2_r3):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  R3_d{t['dropout']}_wd{t['wd']}__{t['sym']}_s0")
        return
    if bench:
        # SPEED+VALIDITY bench (science/speed objective, not budget): same cell
        # (IC, terminal, d0.1/wd1e-3) on H100 at larger batch (+scaled LR). Compare
        # elapsed_s (speed) + cap_edge_gross (validity) vs the R3 baseline
        # (L4, bp=2, lr=1e-3, R3_d0.1_wd0.001). Pick fastest config whose cap_edge
        # matches baseline -> use for rounds 4+.
        cfgs = [(4, 1.4e-3), (8, 2e-3)]        # (batch_periods, lr) on H100
        tasks = [{"sym": s, "target": "terminal", "loss": "IC", "seed": 0,
                  "epochs": 10, "round": "bench", "gpu": "H100",
                  "batch_periods": bp_, "lr": lr_, "force": True}
                 for (bp_, lr_) in cfgs for s in SYMS]
        handles = [train_h100.spawn(t) for t in tasks]
        print(f"BENCH SPAWNED {len(handles)} H100 units (detached -> /cache/results/hd2_bench):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  BENCH_H100_bp{t['batch_periods']}_lr{t['lr']}__{t['sym']}")
        return
    if tsvalidate:
        # plumbing check: 1 unit, 2 epochs, NON-default timescales on H100.
        t = {"sym": "SOL-USDT-PERP", "target": "terminal", "loss": "IC", "seed": 0,
             "epochs": 2, "round": "r4", "gpu": "H100", "ts": (1e-4, 1e-2, 1, 4),
             "force": True}
        print("R4 VALIDATE:", t)
        print(json.dumps(train_h100.remote(t), indent=2, default=float))
        return
    if tsweep:
        # ROUND 4 (rev13): SSM timescales. IC objective + default reg (0.1,1e-3)
        # frozen; data-insensitive -> full 500d x 2sym. Sweep (dt_min,dt_max) x
        # A_init_range; select by mean cap_edge_gross. H100, bp=default(2).
        dts = [(1e-4, 1e-2), (1e-3, 1e-1), (1e-2, 1.0)]   # slow / default / fast
        As = [(1, 4), (1, 16), (1, 64)]                   # long / default / short memory
        tasks = [{"sym": s, "target": "terminal", "loss": "IC", "seed": 0,
                  "epochs": 10, "round": "r4", "gpu": "H100",
                  "ts": (dmn, dmx, alo, ahi)}
                 for (dmn, dmx) in dts for (alo, ahi) in As for s in SYMS]
        handles = [train_h100.spawn(t) for t in tasks]
        print(f"R4 TIMESCALE SWEEP SPAWNED {len(handles)} H100 units (detached -> "
              f"/cache/results/hd2_r4):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  R4_dt{t['ts'][0]}-{t['ts'][1]}_A{t['ts'][2]}-{t['ts'][3]}__{t['sym']}")
        return
    if dconvsweep:
        # ROUND 4b (rev15): local causal-conv width d_conv in {2,3,4}
        # (causal_conv1d-supported). IC + default reg/timescales frozen;
        # data-insensitive; screen-only (cheap) -> confirm only if clear winner. H100.
        widths = [2, 3, 4]
        tasks = [{"sym": s, "target": "terminal", "loss": "IC", "seed": 0,
                  "epochs": 10, "round": "r4b", "gpu": "H100", "dconv": w}
                 for w in widths for s in SYMS]
        handles = [train_h100.spawn(t) for t in tasks]
        print(f"R4b D_CONV SCREEN SPAWNED {len(handles)} H100 units (detached -> "
              f"/cache/results/hd2_r4b):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  R4B_dconv{t['dconv']}__{t['sym']}")
        return
    if r4confirm:
        # seed-stability confirm of the R4 winner (slow dt 1e-4..1e-2, A(1,16)) vs
        # the mamba default (1e-3..1e-1, A(1,16)), 3 seeds, native cap_edge. H100.
        cfgs = [(1e-4, 1e-2, 1, 16), (1e-3, 1e-1, 1, 16)]   # winner(slow) , default
        tasks = [{"sym": s, "target": "terminal", "loss": "IC", "seed": sd,
                  "epochs": 10, "round": "r4", "gpu": "H100", "confirm": True,
                  "ts": ts_}
                 for ts_ in cfgs for s in SYMS for sd in (0, 1, 2)]
        handles = [train_h100.spawn(t) for t in tasks]
        print(f"R4 CONFIRM SPAWNED {len(handles)} units (detached -> "
              f"/cache/results/hd2_r4_confirm):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  R4_dt{t['ts'][0]}-{t['ts'][1]}_A{t['ts'][2]}-{t['ts'][3]}__{t['sym']}_s{t['seed']}")
        return
    if poolrun:
        # DATA-SCALE comparison (rev17): all-8 pooled vs LTC-only pooled, both
        # eval LTC (>= split_date), 10 epochs -> does cross-symbol data lift LTC
        # captured alpha / direction skill? Fresh tags (no resume). H100, spawn.
        cfgs = [("all8", list(ALL_SYMS_POOL)), ("ltconly", ["LTC-USDT-PERP"])]
        tasks = [{"round": "pool", "pooled": True, "tag_extra": ex,
                  "train_symbols": syms, "eval_symbol": "LTC-USDT-PERP",
                  "loss": "IC", "target": "terminal", "sym": "LTC-USDT-PERP",
                  "seed": 0, "epochs": 10, "gpu": "H100"}
                 for (ex, syms) in cfgs]
        handles = [train_h100.spawn(t) for t in tasks]
        print(f"POOLRUN SPAWNED {len(handles)} (all8 vs ltconly, 10ep, eval LTC) -> "
              f"/cache/results/hd2_pool:")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  POOL_{t['tag_extra']}_s0")
        return
    if poolvalidate:
        # DATA-SCALE plumbing check: 1 pooled unit (train all 8, eval LTC), 2 ep, H100.
        t = {"round": "pool", "pooled": True, "tag_extra": "validate",
             "loss": "IC", "target": "terminal", "sym": "LTC-USDT-PERP",
             "seed": 0, "epochs": 2, "gpu": "H100", "force": True}
        print("POOL VALIDATE:", t)
        print(json.dumps(train_h100.remote(t), indent=2, default=float))
        return
    if capvalidate:
        # CAPACITY plumbing check: 1 pooled unit with a NON-default d_state (256)
        # to exercise the new d_state override end-to-end, 2 ep, H100. force.
        t = {"round": "cap", "pooled": True, "d_state": 256,
             "train_symbols": list(ALL_SYMS_POOL), "eval_symbol": "LTC-USDT-PERP",
             "loss": "IC", "target": "terminal", "sym": "LTC-USDT-PERP",
             "seed": 0, "epochs": 2, "gpu": "H100", "force": True}
        print("CAP VALIDATE:", t)
        print(json.dumps(train_h100.remote(t), indent=2, default=float))
        return
    if capsweep:
        # CAPACITY round 1 (rev20): d_state {64,128,256,512} on POOLED all-8
        # (train < split_date 2025-12-10, eval LTC >=). IC + frozen reg/timescales/
        # d_conv. Select by RELATIVE cap_edge_gross -- NO econ gate (rev19). screen
        # seed0 -> confirm winner on 3 seeds. H100, spawn+detach, idempotent -> poll.
        states = [64, 128, 256, 512]
        tasks = [{"round": "cap", "pooled": True, "d_state": ds_,
                  "train_symbols": list(ALL_SYMS_POOL), "eval_symbol": "LTC-USDT-PERP",
                  "loss": "IC", "target": "terminal", "sym": "LTC-USDT-PERP",
                  "seed": 0, "epochs": 10, "gpu": "H100"}
                 for ds_ in states]
        handles = [train_h100.spawn(t) for t in tasks]
        print(f"CAPSWEEP d_state SPAWNED {len(handles)} H100 units (detached -> "
              f"/cache/results/hd2_cap):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  CAP_ds{t['d_state']}_s0")
        return
    if capconfirm:
        # CAPACITY round 1 CONFIRM (rev20): screen winner d_state=256 vs default
        # 128 on 3 seeds, pooled all-8 (train < 2025-12-10, eval LTC >=), IC +
        # frozen reg/timescales/d_conv. Native cap_edge on best-val model. Select
        # by RELATIVE cap_edge_gross + seed-sd; NO econ gate (rev19). H100, spawn+detach.
        states = [256, 128]
        tasks = [{"round": "cap", "pooled": True, "d_state": ds_, "confirm": True,
                  "train_symbols": list(ALL_SYMS_POOL), "eval_symbol": "LTC-USDT-PERP",
                  "loss": "IC", "target": "terminal", "sym": "LTC-USDT-PERP",
                  "seed": sd, "epochs": 10, "gpu": "H100"}
                 for ds_ in states for sd in (0, 1, 2)]
        handles = [train_a100.spawn(t) for t in tasks]   # A100 fallback (H100 scarce)
        print(f"CAPCONFIRM SPAWNED {len(handles)} A100 units (detached -> "
              f"/cache/results/hd2_cap_confirm):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  CAP_ds{t['d_state']}_s{t['seed']}")
        return
    if nlsweep:
        # CAPACITY round 2 (rev22): n_layers {2,6,8} on POOLED all-8 (train <
        # 2025-12-10, eval LTC >=), d_state=256 carried in (rev21 provisional),
        # IC + frozen reg/timescales/d_conv. nl=4 point REUSES the d_state-screen
        # ds256 result (CAP_ds256_s0 = d_state256/n_layers4 = 0.0146) -> only run
        # {2,6,8} to save the last account. Select by RELATIVE cap_edge_gross, NO
        # econ gate (rev19). A100 (H100 scarce). screen seed0; spawn+detach, idempotent.
        layers = [2, 6, 8]
        tasks = [{"round": "cap", "pooled": True, "n_layers": nl_, "d_state": 256,
                  "train_symbols": list(ALL_SYMS_POOL), "eval_symbol": "LTC-USDT-PERP",
                  "loss": "IC", "target": "terminal", "sym": "LTC-USDT-PERP",
                  "seed": 0, "epochs": 10, "gpu": "H100"}
                 for nl_ in layers]
        handles = [train_big.spawn(t) for t in tasks]   # H100 pref, A100 fallback
        print(f"NLSWEEP n_layers SPAWNED {len(handles)} H100/A100 units (detached -> "
              f"/cache/results/hd2_cap; nl=4 reuses CAP_ds256_s0=0.0146):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  CAP_ds256_nl{t['n_layers']}_s0")
        return
    if stride5:
        # COMPUTE side-task (NOT alpha): does training on every 5th day (~5x cheaper:
        # 5x fewer L-period encodes) preserve cap_edge vs full days (rev18 LTC-only
        # +0.0158)? Validates a cheap training PROXY for future budget-bound runs.
        # LTC-only, defaults (d_state128/n_layers4), IC, eval FULL LTC for comparability.
        t = {"round": "pool", "pooled": True, "tag_extra": "LTConly_stride5",
             "train_symbols": ["LTC-USDT-PERP"], "eval_symbol": "LTC-USDT-PERP",
             "loss": "IC", "target": "terminal", "sym": "LTC-USDT-PERP",
             "seed": 0, "epochs": 10, "gpu": "H100", "train_day_stride": 5}
        h = train_big.spawn(t)
        print(f"STRIDE5 SPAWNED {h.object_id}  POOL_LTConly_stride5_s0 "
              f"(day-stride 5, eval full LTC; cf. stride1 +0.0158) -> /cache/results/hd2_pool")
        return
    if regpass:
        # rev25 reg re-pass: dropout x wd on LTC-only (n_layers=4, d_state=128 defaults,
        # STRIDE1 full days). Does stronger reg control the fast overfit (val peaks
        # ep0-2) and lift cap_edge vs baseline (0.1,1e-3)~+0.0158? screen seed0.
        grid = [(0.1, 1e-3), (0.3, 1e-3), (0.5, 1e-3), (0.3, 1e-2)]   # (dropout, wd)
        tasks = [{"round": "pool", "pooled": True, "tag_extra": f"reg_d{d}_wd{w}",
                  "train_symbols": ["LTC-USDT-PERP"], "eval_symbol": "LTC-USDT-PERP",
                  "loss": "IC", "target": "terminal", "sym": "LTC-USDT-PERP",
                  "seed": 0, "epochs": 10, "gpu": "H100", "dropout": d, "wd": w}
                 for (d, w) in grid]
        handles = [train_big.spawn(t) for t in tasks]
        print(f"REGPASS SPAWNED {len(handles)} LTC-only units (detached -> /cache/results/hd2_pool):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  POOL_{t['tag_extra']}_s0")
        return
    if regconfirm:
        # rev25 confirm: dropout 0.3 (screen winner 0.0167) vs 0.1 (baseline 0.0089)
        # on 3 seeds + dropout 0.4 (seed0, refine the 0.3-0.5 peak). LTC-only stride1.
        # seed0 of 0.3/0.1 already exist (screen) -> idempotent skip; seeds 1,2 + 0.4 run.
        cfgs = ([(0.3, 1e-3, sd) for sd in (0, 1, 2)]
                + [(0.1, 1e-3, sd) for sd in (0, 1, 2)]
                + [(0.4, 1e-3, 0)])
        tasks = [{"round": "pool", "pooled": True, "tag_extra": f"reg_d{d}_wd{w}",
                  "train_symbols": ["LTC-USDT-PERP"], "eval_symbol": "LTC-USDT-PERP",
                  "loss": "IC", "target": "terminal", "sym": "LTC-USDT-PERP",
                  "seed": sd, "epochs": 10, "gpu": "H100", "dropout": d, "wd": w}
                 for (d, w, sd) in cfgs]
        handles = [train_big.spawn(t) for t in tasks]
        print(f"REGCONFIRM SPAWNED {len(handles)} units (0.3/0.1 seed0 skip; -> /cache/results/hd2_pool):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  POOL_{t['tag_extra']}_s{t['seed']}")
        return
    if dumprl:
        # RL side-experiment data-prep: eval-only on the existing LTC d0.1 ckpt
        # (POOL_reg_d0.1_wd0.001_s0.ckpt) -> dump 10s (mid, Mamba logits) OOS series
        # to /cache/results/hd2_pool/POOL_reg_d0.1_wd0.001_s0.rlseries.npz.
        t = {"round": "pool", "pooled": True, "tag_extra": "reg_d0.1_wd0.001",
             "train_symbols": ["LTC-USDT-PERP"], "eval_symbol": "LTC-USDT-PERP",
             "loss": "IC", "target": "terminal", "sym": "LTC-USDT-PERP",
             "seed": 0, "epochs": 10, "gpu": "H100", "dropout": 0.1, "wd": 1e-3,
             "eval_only": True, "dump_rl": True, "force": True}
        h = train_big.spawn(t)
        print(f"DUMPRL SPAWNED {h.object_id}  -> POOL_reg_d0.1_wd0.001_s0.rlseries.npz "
              f"(10s mid+logits, LTC OOS)")
        return
    if profit3:
        # LEGACY REPRO (user 2026-05-26): same HD2 streaming Mamba + raw 80-ch LOB
        # + L, but TARGET swapped binary sign(r_H) -> 3-class UP/DN/FL PROFITABILITY
        # (old-repo formulation that gave standalone Mamba prec_NF~0.33 = ~1.6x base)
        # + weighted CE + prec_NF metric. Isolates the binary->3-class variable.
        # LTC single-symbol, honest 70/30. H=130s, TP0.20/SL0.10 (-> ~20/20/60 base).
        t = {"profit3": True, "sym": "LTC-USDT-PERP", "seed": 0, "epochs": 12,
             "gpu": "H100", "p3_H": 130, "p3_tp": 0.20, "p3_sl": 0.10, "force": True}
        h = train_big.spawn(t)
        print(f"PROFIT3 SPAWNED {h.object_id}  -> /cache/results/hd2_profit3/"
              f"P3_L{L_FROZEN}_H130_tp0.2_sl0.1__LTC-USDT-PERP_s0.json  (prec_NF vs ~0.20 coin base)")
        return
    if profit3short:
        # CONTEXT/ARCH variable: profit3 at SHORT L=6000 (~old short-window regime)
        # vs the L=216000 run. Tests whether long streaming context dilutes the
        # short-horizon tradeable signal (the other remaining variable after target).
        t = {"profit3": True, "sym": "LTC-USDT-PERP", "seed": 0, "epochs": 12,
             "gpu": "H100", "p3_H": 130, "p3_tp": 0.20, "p3_sl": 0.10,
             "L": 6000, "force": True}
        h = train_big.spawn(t)
        print(f"PROFIT3SHORT SPAWNED {h.object_id}  L=6000 -> /cache/results/hd2_profit3/"
              f"P3_L6000_H130_tp0.2_sl0.1__LTC-USDT-PERP_s0.json")
        return
    if r2validate:
        # target="terminal" exercises the exact midts-relabel path the full r2
        # run uses (continuous r_H), so this validates streams+midts+secret+train.
        t = {"sym": "SOL-USDT-PERP", "target": "terminal", "loss": "IC",
             "seed": 0, "epochs": 2, "force": True}
        print("R2 VALIDATE:", t)
        print(json.dumps(train_obj.remote(t), indent=2, default=float))
        return
    if r2:
        # TARGET = continuous unbounded r_H ("terminal"), NOT a barrier (rev8
        # correction: alpha-search is execution-neutral, max-alpha; barrier was
        # a capped/execution-laden detour). rank-IC vs r_H is the program metric.
        # SPAWN (not map): preemption-robust. With `modal run --detach` the app
        # outlives this entrypoint and the spawned calls run server-side, even if
        # the local caller disconnects (map() calls would be canceled). train_obj
        # is idempotent + writes each result JSON to the Volume -> poll the Volume.
        losses = list(LOSS.VARIANTS.keys())
        tasks = [{"sym": s, "target": "terminal", "loss": l, "seed": 0, "epochs": 10}
                 for l in losses for s in SYMS]
        handles = [train_obj.spawn(t) for t in tasks]
        print(f"R2 SPAWNED {len(handles)} units (detached; results -> Volume "
              f"/cache/results/hd2_r2):")
        for t, h in zip(tasks, handles):
            print(f"  {h.object_id}  R2_{t['loss']}__terminal__{t['sym']}_s0")
        return
    if confirm:
        # seed-stability confirm of the barrier-round top candidates
        forms = ["fp_0.30", "fp_0.13"]
        tasks = [{"sym": s, "target": v, "seed": sd, "epochs": 10}
                 for v in forms for s in SYMS for sd in (1, 2)]
        print(f"CONFIRM: {len(tasks)} units ({forms} x seeds 1,2)")
        res = list(train_obj.map(tasks, order_outputs=False))
        print("CONFIRM_DONE " + json.dumps(res, default=float))
        return
    if not full:
        t = {"sym": "SOL-USDT-PERP", "target": "fp_0.13", "seed": 0,
             "epochs": 2, "force": True}
        print("VALIDATE relabel:", t)
        print(json.dumps(train_obj.remote(t), indent=2, default=float))
        print(f"(ok; {len(variants)} variants x {len(SYMS)} sym = "
              f"{len(variants)*len(SYMS)} screen units for --full)")
        return
    tasks = [{"sym": s, "target": v, "seed": 0, "epochs": 10}
             for v in variants for s in SYMS]
    print(f"SCREEN: {len(tasks)} units ({len(variants)} forms x {len(SYMS)} sym)")
    res = list(train_obj.map(tasks, order_outputs=False))
    print("OBJSWEEP_DONE " + json.dumps(res, default=float))
