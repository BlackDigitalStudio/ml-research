"""EV-C (HDATA cell EV-C): scheduled MACRO event-window reaction on BTC/ETH.
Free historical consensus/datetimes are paywalled or network-blocked from here
(FRED times out; ForexFactory JSON = current week only; DeFiLlama emissions = 402;
token.unlocks = no open API). So macro release DATETIMES are hardcoded best-effort
(FOMC 14:00 ET; CPI & NFP 08:30 ET; America/New_York -> UTC via zoneinfo, DST-correct)
and DIRECTION is tested WITHOUT consensus via continuation (does the initial impulse
persist?) + vol amplification vs same-symbol random-time baseline. Token unlocks +
consensus-surprise DEFERRED (paywalled) -> forward-log path (FF) noted in plan.

CAVEAT: hardcoded dates may have minor errors (esp 2025-26 CPI/NFP); a wrong date
samples a non-event minute -> dilutes toward null (conservative, not false-positive).
Small n (FOMC~17, CPI~24, NFP~24)."""
import json
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.fs as pafs
from google.cloud import storage
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJ = "project-26a24ad0-1059-4f73-93b"
BUCKET = "blackdigital-scalper-data"
KLINES = "free_v1/klines_1m/"
COST = 0.0013
HS = [5, 15, 30, 60]
MOM_K = [5, 15]
TOL_MS = 120_000
MIN_MS = 60_000
ET = ZoneInfo("America/New_York")
rng = np.random.default_rng(0)

FOMC = [(2024, 5, 1), (2024, 6, 12), (2024, 7, 31), (2024, 9, 18), (2024, 11, 7), (2024, 12, 18),
        (2025, 1, 29), (2025, 3, 19), (2025, 5, 7), (2025, 6, 18), (2025, 7, 30), (2025, 9, 17), (2025, 10, 29), (2025, 12, 10),
        (2026, 1, 28), (2026, 3, 18), (2026, 4, 29)]  # 14:00 ET announcement
CPI = [(2024, 5, 15), (2024, 6, 12), (2024, 7, 11), (2024, 8, 14), (2024, 9, 11), (2024, 10, 10), (2024, 11, 13), (2024, 12, 11),
       (2025, 1, 15), (2025, 2, 12), (2025, 3, 12), (2025, 4, 10), (2025, 5, 13), (2025, 6, 11), (2025, 7, 15), (2025, 8, 12), (2025, 9, 11), (2025, 10, 15), (2025, 11, 13), (2025, 12, 10),
       (2026, 1, 13), (2026, 2, 11), (2026, 3, 11), (2026, 4, 10)]  # 08:30 ET
NFP = [(2024, 5, 3), (2024, 6, 7), (2024, 7, 5), (2024, 8, 2), (2024, 9, 6), (2024, 10, 4), (2024, 11, 1), (2024, 12, 6),
       (2025, 1, 10), (2025, 2, 7), (2025, 3, 7), (2025, 4, 4), (2025, 5, 2), (2025, 6, 6), (2025, 7, 3), (2025, 8, 1), (2025, 9, 5), (2025, 10, 3), (2025, 11, 7), (2025, 12, 5),
       (2026, 1, 9), (2026, 2, 6), (2026, 3, 6), (2026, 4, 3)]  # 08:30 ET


def et_ms(ymd, hh, mm):
    t = pd.Timestamp(dt.datetime(ymd[0], ymd[1], ymd[2], hh, mm), tz=ET).tz_convert("UTC")
    return int(t.value // 1_000_000)


EVENTS = ([("FOMC", et_ms(d, 14, 0)) for d in FOMC]
          + [("CPI", et_ms(d, 8, 30)) for d in CPI]
          + [("NFP", et_ms(d, 8, 30)) for d in NFP])
print(f"events: FOMC={len(FOMC)} CPI={len(CPI)} NFP={len(NFP)} total={len(EVENTS)}", flush=True)

fs = pafs.GcsFileSystem()


def load_close(sym):
    pf = pq.ParquetFile(fs.open_input_file(f"{BUCKET}/{KLINES}{sym}.parquet"))
    t = pf.read(columns=["open_time", "close"])
    ot = pd.to_numeric(pd.Series(t.column("open_time").to_numpy()), errors="coerce").to_numpy()
    cl = pd.to_numeric(pd.Series(t.column("close").to_numpy()), errors="coerce").to_numpy()
    m = np.isfinite(ot) & np.isfinite(cl)
    ot, cl = ot[m], cl[m]
    o = np.argsort(ot, kind="stable")
    ot, cl = ot[o], cl[o]
    uq = np.concatenate(([True], np.diff(ot) > 0))
    return ot[uq], cl[uq]


def cl_at(ot, cl, tt):
    j = np.searchsorted(ot, tt, "left")
    if j < len(ot) and abs(ot[j] - tt) <= TOL_MS:
        return cl[j]
    if j > 0 and abs(ot[j - 1] - tt) <= TOL_MS:
        return cl[j - 1]
    return np.nan


def auc(score, label):
    label = np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def baseline_absmed(ot, cl, h, n=4000):
    lo, hi = ot[0], ot[-1] - h * MIN_MS
    tt = rng.integers(lo, hi, n)
    j0 = np.clip(np.searchsorted(ot, tt, "left"), 0, len(ot) - 1)
    jh = np.clip(np.searchsorted(ot, tt + h * MIN_MS, "left"), 0, len(ot) - 1)
    return float(np.median(np.abs(np.log(cl[jh] / cl[j0]))))


store = {}
for sym in ["BTCUSDT", "ETHUSDT"]:
    ot, cl = load_close(sym)
    print(f"\n######## {sym} (klines {len(ot)}) ########", flush=True)
    base = {h: baseline_absmed(ot, cl, h) for h in HS}
    for cls in ["FOMC", "CPI", "NFP", "ALL"]:
        evs = [t for c, t in EVENTS if cls == "ALL" or c == cls]
        # closes at offsets
        offs = sorted(set([0] + HS + MOM_K + [2 * k for k in MOM_K]))
        C = {o: np.array([cl_at(ot, cl, t + o * MIN_MS) for t in evs]) for o in offs}
        c0 = C[0]
        print(f"\n=== {sym} {cls} (n={len(evs)}) ===")
        print("  vol amplification (event med|r| vs random-time baseline):")
        d = {"n": len(evs), "vol": {}, "mom": {}}
        for h in HS:
            r = np.log(C[h] / c0)
            r = r[np.isfinite(r)]
            if len(r) == 0:
                continue
            amed = float(np.median(np.abs(r)))
            d["vol"][h] = {"event_absmed_pct": round(amed * 100, 3), "base_absmed_pct": round(base[h] * 100, 3),
                           "amp_x": round(amed / base[h], 2), "clear_cost": round(float(np.mean(np.abs(r) > COST)), 3), "n": len(r)}
            print(f"   h={h:>3}m event|r|={amed*100:.3f}% base={base[h]*100:.3f}% amp={amed/base[h]:.2f}x "
                  f"clear={d['vol'][h]['clear_cost']:.3f} n={len(r)}")
        print("  continuation rank_IC(early[0,k], next[k,2k]):")
        for k in MOM_K:
            early = np.log(C[k] / c0)
            nxt = np.log(C[2 * k] / C[k])
            ok = np.isfinite(early) & np.isfinite(nxt)
            if ok.sum() < 5:
                continue
            ric = auc(early[ok], nxt[ok] > 0) - 0.5
            dr = np.sign(early[ok]) * nxt[ok]
            d["mom"][k] = {"rank_IC": round(float(ric), 4), "dir_ret_bp": round(float(np.mean(dr) * 1e4), 1),
                           "win": round(float(np.mean(dr > 0)), 3), "n": int(ok.sum())}
            print(f"   k={k:>3}m rank_IC={ric:+.4f} dir_ret={np.mean(dr)*1e4:+.1f}bp win={np.mean(dr>0):.3f} n={int(ok.sum())}")
        store[f"{sym}/{cls}"] = d

json.dump(store, open("C:/Dev/sb-data-poc/ev_c_macro.json", "w"), indent=2)
print("\nEV_C_MACRO_DONE", flush=True)
