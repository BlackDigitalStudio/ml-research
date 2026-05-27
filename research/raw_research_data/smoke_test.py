import analysis as a
import numpy as np

sym = "BTC"
days = a.list_days(sym)
print("n_days listed:", len(days), "range:", days[0], "->", days[-1])
b = t = day = None
for d in reversed(days):
    try:
        b = a.read_book(sym, d)
        t = a.read_trades(sym, d)
        day = d
        break
    except Exception as e:
        print("skip", d, repr(e))
print("using day:", day)
print("book rows:", len(b), "trades rows:", len(t))
print("book lat_ms p50/p99:", np.nanpercentile(b["lat_ms"], [50, 99]))
print("trades lat_ms p50/p99:", np.nanpercentile(t["lat_ms"], [50, 99]))
print("spread_bp p50/p90:", np.nanpercentile(b["spread_bp"], [50, 90]))
g = a.grid_day(b, t)
print("grid shape:", g.shape, "mid NaN frac:", float(g["mid"].isna().mean()))
mid = g["mid"]
for h in [1, 30, 60]:
    k = max(1, round(h / a.STEP_S))
    r = (mid.shift(-k) / mid - 1.0) * 1e4
    print(f"  |ret| h={h}s  p50/p90/p99:", np.nanpercentile(np.abs(r.dropna()), [50, 90, 99]))
for h in [0.5, 5, 60]:
    k = max(1, round(h / a.STEP_S))
    fwd = mid.shift(-k) / mid - 1.0
    print(f"  IC obi/tfi h={h}s:", round(a.rank_ic(g['obi'], fwd), 4), round(a.rank_ic(g['tfi'], fwd), 4))
print("SMOKE_OK")
