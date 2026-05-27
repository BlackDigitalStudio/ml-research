import json

d = json.load(open(r"C:\Dev\sub60s-hold-feasibility\results.json"))
syms = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "LINK", "LTC"]
HZ = [5, 15, 30, 60]

# Perfect foresight, always trade, always right direction:
# gross capture per trade = E[|r_h|]; net = E[|r_h|] - round_trip_cost.
print("E[|move|] (bp) by horizon  (perfect-direction gross capture per trade)")
print("sym   spread " + " ".join(f"E|r|{h:>2}".rjust(8) for h in HZ))
rows = []
for s in syms:
    r = d[s]
    sp = r["spread_bp"]["p50"]
    means = {h: r["abs_ret_bp"][str(h)]["mean"] for h in HZ}
    rows.append((s, sp, means))
    print(f"{s:5} {sp:6.3f} " + " ".join(f"{means[h]:8.2f}" for h in HZ))

for h in [30, 60]:
    print(f"\n-- h={h}s : E|r|  /  net maker(-4)  /  net taker(-10-spread)  (sorted) --")
    tbl = []
    for s, sp, means in rows:
        m = means[h]
        tbl.append((s, m, m - 4.0, m - 10.0 - sp))
    tbl.sort(key=lambda x: x[1], reverse=True)
    for s, m, nm, nt in tbl:
        print(f"{s:5}  E={m:6.2f}   maker {nm:+6.2f}   taker {nt:+6.2f}")
