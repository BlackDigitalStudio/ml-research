import pyarrow.parquet as pq
import numpy as np

dt = pq.read_table("data/depth/20260409_00.parquet").to_pandas()
print(f"Total rows: {len(dt)}")

jumps = []
for i in range(min(1000, len(dt) - 1)):
    r0 = dt.iloc[i]
    r1 = dt.iloc[i + 1]
    bid0 = r0.bids[0][0]
    bid1 = r1.bids[0][0]
    ask0 = r0.asks[0][0]
    ask1 = r1.asks[0][0]
    bid_ticks = abs(bid1 - bid0) / 0.10
    ask_ticks = abs(ask1 - ask0) / 0.10
    max_ticks = max(bid_ticks, ask_ticks)
    jumps.append(max_ticks)
    if max_ticks > 5:
        print(f"  Row {i}: bid ${bid0:.1f}->${bid1:.1f} ({bid_ticks:.0f}t)  ask ${ask0:.1f}->${ask1:.1f} ({ask_ticks:.0f}t)")

jumps = np.array(jumps)
print(f"\nJump stats (ticks, BTC tick=$0.10):")
print(f"  mean={jumps.mean():.1f}  median={np.median(jumps):.1f}  max={jumps.max():.0f}")
print(f"  >3 ticks: {(jumps > 3).sum()} / {len(jumps)} = {(jumps > 3).mean():.1%}")

# The trainer sweep_intensity uses $0.01 as tick size — wrong for BTC!
# BTC tick = $0.10, not $0.01
print(f"\nNote: trainer uses tick=$0.01 but BTC tick=$0.10")
print(f"  A $1 move = 10 ticks at $0.10, but 100 ticks at $0.01 → inflated sweep!")
