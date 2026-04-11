"""Deep data quality check: distributions, outliers, snapshot integrity."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

data_dir = Path("data")

# ============================================================
print("=" * 60)
print("  DEEP DATA QUALITY CHECK — BTCUSDT")
print("=" * 60)

# --- DEPTH ---
print("\n[1] DEPTH SNAPSHOT INTEGRITY")
depth_files = sorted((data_dir / "depth").glob("*.parquet"))
total_depth = 0
errors = 0
gaps = 0
prev_ts = None

for f in depth_files:
    df = pq.read_table(f).to_pandas()
    total_depth += len(df)

    for i, row in enumerate(df.itertuples()):
        bids = row.bids
        asks = row.asks
        ts = row.timestamp

        # Check 20 levels
        if len(bids) != 20:
            errors += 1
            if errors <= 3:
                print(f"  ERROR: {f.name} row {i}: bids has {len(bids)} levels (expected 20)")
        if len(asks) != 20:
            errors += 1
            if errors <= 3:
                print(f"  ERROR: {f.name} row {i}: asks has {len(asks)} levels (expected 20)")

        # Check bid > ask (no crossing)
        if len(bids) > 0 and len(asks) > 0:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            if best_bid >= best_ask:
                errors += 1
                if errors <= 3:
                    print(f"  ERROR: {f.name} row {i}: crossed book bid={best_bid} >= ask={best_ask}")

        # Check bids sorted descending
        if len(bids) >= 2:
            prices = [b[0] for b in bids if b[1] > 0]
            if prices != sorted(prices, reverse=True):
                errors += 1
                if errors <= 3:
                    print(f"  ERROR: {f.name} row {i}: bids not sorted descending")

        # Check timestamp gaps
        if prev_ts is not None:
            gap_ms = ts - prev_ts
            if gap_ms > 500:  # > 500ms gap
                gaps += 1
            if gap_ms < 0:
                errors += 1
                if errors <= 3:
                    print(f"  ERROR: {f.name} row {i}: timestamp went backwards ({gap_ms}ms)")
        prev_ts = ts

print(f"  Total snapshots: {total_depth}")
print(f"  Level errors: {errors}")
print(f"  Gaps > 500ms: {gaps} ({gaps/max(total_depth,1)*100:.1f}%)")
print(f"  Result: {'PASS' if errors == 0 else 'FAIL'}")

# --- Spread distribution ---
print("\n[2] SPREAD DISTRIBUTION")
spreads = []
for f in depth_files:
    df = pq.read_table(f).to_pandas()
    for row in df.itertuples():
        s = row.asks[0][0] - row.bids[0][0]
        spreads.append(s)

spreads = np.array(spreads)
print(f"  Count: {len(spreads)}")
print(f"  Min: ${spreads.min():.2f}  Max: ${spreads.max():.2f}  Mean: ${spreads.mean():.2f}")
print(f"  Median: ${np.median(spreads):.2f}")
print(f"  $0.10 (1 tick): {(spreads == 0.1).sum() / len(spreads):.1%}")
print(f"  > $1.00: {(spreads > 1.0).sum()} ({(spreads > 1.0).sum() / len(spreads):.2%})")

# --- Price continuity ---
print("\n[3] PRICE CONTINUITY")
all_mids = []
for f in depth_files:
    df = pq.read_table(f).to_pandas()
    for row in df.itertuples():
        mid = (row.bids[0][0] + row.asks[0][0]) / 2
        all_mids.append(mid)

mids = np.array(all_mids)
returns = np.diff(mids) / mids[:-1]
big_jumps = np.abs(returns) > 0.001  # > 0.1% in one tick
print(f"  Total mid prices: {len(mids)}")
print(f"  Price range: ${mids.min():,.2f} — ${mids.max():,.2f}")
print(f"  Max single-tick return: {np.abs(returns).max()*100:.4f}%")
print(f"  Jumps > 0.1%: {big_jumps.sum()} ({big_jumps.sum()/len(returns)*100:.2f}%)")

# --- TRADES ---
print("\n[4] TRADES QUALITY")
trade_files = sorted((data_dir / "trades").glob("*.parquet"))
all_trades = pd.concat([pd.read_parquet(f) for f in trade_files]).sort_values("timestamp").reset_index(drop=True)
print(f"  Total trades: {len(all_trades)}")
print(f"  Price range: ${all_trades.price.min():,.2f} — ${all_trades.price.max():,.2f}")
print(f"  Qty: min={all_trades.quantity.min():.4f}  max={all_trades.quantity.max():.3f}  median={all_trades.quantity.median():.4f}")
print(f"  Seller-initiated: {all_trades.is_buyer_maker.mean():.1%}")
print(f"  Trades/second: {len(all_trades) / ((all_trades.timestamp.max() - all_trades.timestamp.min()) / 1000):.1f}")

# Check for duplicates
dupes = all_trades.duplicated(subset=["timestamp", "price", "quantity"]).sum()
print(f"  Duplicates: {dupes}")

# Check zero qty
zero_qty = (all_trades.quantity == 0).sum()
print(f"  Zero quantity: {zero_qty}")

# --- TRAINER SAMPLE QUALITY ---
print("\n[5] TRAINER SAMPLE QUALITY")
from src.config import load_config
from src.trainer import Trainer

cfg = load_config("config.env")
trainer = Trainer(cfg)
X_lob, X_feat, y, _mids, _target_pnl = trainer.build_samples(hours=24)

print(f"  Samples: {len(y)}")
print(f"  Classes: UP={int((y==0).sum())} DOWN={int((y==1).sum())} FLAT={int((y==2).sum())}")
print(f"  X_feat NaN: {np.isnan(X_feat).sum()}")
print(f"  X_feat Inf: {np.isinf(X_feat).sum()}")
print(f"  X_lob NaN: {np.isnan(X_lob).sum()}")
print(f"  X_lob zeros (all channels): {(X_lob.sum(axis=(1,2,3)) == 0).sum()}")

# Feature stats
from src.features import FEATURE_KEYS
print(f"\n  Feature distributions:")
for i, key in enumerate(FEATURE_KEYS):
    col = X_feat[:, i]
    nonzero = (col != 0).sum()
    print(f"    [{i:2d}] {key:30s}  min={col.min():12.4f}  max={col.max():12.4f}  nonzero={nonzero:6d}/{len(col)} ({nonzero/len(col)*100:5.1f}%)")

print(f"\n{'='*60}")
all_ok = errors == 0 and zero_qty == 0 and np.isnan(X_feat).sum() == 0
print(f"  OVERALL: {'ALL CHECKS PASSED' if all_ok else 'ISSUES FOUND'}")
print(f"{'='*60}")
