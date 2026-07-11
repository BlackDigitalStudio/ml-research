"""Check that recorded data parses correctly and trainer can build samples."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

data_dir = Path("data")

print("=== DEPTH FILES ===")
for f in sorted((data_dir / "depth").glob("*.parquet")):
    table = pq.read_table(f)
    df = table.to_pandas()
    size_kb = f.stat().st_size / 1024
    print(f"\n{f.name}: {len(df)} rows, {size_kb:.1f} KB")
    print(f"  Columns: {list(df.columns)}")
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    print(f"  Timestamps: {ts_min} -> {ts_max}")

    row = df.iloc[0]
    bids = row["bids"]
    asks = row["asks"]
    print(f"  bids: type={type(bids).__name__}, len={len(bids)}")
    if len(bids) > 0:
        print(f"    first entry: {bids[0]} (type={type(bids[0]).__name__})")
    asks_last = df.iloc[-1]["asks"]
    print(f"  last row asks: len={len(asks_last)}")

    nulls = df.isnull().sum()
    bad = nulls[nulls > 0]
    if len(bad):
        print(f"  NULLS FOUND: {bad.to_dict()}")
    else:
        print(f"  No nulls")

print("\n=== TRADE FILES ===")
for f in sorted((data_dir / "trades").glob("*.parquet")):
    df = pd.read_parquet(f)
    size_kb = f.stat().st_size / 1024
    print(f"\n{f.name}: {len(df)} rows, {size_kb:.1f} KB")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Timestamps: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"  Price: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"  Qty: {df['quantity'].min():.4f} - {df['quantity'].max():.4f}")
    print(f"  Seller-initiated: {df['is_buyer_maker'].mean():.1%}")
    nulls = df.isnull().sum()
    bad = nulls[nulls > 0]
    if len(bad):
        print(f"  NULLS FOUND: {bad.to_dict()}")
    else:
        print(f"  No nulls")

print("\n=== TRAINER SAMPLE BUILD ===")
from src.config import load_config
from src.trainer import Trainer

cfg = load_config("config.env")
trainer = Trainer(cfg)
try:
    depth_df = trainer.load_depth_data(hours=24)
    trade_df = trainer.load_trade_data(hours=24)
    print(f"  Depth rows: {len(depth_df)}")
    print(f"  Trade rows: {len(trade_df)}")

    X_lob, X_feat, y = trainer.build_samples(depth_df, trade_df)
    print(f"  X_lob:  {X_lob.shape}  dtype={X_lob.dtype}")
    print(f"  X_feat: {X_feat.shape}  dtype={X_feat.dtype}")
    print(f"  y:      {y.shape}  classes={np.bincount(y).tolist()}")

    nan_count = np.isnan(X_feat).sum()
    inf_count = np.isinf(X_feat).sum()
    print(f"  NaN in X_feat: {nan_count}")
    print(f"  Inf in X_feat: {inf_count}")

    if nan_count == 0 and inf_count == 0:
        print("  PASSED: data is clean")
    else:
        print("  WARNING: data has NaN/Inf!")

    # Show first sample features
    print(f"\n  Sample X_feat[0]:")
    from src.features import FEATURE_KEYS
    for i, key in enumerate(FEATURE_KEYS):
        print(f"    [{i:2d}] {key:30s} = {X_feat[0, i]:.6f}")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
