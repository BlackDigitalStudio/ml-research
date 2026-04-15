//! scalper_ingest — Rust training pipeline (depth parse + features + live_sim).
//!
//! Session 1 scope: flat-schema depth parquet reader. Features + live_sim
//! land in later sessions — see task list in repo and the multi-session plan
//! in the session handoff memory.
//!
//! Parquet schema expected (written by scripts/ingest_tardis.py):
//!
//!     timestamp:   Int64               (ms)
//!     bid_prices:  FixedSizeList<f64, 20>
//!     bid_qtys:    FixedSizeList<f64, 20>
//!     ask_prices:  FixedSizeList<f64, 20>
//!     ask_qtys:    FixedSizeList<f64, 20>
//!
//! Sides are pre-sorted: bids descending (highest first), asks ascending.
//! Padding is 0.0 for rows with fewer than 20 levels on a side.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, BooleanArray, FixedSizeListArray, Float64Array, Int64Array};
use ndarray::{Array1, Array2};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub const DEPTH_LEVELS: usize = 20;

pub mod features;
pub mod live_sim;

/// All depth data for a single parquet file, materialized as contiguous ndarrays.
/// Memory cost per day of data: ~845k rows × (8 + 4*160) bytes ≈ 540 MB — fits RAM.
pub struct DepthData {
    pub timestamps: Array1<i64>,          // (n,) ms
    pub bid_prices: Array2<f64>,          // (n, 20) highest→lowest, 0.0 pad
    pub bid_qtys: Array2<f64>,            // (n, 20)
    pub ask_prices: Array2<f64>,          // (n, 20) lowest→highest, 0.0 pad
    pub ask_qtys: Array2<f64>,            // (n, 20)
}

impl DepthData {
    pub fn n_rows(&self) -> usize {
        self.timestamps.len()
    }

    /// mid = (best_bid + best_ask) / 2 as Array1<f64>, shape (n,).
    /// Rows with either side empty (0.0) get mid = 0.0 — caller must filter.
    pub fn mid_prices(&self) -> Array1<f64> {
        let n = self.n_rows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let bb = self.bid_prices[[i, 0]];
            let aa = self.ask_prices[[i, 0]];
            if bb > 0.0 && aa > 0.0 {
                out[i] = 0.5 * (bb + aa);
            }
        }
        out
    }
}

/// Load a flat-schema depth parquet into contiguous ndarrays.
pub fn read_depth_parquet(path: &Path) -> Result<DepthData> {
    let file = File::open(path).with_context(|| format!("open {:?}", path))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    // First pass: collect all batches; second pass: concat. We collect first
    // because we need the total row count to preallocate ndarrays.
    let mut batches = Vec::new();
    let mut total_rows: usize = 0;
    for b in reader {
        let b = b?;
        total_rows += b.num_rows();
        batches.push(b);
    }
    if batches.is_empty() {
        return Err(anyhow!("parquet has no batches: {:?}", path));
    }

    let schema = batches[0].schema();
    let idx = |name: &str| -> Result<usize> {
        schema
            .index_of(name)
            .with_context(|| format!("column {} missing", name))
    };
    let ts_idx = idx("timestamp")?;
    let bp_idx = idx("bid_prices")?;
    let bq_idx = idx("bid_qtys")?;
    let ap_idx = idx("ask_prices")?;
    let aq_idx = idx("ask_qtys")?;

    let mut timestamps = Array1::<i64>::zeros(total_rows);
    let mut bid_prices = Array2::<f64>::zeros((total_rows, DEPTH_LEVELS));
    let mut bid_qtys = Array2::<f64>::zeros((total_rows, DEPTH_LEVELS));
    let mut ask_prices = Array2::<f64>::zeros((total_rows, DEPTH_LEVELS));
    let mut ask_qtys = Array2::<f64>::zeros((total_rows, DEPTH_LEVELS));

    let mut off: usize = 0;
    for b in batches {
        let n = b.num_rows();

        let ts = b
            .column(ts_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow!("timestamp not Int64"))?;
        for i in 0..n {
            timestamps[off + i] = ts.value(i);
        }

        copy_fsl_into(b.column(bp_idx), &mut bid_prices, off, n, "bid_prices")?;
        copy_fsl_into(b.column(bq_idx), &mut bid_qtys, off, n, "bid_qtys")?;
        copy_fsl_into(b.column(ap_idx), &mut ask_prices, off, n, "ask_prices")?;
        copy_fsl_into(b.column(aq_idx), &mut ask_qtys, off, n, "ask_qtys")?;

        off += n;
    }

    Ok(DepthData {
        timestamps,
        bid_prices,
        bid_qtys,
        ask_prices,
        ask_qtys,
    })
}

/// Copy a FixedSizeList<f64, 20> column into rows [off..off+n] of `dst`.
/// Uses the underlying Float64 buffer directly — one memcpy per batch.
fn copy_fsl_into(
    col: &Arc<dyn Array>,
    dst: &mut Array2<f64>,
    off: usize,
    n: usize,
    name: &str,
) -> Result<()> {
    let fsl = col
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| anyhow!("{} not FixedSizeList", name))?;
    if fsl.value_length() as usize != DEPTH_LEVELS {
        return Err(anyhow!(
            "{} has list size {}, expected {}",
            name,
            fsl.value_length(),
            DEPTH_LEVELS
        ));
    }
    let values = fsl
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| anyhow!("{} child not Float64", name))?;
    // FixedSizeList child is a flat (n * 20) Float64; offset accounts for sliced batches.
    let child_off = fsl.offset() * DEPTH_LEVELS;
    let slice = &values.values()[child_off..child_off + n * DEPTH_LEVELS];
    // dst is row-major (n, 20); rows [off..off+n] is contiguous in memory.
    let dst_slice = dst
        .as_slice_mut()
        .ok_or_else(|| anyhow!("dst not contiguous"))?;
    let row_bytes = DEPTH_LEVELS;
    dst_slice[off * row_bytes..(off + n) * row_bytes].copy_from_slice(slice);
    Ok(())
}

/// Trades for feature-calc side: timestamps + quantities + is_sell flag.
/// `is_sell[i] = true` when aggressor was seller (matches Python
/// trade_side convention: `cum_buy = cumsum(qty * ~side)`).
/// For Binance trades parquet this is the `is_buyer_maker` column directly.
pub struct TradesData {
    pub timestamps: Array1<i64>,
    pub prices: Array1<f64>,
    pub quantities: Array1<f64>,
    pub is_sell: Vec<bool>,
}

impl TradesData {
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Load a trades parquet.
///
/// Expected columns:
///   timestamp: Int64 (ms)
///   quantity:  Float64
///   is_buyer_maker: Bool  (taker was seller → this trade is a sell)
pub fn read_trades_parquet(path: &Path) -> Result<TradesData> {
    let file = File::open(path).with_context(|| format!("open {:?}", path))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut batches = Vec::new();
    let mut total: usize = 0;
    for b in reader {
        let b = b?;
        total += b.num_rows();
        batches.push(b);
    }
    if batches.is_empty() {
        return Ok(TradesData {
            timestamps: Array1::zeros(0),
            prices: Array1::zeros(0),
            quantities: Array1::zeros(0),
            is_sell: Vec::new(),
        });
    }

    let schema = batches[0].schema();
    let idx = |name: &str| -> Result<usize> {
        schema
            .index_of(name)
            .with_context(|| format!("trades column {} missing", name))
    };
    let ts_idx = idx("timestamp")?;
    let price_idx = idx("price")?;
    let qty_idx = idx("quantity")?;
    let side_idx = idx("is_buyer_maker")?;

    let mut ts = Array1::<i64>::zeros(total);
    let mut price = Array1::<f64>::zeros(total);
    let mut qty = Array1::<f64>::zeros(total);
    let mut is_sell = vec![false; total];

    let mut off = 0usize;
    for b in batches {
        let n = b.num_rows();

        let ts_col = b
            .column(ts_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow!("trades.timestamp not Int64"))?;
        let price_col = b
            .column(price_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow!("trades.price not Float64"))?;
        let qty_col = b
            .column(qty_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow!("trades.quantity not Float64"))?;
        let side_col = b
            .column(side_idx)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| anyhow!("trades.is_buyer_maker not Bool"))?;

        ts.as_slice_mut().unwrap()[off..off + n]
            .copy_from_slice(&ts_col.values()[..n]);
        price.as_slice_mut().unwrap()[off..off + n]
            .copy_from_slice(&price_col.values()[..n]);
        qty.as_slice_mut().unwrap()[off..off + n]
            .copy_from_slice(&qty_col.values()[..n]);
        for i in 0..n {
            is_sell[off + i] = side_col.value(i);
        }
        off += n;
    }

    Ok(TradesData {
        timestamps: ts,
        prices: price,
        quantities: qty,
        is_sell,
    })
}

/// Funding parquet — schema: {timestamp:i64, funding_rate:f64, mark_price:f64}.
/// `mark_price` is loaded for the horizon-tier basis feature (col 44).
pub struct FundingData {
    pub timestamps: Array1<i64>,
    pub funding_rate: Array1<f64>,
    pub mark_price: Array1<f64>,
}

pub fn read_funding_parquet(path: &Path) -> Result<FundingData> {
    let (ts, cols) = load_scalar_parquet(path, &["funding_rate", "mark_price"])?;
    let mut it = cols.into_iter();
    Ok(FundingData {
        timestamps: ts,
        funding_rate: it.next().unwrap(),
        mark_price: it.next().unwrap(),
    })
}

/// Derivatives parquet — schema: {timestamp:i64, open_interest:f64, long_short_ratio:f64}.
pub struct DerivativesData {
    pub timestamps: Array1<i64>,
    pub open_interest: Array1<f64>,
    pub long_short_ratio: Array1<f64>,
}

pub fn read_derivatives_parquet(path: &Path) -> Result<DerivativesData> {
    let (ts, cols) = load_scalar_parquet(path, &["open_interest", "long_short_ratio"])?;
    let mut it = cols.into_iter();
    Ok(DerivativesData {
        timestamps: ts,
        open_interest: it.next().unwrap(),
        long_short_ratio: it.next().unwrap(),
    })
}

/// Cross-exchange trades: {ts, signed_qty} where signed_qty = +qty for buy,
/// -qty for sell, matching Python trainer semantics. For `exchange == "gateio"`
/// the raw quantity is taken as abs (see trainer.py comment: Gate.io stores
/// signed contract size in `quantity`; feature 30 only cares about the sign
/// of the net sum, so we use `is_seller` to determine sign and |qty| for mag).
pub struct CrossExTrades {
    pub timestamps: Array1<i64>,
    pub signed_qty: Array1<f64>,
}

pub fn read_cross_ex_parquet(path: &Path, exchange: &str) -> Result<CrossExTrades> {
    let file = File::open(path).with_context(|| format!("open {:?}", path))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut batches = Vec::new();
    let mut total: usize = 0;
    for b in reader {
        let b = b?;
        total += b.num_rows();
        batches.push(b);
    }
    if batches.is_empty() {
        return Ok(CrossExTrades {
            timestamps: Array1::zeros(0),
            signed_qty: Array1::zeros(0),
        });
    }
    let schema = batches[0].schema();
    let ts_idx = schema
        .index_of("timestamp")
        .context("cross-ex missing timestamp")?;
    let qty_idx = schema
        .index_of("quantity")
        .context("cross-ex missing quantity")?;
    let seller_idx = schema
        .index_of("is_seller")
        .context("cross-ex missing is_seller")?;

    let gateio = exchange == "gateio";
    let mut ts = Array1::<i64>::zeros(total);
    let mut sq = Array1::<f64>::zeros(total);

    let mut off = 0usize;
    for b in batches {
        let n = b.num_rows();
        let ts_col = b
            .column(ts_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow!("cross-ex.ts not Int64"))?;
        let qty_col = b
            .column(qty_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow!("cross-ex.qty not Float64"))?;
        let sel_col = b
            .column(seller_idx)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| anyhow!("cross-ex.is_seller not Bool"))?;
        ts.as_slice_mut().unwrap()[off..off + n].copy_from_slice(&ts_col.values()[..n]);
        let qv = &qty_col.values()[..n];
        let dst = &mut sq.as_slice_mut().unwrap()[off..off + n];
        for i in 0..n {
            let raw = if gateio { qv[i].abs() } else { qv[i] };
            dst[i] = if sel_col.value(i) { -raw } else { raw };
        }
        off += n;
    }
    Ok(CrossExTrades {
        timestamps: ts,
        signed_qty: sq,
    })
}

fn load_scalar_parquet(path: &Path, f64_cols: &[&str]) -> Result<(Array1<i64>, Vec<Array1<f64>>)> {
    let file = File::open(path).with_context(|| format!("open {:?}", path))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut batches = Vec::new();
    let mut total: usize = 0;
    for b in reader {
        let b = b?;
        total += b.num_rows();
        batches.push(b);
    }
    if batches.is_empty() {
        return Ok((Array1::zeros(0), vec![Array1::zeros(0); f64_cols.len()]));
    }
    let schema = batches[0].schema();
    let ts_idx = schema
        .index_of("timestamp")
        .context("scalar parquet missing timestamp")?;
    let f64_idxs: Vec<usize> = f64_cols
        .iter()
        .map(|n| {
            schema
                .index_of(n)
                .with_context(|| format!("scalar parquet missing {}", n))
        })
        .collect::<Result<_>>()?;

    let mut ts = Array1::<i64>::zeros(total);
    let mut outs: Vec<Array1<f64>> = f64_cols.iter().map(|_| Array1::zeros(total)).collect();

    let mut off = 0usize;
    for b in batches {
        let n = b.num_rows();
        let ts_col = b
            .column(ts_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow!("timestamp not Int64"))?;
        ts.as_slice_mut().unwrap()[off..off + n].copy_from_slice(&ts_col.values()[..n]);
        for (j, &ci) in f64_idxs.iter().enumerate() {
            let c = b
                .column(ci)
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| anyhow!("{} not Float64", f64_cols[j]))?;
            outs[j].as_slice_mut().unwrap()[off..off + n].copy_from_slice(&c.values()[..n]);
        }
        off += n;
    }
    Ok((ts, outs))
}
