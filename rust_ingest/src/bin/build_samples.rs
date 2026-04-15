//! build_samples — assemble per-sample arrays for the training cache
//! without going through Python.
//!
//! Inputs:
//!   --depth D.parquet       (flat schema: ts + 4 FixedSizeList<f64,20>)
//!   --trades T.parquet      (ts, price, quantity, is_buyer_maker)
//!   --out-dir OUT           (directory where .npy files are written)
//!   --window W              (default 50)  LOB tensor window size in ticks
//!   --horizon H             (default 1300) forward sim_horizon in ticks
//!   --step S                (default 2)   between consecutive sample_starts
//!   --max-samples N         (default 2_000_000)  cap; auto-inflates step
//!
//! Outputs written into --out-dir (all in numpy .npy format):
//!   sample_starts.npy       (N,)           i64  — sample_starts for Python cache key
//!   end_indices.npy         (N,)           i64  — sample_starts + W - 1
//!   X_lob.npy               (N, 3, 20, W)  f32
//!   mid.npy                 (N,)           f64  — mid[end_indices]
//!   entry_long.npy          (N,)           f64  — bid[end_indices, 0]
//!   entry_short.npy         (N,)           f64  — ask[end_indices, 0]
//!   mid_paths.npy           (N, H)         f64  — mid[end_indices+1 .. end_indices+H+1]
//!   sample_ts.npy           (N,)           i64  — depth_ts[end_indices]
//!
//! Memory envelope: the depth parquet is memmap'd through pyarrow's Rust
//! `parquet` crate (row-group streaming). Peak RSS ≈ (n × 640 B) for the
//! raw depth arrays held by `DepthData` + X_lob in chunks. No 40 GB pyarrow
//! transient, no Python copies.
//!
//! Semantics parity with `src/trainer.py::build_samples`:
//!   * sample_starts = np.arange(0, n - W - H, step) capped by max_samples.
//!   * end_indices   = sample_starts + W - 1
//!   * X_lob[i, 0, :, :] = bid_qtys[sample_starts[i] .. sample_starts[i]+W, :20].T
//!   * X_lob[i, 1, :, :] = ask_qtys similarly
//!   * X_lob[i, 2, 0, :] = tick_buy_vol[sample_starts[i] .. +W]
//!   * X_lob[i, 2, 1, :] = tick_sell_vol[sample_starts[i] .. +W]
//!   * tick_buy_vol[t]  = sum of trade_qty where is_buyer_maker==false AND
//!                         depth_ts_idx(trade_ts) == t
//!   * tick_sell_vol analogous.
//!   * mid_paths[i] = mid[end_indices[i]+1 .. end_indices[i]+H+1]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::{s, Array1, Array2, Array4};
use ndarray_npy::write_npy;

use scalper_ingest::{read_depth_parquet, read_trades_parquet};

#[derive(Parser, Debug)]
#[command(about = "Build per-sample training cache arrays directly from parquets.")]
struct Args {
    #[arg(long)]
    depth: PathBuf,
    #[arg(long)]
    trades: Option<PathBuf>,
    #[arg(long, value_name = "OUT_DIR")]
    out_dir: PathBuf,
    #[arg(long, default_value_t = 50)]
    window: i64,
    #[arg(long, default_value_t = 1300)]
    horizon: i64,
    #[arg(long, default_value_t = 2)]
    step: i64,
    #[arg(long, default_value_t = 2_000_000)]
    max_samples: i64,
}

fn main() -> Result<()> {
    let a = Args::parse();
    std::fs::create_dir_all(&a.out_dir)
        .with_context(|| format!("mkdir {:?}", a.out_dir))?;

    let t0 = Instant::now();
    let depth = read_depth_parquet(&a.depth)?;
    let n = depth.n_rows();
    let t_depth = t0.elapsed();
    eprintln!("build_samples: depth loaded n={} in {:.2}s", n, t_depth.as_secs_f64());

    let w = a.window;
    let h = a.horizon;
    if n as i64 <= w + h + 1 {
        anyhow::bail!("depth has {} rows, need > {}", n, w + h + 1);
    }
    let total = n as i64 - w - h;

    // Auto-step: inflate to keep num_samples ≤ max_samples.
    let mut step = a.step.max(1);
    if total > a.max_samples * 2 {
        step = 2 * ((total + a.max_samples - 1) / a.max_samples);
        step = step.max(2);
    }
    let sample_starts: Vec<i64> = (0..total).step_by(step as usize).collect();
    let ns = sample_starts.len();
    let end_indices: Vec<i64> = sample_starts.iter().map(|&s| s + w - 1).collect();
    eprintln!(
        "build_samples: step={} num_samples={} (max={}, total={})",
        step, ns, a.max_samples, total
    );

    let depth_ts = depth.timestamps.as_slice().unwrap();
    let mid = depth.mid_prices();

    // --- tick_buy_vol / tick_sell_vol: map trades to depth ticks via searchsorted_right-1 ---
    let t1 = Instant::now();
    let mut tick_buy_vol = vec![0f32; n];
    let mut tick_sell_vol = vec![0f32; n];
    if let Some(trades_path) = a.trades.as_ref() {
        let trades = read_trades_parquet(trades_path)?;
        let nt = trades.len();
        let t_ts = trades.timestamps.as_slice().unwrap();
        let t_qty = trades.quantities.as_slice().unwrap();
        let is_sell = &trades.is_sell;
        for k in 0..nt {
            // t_idx = clip(searchsorted_right(depth_ts, trade_ts) - 1, 0, n-1)
            let r = match depth_ts.binary_search_by(|probe| {
                if probe <= &t_ts[k] { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater }
            }) {
                Ok(i) => i,
                Err(i) => i,
            };
            let idx = if r == 0 { 0 } else { (r - 1).min(n - 1) };
            if is_sell[k] {
                tick_sell_vol[idx] += t_qty[k] as f32;
            } else {
                tick_buy_vol[idx] += t_qty[k] as f32;
            }
        }
        eprintln!(
            "build_samples: {} trades aggregated onto {} depth ticks in {:.2}s",
            nt, n, t1.elapsed().as_secs_f64()
        );
    }

    // --- entry_long / entry_short / sample_mid / sample_ts ---
    let t2 = Instant::now();
    let bp = &depth.bid_prices;
    let ap = &depth.ask_prices;
    let mut entry_long = Array1::<f64>::zeros(ns);
    let mut entry_short = Array1::<f64>::zeros(ns);
    let mut mid_at_sample = Array1::<f64>::zeros(ns);
    let mut sample_ts = Array1::<i64>::zeros(ns);
    let mut top5_bid = Array1::<f64>::zeros(ns);
    let mut top5_ask = Array1::<f64>::zeros(ns);
    let bq = &depth.bid_qtys;
    let aq = &depth.ask_qtys;
    for (i, &ei) in end_indices.iter().enumerate() {
        let e = ei as usize;
        entry_long[i] = bp[[e, 0]];
        entry_short[i] = ap[[e, 0]];
        mid_at_sample[i] = mid[e];
        sample_ts[i] = depth_ts[e];
        let mut sb = 0.0;
        let mut sa = 0.0;
        for k in 0..5 {
            sb += bq[[e, k]];
            sa += aq[[e, k]];
        }
        top5_bid[i] = sb;
        top5_ask[i] = sa;
    }

    // --- mid_paths[i, k] = mid[end_indices[i] + 1 + k]  for k in 0..h ---
    let mut mid_paths = Array2::<f64>::zeros((ns, h as usize));
    for (i, &ei) in end_indices.iter().enumerate() {
        let e = ei as usize;
        let start = e + 1;
        let end = start + h as usize;
        for (k, src) in (start..end).enumerate() {
            mid_paths[[i, k]] = mid[src];
        }
    }
    eprintln!(
        "build_samples: entry/mid_paths assembled in {:.2}s",
        t2.elapsed().as_secs_f64()
    );

    // --- X_lob tensor: (N, 3, 20, W) f32 ---
    // Ch 0: bid_qtys[sample_starts[i] + offset, k] with offset in 0..W, k in 0..20.
    // Ch 1: ask_qtys similarly.
    // Ch 2: [0, offset] = tick_buy_vol[sample_start+offset]; [1, offset] = tick_sell_vol.
    let t3 = Instant::now();
    let mut x_lob = Array4::<f32>::zeros((ns, 3, 20, w as usize));
    // Vectorised fill by iterating samples outer, channels inner.
    for (i, &ss) in sample_starts.iter().enumerate() {
        let ss = ss as usize;
        for off in 0..w as usize {
            let row = ss + off;
            for k in 0..20 {
                x_lob[[i, 0, k, off]] = bq[[row, k]] as f32;
                x_lob[[i, 1, k, off]] = aq[[row, k]] as f32;
            }
            x_lob[[i, 2, 0, off]] = tick_buy_vol[row];
            x_lob[[i, 2, 1, off]] = tick_sell_vol[row];
        }
    }
    eprintln!(
        "build_samples: X_lob filled in {:.2}s",
        t3.elapsed().as_secs_f64()
    );

    // --- Write npy outputs ---
    let t4 = Instant::now();
    let sample_starts_arr = Array1::<i64>::from_vec(sample_starts.clone());
    let end_indices_arr = Array1::<i64>::from_vec(end_indices.clone());

    write_npy(a.out_dir.join("sample_starts.npy"), &sample_starts_arr)?;
    write_npy(a.out_dir.join("end_indices.npy"), &end_indices_arr)?;
    write_npy(a.out_dir.join("sample_ts.npy"), &sample_ts)?;
    write_npy(a.out_dir.join("mid.npy"), &mid_at_sample)?;
    write_npy(a.out_dir.join("entry_long.npy"), &entry_long)?;
    write_npy(a.out_dir.join("entry_short.npy"), &entry_short)?;
    write_npy(a.out_dir.join("top5_bid.npy"), &top5_bid)?;
    write_npy(a.out_dir.join("top5_ask.npy"), &top5_ask)?;
    write_npy(a.out_dir.join("mid_paths.npy"), &mid_paths)?;
    write_npy(a.out_dir.join("X_lob.npy"), &x_lob)?;
    // Free in reverse order of allocation to smooth RSS before Python resumes.
    drop(x_lob);
    drop(mid_paths);
    eprintln!(
        "build_samples: wrote 8 npy files in {:.2}s ({} samples)",
        t4.elapsed().as_secs_f64(), ns
    );

    let _ = (s![..],); // keep `ndarray::s` import lint-happy
    Ok(())
}
