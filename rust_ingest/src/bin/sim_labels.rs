//! sim_labels — batch forward-simulate trades for label construction.
//!
//! For every sample i, simulate a LONG at entry_px_long[i] and a SHORT at
//! entry_px_short[i] against mid_paths[i, :], then pick the best-PnL
//! direction as the training label.
//!
//! Rayon-parallel; matches Python `live_sim.simulate_trade` + `label_from_outcomes`
//! byte-for-byte. Used by trainer.py via feature_builder workflow.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::{Array2, Array3};
use ndarray_npy::{read_npy, write_npy};
use rayon::prelude::*;
use scalper_ingest::live_sim::{
    label_from_outcomes, simulate_trade, simulate_trade_book, BookL1, LiveSimConfig, SimDirection,
};

#[derive(Parser, Debug)]
#[command(about = "Batch forward-sim for training labels")]
struct Args {
    /// (n_samples,) f64
    #[arg(long)]
    entry_long: PathBuf,
    #[arg(long)]
    entry_short: PathBuf,
    /// (n_samples, horizon) f64, row per sample. Used when --book-paths is NOT set.
    #[arg(long)]
    mid_paths: PathBuf,
    /// Optional (n_samples, horizon, 2) f64 [best_bid, best_ask] forward path.
    /// When set, uses the book-aware simulator (simulate_trade_book) which
    /// walks the top-1 L1 book instead of mid. TP/SL distances get computed
    /// relative to entry fill (ask for long / bid for short), and stops pay
    /// realistic bid/ask slippage on trigger.
    #[arg(long)]
    book_paths: Option<PathBuf>,
    /// Optional (n_samples, 2) f64 [bid_at_entry, ask_at_entry]. Required when
    /// --book-paths is set. If absent, entry_long/entry_short are assumed to
    /// already encode the (bid, ask) pair at entry tick.
    #[arg(long)]
    entry_book: Option<PathBuf>,
    /// (n_samples,) f64
    #[arg(long)]
    tp_pct: PathBuf,
    /// (n_samples,) f64
    #[arg(long)]
    sl_pct: PathBuf,
    /// (n_samples,) i64
    #[arg(long)]
    timeout_ticks: PathBuf,

    #[arg(long, default_value_t = 0.04)]
    commission_win_pct: f64,
    #[arg(long, default_value_t = 0.07)]
    commission_loss_pct: f64,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    partial_enabled: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    trailing_enabled: bool,
    #[arg(long, default_value_t = 150.0)]
    fill_latency_ms: f64,
    /// Optional (n_samples,) f64 array of per-sample fill latency in ms.
    /// Overrides --fill-latency-ms when present. Generate from an empirical
    /// RTT distribution (see scripts/build_latency_dist.py) for live parity.
    #[arg(long)]
    fill_latency_ms_array: Option<PathBuf>,

    /// Output prefix. Writes <prefix>_y.npy (u8), <prefix>_target_pnl.npy (f64),
    /// <prefix>_reason_long.npy (u8), <prefix>_reason_short.npy (u8).
    #[arg(long)]
    out_prefix: String,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let t0 = Instant::now();
    let entry_long: ndarray::Array1<f64> =
        read_npy(&a.entry_long).context("entry_long")?;
    let entry_short: ndarray::Array1<f64> =
        read_npy(&a.entry_short).context("entry_short")?;
    let mid_paths: Array2<f64> = read_npy(&a.mid_paths).context("mid_paths")?;
    let tp_pct: ndarray::Array1<f64> = read_npy(&a.tp_pct).context("tp_pct")?;
    let sl_pct: ndarray::Array1<f64> = read_npy(&a.sl_pct).context("sl_pct")?;
    let timeout_ticks: ndarray::Array1<i64> =
        read_npy(&a.timeout_ticks).context("timeout_ticks")?;
    let ns = entry_long.len();
    anyhow::ensure!(
        entry_short.len() == ns
            && mid_paths.nrows() == ns
            && tp_pct.len() == ns
            && sl_pct.len() == ns
            && timeout_ticks.len() == ns,
        "array length mismatch"
    );

    // Optional book-aware inputs.
    let book_paths: Option<Array3<f64>> = match &a.book_paths {
        Some(p) => {
            let arr: Array3<f64> = read_npy(p).context("book_paths")?;
            anyhow::ensure!(
                arr.shape()[0] == ns && arr.shape()[2] == 2,
                "book_paths shape mismatch: got {:?}, expected ({}, H, 2)", arr.shape(), ns
            );
            Some(arr)
        }
        None => None,
    };
    let entry_book: Option<Array2<f64>> = match &a.entry_book {
        Some(p) => {
            let arr: Array2<f64> = read_npy(p).context("entry_book")?;
            anyhow::ensure!(
                arr.shape() == [ns, 2],
                "entry_book shape mismatch: got {:?}, expected ({}, 2)", arr.shape(), ns
            );
            Some(arr)
        }
        None => None,
    };
    let fill_latency_arr: Option<ndarray::Array1<f64>> = match &a.fill_latency_ms_array {
        Some(p) => {
            let arr: ndarray::Array1<f64> =
                read_npy(p).context("fill_latency_ms_array")?;
            anyhow::ensure!(
                arr.len() == ns,
                "fill_latency_ms_array length mismatch: {} vs {}", arr.len(), ns
            );
            Some(arr)
        }
        None => None,
    };

    let use_book = book_paths.is_some();
    if use_book {
        eprintln!(
            "sim_labels: book-aware path (horizon={}, entry_book={})",
            book_paths.as_ref().unwrap().shape()[1],
            entry_book.is_some()
        );
    }
    if fill_latency_arr.is_some() {
        eprintln!("sim_labels: per-sample fill latency array active");
    }

    let base = LiveSimConfig {
        commission_win_pct: a.commission_win_pct,
        commission_loss_pct: a.commission_loss_pct,
        partial_enabled: a.partial_enabled,
        trailing_enabled: a.trailing_enabled,
        ..LiveSimConfig::default()
    };

    // Parallel per-sample sim.
    let results: Vec<(u8, f64, u8, u8, f64, f64)> = (0..ns)
        .into_par_iter()
        .map(|i| {
            let cfg = LiveSimConfig {
                tp_pct: tp_pct[i],
                sl_pct: sl_pct[i],
                timeout_ticks: timeout_ticks[i],
                ..base.clone()
            };
            let lat = fill_latency_arr
                .as_ref()
                .map(|a| a[i])
                .unwrap_or(a.fill_latency_ms);

            let (long_o, short_o) = if use_book {
                let bp = book_paths.as_ref().unwrap();
                // Book path for this sample: slice along axis 0, reshape to
                // a flat BookL1 slice. bp has shape (ns, H, 2).
                let row = bp.index_axis(ndarray::Axis(0), i);
                let h = row.shape()[0];
                let mut path: Vec<BookL1> = Vec::with_capacity(h);
                for t in 0..h {
                    path.push(BookL1 {
                        bid: row[[t, 0]],
                        ask: row[[t, 1]],
                        bid_qty: 0.0,
                        ask_qty: 0.0,
                    });
                }
                let eb = match entry_book.as_ref() {
                    Some(arr) => BookL1 {
                        bid: arr[[i, 0]],
                        ask: arr[[i, 1]],
                        bid_qty: 0.0,
                        ask_qty: 0.0,
                    },
                    None => BookL1 {
                        // Fallback: entry_long already = bid, entry_short = ask
                        // per the build_samples convention.
                        bid: entry_long[i],
                        ask: entry_short[i],
                        bid_qty: 0.0,
                        ask_qty: 0.0,
                    },
                };
                let l = simulate_trade_book(SimDirection::Long, eb, &path, &cfg, lat);
                let s = simulate_trade_book(SimDirection::Short, eb, &path, &cfg, lat);
                (l, s)
            } else {
                let path = mid_paths.row(i);
                let path_slice = path.as_slice().unwrap();
                let l = simulate_trade(
                    SimDirection::Long,
                    entry_long[i],
                    path_slice,
                    &cfg,
                    lat,
                );
                let s = simulate_trade(
                    SimDirection::Short,
                    entry_short[i],
                    path_slice,
                    &cfg,
                    lat,
                );
                (l, s)
            };
            let (label, target) = label_from_outcomes(&long_o, &short_o);
            (
                label,
                target,
                long_o.exit_reason.id(),
                short_o.exit_reason.id(),
                long_o.net_pnl_pct,
                short_o.net_pnl_pct,
            )
        })
        .collect();

    let mut y = ndarray::Array1::<u8>::zeros(ns);
    let mut target_pnl = ndarray::Array1::<f64>::zeros(ns);
    let mut reason_long = ndarray::Array1::<u8>::zeros(ns);
    let mut reason_short = ndarray::Array1::<u8>::zeros(ns);
    let mut pnl_long = ndarray::Array1::<f64>::zeros(ns);
    let mut pnl_short = ndarray::Array1::<f64>::zeros(ns);
    for (i, r) in results.iter().enumerate() {
        y[i] = r.0;
        target_pnl[i] = r.1;
        reason_long[i] = r.2;
        reason_short[i] = r.3;
        pnl_long[i] = r.4;
        pnl_short[i] = r.5;
    }

    let p = &a.out_prefix;
    write_npy(format!("{}_y.npy", p), &y)?;
    write_npy(format!("{}_target_pnl.npy", p), &target_pnl)?;
    write_npy(format!("{}_reason_long.npy", p), &reason_long)?;
    write_npy(format!("{}_reason_short.npy", p), &reason_short)?;
    write_npy(format!("{}_pnl_long.npy", p), &pnl_long)?;
    write_npy(format!("{}_pnl_short.npy", p), &pnl_short)?;
    eprintln!(
        "sim_labels: {} samples in {:.2}s ({:.0}/s)",
        ns,
        t0.elapsed().as_secs_f64(),
        ns as f64 / t0.elapsed().as_secs_f64().max(1e-9),
    );
    Ok(())
}
