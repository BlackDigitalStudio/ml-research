//! grid_sim — fused outer-grid forward simulator.
//!
//! Replaces the "27× call sim_labels with different (tp, sl, timeout)" pattern
//! with a single sweep: load mid_paths once, then for every sample run
//! ALL outer configs back-to-back. Data locality wins big — mid_paths.row(i)
//! stays L1/L2-resident across all configs for sample i, so the per-sample
//! cost grows ~linearly with n_configs instead of ~quadratically (each old
//! call also re-paid mmap/page-cache fault overhead).
//!
//! Reuses live_sim::simulate_trade* directly — no parity drift from the
//! single-config sim_labels binary.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use memmap2::Mmap;
use ndarray::{Array1, Array2, Array3, ArrayView2};
use ndarray_npy::{read_npy, write_npy};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use scalper_ingest::live_sim::{
    simulate_trade, simulate_trade_book, BookL1, LiveSimConfig, SimDirection,
};

#[derive(Parser, Debug)]
#[command(about = "Fused outer-grid forward simulator")]
struct Args {
    /// (n_samples,) f64
    #[arg(long)]
    entry_long: PathBuf,
    #[arg(long)]
    entry_short: PathBuf,
    /// (n_samples, horizon) f64. Mid-only path — for book mode see --book-paths.
    #[arg(long)]
    mid_paths: PathBuf,
    /// Optional (n_samples, horizon, 2) f64 [bid, ask] for book-aware sim.
    #[arg(long)]
    book_paths: Option<PathBuf>,
    /// Optional (n_samples, 2) f64 [bid_at_entry, ask_at_entry] for book mode.
    #[arg(long)]
    entry_book: Option<PathBuf>,

    /// Path to JSON file containing the outer-config list. Format:
    ///   [{"tp": 0.20, "sl": 0.10, "to": 1200, "par": false, "tr": false}, ...]
    /// `tp` and `sl` are percent (e.g. 0.20 = 0.20%), `to` is timeout in ticks.
    #[arg(long)]
    configs: PathBuf,

    #[arg(long, default_value_t = 0.04)]
    commission_win_pct: f64,
    #[arg(long, default_value_t = 0.07)]
    commission_loss_pct: f64,
    #[arg(long, default_value_t = 150.0)]
    fill_latency_ms: f64,
    /// Optional (n_samples,) f64 per-sample fill latency override.
    #[arg(long)]
    fill_latency_ms_array: Option<PathBuf>,

    /// Output prefix. Writes <prefix>_pnl_long.npy and <prefix>_pnl_short.npy
    /// of shape (n_configs, n_samples) f64.
    #[arg(long)]
    out_prefix: String,

    // ─── Inner-grid (realise) options. When provided, the binary additionally
    //     sweeps every (outer × min_prob × spread × fill_prob × kelly_frac)
    //     combo on the holdout slice and writes results JSON to --inner-out.
    /// (n_samples,) i64 — predicted class per sample. 0=UP, 1=DN, 2=FL.
    #[arg(long)]
    pred: Option<PathBuf>,
    /// (n_samples,) f64 — max softmax probability per sample.
    #[arg(long)]
    max_prob: Option<PathBuf>,
    /// First sample index of the holdout slice (inclusive).
    #[arg(long, default_value_t = 0)]
    holdout_start: usize,
    /// Effective days in the holdout (for trades_per_day reporting).
    #[arg(long, default_value_t = 1.0)]
    n_eff_days: f64,
    /// Inner sweep — comma-separated f64 values.
    #[arg(long, default_value = "0.40,0.45,0.50,0.55", value_delimiter = ',')]
    inner_min_probs: Vec<f64>,
    #[arg(long, default_value = "0.0,0.02,0.04", value_delimiter = ',')]
    inner_spreads: Vec<f64>,
    #[arg(long, default_value = "1.0,0.8,0.6", value_delimiter = ',')]
    inner_fill_probs: Vec<f64>,
    #[arg(long, default_value = "0.10,0.25,0.50", value_delimiter = ',')]
    inner_kelly_fracs: Vec<f64>,
    #[arg(long, default_value_t = 19.0)]
    inner_kelly_cap: f64,
    #[arg(long, default_value_t = 100.0)]
    inner_initial_capital: f64,
    #[arg(long, default_value_t = 42)]
    inner_seed: u64,
    /// Output JSON path for inner sweep results. When set with --pred and
    /// --max-prob, the inner sweep runs and writes its results here.
    #[arg(long)]
    inner_out: Option<PathBuf>,
}

/// Mmap an .npy file containing little-endian f64 data, return (mmap, data_offset, shape).
/// The mmap MUST outlive any view into it.
fn mmap_npy_f64_2d(path: &Path) -> Result<(Mmap, usize, (usize, usize))> {
    let f = File::open(path).with_context(|| format!("open {:?}", path))?;
    let mmap = unsafe { Mmap::map(&f).with_context(|| format!("mmap {:?}", path))? };
    anyhow::ensure!(mmap.len() >= 12, "npy too short: {:?}", path);
    anyhow::ensure!(&mmap[0..6] == b"\x93NUMPY", "bad npy magic in {:?}", path);
    let major = mmap[6];
    let (header_len, header_start) = match major {
        1 => {
            let h = u16::from_le_bytes([mmap[8], mmap[9]]) as usize;
            (h, 10)
        }
        2 => {
            let h = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize;
            (h, 12)
        }
        v => anyhow::bail!("unsupported npy version {} in {:?}", v, path),
    };
    let header = std::str::from_utf8(&mmap[header_start..header_start + header_len])
        .context("npy header not utf8")?;
    anyhow::ensure!(
        header.contains("'<f8'") || header.contains("\"<f8\""),
        "expected dtype '<f8' in {:?}, got: {}", path, header
    );
    anyhow::ensure!(
        !header.contains("'fortran_order': True"),
        "fortran_order=True not supported in {:?}", path
    );
    let shape_pos = header.find("'shape':").context("no 'shape' key in npy header")?;
    let after = &header[shape_pos..];
    let lp = after.find('(').context("no ( in shape")?;
    let rp = after.find(')').context("no ) in shape")?;
    let dims: Vec<usize> = after[lp + 1..rp]
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    anyhow::ensure!(dims.len() == 2, "expected 2D shape in {:?}, got {:?}", path, dims);
    Ok((mmap, header_start + header_len, (dims[0], dims[1])))
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct OuterCfg {
    tp: f64,
    sl: f64,
    to: i64,
    #[serde(default)]
    par: bool,
    #[serde(default)]
    tr: bool,
    // Partial-TP / Trailing-SL внутренности. Optional — если ключ не передан,
    // берётся LiveSimConfig::default(). Используются только при par=True / tr=True
    // соответственно, иначе игнорируются simulate_trade.
    #[serde(default)]
    partial_tp_progress: Option<f64>,
    #[serde(default)]
    trailing_step1_progress: Option<f64>,
    #[serde(default)]
    trailing_step1_sl_floor_pct: Option<f64>,
    #[serde(default)]
    trailing_step1_sl_ratio: Option<f64>,
    #[serde(default)]
    trailing_step2_progress: Option<f64>,
    #[serde(default)]
    trailing_step2_sl_ratio: Option<f64>,
}

/// Mirror of grid_ensemble_b300.realise output. Fields mirror the Python
/// dict keys so downstream ranking code stays identical. PT/TS effective
/// values are always serialized — каждый top-config показывает фактически
/// использованные параметры trailing/partial, без ссылки на default-таблицу.
#[derive(Serialize, Debug, Clone)]
struct InnerResult {
    outer_idx: usize,
    tp: f64,
    sl: f64,
    timeout: i64,
    partial: bool,
    trailing: bool,
    partial_tp_progress: f64,
    trailing_step1_progress: f64,
    trailing_step1_sl_floor_pct: f64,
    trailing_step1_sl_ratio: f64,
    trailing_step2_progress: f64,
    trailing_step2_sl_ratio: f64,
    min_prob: f64,
    spread_pct: f64,
    fill_prob: f64,
    kelly_frac: f64,
    n_trades: i64,
    win_rate_pct: f64,
    net_return_pct: f64,
    sum_pnl_pct: f64,
    max_dd_pct: f64,
    sharpe: f64,
    mean_size: f64,
    ev_per_trade_pct: f64,
    trades_per_day: f64,
}

/// Tiny xorshift64 PRNG — adequate for Bernoulli fill masks. Deterministic
/// from seed; same (seed, n) produces same mask sequence across runs.
struct XorShift64 {
    s: u64,
}
impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { s: seed.max(1) }
    }
    /// Returns f64 in [0, 1).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        let mut x = self.s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.s = x;
        // Top 53 bits → [0, 1).
        ((x >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

const UP_CLS: i64 = 0;
const DN_CLS: i64 = 1;
const FL_CLS: i64 = 2;

#[inline]
fn kelly_size(p: f64, b: f64, cap: f64) -> f64 {
    let p = p.clamp(0.01, 0.99);
    let raw = (b * p - (1.0 - p)) / b;
    raw.clamp(0.0, cap)
}

/// Compute the 8 metrics for one (outer, min_prob, spread, fill_prob, kelly) combo.
/// `pnl_long_h` / `pnl_short_h` are the holdout slice (length n_h).
/// `pred_h`, `max_prob_h` likewise.
/// `fill_mask_h` is precomputed per fill_prob to avoid re-rolling RNG per inner.
fn compute_metrics(
    pnl_long_h: &[f64],
    pnl_short_h: &[f64],
    pred_h: &[i64],
    max_prob_h: &[f64],
    fill_mask_h: &[bool],
    min_prob: f64,
    spread_pct: f64,
    kelly_frac: f64,
    kelly_cap: f64,
    tp_pct: f64,
    sl_pct: f64,
    initial_capital: f64,
) -> (i64, f64, f64, f64, f64, f64, f64, f64) {
    let n = pnl_long_h.len();
    let b = tp_pct / sl_pct.max(1e-6);
    let kelly_active = kelly_frac > 0.0;

    // Single-pass: for each i compute take/realised, accumulate stats and equity.
    let mut n_trades: i64 = 0;
    let mut sum_realised: f64 = 0.0;
    let mut sum_realised_sq: f64 = 0.0;
    let mut wins: i64 = 0;
    let mut sum_size: f64 = 0.0;
    let mut eq = initial_capital;
    let mut peak = eq;
    let mut max_dd_frac: f64 = 0.0;

    for i in 0..n {
        let p = pred_h[i];
        let mp = max_prob_h[i];
        let gate = p != FL_CLS && mp >= min_prob;
        let take = gate && fill_mask_h[i];
        let mut realised = 0.0;
        let size = if kelly_active {
            kelly_frac * kelly_size(mp, b, kelly_cap)
        } else {
            kelly_frac
        };
        if take {
            let dir_pnl = if p == UP_CLS {
                pnl_long_h[i]
            } else if p == DN_CLS {
                pnl_short_h[i]
            } else {
                0.0
            };
            let real = dir_pnl - spread_pct;
            realised = size * real;
            n_trades += 1;
            sum_realised += realised;
            sum_realised_sq += realised * realised;
            if realised > 0.0 {
                wins += 1;
            }
            sum_size += size;
        }
        // Equity always advances (realised=0 if no trade).
        let ret = realised / 100.0;
        eq *= 1.0 + ret;
        if eq > peak {
            peak = eq;
        }
        let dd = (peak - eq) / peak.max(1e-12);
        if dd > max_dd_frac {
            max_dd_frac = dd;
        }
    }

    if n_trades == 0 {
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let nt_f = n_trades as f64;
    let mean = sum_realised / nt_f;
    let var = (sum_realised_sq / nt_f - mean * mean).max(0.0);
    let std = var.sqrt();
    let sharpe = if n_trades > 1 { mean / (std + 1e-9) } else { 0.0 };
    let win_rate_pct = (wins as f64 / nt_f) * 100.0;
    let net_return_pct = 100.0 * (eq / initial_capital - 1.0);
    let sum_pnl_pct = sum_realised;
    let mean_size = sum_size / nt_f;
    let ev_per_trade_pct = mean;

    (
        n_trades,
        win_rate_pct,
        net_return_pct,
        sum_pnl_pct,
        max_dd_frac * 100.0,
        sharpe,
        mean_size,
        ev_per_trade_pct,
    )
}

fn main() -> Result<()> {
    let a = Args::parse();
    let t0 = Instant::now();

    let entry_long: Array1<f64> = read_npy(&a.entry_long).context("entry_long")?;
    let entry_short: Array1<f64> = read_npy(&a.entry_short).context("entry_short")?;

    // Mmap mid_paths zero-copy. The OS pages stay shared between this process
    // and the page cache; rayon threads page-fault them in on demand. For a
    // 15 GB tensor this saves ~50s of `read_npy` memcpy.
    let (mmap_mid, mid_off, mid_shape) = mmap_npy_f64_2d(&a.mid_paths)?;
    let mid_data: &[f64] = unsafe {
        let bytes = &mmap_mid[mid_off..];
        let n_elems = mid_shape.0 * mid_shape.1;
        std::slice::from_raw_parts(bytes.as_ptr() as *const f64, n_elems)
    };
    let mid_paths: ArrayView2<f64> = ArrayView2::from_shape(mid_shape, mid_data)
        .context("ArrayView2 from mmap")?;
    let ns = entry_long.len();
    anyhow::ensure!(
        entry_short.len() == ns && mid_paths.nrows() == ns,
        "array length mismatch: el={}, es={}, mid.nrows={}",
        ns, entry_short.len(), mid_paths.nrows()
    );

    let book_paths: Option<Array3<f64>> = match &a.book_paths {
        Some(p) => Some(read_npy(p).context("book_paths")?),
        None => None,
    };
    let entry_book: Option<Array2<f64>> = match &a.entry_book {
        Some(p) => Some(read_npy(p).context("entry_book")?),
        None => None,
    };
    let fill_latency_arr: Option<Array1<f64>> = match &a.fill_latency_ms_array {
        Some(p) => Some(read_npy(p).context("fill_latency_ms_array")?),
        None => None,
    };

    let configs_raw = fs::read_to_string(&a.configs).context("read configs json")?;
    let configs: Vec<OuterCfg> =
        serde_json::from_str(&configs_raw).context("parse configs json")?;
    let nc = configs.len();
    anyhow::ensure!(nc > 0, "configs list is empty");

    eprintln!(
        "grid_sim: {} samples × {} configs ({} mode) — load {:.2}s",
        ns,
        nc,
        if book_paths.is_some() { "book-aware" } else { "mid" },
        t0.elapsed().as_secs_f64(),
    );

    let base = LiveSimConfig {
        commission_win_pct: a.commission_win_pct,
        commission_loss_pct: a.commission_loss_pct,
        ..LiveSimConfig::default()
    };
    let scalar_lat = a.fill_latency_ms;
    let use_book = book_paths.is_some();

    let t_sim = Instant::now();

    // Per-sample result: Vec<(pnl_long, pnl_short)> of length nc.
    // Rayon-parallel across samples; inner config loop walks the same
    // mid_path row 27 times → mid stays L1/L2-resident.
    let per_sample: Vec<Vec<(f64, f64)>> = (0..ns)
        .into_par_iter()
        .map(|i| {
            let lat = fill_latency_arr.as_ref().map(|a| a[i]).unwrap_or(scalar_lat);
            let mut row = Vec::with_capacity(nc);

            if use_book {
                let bp = book_paths.as_ref().unwrap();
                let path_view = bp.index_axis(ndarray::Axis(0), i);
                let h = path_view.shape()[0];
                let mut path: Vec<BookL1> = Vec::with_capacity(h);
                for t in 0..h {
                    path.push(BookL1 {
                        bid: path_view[[t, 0]],
                        ask: path_view[[t, 1]],
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
                        bid: entry_long[i],
                        ask: entry_short[i],
                        bid_qty: 0.0,
                        ask_qty: 0.0,
                    },
                };
                for c in configs.iter() {
                    let cfg = LiveSimConfig {
                        tp_pct: c.tp,
                        sl_pct: c.sl,
                        timeout_ticks: c.to,
                        partial_enabled: c.par,
                        trailing_enabled: c.tr,
                        partial_tp_progress: c.partial_tp_progress.unwrap_or(base.partial_tp_progress),
                        trailing_step1_progress: c.trailing_step1_progress.unwrap_or(base.trailing_step1_progress),
                        trailing_step1_sl_floor_pct: c.trailing_step1_sl_floor_pct.unwrap_or(base.trailing_step1_sl_floor_pct),
                        trailing_step1_sl_ratio: c.trailing_step1_sl_ratio.unwrap_or(base.trailing_step1_sl_ratio),
                        trailing_step2_progress: c.trailing_step2_progress.unwrap_or(base.trailing_step2_progress),
                        trailing_step2_sl_ratio: c.trailing_step2_sl_ratio.unwrap_or(base.trailing_step2_sl_ratio),
                        ..base.clone()
                    };
                    let l = simulate_trade_book(SimDirection::Long, eb, &path, &cfg, lat);
                    let s = simulate_trade_book(SimDirection::Short, eb, &path, &cfg, lat);
                    row.push((l.net_pnl_pct, s.net_pnl_pct));
                }
            } else {
                let path_row = mid_paths.row(i);
                let path_slice = path_row.as_slice().unwrap();
                let el = entry_long[i];
                let es = entry_short[i];
                for c in configs.iter() {
                    let cfg = LiveSimConfig {
                        tp_pct: c.tp,
                        sl_pct: c.sl,
                        timeout_ticks: c.to,
                        partial_enabled: c.par,
                        trailing_enabled: c.tr,
                        partial_tp_progress: c.partial_tp_progress.unwrap_or(base.partial_tp_progress),
                        trailing_step1_progress: c.trailing_step1_progress.unwrap_or(base.trailing_step1_progress),
                        trailing_step1_sl_floor_pct: c.trailing_step1_sl_floor_pct.unwrap_or(base.trailing_step1_sl_floor_pct),
                        trailing_step1_sl_ratio: c.trailing_step1_sl_ratio.unwrap_or(base.trailing_step1_sl_ratio),
                        trailing_step2_progress: c.trailing_step2_progress.unwrap_or(base.trailing_step2_progress),
                        trailing_step2_sl_ratio: c.trailing_step2_sl_ratio.unwrap_or(base.trailing_step2_sl_ratio),
                        ..base.clone()
                    };
                    let l = simulate_trade(SimDirection::Long, el, path_slice, &cfg, lat);
                    let s = simulate_trade(SimDirection::Short, es, path_slice, &cfg, lat);
                    row.push((l.net_pnl_pct, s.net_pnl_pct));
                }
            }
            row
        })
        .collect();

    eprintln!(
        "grid_sim: simulated {} samples × {} configs in {:.2}s ({:.0} sample×cfg/s)",
        ns,
        nc,
        t_sim.elapsed().as_secs_f64(),
        (ns as f64 * nc as f64) / t_sim.elapsed().as_secs_f64().max(1e-9),
    );

    // (nc, ns) — каждый config'а полный pnl вектор contiguous on disk.
    let mut pnl_long = Array2::<f64>::zeros((nc, ns));
    let mut pnl_short = Array2::<f64>::zeros((nc, ns));
    for (i, row) in per_sample.iter().enumerate() {
        for (k, &(l, s)) in row.iter().enumerate() {
            pnl_long[[k, i]] = l;
            pnl_short[[k, i]] = s;
        }
    }

    let p = &a.out_prefix;
    write_npy(format!("{}_pnl_long.npy", p), &pnl_long)?;
    write_npy(format!("{}_pnl_short.npy", p), &pnl_short)?;

    // ─── Inner sweep (realise) — when caller provides pred + max_prob + inner_out ────
    if let (Some(pred_path), Some(max_prob_path), Some(inner_out)) =
        (&a.pred, &a.max_prob, &a.inner_out)
    {
        let pred: Array1<i64> = read_npy(pred_path).context("pred")?;
        let max_prob: Array1<f64> = read_npy(max_prob_path).context("max_prob")?;
        anyhow::ensure!(
            pred.len() == ns && max_prob.len() == ns,
            "pred/max_prob length mismatch with samples ({} vs {})",
            pred.len(), ns
        );
        anyhow::ensure!(a.holdout_start < ns, "holdout_start {} >= ns {}", a.holdout_start, ns);
        let h_start = a.holdout_start;
        let n_h = ns - h_start;
        eprintln!("grid_sim: inner sweep on holdout slice [{}..{}] = {} samples", h_start, ns, n_h);

        let pred_h_full = pred.as_slice().unwrap();
        let max_prob_h_full = max_prob.as_slice().unwrap();
        let pred_h: Vec<i64> = pred_h_full[h_start..].to_vec();
        let max_prob_h: Vec<f64> = max_prob_h_full[h_start..].to_vec();

        // Pre-roll fill masks once per fill_prob (saves nc·n_mp·n_sp·n_kf RNG passes).
        let fill_masks: Vec<Vec<bool>> = a.inner_fill_probs.iter().map(|&fp| {
            if fp >= 1.0 {
                vec![true; n_h]
            } else {
                let mut rng = XorShift64::new(a.inner_seed);
                (0..n_h).map(|_| rng.next_f64() < fp).collect()
            }
        }).collect();

        // Cartesian product (k, mp_i, sp_i, fp_i, kf_i).
        let mut combos: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
        for k in 0..nc {
            for mp_i in 0..a.inner_min_probs.len() {
                for sp_i in 0..a.inner_spreads.len() {
                    for fp_i in 0..a.inner_fill_probs.len() {
                        for kf_i in 0..a.inner_kelly_fracs.len() {
                            combos.push((k, mp_i, sp_i, fp_i, kf_i));
                        }
                    }
                }
            }
        }
        eprintln!("grid_sim: inner combos = {}", combos.len());

        let inner_t0 = Instant::now();
        // pnl_long/short are row-contiguous (we built them as row-major zeros).
        let pnl_long_view = pnl_long.view();
        let pnl_short_view = pnl_short.view();

        let inner_results: Vec<InnerResult> = combos.par_iter().map(|&(k, mp_i, sp_i, fp_i, kf_i)| {
            let cfg = &configs[k];
            let pl_row = pnl_long_view.row(k);
            let ps_row = pnl_short_view.row(k);
            let pnl_long_h = &pl_row.as_slice().expect("pnl_long row contig")[h_start..];
            let pnl_short_h = &ps_row.as_slice().expect("pnl_short row contig")[h_start..];

            let min_prob = a.inner_min_probs[mp_i];
            let spread = a.inner_spreads[sp_i];
            let fill_prob = a.inner_fill_probs[fp_i];
            let kelly = a.inner_kelly_fracs[kf_i];
            let fm = &fill_masks[fp_i];

            let (nt, wr, net, sum_pnl, dd, sharpe, mean_size, ev) = compute_metrics(
                pnl_long_h, pnl_short_h, &pred_h, &max_prob_h, fm,
                min_prob, spread, kelly, a.inner_kelly_cap,
                cfg.tp, cfg.sl, a.inner_initial_capital,
            );
            InnerResult {
                outer_idx: k,
                tp: cfg.tp, sl: cfg.sl, timeout: cfg.to,
                partial: cfg.par, trailing: cfg.tr,
                partial_tp_progress: cfg.partial_tp_progress.unwrap_or(base.partial_tp_progress),
                trailing_step1_progress: cfg.trailing_step1_progress.unwrap_or(base.trailing_step1_progress),
                trailing_step1_sl_floor_pct: cfg.trailing_step1_sl_floor_pct.unwrap_or(base.trailing_step1_sl_floor_pct),
                trailing_step1_sl_ratio: cfg.trailing_step1_sl_ratio.unwrap_or(base.trailing_step1_sl_ratio),
                trailing_step2_progress: cfg.trailing_step2_progress.unwrap_or(base.trailing_step2_progress),
                trailing_step2_sl_ratio: cfg.trailing_step2_sl_ratio.unwrap_or(base.trailing_step2_sl_ratio),
                min_prob, spread_pct: spread, fill_prob, kelly_frac: kelly,
                n_trades: nt,
                win_rate_pct: wr,
                net_return_pct: net,
                sum_pnl_pct: sum_pnl,
                max_dd_pct: dd,
                sharpe,
                mean_size,
                ev_per_trade_pct: ev,
                trades_per_day: nt as f64 / a.n_eff_days,
            }
        }).collect();

        eprintln!(
            "grid_sim: inner sweep {} combos in {:.2}s",
            inner_results.len(),
            inner_t0.elapsed().as_secs_f64()
        );

        let json_str = serde_json::to_string(&inner_results).context("serialize inner results")?;
        let mut f = File::create(inner_out).with_context(|| format!("create {:?}", inner_out))?;
        f.write_all(json_str.as_bytes())?;
        eprintln!("grid_sim: wrote inner results → {:?}", inner_out);
    }

    eprintln!("grid_sim: total {:.2}s", t0.elapsed().as_secs_f64());
    Ok(())
}
