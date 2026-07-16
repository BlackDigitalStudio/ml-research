//! Golden-parity harness for the incremental feature engine (`features_incr`).
//!
//! Loads a recorder day exactly like `feature_builder`, computes the (n_samples, 64)
//! matrix twice — (a) the frozen BATCH path (features.rs fill_* sequence, byte-frozen)
//! and (b) the INCREMENTAL path (merged-ts replay through FeatState + compute64 per
//! index) — and requires bit-equality per f32. Also reports compute64 latency stats
//! (the sub-ms budget evidence).
//!
//! Input scope = the deployed h150 config: depth+trades+funding(day-anchor)+eth+
//! liquidations+open-interest. Cols 17-19/30/50-53 are zero in both paths.
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array1;

use scalper_ingest::features::{
    compute_lob_features, fill_deep_book, fill_eth_features, fill_funding_features,
    fill_horizon_features, fill_horizon_features_b, fill_horizon_features_c,
    fill_horizon_features_d, fill_liquidation_features, fill_microstructure_depth,
    fill_microstructure_trades, fill_oi_features, fill_trade_features,
};
use scalper_ingest::features_incr::FeatState;
use scalper_ingest::{
    read_depth_parquet, read_funding_parquet, read_liquidations_parquet,
    read_open_interest_parquet, read_trades_parquet, DEPTH_LEVELS,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    depth: PathBuf,
    #[arg(long)]
    trades: Option<PathBuf>,
    #[arg(long)]
    funding: Option<PathBuf>,
    #[arg(long)]
    eth: Option<PathBuf>,
    #[arg(long)]
    liquidations: Option<PathBuf>,
    #[arg(long, name = "open-interest")]
    open_interest: Option<PathBuf>,
    #[arg(long)]
    indices: PathBuf,
    /// "anchor" (single-row day-anchor contract) | "true" (stream funding rows;
    /// col13/col44 batch semantics, HD3 rev10 BTC x true-funding policy).
    #[arg(long, default_value = "anchor")]
    funding_mode: String,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let depth = read_depth_parquet(&a.depth)?;
    let trades = a.trades.as_ref().map(|p| read_trades_parquet(p)).transpose()?;
    let funding = a.funding.as_ref().map(|p| read_funding_parquet(p)).transpose()?;
    let eth = a.eth.as_ref().map(|p| read_trades_parquet(p)).transpose()?;
    let liq = a.liquidations.as_ref().map(|p| read_liquidations_parquet(p)).transpose()?;
    let oi = a.open_interest.as_ref().map(|p| read_open_interest_parquet(p)).transpose()?;
    let idx: Array1<i64> = ndarray_npy::read_npy(&a.indices).context("indices")?;
    let is = idx.as_slice().unwrap();
    let n = depth.n_rows();
    eprintln!("harness: depth={} trades={} eth={} liq={} oi={} samples={}",
        n,
        trades.as_ref().map(|t| t.len()).unwrap_or(0),
        eth.as_ref().map(|t| t.len()).unwrap_or(0),
        liq.as_ref().map(|l| l.timestamps.len()).unwrap_or(0),
        oi.as_ref().map(|(t, _)| t.len()).unwrap_or(0),
        is.len());

    // ---------------- (a) frozen batch path — same sequence as feature_builder ----
    let t0 = Instant::now();
    let mut fb = compute_lob_features(&depth, is);
    fill_microstructure_depth(&mut fb, &depth, is);
    fill_horizon_features(&mut fb, &depth, is);
    fill_horizon_features_b(&mut fb, &depth, is, trades.as_ref(), funding.as_ref());
    fill_horizon_features_c(&mut fb, &depth, is, trades.as_ref());
    fill_horizon_features_d(&mut fb, &depth, is, None, eth.as_ref(), None, None, None);
    if let Some(tr) = trades.as_ref() {
        fill_trade_features(&mut fb, &depth, is, tr);
        fill_microstructure_trades(&mut fb, &depth, is, tr);
    }
    if let Some(f) = funding.as_ref() {
        fill_funding_features(&mut fb, &depth, is, f);
    }
    if let Some(e) = eth.as_ref() {
        fill_eth_features(&mut fb, &depth, is, e);
    }
    fill_deep_book(&mut fb, &depth, is);
    if let Some(l) = liq.as_ref() {
        fill_liquidation_features(&mut fb, &depth, is, l);
    }
    if let Some((ots, ov)) = oi.as_ref() {
        fill_oi_features(&mut fb, &depth, is, ots.as_slice().unwrap(), ov.as_slice().unwrap());
    }
    eprintln!("batch path: {:.2}s", t0.elapsed().as_secs_f64());

    // ---------------- (b) incremental replay ----
    let t1 = Instant::now();
    let mut st = FeatState::new();
    let funding_true = a.funding_mode == "true";
    if let Some(f) = funding.as_ref() {
        if !f.timestamps.is_empty() {
            if funding_true {
                st.funding_true = true; // rows streamed in the replay loop below
            } else {
                st.anchor_rate = Some(f.funding_rate[0]); // day-anchor single-row contract
            }
        }
    }
    let d_ts = depth.timestamps.as_slice().unwrap();
    // stream pointers (commit contract: events with ts <= tick ts BEFORE the tick)
    let (mut pt, mut pe, mut pl, mut po, mut pf) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut bids = [(0f64, 0f64); DEPTH_LEVELS];
    let mut asks = [(0f64, 0f64); DEPTH_LEVELS];
    for i in 0..n {
        let ts = d_ts[i];
        if let Some(tr) = trades.as_ref() {
            let tt = tr.timestamps.as_slice().unwrap();
            while pt < tr.len() && tt[pt] <= ts {
                st.push_trade(tt[pt], tr.prices[pt], tr.quantities[pt], tr.is_sell[pt]);
                pt += 1;
            }
        }
        if let Some(e) = eth.as_ref() {
            let et = e.timestamps.as_slice().unwrap();
            while pe < e.len() && et[pe] <= ts {
                st.push_eth(et[pe], e.prices[pe], e.quantities[pe], e.is_sell[pe]);
                pe += 1;
            }
        }
        if let Some(l) = liq.as_ref() {
            let lt = l.timestamps.as_slice().unwrap();
            while pl < lt.len() && lt[pl] <= ts {
                st.push_liq(lt[pl], l.signed_notional[pl], l.abs_notional[pl]);
                pl += 1;
            }
        }
        if let Some((ots, ov)) = oi.as_ref() {
            let ot = ots.as_slice().unwrap();
            while po < ot.len() && ot[po] <= ts {
                st.push_oi(ot[po], ov[po]);
                po += 1;
            }
        }
        if funding_true {
            if let Some(f) = funding.as_ref() {
                let ft = f.timestamps.as_slice().unwrap();
                while pf < ft.len() && ft[pf] <= ts {
                    st.push_funding(ft[pf], f.funding_rate[pf], f.mark_price[pf]);
                    pf += 1;
                }
            }
        }
        for k in 0..DEPTH_LEVELS {
            bids[k] = (depth.bid_prices[[i, k]], depth.bid_qtys[[i, k]]);
            asks[k] = (depth.ask_prices[[i, k]], depth.ask_qtys[[i, k]]);
        }
        st.push_book(ts, &bids, &asks);
    }
    let replay_s = t1.elapsed().as_secs_f64();

    // compute64 at each index + latency stats
    let mut lat_us: Vec<f64> = Vec::with_capacity(is.len());
    let mut fi = vec![[0f32; 64]; is.len()];
    for (s_idx, &raw) in is.iter().enumerate() {
        let t = Instant::now();
        fi[s_idx] = st.compute64(raw as usize);
        lat_us.push(t.elapsed().as_secs_f64() * 1e6);
    }
    lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| lat_us[((lat_us.len() as f64 - 1.0) * p) as usize];
    eprintln!("incremental: replay {:.2}s | compute64 p50={:.1}us p99={:.1}us max={:.1}us",
        replay_s, q(0.5), q(0.99), lat_us[lat_us.len() - 1]);

    // ---------------- byte comparison ----------------
    let mut bad_cols = [0usize; 64];
    let mut max_abs = [0f64; 64];
    let mut n_bad = 0usize;
    let mut first_bad: Option<(usize, usize, f32, f32)> = None;
    for s_idx in 0..is.len() {
        for c in 0..64 {
            let b = fb[[s_idx, c]];
            let v = fi[s_idx][c];
            if b.to_bits() != v.to_bits() {
                n_bad += 1;
                bad_cols[c] += 1;
                let d = ((b as f64) - (v as f64)).abs();
                if d > max_abs[c] {
                    max_abs[c] = d;
                }
                if first_bad.is_none() {
                    first_bad = Some((s_idx, c, b, v));
                }
            }
        }
    }
    if n_bad == 0 {
        println!("PARITY: BYTE-EXACT over {} samples x 64 cols", is.len());
    } else {
        println!("PARITY: {} mismatched cells / {} total", n_bad, is.len() * 64);
        for c in 0..64 {
            if bad_cols[c] > 0 {
                println!("  col {c:2}: {} cells, max|d|={:.3e}", bad_cols[c], max_abs[c]);
            }
        }
        if let Some((s, c, b, v)) = first_bad {
            println!("  first: sample {s} col {c}: batch={b:?} incr={v:?}");
        }
        std::process::exit(1);
    }
    Ok(())
}
