//! feature_builder — full 34-feature Rust pipeline, byte-parity with Python
//! `Trainer._calc_features_batch`.
//!
//! Coverage:
//!   [0..5]  depth                    [17..19] derivatives
//!   [6..9]  BTC trades               [20..29] microstructure depth + OFI
//!   [10,11] volatility, VWAP-dev     [30]     cross-exchange momentum
//!   [12]    momentum 5s              [31]     queue-pressure EMA
//!   [13]    funding                  [32]     top3 asymmetry
//!   [14..16] ETH leading signals     [33]     effective spread EMA
//!
//! Usage:
//!   feature_builder --depth D --indices I --out O
//!     [--trades T] [--funding F] [--derivs D] [--eth E]
//!     [--bybit B] [--okx O] [--bitget G] [--gateio G]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use ndarray_npy::{read_npy, write_npy};
use scalper_ingest::{
    features::{
        compute_lob_features, fill_cross_ex_feature, fill_deriv_features, fill_eth_features,
        fill_funding_features, fill_microstructure_depth, fill_microstructure_trades,
        fill_trade_features,
    },
    read_cross_ex_parquet, read_depth_parquet, read_derivatives_parquet, read_funding_parquet,
    read_trades_parquet,
};

#[derive(Parser, Debug)]
#[command(about = "Rust port of Trainer._calc_features_batch", long_about = None)]
struct Args {
    #[arg(long)]
    depth: PathBuf,
    #[arg(long)]
    indices: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    trades: Option<PathBuf>,
    #[arg(long)]
    funding: Option<PathBuf>,
    #[arg(long)]
    derivs: Option<PathBuf>,
    #[arg(long)]
    eth: Option<PathBuf>,
    #[arg(long)]
    bybit: Option<PathBuf>,
    #[arg(long)]
    okx: Option<PathBuf>,
    #[arg(long)]
    bitget: Option<PathBuf>,
    #[arg(long)]
    gateio: Option<PathBuf>,
}

fn main() -> Result<()> {
    let a = Args::parse();

    let t0 = Instant::now();
    let depth = read_depth_parquet(&a.depth)?;
    let trades = a.trades.as_ref().map(|p| read_trades_parquet(p)).transpose()?;
    let funding = a.funding.as_ref().map(|p| read_funding_parquet(p)).transpose()?;
    let derivs = a.derivs.as_ref().map(|p| read_derivatives_parquet(p)).transpose()?;
    let eth = a.eth.as_ref().map(|p| read_trades_parquet(p)).transpose()?;
    let bybit = a.bybit.as_ref().map(|p| read_cross_ex_parquet(p, "bybit")).transpose()?;
    let okx = a.okx.as_ref().map(|p| read_cross_ex_parquet(p, "okx")).transpose()?;
    let bitget = a.bitget.as_ref().map(|p| read_cross_ex_parquet(p, "bitget")).transpose()?;
    let gateio = a.gateio.as_ref().map(|p| read_cross_ex_parquet(p, "gateio")).transpose()?;
    let t_load = t0.elapsed();

    let idx: ndarray::Array1<i64> =
        read_npy(&a.indices).with_context(|| format!("read indices {:?}", a.indices))?;
    let n = depth.n_rows() as i64;
    for &v in idx.iter() {
        anyhow::ensure!(v >= 0 && v < n, "index {} out of range [0, {})", v, n);
    }
    let is = idx.as_slice().unwrap();

    let t1 = Instant::now();
    let mut feat = compute_lob_features(&depth, is);
    fill_microstructure_depth(&mut feat, &depth, is);
    if let Some(tr) = trades.as_ref() {
        fill_trade_features(&mut feat, &depth, is, tr);
        fill_microstructure_trades(&mut feat, &depth, is, tr);
    }
    if let Some(f) = funding.as_ref() {
        fill_funding_features(&mut feat, &depth, is, f);
    }
    if let Some(d) = derivs.as_ref() {
        fill_deriv_features(&mut feat, &depth, is, d);
    }
    if let Some(e) = eth.as_ref() {
        fill_eth_features(&mut feat, &depth, is, e);
    }
    let mut cross: Vec<&scalper_ingest::CrossExTrades> = Vec::new();
    for o in [bybit.as_ref(), okx.as_ref(), bitget.as_ref(), gateio.as_ref()] {
        if let Some(x) = o {
            cross.push(x);
        }
    }
    if !cross.is_empty() {
        fill_cross_ex_feature(&mut feat, &depth, is, &cross);
    }
    let t_feat = t1.elapsed();

    write_npy(&a.out, &feat).with_context(|| format!("write {:?}", a.out))?;
    eprintln!(
        "feature_builder: depth={} trades={} funding={} derivs={} eth={} cross={} samples={} load={:.2}s feat={:.2}s",
        depth.n_rows(),
        trades.as_ref().map(|t| t.len()).unwrap_or(0),
        funding.as_ref().map(|t| t.timestamps.len()).unwrap_or(0),
        derivs.as_ref().map(|t| t.timestamps.len()).unwrap_or(0),
        eth.as_ref().map(|t| t.len()).unwrap_or(0),
        cross.iter().map(|c| c.timestamps.len()).sum::<usize>(),
        feat.nrows(),
        t_load.as_secs_f64(),
        t_feat.as_secs_f64(),
    );
    Ok(())
}
