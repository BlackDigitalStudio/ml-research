//! Golden-parity harness for the full DECISION path: FeatState features -> feat71
//! (btc_lead + ToD) -> per-seed (x-mu)/sd f64->f32 normalize -> Gbt predict ->
//! rank-CDF -> 4-seed ensemble score. Compares bitwise against the Python golden
//! outputs produced on hd2 (`score_golden/`: F71/pA{s}/pBg{s}/score + mu/sd/sA/sBg).
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

use scalper_ingest::features_incr::FeatState;
use scalper_ingest::gbt::Gbt;
use scalper_ingest::{
    read_depth_parquet, read_funding_parquet, read_liquidations_parquet,
    read_open_interest_parquet, read_trades_parquet, DEPTH_LEVELS,
};

#[derive(Parser, Debug)]
struct Args {
    /// Directory with day parquets (book/trades/eth/liq/fund/oi) + idx.npy
    #[arg(long)]
    day_dir: PathBuf,
    /// Directory with golden npys + model jsons (score_golden)
    #[arg(long)]
    golden: PathBuf,
}

const NS: i64 = 1_000_000_000;

fn cdf(refs: &[f64], v: f64) -> f64 {
    refs.partition_point(|&x| x <= v) as f64 / refs.len().max(1) as f64
}

fn main() -> Result<()> {
    let a = Args::parse();
    let d = |n: &str| a.day_dir.join(n);
    let g = |n: &str| a.golden.join(n);

    // ---- replay the day through FeatState (same as fb_incr_harness) ----
    let depth = read_depth_parquet(&d("book.parquet"))?;
    let trades = read_trades_parquet(&d("trades.parquet"))?;
    let eth = read_trades_parquet(&d("eth.parquet"))?;
    let liq = read_liquidations_parquet(&d("liq.parquet"))?;
    let (oi_ts, oi_v) = read_open_interest_parquet(&d("oi.parquet"))?;
    let funding = read_funding_parquet(&d("fund.parquet"))?;
    let idx: Array1<i64> = read_npy(d("idx.npy")).context("idx.npy")?;
    let is = idx.as_slice().unwrap();

    let mut st = FeatState::new();
    st.anchor_rate = Some(funding.funding_rate[0]);
    let d_ts = depth.timestamps.as_slice().unwrap();
    let (mut pt, mut pe, mut pl, mut po) = (0usize, 0usize, 0usize, 0usize);
    let tt = trades.timestamps.as_slice().unwrap();
    let et = eth.timestamps.as_slice().unwrap();
    let lt = liq.timestamps.as_slice().unwrap();
    let ot = oi_ts.as_slice().unwrap();
    let mut bids = [(0f64, 0f64); DEPTH_LEVELS];
    let mut asks = [(0f64, 0f64); DEPTH_LEVELS];
    for i in 0..depth.n_rows() {
        let ts = d_ts[i];
        while pt < trades.len() && tt[pt] <= ts {
            st.push_trade(tt[pt], trades.prices[pt], trades.quantities[pt], trades.is_sell[pt]);
            pt += 1;
        }
        while pe < eth.len() && et[pe] <= ts {
            st.push_eth(et[pe], eth.prices[pe], eth.quantities[pe], eth.is_sell[pe]);
            pe += 1;
        }
        while pl < lt.len() && lt[pl] <= ts {
            st.push_liq(lt[pl], liq.signed_notional[pl], liq.abs_notional[pl]);
            pl += 1;
        }
        while po < ot.len() && ot[po] <= ts {
            st.push_oi(ot[po], oi_v[po]);
            po += 1;
        }
        for k in 0..DEPTH_LEVELS {
            bids[k] = (depth.bid_prices[[i, k]], depth.bid_qtys[[i, k]]);
            asks[k] = (depth.ask_prices[[i, k]], depth.ask_qtys[[i, k]]);
        }
        st.push_book(ts, &bids, &asks);
    }

    // ---- feat71 (btc arrays from golden to share the identical input series) ----
    let btc_ts: Array1<i64> = read_npy(g("btc_ts.npy"))?;
    let btc_mid: Array1<f64> = read_npy(g("btc_mid.npy"))?;
    let bts = btc_ts.as_slice().unwrap();
    let bm = btc_mid.as_slice().unwrap();
    let f71_golden: Array2<f32> = read_npy(g("F71.npy"))?;
    let ns_samples = is.len();
    let mut f71 = vec![[0f32; 71]; ns_samples];
    let t_feat = Instant::now();
    for (s, &raw) in is.iter().enumerate() {
        let x64 = st.compute64(raw as usize);
        let dtd = d_ts[raw as usize] * 1_000_000; // ms -> ns (Python dtd = bt[se] in ns)
        let mut row = [0f64; 71];
        for c in 0..64 {
            row[c] = x64[c] as f64;
        }
        // btc_lead cols 64-66 (exact port of the live/golden feat71)
        let nb = bts.len();
        let i_now = bts.partition_point(|&x| x <= dtd).saturating_sub(1).min(nb - 1);
        let b_now = bm[i_now];
        for (k, wd) in [(0usize, 5i64), (1, 30), (2, 60)] {
            let j = bts.partition_point(|&x| x <= dtd - wd * NS).saturating_sub(1).min(nb - 1);
            let a_ = bm[j];
            row[64 + k] = if a_ > 0.0 && b_now > 0.0 { (b_now / a_).ln() * 1e4 } else { 0.0 };
        }
        let h = ((dtd as f64 / NS as f64) % 86400.0) / 3600.0;
        let hf = h % 8.0;
        let pi = std::f64::consts::PI;
        row[67] = (2.0 * pi * h / 24.0).sin();
        row[68] = (2.0 * pi * h / 24.0).cos();
        row[69] = (2.0 * pi * hf / 8.0).sin();
        row[70] = (2.0 * pi * hf / 8.0).cos();
        for c in 0..71 {
            f71[s][c] = row[c] as f32;
        }
    }
    let feat_el = t_feat.elapsed().as_secs_f64();
    let mut bad71 = 0usize;
    for s in 0..ns_samples {
        for c in 0..71 {
            if f71[s][c].to_bits() != f71_golden[[s, c]].to_bits() {
                if bad71 == 0 {
                    eprintln!("F71 first mismatch: sample {s} col {c}: rust={:?} py={:?}",
                        f71[s][c], f71_golden[[s, c]]);
                }
                bad71 += 1;
            }
        }
    }
    println!("F71: {} mismatched cells / {}", bad71, ns_samples * 71);

    // ---- per-seed normalize + predict + cdf + ensemble ----
    let mut score = vec![0f64; ns_samples];
    let mut pbg_mean = vec![0f64; ns_samples];
    let mut pred_bad = 0usize;
    let mut t_pred = 0f64;
    for s in 0..4usize {
        let mu: Array1<f64> = read_npy(g(&format!("mu{s}.npy")))?;
        let sd: Array1<f64> = read_npy(g(&format!("sd{s}.npy")))?;
        let s_a: Array1<f64> = read_npy(g(&format!("sA{s}.npy")))?;
        let s_bg: Array1<f64> = read_npy(g(&format!("sBg{s}.npy")))?;
        let mut ga = Gbt::load_json(&g(&format!("A{s}.json")))?;
        let mut gb = Gbt::load_json(&g(&format!("Bg{s}.json")))?;
        let ba: Array1<f32> = read_npy(g(&format!("base_A{s}.npy")))?;
        let bb: Array1<f32> = read_npy(g(&format!("base_Bg{s}.npy")))?;
        ga.set_base_margin(ba[0]);
        gb.set_base_margin(bb[0]);
        let pa_g: Array1<f32> = read_npy(g(&format!("pA{s}.npy")))?;
        let pb_g: Array1<f32> = read_npy(g(&format!("pBg{s}.npy")))?;
        let mut fn_row = [0f32; 71];
        for i in 0..ns_samples {
            for c in 0..71 {
                fn_row[c] = ((f71[i][c] as f64 - mu[c]) / sd[c]) as f32;
            }
            let t0 = Instant::now();
            let pa = ga.predict_prob(&fn_row);
            let pb = gb.predict_prob(&fn_row);
            t_pred += t0.elapsed().as_secs_f64();
            if pa.to_bits() != pa_g[i].to_bits() || pb.to_bits() != pb_g[i].to_bits() {
                if pred_bad == 0 {
                    eprintln!("pred first mismatch seed{s} sample {i}: pA rust={:?} py={:?} | pBg rust={:?} py={:?}",
                        pa, pa_g[i], pb, pb_g[i]);
                }
                pred_bad += 1;
            }
            score[i] += cdf(s_a.as_slice().unwrap(), pa as f64)
                * cdf(s_bg.as_slice().unwrap(), (pb - 0.5f32).abs() as f64);
            pbg_mean[i] += pb as f64;
        }
    }
    println!("predictions: {} mismatched / {}", pred_bad, ns_samples * 8);
    let sc_g: Array1<f64> = read_npy(g("score.npy"))?;
    let mut sc_bad = 0usize;
    let mut max_d = 0f64;
    for i in 0..ns_samples {
        let v = score[i] / 4.0;
        if v.to_bits() != sc_g[i].to_bits() {
            sc_bad += 1;
            max_d = max_d.max((v - sc_g[i]).abs());
        }
    }
    println!("ensemble score: {} mismatched / {} (max|d|={:.3e})", sc_bad, ns_samples, max_d);
    println!("timing: feat71+compute64 {:.1}us/sample | 8x predict {:.1}us/sample",
        feat_el / ns_samples as f64 * 1e6, t_pred / ns_samples as f64 * 1e6);
    if bad71 == 0 && pred_bad == 0 && sc_bad == 0 {
        println!("SCORE PATH: BYTE-EXACT");
    } else {
        std::process::exit(1);
    }
    Ok(())
}
