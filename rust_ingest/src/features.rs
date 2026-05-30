//! Hand-crafted features — Rust port of `src/trainer.py::_calc_features_batch`.
//!
//! Session 2 scope: LOB-only features [0,1,2,3,4,5,10,11].
//! Non-LOB slots ([6..9], [12..33]) are left as 0.0 and MUST be filled by
//! later sessions (trade flow, ETH, funding, derivs, microstructure).
//!
//! Parity contract: for the LOB-only cols, output byte-matches Python
//! `_calc_features_batch` when identical inputs are passed.

use ndarray::{s, Array1, Array2};

use crate::{CrossExTrades, DepthData, DerivativesData, FundingData, LiquidationData, TradesData, DEPTH_LEVELS};

// 0..=55 legacy set; 56..=63 sub-60s additions from the FULL raw feeds:
//   56 liq_signed_5s  57 liq_signed_30s  58 liq_intensity_60s
//   59 oi_delta_30s   60 oi_delta_300s   61 OBI_L20
//   62 OBI_L1 (top-of-book)   63 OBI_L10   (OBI@L5 already = col1 imbalance_ratio)
pub const NUM_FEATURES: usize = 64;

/// Must match `src/features.py::QUEUE_DECAY_ALPHA`.
pub const QUEUE_DECAY_ALPHA: f64 = 0.1;

/// sub-60s stream-1: raw 20-level L2 -> (n_ticks, 80) f32. Byte-parity with the
/// Python `_lob_stream_80` (hd2_stream_build) encoding: order [bid_p|bid_s|ask_p|
/// ask_s]; prices (p-mid)/mid; sizes sign*log1p(|s|); non-finite -> 0. Stored f32
/// (caller may cast to f16: (p-mid)/mid ~1e-3, sign*log1p|s| <~15 are f16-safe).
pub fn lob_stream_80(depth: &DepthData) -> Array2<f32> {
    let n = depth.n_rows();
    let mid = depth.mid_prices();
    let mut out = Array2::<f32>::zeros((n, 80));
    for i in 0..n {
        let m = mid[i];
        let pos = m > 0.0;
        let safe = if pos { m } else { 1.0 };
        for k in 0..DEPTH_LEVELS {
            let bp = depth.bid_prices[[i, k]];
            let ap = depth.ask_prices[[i, k]];
            let bs = depth.bid_qtys[[i, k]];
            let aq = depth.ask_qtys[[i, k]];
            let v_bp = if pos { ((bp - m) / safe) as f32 } else { 0.0 };
            let v_ap = if pos { ((ap - m) / safe) as f32 } else { 0.0 };
            let v_bs = (bs.signum() * bs.abs().ln_1p()) as f32;
            let v_as = (aq.signum() * aq.abs().ln_1p()) as f32;
            out[[i, k]] = if v_bp.is_finite() { v_bp } else { 0.0 };
            out[[i, 20 + k]] = if v_bs.is_finite() { v_bs } else { 0.0 };
            out[[i, 40 + k]] = if v_ap.is_finite() { v_ap } else { 0.0 };
            out[[i, 60 + k]] = if v_as.is_finite() { v_as } else { 0.0 };
        }
    }
    out
}

/// Compute features for `indices` into `depth`. Output shape (n_samples, NUM_FEATURES) f32.
///
/// `indices` must be in range [0, depth.n_rows()). Caller's responsibility.
pub fn compute_lob_features(depth: &DepthData, indices: &[i64]) -> Array2<f32> {
    let n = depth.n_rows();
    let ns = indices.len();
    let mut feat = Array2::<f32>::zeros((ns, NUM_FEATURES));

    if n == 0 || ns == 0 {
        return feat;
    }

    // --- pre-compute full-array quantities over all n rows ---

    // bv5, av5 — sum of first 5 levels on each side
    let mut bv5 = Array1::<f64>::zeros(n);
    let mut av5 = Array1::<f64>::zeros(n);
    // large_bid/ask[i] = any of first 5 qty levels > 100
    let mut large_bid = vec![false; n];
    let mut large_ask = vec![false; n];
    for i in 0..n {
        let mut sb = 0.0;
        let mut sa = 0.0;
        let mut lb = false;
        let mut la = false;
        for k in 0..5 {
            let q_b = depth.bid_qtys[[i, k]];
            let q_a = depth.ask_qtys[[i, k]];
            sb += q_b;
            sa += q_a;
            if q_b > 100.0 {
                lb = true;
            }
            if q_a > 100.0 {
                la = true;
            }
        }
        bv5[i] = sb;
        av5[i] = sa;
        large_bid[i] = lb;
        large_ask[i] = la;
    }

    // imb_all = (bv5 - av5) / (bv5 + av5), 0 where denom<=0
    let mut imb_all = Array1::<f64>::zeros(n);
    for i in 0..n {
        let tot = bv5[i] + av5[i];
        if tot > 0.0 {
            imb_all[i] = (bv5[i] - av5[i]) / tot;
        }
    }

    // [0] OFI — np.diff with prepend first value
    //   d_bid[0] = bid_vols[0,0] - bid_vols[0,0] = 0
    //   d_bid[i] = bid_vols[i,0] - bid_vols[i-1,0]
    //   feat[:,0] = (d_bid - d_ask)[indices]
    let mut ofi = Array1::<f64>::zeros(n);
    for i in 1..n {
        let db = depth.bid_qtys[[i, 0]] - depth.bid_qtys[[i - 1, 0]];
        let da = depth.ask_qtys[[i, 0]] - depth.ask_qtys[[i - 1, 0]];
        ofi[i] = db - da;
    }

    // [3] spread_all
    let mut spread = Array1::<f64>::zeros(n);
    for i in 0..n {
        spread[i] = depth.ask_prices[[i, 0]] - depth.bid_prices[[i, 0]];
    }

    // [10] volatility 1s — std of 10-sample sliding window over returns
    // returns_all[i] = (mid[i+1] - mid[i]) / (mid[i] or 1.0 if mid[i]<=0); length n-1.
    // vol_all[j] = std(returns_all[j..j+10]); length max(0, n-10).
    // feat[m10, 10] = vol_all[clip(indices[m10]-10, 0, len-1)]  where m10 = indices>=10.
    let mids = depth.mid_prices(); // Array1<f64>
    let mut returns_all = Array1::<f64>::zeros(n.saturating_sub(1));
    for i in 0..returns_all.len() {
        let base = if mids[i] > 0.0 { mids[i] } else { 1.0 };
        returns_all[i] = (mids[i + 1] - mids[i]) / base;
    }
    let vol_len = returns_all.len().saturating_sub(9); // n-10 (or 0)
    let mut vol_all = Array1::<f64>::zeros(vol_len);
    for j in 0..vol_len {
        let w = &returns_all.slice(s![j..j + 10]);
        // Population std matching numpy default (ddof=0).
        let mean: f64 = w.sum() / 10.0;
        let var: f64 = w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 10.0;
        vol_all[j] = var.sqrt();
    }

    // [11] VWAP deviation — rolling mean of mid over 60000ms window.
    //   left_60s[i_sample] = searchsorted(depth_ts, sample_ts - 60000, side="left")
    //   hi = idx + 1, lo = clip(left_60s, 0, n)
    //   vwap = (cum_mid[hi]-cum_mid[lo]) / counts   where counts = hi-lo
    //   feat[i, 11] = (mid[idx]-vwap)/vwap  if counts>0 & vwap>0 else 0
    let mut cum_mid = Array1::<f64>::zeros(n + 1);
    for i in 0..n {
        cum_mid[i + 1] = cum_mid[i] + mids[i];
    }

    // --- per-sample features ---
    for (s_idx, &raw_idx) in indices.iter().enumerate() {
        let idx = raw_idx as usize;

        // [0] OFI
        feat[[s_idx, 0]] = ofi[idx] as f32;
        // [1] imbalance
        feat[[s_idx, 1]] = imb_all[idx] as f32;
        // [2] imbalance velocity
        if idx >= 5 {
            feat[[s_idx, 2]] = (imb_all[idx] - imb_all[idx - 5]) as f32;
        }
        // [3] spread
        feat[[s_idx, 3]] = spread[idx] as f32;
        // [4] depth ratio L5
        feat[[s_idx, 4]] = if av5[idx] > 0.0 {
            (bv5[idx] / av5[idx]) as f32
        } else {
            10.0
        };
        // [5] large order
        feat[[s_idx, 5]] = if large_bid[idx] || large_ask[idx] {
            1.0
        } else {
            0.0
        };

        // [12] Momentum 5s — (mid[idx] - mid[idx-50]) / mid[idx-50]  if idx>=50 & prev>0
        if idx >= 50 {
            let prev = mids[idx - 50];
            if prev > 0.0 {
                feat[[s_idx, 12]] = ((mids[idx] - prev) / prev) as f32;
            }
        }

        // [10] volatility
        if idx >= 10 && vol_len > 0 {
            let j = (idx - 10).min(vol_len - 1);
            feat[[s_idx, 10]] = vol_all[j] as f32;
        }

        // [11] VWAP deviation — searchsorted(depth_ts, sample_ts - 60000, "left")
        let sample_ts = depth.timestamps[idx];
        let target = sample_ts - 60_000;
        // left: first index i where depth_ts[i] >= target
        let lo = searchsorted_left_i64(depth.timestamps.as_slice().unwrap(), target);
        let hi = idx + 1;
        let count = hi.saturating_sub(lo);
        if count > 0 {
            let vwap = (cum_mid[hi] - cum_mid[lo]) / (count as f64);
            if vwap > 0.0 {
                feat[[s_idx, 11]] = ((mids[idx] - vwap) / vwap) as f32;
            }
        }
    }

    feat
}

/// Fill cols [6,7,8,9] from trades — trade-flow features.
///
///   [6] trade flow imbalance, 5s window: (buys-sells)/(buys+sells)
///   [7] trade intensity, 1s window:       count(trades in [ts-1000, ts])
///   [8] large trade flag, 5s window:      any(qty > 10) ? 1.0 : 0.0
///   [9] CVD, 30s window:                  cumsum(buy) - cumsum(sell)
///
/// Python semantics (from `_calc_features_batch`):
///   cum_buy  = cumsum(qty * ~is_sell)
///   cum_sell = cumsum(qty * is_sell)
///   cum_large = cumsum(qty > 10)
///   right = searchsorted(trade_ts, sample_ts, "right")
///   left_Ns = searchsorted(trade_ts, sample_ts - N_ms, "left")
pub fn fill_trade_features(
    feat: &mut Array2<f32>,
    depth: &DepthData,
    indices: &[i64],
    trades: &TradesData,
) {
    let nt = trades.len();
    if nt == 0 {
        return;
    }

    let t_ts = trades.timestamps.as_slice().unwrap();
    let t_qty = trades.quantities.as_slice().unwrap();
    let is_sell = &trades.is_sell;

    let mut cum_buy = Array1::<f64>::zeros(nt + 1);
    let mut cum_sell = Array1::<f64>::zeros(nt + 1);
    let mut cum_large = Array1::<f64>::zeros(nt + 1);
    for i in 0..nt {
        let q = t_qty[i];
        if is_sell[i] {
            cum_sell[i + 1] = cum_sell[i] + q;
            cum_buy[i + 1] = cum_buy[i];
        } else {
            cum_buy[i + 1] = cum_buy[i] + q;
            cum_sell[i + 1] = cum_sell[i];
        }
        cum_large[i + 1] = cum_large[i] + if q > 10.0 { 1.0 } else { 0.0 };
    }

    for (s_idx, &raw_idx) in indices.iter().enumerate() {
        let idx = raw_idx as usize;
        let sample_ts = depth.timestamps[idx];
        let right = searchsorted_right_i64(t_ts, sample_ts);
        let left_5s = searchsorted_left_i64(t_ts, sample_ts - 5_000);
        let left_1s = searchsorted_left_i64(t_ts, sample_ts - 1_000);
        let left_30s = searchsorted_left_i64(t_ts, sample_ts - 30_000);

        // [6]
        let buys5 = cum_buy[right] - cum_buy[left_5s];
        let sells5 = cum_sell[right] - cum_sell[left_5s];
        let tot5 = buys5 + sells5;
        feat[[s_idx, 6]] = if tot5 > 0.0 {
            ((buys5 - sells5) / tot5) as f32
        } else {
            0.0
        };
        // [7]
        feat[[s_idx, 7]] = (right as f64 - left_1s as f64) as f32;
        // [8]
        feat[[s_idx, 8]] = if cum_large[right] - cum_large[left_5s] > 0.0 {
            1.0
        } else {
            0.0
        };
        // [9]
        feat[[s_idx, 9]] =
            ((cum_buy[right] - cum_buy[left_30s]) - (cum_sell[right] - cum_sell[left_30s])) as f32;
    }
}

/// Fill microstructure features computed from depth only:
///   [20] spoof approximation
///   [21] volatility ratio (curr / 30-tick rolling avg)
///   [23] Hurst exponent (R/S, 100-tick window)
///   [24] sweep intensity
///   [25] cancel rate diff (ask - bid, 10-tick window)
///   [26] OFI 1s  (sum of 10 raw OFI ticks)
///   [27] OFI 5s  (50 ticks)
///   [28] OFI 30s (300 ticks)
///   [29] OFI divergence (1s vs 30s when signs differ)
///   [31] queue pressure EMA diff (ask_decay - bid_decay)
///   [32] top3 asymmetry  (bid_share_top3 - ask_share_top3)
pub fn fill_microstructure_depth(feat: &mut Array2<f32>, depth: &DepthData, indices: &[i64]) {
    let n = depth.n_rows();
    if n == 0 {
        return;
    }
    let ns = indices.len();
    let bp = &depth.bid_prices;
    let ap = &depth.ask_prices;
    let bq = &depth.bid_qtys;
    let aq = &depth.ask_qtys;
    let mids = depth.mid_prices();

    // ofi_raw[i] = (bid_qty[i,0]-bid_qty[i-1,0]) - (ask_qty[i,0]-ask_qty[i-1,0]); ofi_raw[0]=0
    let mut ofi_raw = vec![0f64; n];
    for i in 1..n {
        ofi_raw[i] = (bq[[i, 0]] - bq[[i - 1, 0]]) - (aq[[i, 0]] - aq[[i - 1, 0]]);
    }

    // vol_all (needed for [21]) — same as [10] logic.
    let mut returns_all = vec![0f64; n.saturating_sub(1)];
    for i in 0..returns_all.len() {
        let base = if mids[i] > 0.0 { mids[i] } else { 1.0 };
        returns_all[i] = (mids[i + 1] - mids[i]) / base;
    }
    let vol_len = returns_all.len().saturating_sub(9); // n-10 when n>=10
    let mut vol_all = vec![0f64; vol_len];
    for j in 0..vol_len {
        let w = &returns_all[j..j + 10];
        let mean = w.iter().sum::<f64>() / 10.0;
        let var = w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 10.0;
        vol_all[j] = var.sqrt();
    }
    // vol_mean_all[k] = mean(vol_all[k..k+30]); len = vol_len-29 when vol_len>=30.
    let vol_mean_len = vol_len.saturating_sub(29);
    let mut vol_mean_all = vec![0f64; vol_mean_len];
    for k in 0..vol_mean_len {
        vol_mean_all[k] = vol_all[k..k + 30].iter().sum::<f64>() / 30.0;
    }

    // log_ret for Hurst
    let mut log_ret = vec![0f64; n.saturating_sub(1)];
    for i in 0..log_ret.len() {
        log_ret[i] = (mids[i + 1] + 1e-10).ln() - (mids[i] + 1e-10).ln();
    }
    // all_hurst[j] = Hurst over log_ret[j..j+100]; default 0.5; len = lr_len-99.
    let hurst_len = log_ret.len().saturating_sub(99);
    let mut all_hurst = vec![0.5f64; hurst_len];
    for j in 0..hurst_len {
        let chunk = &log_ret[j..j + 100];
        let mean = chunk.iter().sum::<f64>() / 100.0;
        let mut dev = 0.0;
        let mut dmin = f64::INFINITY;
        let mut dmax = f64::NEG_INFINITY;
        for &x in chunk {
            dev += x - mean;
            if dev < dmin {
                dmin = dev;
            }
            if dev > dmax {
                dmax = dev;
            }
        }
        let var = chunk.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 100.0;
        let s = var.sqrt();
        if s > 0.0 {
            let r = dmax - dmin;
            // numpy: np.log(r/(s+1e-10))/np.log(100), clipped to [0, 1]
            let h = (r / (s + 1e-10)).ln() / 100f64.ln();
            all_hurst[j] = h.clamp(0.0, 1.0);
        } else {
            all_hurst[j] = 0.5;
        }
    }

    // Cancel-rate rolling 10-tick sums (ask_cancel - bid_cancel).
    // bid_cancel_tick[0]=0; bid_cancel_tick[i] = max(0, bid_qty[i-1,:5].sum - bid_qty[i,:5].sum_where).
    // But Python does per-level: bid_vol_diff = np.diff(bid_vols[:,:5], axis=0); cancel = max(0, -diff).sum(axis=1).
    // So sum over levels of max(0, -(bq[i,k]-bq[i-1,k])).
    let mut bid_cancel_tick = vec![0f64; n];
    let mut ask_cancel_tick = vec![0f64; n];
    for i in 1..n {
        let mut sb = 0.0;
        let mut sa = 0.0;
        for k in 0..5 {
            let db = bq[[i, k]] - bq[[i - 1, k]];
            let da = aq[[i, k]] - aq[[i - 1, k]];
            if db < 0.0 {
                sb += -db;
            }
            if da < 0.0 {
                sa += -da;
            }
        }
        bid_cancel_tick[i] = sb;
        ask_cancel_tick[i] = sa;
    }
    // Rolling 10-tick sums — length n-9 when n>=10.
    let cancel_win_len = n.saturating_sub(9);
    let mut bc_win = vec![0f64; cancel_win_len];
    let mut ac_win = vec![0f64; cancel_win_len];
    if cancel_win_len > 0 {
        let mut sb = 0.0;
        let mut sa = 0.0;
        for i in 0..10 {
            sb += bid_cancel_tick[i];
            sa += ask_cancel_tick[i];
        }
        bc_win[0] = sb;
        ac_win[0] = sa;
        for k in 1..cancel_win_len {
            sb += bid_cancel_tick[k + 9] - bid_cancel_tick[k - 1];
            sa += ask_cancel_tick[k + 9] - ask_cancel_tick[k - 1];
            bc_win[k] = sb;
            ac_win[k] = sa;
        }
    }

    // OFI rolling sums for windows 10, 50, 300.
    let rolling_sum = |data: &[f64], w: usize| -> Vec<f64> {
        let len = data.len().saturating_sub(w - 1);
        let mut out = vec![0f64; len];
        if len == 0 {
            return out;
        }
        let mut s = 0.0;
        for i in 0..w {
            s += data[i];
        }
        out[0] = s;
        for k in 1..len {
            s += data[k + w - 1] - data[k - 1];
            out[k] = s;
        }
        out
    };
    let ofi_1s = rolling_sum(&ofi_raw, 10);
    let ofi_5s = rolling_sum(&ofi_raw, 50);
    let ofi_30s = rolling_sum(&ofi_raw, 300);

    // [31] queue pressure EMA (over all n rows), using L1 decay.
    let mut bid_ema = vec![0f64; n];
    let mut ask_ema = vec![0f64; n];
    let a = QUEUE_DECAY_ALPHA;
    let mut b_acc = 0.0;
    let mut s_acc = 0.0;
    for i in 0..n {
        // bid_decay[0]=0; bid_decay[i] = max(0, bid_l1[i-1] - bid_l1[i]).
        let (bd, ad) = if i == 0 {
            (0.0, 0.0)
        } else {
            (
                (bq[[i - 1, 0]] - bq[[i, 0]]).max(0.0),
                (aq[[i - 1, 0]] - aq[[i, 0]]).max(0.0),
            )
        };
        b_acc = a * bd + (1.0 - a) * b_acc;
        s_acc = a * ad + (1.0 - a) * s_acc;
        bid_ema[i] = b_acc;
        ask_ema[i] = s_acc;
    }

    // [32] top3 asymmetry — bid_share - ask_share.
    let mut top3_asym = vec![0f64; n];
    for i in 0..n {
        let mut t3b = 0.0;
        let mut t20b = 0.0;
        let mut t3a = 0.0;
        let mut t20a = 0.0;
        for k in 0..20 {
            let qb = bq[[i, k]];
            let qa = aq[[i, k]];
            t20b += qb;
            t20a += qa;
            if k < 3 {
                t3b += qb;
                t3a += qa;
            }
        }
        top3_asym[i] = t3b / (t20b + 1e-9) - t3a / (t20a + 1e-9);
    }

    // --- per-sample fills ---
    for (s_idx, &raw_idx) in indices.iter().enumerate() {
        let idx = raw_idx as usize;

        // [20] Spoof
        if idx >= 25 && feat[[s_idx, 5]] > 0.0 {
            let prev = if idx >= 25 { mids[idx - 25] } else { mids[0] };
            let pc = (mids[idx] - prev).abs();
            if pc < 0.10 {
                feat[[s_idx, 20]] = 1.0;
            }
        }

        // [21] Volatility ratio — requires idx>=40 and vol_mean_all available.
        if idx >= 40 && vol_mean_len > 0 && vol_len > 0 {
            let adj_vr = (idx - 40).min(vol_mean_len - 1);
            let adj_v = (idx - 10).min(vol_len - 1);
            let vm = vol_mean_all[adj_vr];
            feat[[s_idx, 21]] = if vm > 0.0 {
                (vol_all[adj_v] / (vm + 1e-10)) as f32
            } else {
                1.0
            };
        }

        // [23] Hurst
        if idx >= 100 && hurst_len > 0 {
            let adj = (idx - 100).min(hurst_len - 1);
            feat[[s_idx, 23]] = all_hurst[adj] as f32;
        } else {
            feat[[s_idx, 23]] = 0.5;
        }

        // [24] Sweep — max(bid_jump, ask_jump) - 1, clamped to >=0.
        if idx >= 1 {
            let tick = 0.10;
            let bj = (bp[[idx, 0]] - bp[[idx - 1, 0]]).abs() / tick;
            let aj = (ap[[idx, 0]] - ap[[idx - 1, 0]]).abs() / tick;
            let mx = bj.max(aj) - 1.0;
            feat[[s_idx, 24]] = mx.max(0.0) as f32;
        }

        // [25] Cancel diff — ask_win - bid_win at idx-10.
        if idx >= 10 && cancel_win_len > 0 {
            let adj = (idx - 10).min(cancel_win_len - 1);
            feat[[s_idx, 25]] = (ac_win[adj] - bc_win[adj]) as f32;
        }

        // [26-28] OFI windows
        if idx >= 10 && !ofi_1s.is_empty() {
            let a = (idx - 10).min(ofi_1s.len() - 1);
            feat[[s_idx, 26]] = ofi_1s[a] as f32;
        }
        if idx >= 50 && !ofi_5s.is_empty() {
            let a = (idx - 50).min(ofi_5s.len() - 1);
            feat[[s_idx, 27]] = ofi_5s[a] as f32;
        }
        if idx >= 300 && !ofi_30s.is_empty() {
            let a = (idx - 300).min(ofi_30s.len() - 1);
            feat[[s_idx, 28]] = ofi_30s[a] as f32;
        }
        // [29] divergence (needs 300 history for the 30s window)
        if idx >= 300 {
            let short = feat[[s_idx, 26]];
            let long_ = feat[[s_idx, 28]];
            if (short as f64) * (long_ as f64) < 0.0 {
                feat[[s_idx, 29]] = short - long_;
            }
        }

        // [31]
        feat[[s_idx, 31]] = (ask_ema[idx] - bid_ema[idx]) as f32;
        // [32]
        feat[[s_idx, 32]] = top3_asym[idx] as f32;
    }
}

/// Extend cols [22] and [33] — trade-ratio + effective spread EMA.
/// Must be called AFTER fill_trade_features (shares the same trades).
pub fn fill_microstructure_trades(
    feat: &mut Array2<f32>,
    depth: &DepthData,
    indices: &[i64],
    trades: &TradesData,
) {
    let n = depth.n_rows();
    let nt = trades.len();
    if n == 0 {
        return;
    }
    let d_ts = depth.timestamps.as_slice().unwrap();
    let t_ts = trades.timestamps.as_slice().unwrap();
    let t_price = trades.prices.as_slice().unwrap();

    // [22] trade intensity ratio
    // tick_intensity[i] = # trades with depth_tick == i.
    // t_tick_idx[k] = clip(searchsorted(depth_ts, trade_ts[k], "right") - 1, 0, n-1)
    let mut tick_intensity = vec![0f64; n];
    for k in 0..nt {
        let r = searchsorted_right_i64(d_ts, t_ts[k]);
        let ti = if r == 0 { 0 } else { (r - 1).min(n - 1) };
        tick_intensity[ti] += 1.0;
    }
    // curr_int[k] = sum(tick_intensity[k..k+10]); len n-9.
    let ci_len = n.saturating_sub(9);
    let mut curr_int = vec![0f64; ci_len];
    if ci_len > 0 {
        let mut s = 0.0;
        for i in 0..10 {
            s += tick_intensity[i];
        }
        curr_int[0] = s;
        for k in 1..ci_len {
            s += tick_intensity[k + 9] - tick_intensity[k - 1];
            curr_int[k] = s;
        }
    }
    // int_mean_all[k] = mean(curr_int[k..k+30]); len = ci_len - 29.
    let im_len = ci_len.saturating_sub(29);
    let mut int_mean_all = vec![0f64; im_len];
    if im_len > 0 {
        let mut s = 0.0;
        for i in 0..30 {
            s += curr_int[i];
        }
        int_mean_all[0] = s / 30.0;
        for k in 1..im_len {
            s += curr_int[k + 29] - curr_int[k - 1];
            int_mean_all[k] = s / 30.0;
        }
    }

    // Guarded in Python by `len(tick_intensity) >= 40` and `len(curr_int) >= 30`.
    let col22_valid = n >= 40 && ci_len >= 30;

    // [33] effective spread EMA across all depth ticks.
    let a = QUEUE_DECAY_ALPHA;
    let mut eff_ema = vec![0f64; n];
    let mut e_acc = 0.0;
    let spread = |i: usize| -> f64 {
        let s = depth.ask_prices[[i, 0]] - depth.bid_prices[[i, 0]];
        s.max(1e-9)
    };
    let mid_at = |i: usize| -> f64 {
        let bb = depth.bid_prices[[i, 0]];
        let aa = depth.ask_prices[[i, 0]];
        if bb > 0.0 && aa > 0.0 {
            0.5 * (bb + aa)
        } else {
            0.0
        }
    };
    if nt > 0 {
        // last_trade_idx_for_depth_tick[i] = clip(searchsorted(t_ts, d_ts[i], "right") - 1, 0, nt-1)
        // valid_lt = r > 0 (Python: last_trade_idx >= 0)
        for i in 0..n {
            let r = searchsorted_right_i64(t_ts, d_ts[i]);
            let lt = if r == 0 { 0 } else { (r - 1).min(nt - 1) };
            let m = mid_at(i);
            let sp = spread(i);
            let ratio = if r > 0 {
                (t_price[lt] - m).abs() / sp
            } else {
                0.0
            };
            e_acc = a * ratio + (1.0 - a) * e_acc;
            eff_ema[i] = e_acc;
        }
    } else {
        for i in 0..n {
            e_acc = (1.0 - a) * e_acc;
            eff_ema[i] = e_acc;
        }
    }

    for (s_idx, &raw_idx) in indices.iter().enumerate() {
        let idx = raw_idx as usize;
        if col22_valid && idx >= 40 {
            let adj_ci = (idx - 10).min(ci_len - 1);
            let adj_im = (idx - 40).min(im_len - 1);
            let im = int_mean_all[adj_im];
            feat[[s_idx, 22]] = if im > 0.0 {
                (curr_int[adj_ci] / (im + 1e-10)) as f32
            } else {
                1.0
            };
        }
        feat[[s_idx, 33]] = eff_ema[idx] as f32;
    }
}

/// Fill cols [14,15,16] from ETH trades.
///
///   [14] eth_momentum_1s: change in 1s VWAP vs previous 1s VWAP
///   [15] eth_ofi (500ms):  (buys-sells)/(buys+sells)
///   [16] eth_leading_signal: (btc_mid/eth_mid - mean_ratio) / (mean_ratio + 1e-10)
///        where mean_ratio is computed across all samples with ratio>0.
/// Last trade price at-or-before `t` (0.0 if none). For clean leading-asset
/// point-to-point returns (validated strongest sub-60s directional feature).
fn last_price_at(ts: &[i64], price: &[f64], t: i64) -> f64 {
    let j = searchsorted_right_i64(ts, t);
    if j > 0 { price[j - 1] } else { 0.0 }
}

#[inline]
fn eth_logret(p_now: f64, p_prev: f64) -> f64 {
    if p_now > 0.0 && p_prev > 0.0 { (p_now / p_prev).ln() } else { 0.0 }
}

pub fn fill_eth_features(
    feat: &mut Array2<f32>,
    depth: &DepthData,
    indices: &[i64],
    eth: &TradesData,
) {
    let nt = eth.len();
    if nt == 0 {
        return;
    }
    let ts = eth.timestamps.as_slice().unwrap();
    let qty = eth.quantities.as_slice().unwrap();
    let price = eth.prices.as_slice().unwrap();
    let is_sell = &eth.is_sell;

    // cumulative arrays (length nt+1)
    let mut cum_buy = vec![0f64; nt + 1];
    let mut cum_sell = vec![0f64; nt + 1];
    let mut cum_pv = vec![0f64; nt + 1];
    let mut cum_qty = vec![0f64; nt + 1];
    for i in 0..nt {
        let q = qty[i];
        if is_sell[i] {
            cum_sell[i + 1] = cum_sell[i] + q;
            cum_buy[i + 1] = cum_buy[i];
        } else {
            cum_buy[i + 1] = cum_buy[i] + q;
            cum_sell[i + 1] = cum_sell[i];
        }
        cum_pv[i + 1] = cum_pv[i] + price[i] * q;
        cum_qty[i + 1] = cum_qty[i] + q;
    }

    let mids = depth.mid_prices();
    let ns = indices.len();

    // first pass: compute vwap_1s (for both [14] and [16] it's the eth_mid)
    let mut vwap_1s = vec![0f64; ns];
    let mut vwap_prev = vec![0f64; ns];
    let mut flow_imb = vec![0f64; ns];

    for (s_idx, &raw_idx) in indices.iter().enumerate() {
        let idx = raw_idx as usize;
        let sample_ts = depth.timestamps[idx];
        let right = searchsorted_right_i64(ts, sample_ts);
        let left_1s = searchsorted_left_i64(ts, sample_ts - 1_000);
        let left_2s = searchsorted_left_i64(ts, sample_ts - 2_000);
        let left_500 = searchsorted_left_i64(ts, sample_ts - 500);

        let qn = cum_qty[right] - cum_qty[left_1s];
        let pn = cum_pv[right] - cum_pv[left_1s];
        vwap_1s[s_idx] = if qn > 0.0 { pn / qn } else { 0.0 };

        let qp = cum_qty[left_1s] - cum_qty[left_2s];
        let pp = cum_pv[left_1s] - cum_pv[left_2s];
        vwap_prev[s_idx] = if qp > 0.0 { pp / qp } else { 0.0 };

        let bz = cum_buy[right] - cum_buy[left_1s];
        let sz = cum_sell[right] - cum_sell[left_1s];
        let tot = bz + sz;
        flow_imb[s_idx] = if tot > 0.0 { (bz - sz) / tot } else { 0.0 };

        // CORRECTED sub-60s ETH lead-lag. The old defs (1s-VWAP-diff [14],
        // BTC/ETH-ratio dev [16]) scored IC_30s ~0.03 / ~0.01; the validated
        // strong signal is the clean point-to-point ETH log-return (IC_30s
        // ~0.13). Model consumes stream-2 as an opaque vector, so columns are
        // repurposed: [14]=eth_ret_1s [16]=eth_ret_2s [54]=eth_ret_5s,
        // [15]=eth flow-imbalance (1s window), [55] left as eth_btc_corr.
        let p_now = last_price_at(ts, price, sample_ts);
        feat[[s_idx, 14]] = eth_logret(p_now, last_price_at(ts, price, sample_ts - 1_000)) as f32;
        feat[[s_idx, 16]] = eth_logret(p_now, last_price_at(ts, price, sample_ts - 2_000)) as f32;
        feat[[s_idx, 54]] = eth_logret(p_now, last_price_at(ts, price, sample_ts - 5_000)) as f32;
        feat[[s_idx, 15]] = flow_imb[s_idx] as f32;
    }
}

/// Cols [56,57,58] — liquidation signed-notional imbalance (5s, 30s) + intensity
/// (60s). Raw liquidations are sparse impulse events: side "buy" = short-position
/// liquidation (forced buy -> UP pressure) => +notional; "sell" = long liquidation
/// (down pressure) => -notional. Mostly 0 between cascades (an event feature).
pub fn fill_liquidation_features(
    feat: &mut Array2<f32>,
    depth: &DepthData,
    indices: &[i64],
    liq: &LiquidationData,
) {
    let nt = liq.timestamps.len();
    if nt == 0 {
        return;
    }
    let ts = liq.timestamps.as_slice().unwrap();
    let sn = liq.signed_notional.as_slice().unwrap();
    let an = liq.abs_notional.as_slice().unwrap();
    let mut csn = vec![0f64; nt + 1];
    let mut can = vec![0f64; nt + 1];
    for i in 0..nt {
        csn[i + 1] = csn[i] + sn[i];
        can[i + 1] = can[i] + an[i];
    }
    for (s, &raw) in indices.iter().enumerate() {
        let t = depth.timestamps[raw as usize];
        let r = searchsorted_right_i64(ts, t);
        let l5 = searchsorted_left_i64(ts, t - 5_000);
        let l30 = searchsorted_left_i64(ts, t - 30_000);
        let l60 = searchsorted_left_i64(ts, t - 60_000);
        let a5 = can[r] - can[l5];
        let a30 = can[r] - can[l30];
        feat[[s, 56]] = if a5 > 0.0 { ((csn[r] - csn[l5]) / a5) as f32 } else { 0.0 };
        feat[[s, 57]] = if a30 > 0.0 { ((csn[r] - csn[l30]) / a30) as f32 } else { 0.0 };
        feat[[s, 58]] = ((can[r] - can[l60]).max(0.0) + 1.0).ln() as f32;
    }
}

/// Cols [59,60] — open-interest delta over 30s / 300s: (OI_t - OI_{t-W}) / OI_t.
/// Positioning-regime shift (raw/open_interest, ~4 s cadence).
pub fn fill_oi_features(
    feat: &mut Array2<f32>,
    depth: &DepthData,
    indices: &[i64],
    oi_ts: &[i64],
    oi: &[f64],
) {
    if oi_ts.is_empty() {
        return;
    }
    for (s, &raw) in indices.iter().enumerate() {
        let t = depth.timestamps[raw as usize];
        let r = searchsorted_right_i64(oi_ts, t);
        if r == 0 {
            continue;
        }
        let now = oi[r - 1];
        if now <= 0.0 {
            continue;
        }
        let j30 = searchsorted_right_i64(oi_ts, t - 30_000);
        let j300 = searchsorted_right_i64(oi_ts, t - 300_000);
        if j30 > 0 {
            feat[[s, 59]] = ((now - oi[j30 - 1]) / now) as f32;
        }
        if j300 > 0 {
            feat[[s, 60]] = ((now - oi[j300 - 1]) / now) as f32;
        }
    }
}

/// OBI depth-ladder: [61]=OBI@L20, [62]=OBI@L1 (top-of-book), [63]=OBI@L10.
/// OBI@L5 already exists as col1 (imbalance_ratio). OBI = (Σbid-Σask)/(Σbid+Σask).
pub fn fill_deep_book(feat: &mut Array2<f32>, depth: &DepthData, indices: &[i64]) {
    for (s, &raw) in indices.iter().enumerate() {
        let i = raw as usize;
        let b1 = depth.bid_qtys[[i, 0]];
        let a1 = depth.ask_qtys[[i, 0]];
        let mut b10 = 0f64;
        let mut a10 = 0f64;
        let mut b20 = 0f64;
        let mut a20 = 0f64;
        for k in 0..DEPTH_LEVELS {
            let bq = depth.bid_qtys[[i, k]];
            let aq = depth.ask_qtys[[i, k]];
            b20 += bq;
            a20 += aq;
            if k < 10 {
                b10 += bq;
                a10 += aq;
            }
        }
        let t1 = b1 + a1;
        let t10 = b10 + a10;
        let t20 = b20 + a20;
        feat[[s, 61]] = if t20 > 0.0 { ((b20 - a20) / t20) as f32 } else { 0.0 };
        feat[[s, 62]] = if t1 > 0.0 { ((b1 - a1) / t1) as f32 } else { 0.0 };
        feat[[s, 63]] = if t10 > 0.0 { ((b10 - a10) / t10) as f32 } else { 0.0 };
    }
}

/// Fill col [30] — count of cross-exchanges with net signed volume > 0
/// in the last 500ms.  Adds +1 per exchange whose cumsum diff in the
/// window is strictly positive. Missing feeds contribute 0.
pub fn fill_cross_ex_feature(
    feat: &mut Array2<f32>,
    depth: &DepthData,
    indices: &[i64],
    cross: &[&CrossExTrades],
) {
    for ex in cross {
        if ex.timestamps.is_empty() {
            continue;
        }
        let ex_ts = ex.timestamps.as_slice().unwrap();
        let ex_sq = ex.signed_qty.as_slice().unwrap();
        let nt = ex_ts.len();
        let mut cum = vec![0f64; nt + 1];
        for i in 0..nt {
            cum[i + 1] = cum[i] + ex_sq[i];
        }
        for (s_idx, &raw_idx) in indices.iter().enumerate() {
            let sample_ts = depth.timestamps[raw_idx as usize];
            let right = searchsorted_right_i64(ex_ts, sample_ts);
            let left = searchsorted_left_i64(ex_ts, sample_ts - 500);
            let net = cum[right] - cum[left];
            if net > 0.0 {
                feat[[s_idx, 30]] += 1.0;
            }
        }
    }
}

/// Fill col [13] funding_rate: latest rate at or before sample_ts.
///
/// Python: `fund_idx = clip(searchsorted(ts, sample_ts, "right") - 1, 0, len-1)`
pub fn fill_funding_features(
    feat: &mut Array2<f32>,
    depth: &DepthData,
    indices: &[i64],
    funding: &FundingData,
) {
    if funding.timestamps.is_empty() {
        return;
    }
    let f_ts = funding.timestamps.as_slice().unwrap();
    let f_rate = funding.funding_rate.as_slice().unwrap();
    let n_fund = f_ts.len();

    for (s_idx, &raw_idx) in indices.iter().enumerate() {
        let idx = raw_idx as usize;
        let sample_ts = depth.timestamps[idx];
        let r = searchsorted_right_i64(f_ts, sample_ts);
        let fi = if r == 0 { 0 } else { (r - 1).min(n_fund - 1) };
        feat[[s_idx, 13]] = f_rate[fi] as f32;
    }
}

/// Fill cols [17,18,19] from derivatives parquet.
///
///   [17] OI delta vs previous poll: (oi_now - oi_prev) / oi_prev  if oi_prev>0
///   [18] long_short_ratio
///   [19] liquidation proximity heuristic:
///          -0.015 if ls > 1.2 (longs crowded → cluster below)
///          +0.015 if ls < 0.8 (shorts crowded → cluster above)
///          0      otherwise
pub fn fill_deriv_features(
    feat: &mut Array2<f32>,
    depth: &DepthData,
    indices: &[i64],
    derivs: &DerivativesData,
) {
    // Python gate is `len(deriv_ts) > 1`; respect it so feat[:,17..19] stays 0
    // when only one row is available.
    if derivs.timestamps.len() <= 1 {
        return;
    }
    let d_ts = derivs.timestamps.as_slice().unwrap();
    let d_oi = derivs.open_interest.as_slice().unwrap();
    let d_ls = derivs.long_short_ratio.as_slice().unwrap();
    let nd = d_ts.len();
    const CLUSTER_PCT: f64 = 0.015;

    for (s_idx, &raw_idx) in indices.iter().enumerate() {
        let idx = raw_idx as usize;
        let sample_ts = depth.timestamps[idx];
        let r = searchsorted_right_i64(d_ts, sample_ts);
        let di = if r == 0 { 0 } else { (r - 1).min(nd - 1) };
        let di_prev = if di == 0 { 0 } else { di - 1 };

        let oi_now = d_oi[di];
        let oi_prev = d_oi[di_prev];
        feat[[s_idx, 17]] = if oi_prev > 0.0 {
            ((oi_now - oi_prev) / oi_prev) as f32
        } else {
            0.0
        };
        feat[[s_idx, 18]] = d_ls[di] as f32;
        let ls = d_ls[di];
        feat[[s_idx, 19]] = if ls > 1.2 {
            -CLUSTER_PCT as f32
        } else if ls < 0.8 {
            CLUSTER_PCT as f32
        } else {
            0.0
        };
    }
}

/// Fill cols [34..=39] — horizon-tier momentum / realised vol / bipower.
///
/// Must match `src/features_ext.py::compute_ext_features_batch` bit-for-bit
/// in f32.
///
///   [34] momentum_30s      = (mid[T] - mid[T-300]) / mid[T-300]
///   [35] momentum_60s      = (mid[T] - mid[T-600]) / mid[T-600]
///   [36] momentum_120s     = (mid[T] - mid[T-1200]) / mid[T-1200]
///   [37] realized_vol_60s  = sqrt(Σ r[k]²) for k in [T-600, T-1]
///   [38] realized_vol_120s = sqrt(Σ r[k]²) for k in [T-1200, T-1]
///   [39] bipower_var_120s  = (π/2) · Σ |r[k]|·|r[k-1]| for k in [T-1199, T-1]
///
/// where r[k] = log(mid[k+1]) - log(mid[k]). Features emit 0 until their
/// window is saturated (same convention as [10]/[12]).
pub fn fill_horizon_features(feat: &mut Array2<f32>, depth: &DepthData, indices: &[i64]) {
    const W30: i64 = 300;
    const W60: i64 = 600;
    const W120: i64 = 1200;
    let bv_scale: f64 = std::f64::consts::FRAC_PI_2;

    let mid = depth.mid_prices();
    let n = mid.len();
    if n < 2 {
        return;
    }

    // Per-tick log-returns: r[k] = log(mid[k+1]) - log(mid[k]) for k in [0, n-2].
    // Zero when either side is non-positive, matching the Python gate.
    let mut r = Array1::<f64>::zeros(n - 1);
    let mut abs_r = Array1::<f64>::zeros(n - 1);
    for k in 0..(n - 1) {
        let a = mid[k];
        let b = mid[k + 1];
        if a > 0.0 && b > 0.0 {
            let v = b.ln() - a.ln();
            r[k] = v;
            abs_r[k] = v.abs();
        }
    }

    // cum_sq[i] = Σ_{k=0}^{i-1} r[k]². Length n (so cum_sq[T] - cum_sq[T-W]
    // is the sum over [T-W, T-1], i.e. last W returns ending at tick T).
    let mut cum_sq = Array1::<f64>::zeros(n);
    let mut s = 0.0;
    for k in 0..(n - 1) {
        s += r[k] * r[k];
        cum_sq[k + 1] = s;
    }

    // pair[j] = |r[j+1]| · |r[j]|  for j in [0, n-3], with right-return-index (j+1).
    // cum_pair[k] with cum_pair[0] = cum_pair[1] = 0 and, for k >= 2,
    //   cum_pair[k] = Σ_{j=0}^{k-2} pair[j]
    // so BV sum on window W at tick T = cum_pair[T] - cum_pair[T - W + 1].
    let mut cum_pair = Array1::<f64>::zeros(n);
    if n >= 3 {
        let mut sp = 0.0;
        for k in 2..n {
            // pair[k-2] = |r[k-1]| * |r[k-2]|
            sp += abs_r[k - 1] * abs_r[k - 2];
            cum_pair[k] = sp;
        }
    }

    for (s_idx, &raw_idx) in indices.iter().enumerate() {
        let t = raw_idx;
        let ti = t as usize;
        let cur = mid[ti];

        // [34] momentum_30s
        if t >= W30 {
            let past = mid[(t - W30) as usize];
            if past > 0.0 && cur > 0.0 {
                feat[[s_idx, 34]] = ((cur - past) / past) as f32;
            }
        }
        // [35] momentum_60s
        if t >= W60 {
            let past = mid[(t - W60) as usize];
            if past > 0.0 && cur > 0.0 {
                feat[[s_idx, 35]] = ((cur - past) / past) as f32;
            }
        }
        // [36] momentum_120s
        if t >= W120 {
            let past = mid[(t - W120) as usize];
            if past > 0.0 && cur > 0.0 {
                feat[[s_idx, 36]] = ((cur - past) / past) as f32;
            }
        }
        // [37] realized_vol_60s
        if t >= W60 {
            let rv = cum_sq[ti] - cum_sq[(t - W60) as usize];
            feat[[s_idx, 37]] = rv.max(0.0).sqrt() as f32;
        }
        // [38] realized_vol_120s
        if t >= W120 {
            let rv = cum_sq[ti] - cum_sq[(t - W120) as usize];
            feat[[s_idx, 38]] = rv.max(0.0).sqrt() as f32;
        }
        // [39] bipower_var_120s
        if t >= W120 {
            let bv = cum_pair[ti] - cum_pair[(t - W120 + 1) as usize];
            feat[[s_idx, 39]] = (bv_scale * bv) as f32;
        }
    }
}

/// Fill cols [40..=44] — horizon-tier Stage B (OFI windows + trade flow + funding).
///
/// Must match `src/features_ext.py::compute_ext_features_batch` bit-for-bit
/// in f32. Semantics:
///
///   [40] ofi_60s   = Σ ofi_raw[k] for k in [T-600, T-1] (lagged convention
///                     matching ofi_1s/5s/30s on cols 26-28)
///   [41] ofi_120s  = Σ ofi_raw[k] for k in [T-1200, T-1]
///   [42] trade_flow_imbalance_60s
///                   = (Σ signed_qty − 0) / Σ |qty| over trades whose ts
///                     is in [sample_ts − 60 000, sample_ts] (ms).
///                     signed_qty = +qty if !is_buyer_maker, else −qty.
///   [43] funding_time_to_next_min
///                   = minutes to the next 00/08/16 UTC funding boundary,
///                     from depth_ts[T] in ms.
///   [44] funding_basis_bps
///                   = (mark − mid) / mid × 10 000 where `mark` is the
///                     most recent mark_price at or before sample_ts
///                     and `mid` is the mid at sample index T.
///                     Zero when no funding row precedes sample_ts, or
///                     when mark/mid <= 0.
///
/// If `trades` or `funding` is None the respective columns stay 0.
pub fn fill_horizon_features_b(
    feat: &mut Array2<f32>,
    depth: &DepthData,
    indices: &[i64],
    trades: Option<&TradesData>,
    funding: Option<&FundingData>,
) {
    const W60: i64 = 600;
    const W120: i64 = 1200;
    const TFI_WINDOW_MS: i64 = 60_000;
    const FUNDING_PERIOD_MS: i64 = 8 * 3600 * 1000;

    let n = depth.n_rows();
    if n == 0 || indices.is_empty() {
        return;
    }
    let bq = &depth.bid_qtys;
    let aq = &depth.ask_qtys;
    let mid = depth.mid_prices();
    let d_ts = depth.timestamps.as_slice().unwrap();

    // --- ofi_raw + cumulative sum for 60/120 s windows ---
    let mut cum_ofi = Array1::<f64>::zeros(n + 1);
    {
        let mut s = 0.0;
        for i in 1..n {
            let raw = (bq[[i, 0]] - bq[[i - 1, 0]]) - (aq[[i, 0]] - aq[[i - 1, 0]]);
            s += raw;
            cum_ofi[i + 1] = s;
        }
        // cum_ofi[0] = cum_ofi[1] = 0 by construction (no ofi at t=0).
    }

    // --- trade-flow cumulative sums (signed and abs) ---
    let (t_ts, cum_signed, cum_abs) = if let Some(tr) = trades {
        let nt = tr.len();
        let ts = tr.timestamps.as_slice().unwrap();
        let q = tr.quantities.as_slice().unwrap();
        let is_sell = &tr.is_sell;
        let mut cs = Array1::<f64>::zeros(nt + 1);
        let mut ca = Array1::<f64>::zeros(nt + 1);
        for i in 0..nt {
            let signed = if is_sell[i] { -q[i] } else { q[i] };
            cs[i + 1] = cs[i] + signed;
            ca[i + 1] = ca[i] + q[i].abs();
        }
        (Some(ts), Some(cs), Some(ca))
    } else {
        (None, None, None)
    };

    // --- funding mark slice (searchsorted per-sample) ---
    let (f_ts, f_mark) = if let Some(f) = funding {
        if f.timestamps.is_empty() {
            (None, None)
        } else {
            (
                Some(f.timestamps.as_slice().unwrap()),
                Some(f.mark_price.as_slice().unwrap()),
            )
        }
    } else {
        (None, None)
    };

    for (s_idx, &raw_idx) in indices.iter().enumerate() {
        let t = raw_idx;
        let ti = t as usize;

        // [40] ofi_60s — window is ofi_raw[t-W .. t-1] (t-1 inclusive).
        if t >= W60 {
            let v = cum_ofi[ti] - cum_ofi[(t - W60) as usize];
            feat[[s_idx, 40]] = v as f32;
        }
        if t >= W120 {
            let v = cum_ofi[ti] - cum_ofi[(t - W120) as usize];
            feat[[s_idx, 41]] = v as f32;
        }

        // [42] trade_flow_imbalance_60s
        if let (Some(tts), Some(cs), Some(ca)) = (t_ts, cum_signed.as_ref(), cum_abs.as_ref()) {
            let sample_ts = d_ts[ti];
            let lo = searchsorted_left_i64(tts, sample_ts - TFI_WINDOW_MS);
            let hi = searchsorted_right_i64(tts, sample_ts);
            let signed = cs[hi] - cs[lo];
            let total = ca[hi] - ca[lo];
            feat[[s_idx, 42]] = if total > 0.0 { (signed / total) as f32 } else { 0.0 };
        }

        // [43] funding_time_to_next_min
        {
            let sample_ts = d_ts[ti];
            if sample_ts > 0 {
                let rem = sample_ts.rem_euclid(FUNDING_PERIOD_MS);
                let mins = if rem == 0 {
                    0.0
                } else {
                    (FUNDING_PERIOD_MS - rem) as f64 / 60_000.0
                };
                feat[[s_idx, 43]] = mins as f32;
            }
        }

        // [44] funding_basis_bps
        if let (Some(fts), Some(fmark)) = (f_ts, f_mark) {
            let sample_ts = d_ts[ti];
            let r = searchsorted_right_i64(fts, sample_ts);
            if r > 0 {
                let fi = (r - 1).min(fts.len() - 1);
                let mark = fmark[fi];
                let m = mid[ti];
                if mark > 0.0 && m > 0.0 {
                    feat[[s_idx, 44]] = ((mark - m) / m * 10_000.0) as f32;
                }
            }
        }
    }
}

/// Fill cols [45..=49] — horizon-tier Stage C (structural microstructure).
///
/// Must match `src/features_ext.py::compute_ext_features_batch` bit-for-bit
/// in f32. Semantics:
///
///   [45] microprice_deviation       = (microprice − mid) / max(spread, eps)
///                                     where microprice = (aq0·bp0 + bq0·ap0)/(bq0+aq0)
///   [46] ofi_top5_weighted (30 s)   = Σ_k 1/(k+1)·(Δbq_k − Δaq_k) summed over
///                                     last 30 ticks; lagged window [T-30..T-1]
///   [47] kyle_lambda_60s            = rolling OLS slope β over last 600 ticks:
///                                     x_t = Σ signed_qty in (dts[t-1], dts[t]]
///                                     y_t = log_mid[t] − log_mid[t-1]
///                                     β   = Σxy / (Σxx + 1e-18), lagged [T-600..T-1]
///   [48] vpin_60s                   = Σ|signed_k| / Σ total_k over 6 consecutive
///                                     10 s sub-buckets ending at sample_ts.
///   [49] cancel_to_trade_ratio_30s  = (cancel_sum over 300 ticks) /
///                                     (|trade_qty| sum over 30 s trades).
///                                     cancel_tick = Σ_k max(0, q_prev_k − q_k)
///                                     over top-5 bids + top-5 asks.
///                                     Only emitted once the 300-tick cancel window
///                                     is saturated AND trade_vol > 0.
pub fn fill_horizon_features_c(
    feat: &mut Array2<f32>,
    depth: &DepthData,
    indices: &[i64],
    trades: Option<&TradesData>,
) {
    const W3S: i64 = 30;
    const W30S: i64 = 300;
    const W60S: i64 = 600;
    const CTR_WINDOW_MS: i64 = 30_000;
    const VPIN_NUM_BUCKETS: i64 = 6;
    const VPIN_BUCKET_MS: i64 = 10_000;
    let ofi5_w = [1.0_f64, 0.5, 1.0 / 3.0, 0.25, 0.2];

    let n = depth.n_rows();
    if n == 0 || indices.is_empty() {
        return;
    }
    let bp = &depth.bid_prices;
    let ap = &depth.ask_prices;
    let bq = &depth.bid_qtys;
    let aq = &depth.ask_qtys;
    let mid = depth.mid_prices();
    let d_ts = depth.timestamps.as_slice().unwrap();

    // --- Cumulative sums for ofi_top5_weighted (30-tick lagged window) ---
    let mut cum_ofi5 = Array1::<f64>::zeros(n + 1);
    let mut cum_cancel = Array1::<f64>::zeros(n + 1);
    {
        let mut s_ofi5 = 0.0;
        let mut s_cancel = 0.0;
        cum_ofi5[1] = 0.0;
        cum_cancel[1] = 0.0;
        for i in 1..n {
            let mut raw_ofi5 = 0.0;
            let mut cancel_tick = 0.0;
            for k in 0..5 {
                let db = bq[[i, k]] - bq[[i - 1, k]];
                let da = aq[[i, k]] - aq[[i - 1, k]];
                raw_ofi5 += (db - da) * ofi5_w[k];
                if db < 0.0 {
                    cancel_tick += -db;
                }
                if da < 0.0 {
                    cancel_tick += -da;
                }
            }
            s_ofi5 += raw_ofi5;
            s_cancel += cancel_tick;
            cum_ofi5[i + 1] = s_ofi5;
            cum_cancel[i + 1] = s_cancel;
        }
    }

    // --- log_mid[t] - log_mid[t-1] as y_t for Kyle; per-tick x_t for Kyle + VPIN ---
    // Per-tick signed volume lives inside the `trades` branch below.
    let mut log_mid = Array1::<f64>::zeros(n);
    for i in 0..n {
        let m = mid[i];
        if m > 0.0 {
            log_mid[i] = m.ln();
        }
    }

    // Kyle cumulative sums (only populated when trades are present).
    let (kyle_cum_xy, kyle_cum_xx) = if let Some(tr) = trades {
        let nt = tr.len();
        let t_ts = tr.timestamps.as_slice().unwrap();
        let t_q = tr.quantities.as_slice().unwrap();
        let is_sell = &tr.is_sell;

        // cum_signed[i] = Σ signed up to i-th trade (exclusive).
        let mut cum_signed = Array1::<f64>::zeros(nt + 1);
        for i in 0..nt {
            let signed = if is_sell[i] { -t_q[i] } else { t_q[i] };
            cum_signed[i + 1] = cum_signed[i] + signed;
        }

        // right_cur[t] = searchsorted_right(t_ts, dts[t])
        let mut right_cur = vec![0usize; n];
        for (i, &ts) in d_ts.iter().enumerate() {
            right_cur[i] = searchsorted_right_i64(t_ts, ts);
        }
        // x_per_tick[t] = Σ signed for trades in (dts[t-1], dts[t]];
        // also = cum_signed[right_cur[t]] - cum_signed[right_cur[t-1]].
        let mut xy = Array1::<f64>::zeros(n);
        let mut xx = Array1::<f64>::zeros(n);
        for t in 1..n {
            let x = cum_signed[right_cur[t]] - cum_signed[right_cur[t - 1]];
            let y = log_mid[t] - log_mid[t - 1];
            xy[t] = x * y;
            xx[t] = x * x;
        }
        let mut cxy = Array1::<f64>::zeros(n + 1);
        let mut cxx = Array1::<f64>::zeros(n + 1);
        let mut sxy = 0.0;
        let mut sxx = 0.0;
        for t in 0..n {
            sxy += xy[t];
            sxx += xx[t];
            cxy[t + 1] = sxy;
            cxx[t + 1] = sxx;
        }
        (Some(cxy), Some(cxx))
    } else {
        (None, None)
    };

    // VPIN + cancel-to-trade denominator sums (signed+abs trade cum).
    let (vpin_ready, vpin_t_ts, vpin_cum_signed, vpin_cum_abs) = if let Some(tr) = trades {
        let nt = tr.len();
        let t_ts = tr.timestamps.as_slice().unwrap();
        let t_q = tr.quantities.as_slice().unwrap();
        let is_sell = &tr.is_sell;
        let mut cs = Array1::<f64>::zeros(nt + 1);
        let mut ca = Array1::<f64>::zeros(nt + 1);
        for i in 0..nt {
            let signed = if is_sell[i] { -t_q[i] } else { t_q[i] };
            cs[i + 1] = cs[i] + signed;
            ca[i + 1] = ca[i] + t_q[i].abs();
        }
        (true, Some(t_ts), Some(cs), Some(ca))
    } else {
        (false, None, None, None)
    };

    for (s_idx, &raw_idx) in indices.iter().enumerate() {
        let t = raw_idx;
        let ti = t as usize;

        // [45] microprice_deviation
        {
            let b0 = bp[[ti, 0]];
            let a0 = ap[[ti, 0]];
            let bq0 = bq[[ti, 0]];
            let aq0 = aq[[ti, 0]];
            let tot = bq0 + aq0;
            let spread = a0 - b0;
            if tot > 0.0 && spread > 1e-12 {
                let microprice = (aq0 * b0 + bq0 * a0) / tot;
                let cur = mid[ti];
                feat[[s_idx, 45]] = ((microprice - cur) / spread) as f32;
            }
        }

        // [46] ofi_top5_weighted over 30 ticks (lagged window)
        if t >= W3S {
            let v = cum_ofi5[ti] - cum_ofi5[(t - W3S) as usize];
            feat[[s_idx, 46]] = v as f32;
        }

        // [47] kyle_lambda_60s
        if t >= W60S {
            if let (Some(cxy), Some(cxx)) = (kyle_cum_xy.as_ref(), kyle_cum_xx.as_ref()) {
                let num = cxy[ti] - cxy[(t - W60S) as usize];
                let den = cxx[ti] - cxx[(t - W60S) as usize];
                if den > 1e-18 {
                    feat[[s_idx, 47]] = (num / den) as f32;
                }
            }
        }

        // [48] vpin_60s
        if vpin_ready {
            let t_ts_ref = vpin_t_ts.unwrap();
            let cs = vpin_cum_signed.as_ref().unwrap();
            let ca = vpin_cum_abs.as_ref().unwrap();
            let sample_ts = d_ts[ti];
            let mut sum_abs_net = 0.0_f64;
            let mut sum_total = 0.0_f64;
            for k in 0..VPIN_NUM_BUCKETS {
                let hi_ts = sample_ts - k * VPIN_BUCKET_MS;
                let lo_ts = sample_ts - (k + 1) * VPIN_BUCKET_MS;
                let hi_idx = searchsorted_right_i64(t_ts_ref, hi_ts);
                let lo_idx = searchsorted_right_i64(t_ts_ref, lo_ts);
                let net = cs[hi_idx] - cs[lo_idx];
                let tot = ca[hi_idx] - ca[lo_idx];
                sum_abs_net += net.abs();
                sum_total += tot;
            }
            if sum_total > 0.0 {
                feat[[s_idx, 48]] = (sum_abs_net / sum_total) as f32;
            }

            // [49] cancel_to_trade_ratio_30s (only emit once cancel window saturated)
            if t >= W30S {
                let num = cum_cancel[ti] - cum_cancel[(t - W30S) as usize];
                let lo = searchsorted_left_i64(t_ts_ref, sample_ts - CTR_WINDOW_MS);
                let hi = searchsorted_right_i64(t_ts_ref, sample_ts);
                let den = ca[hi] - ca[lo];
                if den > 0.0 {
                    feat[[s_idx, 49]] = (num / den) as f32;
                }
            }
        }
    }
}

/// Fill cols [50..=55] — horizon-tier Stage D (cross-exchange + ETH).
///
/// Must match `src/features_ext.py::compute_ext_features_batch` bit-for-bit.
///
///   [50] bybit_lead_lag_corr_30s  — corr(btc_ret[t], bybit_ret_lagged[t]) over
///                                    300-tick lagged window; bybit_ret is the
///                                    shifted (by one tick) series.
///   [51] okx_net_flow_30s          — Σ signed_qty for OKX trades in last 30 s.
///   [52] bitget_net_flow_30s       — ditto Bitget.
///   [53] gateio_net_flow_30s       — ditto Gate.io.
///   [54] eth_momentum_60s          — (eth_last[T] − eth_last[T-600]) / eth_last[T-600]
///   [55] eth_btc_corr_30s          — corr(btc_ret, eth_ret) over 300 ticks.
pub fn fill_horizon_features_d(
    feat: &mut Array2<f32>,
    depth: &DepthData,
    indices: &[i64],
    bybit: Option<&CrossExTrades>,
    eth: Option<&TradesData>,
    okx: Option<&CrossExTrades>,
    bitget: Option<&CrossExTrades>,
    gateio: Option<&CrossExTrades>,
) {
    const W30S: i64 = 300;
    const W60S: i64 = 600;
    const CTR_WINDOW_MS: i64 = 30_000;

    let n = depth.n_rows();
    if n == 0 || indices.is_empty() {
        return;
    }
    let d_ts = depth.timestamps.as_slice().unwrap();
    let mid = depth.mid_prices();

    // Helper: cross-exchange net flow over 30 s window.
    let fill_net_flow = |feat: &mut Array2<f32>, col: usize, ex: &CrossExTrades| {
        let e_ts = ex.timestamps.as_slice().unwrap();
        let e_q = ex.signed_qty.as_slice().unwrap();
        let ne = e_ts.len();
        if ne == 0 {
            return;
        }
        let mut cum = Array1::<f64>::zeros(ne + 1);
        for i in 0..ne {
            cum[i + 1] = cum[i] + e_q[i];
        }
        for (s_idx, &raw_idx) in indices.iter().enumerate() {
            let sample_ts = d_ts[raw_idx as usize];
            let lo = searchsorted_left_i64(e_ts, sample_ts - CTR_WINDOW_MS);
            let hi = searchsorted_right_i64(e_ts, sample_ts);
            feat[[s_idx, col]] = (cum[hi] - cum[lo]) as f32;
        }
    };
    if let Some(o) = okx { fill_net_flow(feat, 51, o); }
    if let Some(b) = bitget { fill_net_flow(feat, 52, b); }
    if let Some(g) = gateio { fill_net_flow(feat, 53, g); }

    // Per-tick last price lookups for Bybit (CrossExTrades has no price column;
    // bybit price is not preserved through the bridge).
    // Fallback: if bybit has trades with `price` semantically lost, use signed_qty
    // magnitude? Instead, we require a `bybit_trades: &TradesData` style feed.
    // For now, Bybit lead-lag is skipped unless a TradesData is provided, and
    // we piggy-back on `eth` (Binance) as the only price-bearing trades feed.
    // Batch path handles both Bybit and ETH via prices, but our Rust reader keeps
    // Bybit as signed flow only. This mirrors the existing CrossExTrades schema.
    //
    // Cols [50,55] therefore require a TradesData-style feed for the respective
    // exchange. In practice the Bybit parquet is loaded as `read_trades_parquet`
    // (not CrossExTrades), so we accept a TradesData-like for bybit via the
    // `eth` parameter pattern. To keep the signature tight here we only compute
    // col 55 (ETH) from the provided `eth` TradesData; col 50 stays at 0 in the
    // Rust path if Bybit is passed as CrossExTrades-without-price.
    //
    // TODO: a future commit can thread a `bybit_trades: Option<&TradesData>`
    // through feature_builder when the need arises; today the parity harness
    // already skips col 50 when Bybit price is not available.
    let _ = bybit;

    // ETH per-tick last-price + log-returns.
    if let Some(e) = eth {
        let e_ts = e.timestamps.as_slice().unwrap();
        let e_px = &e.prices;
        let ne = e_ts.len();
        if ne > 0 {
            // Per-tick last ETH price by searchsorted.
            let mut eth_per_tick = Array1::<f64>::zeros(n);
            for t in 0..n {
                let r = searchsorted_right_i64(e_ts, d_ts[t]);
                if r > 0 {
                    eth_per_tick[t] = e_px[r - 1];
                }
            }
            // ETH log-return per tick.
            let mut eth_ret = Array1::<f64>::zeros(n);
            for t in 1..n {
                let a = eth_per_tick[t - 1];
                let b = eth_per_tick[t];
                if a > 0.0 && b > 0.0 {
                    eth_ret[t] = (b / a).ln();
                }
            }
            // BTC log-return per tick (mirrors Python).
            let mut btc_ret = Array1::<f64>::zeros(n);
            for t in 1..n {
                let a = mid[t - 1];
                let b = mid[t];
                if a > 0.0 && b > 0.0 {
                    btc_ret[t] = (b / a).ln();
                }
            }

            // Rolling corr(btc_ret, eth_ret) over 300 ticks (lagged window).
            let rolling_corr = |xs: &Array1<f64>, ys: &Array1<f64>, out_col: usize,
                                feat: &mut Array2<f32>| {
                let mut cx = Array1::<f64>::zeros(n + 1);
                let mut cy = Array1::<f64>::zeros(n + 1);
                let mut cxx = Array1::<f64>::zeros(n + 1);
                let mut cyy = Array1::<f64>::zeros(n + 1);
                let mut cxy = Array1::<f64>::zeros(n + 1);
                for t in 0..n {
                    cx[t + 1] = cx[t] + xs[t];
                    cy[t + 1] = cy[t] + ys[t];
                    cxx[t + 1] = cxx[t] + xs[t] * xs[t];
                    cyy[t + 1] = cyy[t] + ys[t] * ys[t];
                    cxy[t + 1] = cxy[t] + xs[t] * ys[t];
                }
                for (s_idx, &raw_idx) in indices.iter().enumerate() {
                    let t = raw_idx;
                    if t < W30S { continue; }
                    let ti = t as usize;
                    let lo = (t - W30S) as usize;
                    let sx = cx[ti] - cx[lo];
                    let sy = cy[ti] - cy[lo];
                    let sxx = cxx[ti] - cxx[lo];
                    let syy = cyy[ti] - cyy[lo];
                    let sxy = cxy[ti] - cxy[lo];
                    let w = W30S as f64;
                    let num = w * sxy - sx * sy;
                    let den = (w * sxx - sx * sx) * (w * syy - sy * sy);
                    if den > 1e-24 {
                        feat[[s_idx, out_col]] = (num / den.sqrt()) as f32;
                    }
                }
            };
            rolling_corr(&btc_ret, &eth_ret, 55, feat);

            // ETH momentum 60s.
            for (s_idx, &raw_idx) in indices.iter().enumerate() {
                let t = raw_idx;
                if t < W60S { continue; }
                let ti = t as usize;
                let cur = eth_per_tick[ti];
                let past = eth_per_tick[(t - W60S) as usize];
                if past > 0.0 && cur > 0.0 {
                    feat[[s_idx, 54]] = ((cur - past) / past) as f32;
                }
            }
        }
    }
}

/// np.searchsorted(a, v, side="right") — first i with a[i] > v.
#[inline]
fn searchsorted_right_i64(a: &[i64], v: i64) -> usize {
    let mut lo = 0usize;
    let mut hi = a.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if a[mid] <= v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// np.searchsorted(a, v, side="left") — returns first i with a[i] >= v.
/// a MUST be non-decreasing.
#[inline]
fn searchsorted_left_i64(a: &[i64], v: i64) -> usize {
    let mut lo = 0usize;
    let mut hi = a.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if a[mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}
