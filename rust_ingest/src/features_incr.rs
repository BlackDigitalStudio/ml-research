//! Incremental (day-anchored, append-only) mirror of `features.rs` for the sub-ms
//! live engine. NOT a reimplementation of the math: every prefix array is appended
//! in the SAME sequential order the batch builders use, and every per-sample kernel
//! is copied verbatim from the batch fill_* loops — so a full-day replay through
//! `FeatState` is BYTE-IDENTICAL to `feature_builder` on the day files (enforced by
//! `bin/fb_incr_harness`). The frozen batch path in `features.rs` is untouched.
//!
//! Input scope = the deployed h150 pipeline: depth + trades + funding(day-anchor,
//! single row => col13 const / col44 0) + eth + liquidations + open-interest.
//! Cols [17,18,19,30,50,51,52,53] (derivs / cross-exchange) stay 0.
//!
//! Event-commit contract (matches batch searchsorted semantics): before pushing a
//! book tick with timestamp T, the caller must have pushed every trade/eth/liq/oi
//! event with ts <= T. `bin/fb_incr_harness` replays file rows in that merged
//! order; the live engine commits pending stream events on each book-tick arrival.

use crate::features::{NUM_FEATURES, QUEUE_DECAY_ALPHA};

#[inline]
fn searchsorted_left_i64(a: &[i64], v: i64) -> usize {
    a.partition_point(|&x| x < v)
}
#[inline]
fn searchsorted_right_i64(a: &[i64], v: i64) -> usize {
    a.partition_point(|&x| x <= v)
}

/// Day-anchored incremental feature state (one instrument).
#[derive(Default)]
pub struct FeatState {
    // ---- book ticks ----
    pub ts: Vec<i64>,      // exchange ms, ascending
    mids: Vec<f64>,
    log_mid: Vec<f64>,     // ln(mid) or 0 when mid<=0 (stage C convention)
    bp0: Vec<f64>,
    ap0: Vec<f64>,
    bq0: Vec<f64>,
    aq0: Vec<f64>,
    // previous full book row (only the parts diffs need)
    prev_bq5: [f64; 5],
    prev_aq5: [f64; 5],
    prev_bq0: f64,
    prev_aq0: f64,
    prev_bp0: f64,
    prev_ap0: f64,
    // per-tick derived series (index == tick index)
    ofi: Vec<f64>,          // [0] raw OFI (L1)
    bv5: Vec<f64>,
    av5: Vec<f64>,
    imb: Vec<f64>,
    spread: Vec<f64>,
    large: Vec<bool>,
    top3: Vec<f64>,
    bid_ema: Vec<f64>,      // [31]
    ask_ema: Vec<f64>,
    obi1: Vec<f64>,         // [62] numer/denom folded: store final ratio
    obi10: Vec<f64>,
    obi20: Vec<f64>,
    // rolling machinery (identical update order to the batch builders)
    returns: Vec<f64>,      // len n-1
    vol_all: Vec<f64>,      // fresh 10-pass per element (batch does the same)
    vol_mean_all: Vec<f64>, // fresh 30-sum per element (batch does the same)
    log_ret: Vec<f64>,      // len n-1
    hurst: Vec<f64>,        // fresh 100-pass per element
    bid_cancel: Vec<f64>,
    ask_cancel: Vec<f64>,
    bc_win: Vec<f64>,       // sliding (state kept)
    ac_win: Vec<f64>,
    bc_s: f64,
    ac_s: f64,
    ofi_1s: Vec<f64>,       // sliding sums over ofi (10/50/300)
    ofi_5s: Vec<f64>,
    ofi_30s: Vec<f64>,
    ofi_s10: f64,
    ofi_s50: f64,
    ofi_s300: f64,
    cum_mid: Vec<f64>,      // len n+1
    // 1-second grid ([37],[38])
    g0: i64,
    grid_mid: Vec<f64>,
    cum_gsq: Vec<f64>,
    // bipower ([39])
    cum_pair: Vec<f64>,     // len n, cum_pair[0..2]=0
    // stage B [40]/[41] prefix (len n+1; cum_ofi_b[x] = Σ ofi[0..x], ofi[0]=0)
    cum_ofi_b: Vec<f64>,
    // stage C prefixes (len n+1)
    cum_ofi5: Vec<f64>,
    cum_cancel_c: Vec<f64>,
    // kyle ([47]): per-tick right_cur + prefixes (len n+1)
    right_cur: Vec<usize>,
    kyle_cxy: Vec<f64>,
    kyle_cxx: Vec<f64>,
    // [22] trade-intensity per depth tick + rolling. Assignment is deferred: batch
    // maps trade->tick with the FULL day array (a trade between ticks i and i+1 maps
    // to i; one exactly AT tick i+1's ts maps to i+1), so trades are queued and
    // drained on the next book tick; rolling windows are extended only once their
    // last tick's count is FINAL (i.e. when the following tick arrives).
    intens_pending: Vec<i64>,
    tick_intensity: Vec<f64>,
    curr_int: Vec<f64>,     // sliding (state kept)
    int_mean_all: Vec<f64>, // sliding (state kept)
    ci_s: f64,
    im_s: f64,
    // [33] effective-spread EMA
    eff_ema: Vec<f64>,
    eff_acc: f64,
    // eth-vs-mid rolling corr ([55]) prefixes (len n+1) + per-tick eth price
    eth_per_tick: Vec<f64>,
    c55_x: Vec<f64>,
    c55_y: Vec<f64>,
    c55_xx: Vec<f64>,
    c55_yy: Vec<f64>,
    c55_xy: Vec<f64>,

    // ---- trades (DOGE) ----
    t_ts: Vec<i64>,
    t_px: Vec<f64>,
    cum_buy: Vec<f64>,      // len nt+1
    cum_sell: Vec<f64>,
    cum_large: Vec<f64>,
    cum_signed: Vec<f64>,
    cum_abs: Vec<f64>,

    // ---- eth trades ----
    e_ts: Vec<i64>,
    e_px: Vec<f64>,
    e_cum_buy: Vec<f64>,    // len ne+1
    e_cum_sell: Vec<f64>,
    e_cum_pv: Vec<f64>,
    e_cum_qty: Vec<f64>,

    // ---- liquidations ----
    l_ts: Vec<i64>,
    l_csn: Vec<f64>,        // len nl+1
    l_can: Vec<f64>,

    // ---- open interest ----
    oi_ts: Vec<i64>,
    oi_v: Vec<f64>,

    // ---- funding day-anchor ----
    pub anchor_rate: Option<f64>,
}

impl FeatState {
    pub fn new() -> Self {
        let mut s = Self::default();
        // len-(n+1) prefixes start with the 0 element (batch zeros(n+1)).
        for v in [
            &mut s.cum_mid, &mut s.cum_ofi_b, &mut s.cum_ofi5, &mut s.cum_cancel_c,
            &mut s.kyle_cxy, &mut s.kyle_cxx,
            &mut s.c55_x, &mut s.c55_y, &mut s.c55_xx, &mut s.c55_yy, &mut s.c55_xy,
            &mut s.cum_buy, &mut s.cum_sell, &mut s.cum_large, &mut s.cum_signed,
            &mut s.cum_abs, &mut s.e_cum_buy, &mut s.e_cum_sell, &mut s.e_cum_pv,
            &mut s.e_cum_qty, &mut s.l_csn, &mut s.l_can,
        ] {
            v.push(0.0);
        }
        s
    }

    #[inline]
    pub fn n_ticks(&self) -> usize {
        self.ts.len()
    }

    // ---------------------------------------------------------------- events
    pub fn push_trade(&mut self, ts: i64, px: f64, qty: f64, is_sell: bool) {
        self.t_ts.push(ts);
        self.t_px.push(px);
        let i = self.t_ts.len();
        let (pb, ps) = (self.cum_buy[i - 1], self.cum_sell[i - 1]);
        if is_sell {
            self.cum_sell.push(ps + qty);
            self.cum_buy.push(pb);
        } else {
            self.cum_buy.push(pb + qty);
            self.cum_sell.push(ps);
        }
        self.cum_large.push(self.cum_large[i - 1] + if qty > 10.0 { 1.0 } else { 0.0 });
        let signed = if is_sell { -qty } else { qty };
        self.cum_signed.push(self.cum_signed[i - 1] + signed);
        self.cum_abs.push(self.cum_abs[i - 1] + qty.abs());
        // [22] tick_intensity assignment deferred to the next push_book (see field doc).
        self.intens_pending.push(ts);
    }

    pub fn push_eth(&mut self, ts: i64, px: f64, qty: f64, is_sell: bool) {
        self.e_ts.push(ts);
        self.e_px.push(px);
        let i = self.e_ts.len();
        let (pb, ps) = (self.e_cum_buy[i - 1], self.e_cum_sell[i - 1]);
        if is_sell {
            self.e_cum_sell.push(ps + qty);
            self.e_cum_buy.push(pb);
        } else {
            self.e_cum_buy.push(pb + qty);
            self.e_cum_sell.push(ps);
        }
        self.e_cum_pv.push(self.e_cum_pv[i - 1] + px * qty);
        self.e_cum_qty.push(self.e_cum_qty[i - 1] + qty);
    }

    /// signed_notional = +qty.abs()*px for side "buy" (short liq), − for "sell";
    /// abs_notional = qty.abs()*px — the exact `read_liquidations_parquet` encoding.
    pub fn push_liq(&mut self, ts: i64, signed_notional: f64, abs_notional: f64) {
        self.l_ts.push(ts);
        let i = self.l_ts.len();
        self.l_csn.push(self.l_csn[i - 1] + signed_notional);
        self.l_can.push(self.l_can[i - 1] + abs_notional);
    }

    pub fn push_oi(&mut self, ts: i64, v: f64) {
        self.oi_ts.push(ts);
        self.oi_v.push(v);
    }

    // ---------------------------------------------------------------- book tick
    /// `bids`/`asks`: 20 (price, qty) levels, zero-padded — the MirrorBook top20 row.
    pub fn push_book(&mut self, ts: i64, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        let i = self.ts.len(); // index of the new tick

        // -- [22] deferred intensity: trades strictly BEFORE this tick map to the
        // existing ticks (batch clip(searchsorted_right-1, 0, n-1)); tick i-1's
        // count is thereby FINAL, so the rolling windows ending at i-1 extend now.
        let mut eq_pending: Vec<i64> = Vec::new();
        if !self.intens_pending.is_empty() {
            for &tt in &self.intens_pending {
                if tt >= ts {
                    eq_pending.push(tt); // maps to THIS tick (t_ts == ts by contract)
                    continue;
                }
                if self.tick_intensity.is_empty() {
                    continue; // batch clips to tick 0 == the tick being appended below
                }
                let n0 = self.tick_intensity.len();
                let r = searchsorted_right_i64(&self.ts, tt);
                let ti = if r == 0 { 0 } else { (r - 1).min(n0 - 1) };
                self.tick_intensity[ti] += 1.0;
            }
            self.intens_pending.clear();
        }
        if i >= 10 {
            // window [k .. k+10) ending at tick i-1 is now final (k = i-10).
            if self.curr_int.is_empty() {
                let mut s = 0.0;
                for j in 0..10 {
                    s += self.tick_intensity[j];
                }
                self.ci_s = s;
            } else {
                let k = self.curr_int.len();
                self.ci_s += self.tick_intensity[k + 9] - self.tick_intensity[k - 1];
            }
            self.curr_int.push(self.ci_s);
            if self.curr_int.len() >= 30 {
                if self.int_mean_all.is_empty() {
                    let mut s = 0.0;
                    for j in 0..30 {
                        s += self.curr_int[j];
                    }
                    self.im_s = s;
                    self.int_mean_all.push(s / 30.0);
                } else {
                    let k = self.int_mean_all.len();
                    self.im_s += self.curr_int[k + 29] - self.curr_int[k - 1];
                    self.int_mean_all.push(self.im_s / 30.0);
                }
            }
        }

        self.ts.push(ts);

        // -- basic row quantities (batch compute_lob_features precompute loops) --
        let (b0p, b0q) = bids[0];
        let (a0p, a0q) = asks[0];
        let mid = if b0p > 0.0 && a0p > 0.0 { 0.5 * (b0p + a0p) } else { 0.0 };
        self.mids.push(mid);
        self.log_mid.push(if mid > 0.0 { mid.ln() } else { 0.0 });
        self.bp0.push(b0p);
        self.ap0.push(a0p);
        self.bq0.push(b0q);
        self.aq0.push(a0q);

        let mut sb = 0.0;
        let mut sa = 0.0;
        let mut lb = false;
        let mut la = false;
        for k in 0..5 {
            let qb = bids[k].1;
            let qa = asks[k].1;
            sb += qb;
            sa += qa;
            if qb > 100.0 {
                lb = true;
            }
            if qa > 100.0 {
                la = true;
            }
        }
        self.bv5.push(sb);
        self.av5.push(sa);
        self.large.push(lb || la);
        let tot = sb + sa;
        self.imb.push(if tot > 0.0 { (sb - sa) / tot } else { 0.0 });
        self.spread.push(a0p - b0p);

        // [0] OFI raw
        let ofi_i = if i == 0 {
            0.0
        } else {
            (b0q - self.prev_bq0) - (a0q - self.prev_aq0)
        };
        self.ofi.push(ofi_i);

        // [32] top3 asymmetry + [61-63] OBI ladder
        let mut t3b = 0.0;
        let mut t20b = 0.0;
        let mut t3a = 0.0;
        let mut t20a = 0.0;
        let mut b10 = 0.0;
        let mut a10 = 0.0;
        for k in 0..20 {
            let qb = bids[k].1;
            let qa = asks[k].1;
            t20b += qb;
            t20a += qa;
            if k < 3 {
                t3b += qb;
                t3a += qa;
            }
            if k < 10 {
                b10 += qb;
                a10 += qa;
            }
        }
        self.top3.push(t3b / (t20b + 1e-9) - t3a / (t20a + 1e-9));
        let t1 = b0q + a0q;
        let t10 = b10 + a10;
        let t20 = t20b + t20a;
        self.obi1.push(if t1 > 0.0 { (b0q - a0q) / t1 } else { 0.0 });
        self.obi10.push(if t10 > 0.0 { (b10 - a10) / t10 } else { 0.0 });
        self.obi20.push(if t20 > 0.0 { (t20b - t20a) / t20 } else { 0.0 });

        // [31] queue-pressure EMA (batch runs the recurrence over all rows)
        let a = QUEUE_DECAY_ALPHA;
        let (bd, ad) = if i == 0 {
            (0.0, 0.0)
        } else {
            ((self.prev_bq0 - b0q).max(0.0), (self.prev_aq0 - a0q).max(0.0))
        };
        let b_prev = self.bid_ema.last().copied().unwrap_or(0.0);
        let s_prev = self.ask_ema.last().copied().unwrap_or(0.0);
        self.bid_ema.push(a * bd + (1.0 - a) * b_prev);
        self.ask_ema.push(a * ad + (1.0 - a) * s_prev);

        // cancel ticks ([25]) + stage-C ofi5/cancel prefixes
        let (mut cb, mut ca) = (0.0f64, 0.0f64);
        let mut raw_ofi5 = 0.0f64;
        let mut cancel_c = 0.0f64;
        const OFI5_W: [f64; 5] = [1.0, 0.5, 1.0 / 3.0, 0.25, 0.2];
        if i > 0 {
            for k in 0..5 {
                let db = bids[k].1 - self.prev_bq5[k];
                let da = asks[k].1 - self.prev_aq5[k];
                if db < 0.0 {
                    cb += -db;
                }
                if da < 0.0 {
                    ca += -da;
                }
                raw_ofi5 += (db - da) * OFI5_W[k];
                if db < 0.0 {
                    cancel_c += -db;
                }
                if da < 0.0 {
                    cancel_c += -da;
                }
            }
        }
        self.bid_cancel.push(cb);
        self.ask_cancel.push(ca);
        // batch: cum_ofi5[1]=0 explicitly (i=0 contributes nothing), then adds from i=1.
        self.cum_ofi5.push(self.cum_ofi5[i] + raw_ofi5);
        self.cum_cancel_c.push(self.cum_cancel_c[i] + cancel_c);
        self.cum_ofi_b.push(self.cum_ofi_b[i] + ofi_i);

        // [25] rolling 10-tick cancel sums (batch: fresh init at k=0, slide after)
        let n = i + 1;
        if n >= 10 {
            if self.bc_win.is_empty() {
                let mut sbc = 0.0;
                let mut sac = 0.0;
                for j in 0..10 {
                    sbc += self.bid_cancel[j];
                    sac += self.ask_cancel[j];
                }
                self.bc_s = sbc;
                self.ac_s = sac;
            } else {
                let k = self.bc_win.len(); // new window start
                self.bc_s += self.bid_cancel[k + 9] - self.bid_cancel[k - 1];
                self.ac_s += self.ask_cancel[k + 9] - self.ask_cancel[k - 1];
            }
            self.bc_win.push(self.bc_s);
            self.ac_win.push(self.ac_s);
        }

        // OFI rolling sums 10/50/300 (batch rolling_sum: fresh init, then slide)
        macro_rules! roll {
            ($out:ident, $st:ident, $w:expr) => {
                if n >= $w {
                    if self.$out.is_empty() {
                        let mut s = 0.0;
                        for j in 0..$w {
                            s += self.ofi[j];
                        }
                        self.$st = s;
                    } else {
                        let k = self.$out.len();
                        self.$st += self.ofi[k + $w - 1] - self.ofi[k - 1];
                    }
                    let v = self.$st;
                    self.$out.push(v);
                }
            };
        }
        roll!(ofi_1s, ofi_s10, 10);
        roll!(ofi_5s, ofi_s50, 50);
        roll!(ofi_30s, ofi_s300, 300);

        // [10]/[21] returns + vol windows (batch: fresh per-element passes)
        if i >= 1 {
            let pm = self.mids[i - 1];
            let base = if pm > 0.0 { pm } else { 1.0 };
            self.returns.push((mid - pm) / base);
            let lr = (mid + 1e-10).ln() - (pm + 1e-10).ln();
            self.log_ret.push(lr);
        }
        if self.returns.len() >= 10 {
            let j = self.vol_all.len();
            let w = &self.returns[j..j + 10];
            let mean: f64 = w.iter().sum::<f64>() / 10.0;
            let var: f64 = w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 10.0;
            self.vol_all.push(var.sqrt());
        }
        if self.vol_all.len() >= 30 {
            let k = self.vol_mean_all.len();
            self.vol_mean_all
                .push(self.vol_all[k..k + 30].iter().sum::<f64>() / 30.0);
        }
        // [23] Hurst (fresh 100-pass per element, batch-identical)
        if self.log_ret.len() >= 100 {
            let j = self.hurst.len();
            let chunk = &self.log_ret[j..j + 100];
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
            self.hurst.push(if s > 0.0 {
                let r = dmax - dmin;
                ((r / (s + 1e-10)).ln() / 100f64.ln()).clamp(0.0, 1.0)
            } else {
                0.5
            });
        }

        // [11] cum_mid
        self.cum_mid.push(self.cum_mid[i] + mid);

        // [37]/[38] 1-second grid RV (batch forward-fill pointer == last tick <= gsec)
        if i == 0 {
            self.g0 = ts / 1000;
            self.grid_mid.push(mid);
            self.cum_gsq.push(0.0);
        } else {
            let gsec_new = ts / 1000;
            let filled = self.g0 + self.grid_mid.len() as i64 - 1; // last grid second filled
            // extend grid seconds (filled+1 ..= gsec_new): each gets mid of the last
            // tick with ts/1000 <= gsec; ticks are processed in order so the previous
            // tick's mid forward-fills until this tick's second.
            let prev_mid = self.mids[i - 1];
            let mut g = filled + 1;
            while g <= gsec_new {
                // last tick <= g is the previous tick for g < gsec_new; for g == gsec_new
                // it is THIS tick (batch: while ts[p+1]/1000 <= gsec advances onto it).
                let gm = if g == gsec_new { mid } else { prev_mid };
                let a_ = *self.grid_mid.last().unwrap();
                let gr = if a_ > 0.0 && gm > 0.0 { gm.ln() - a_.ln() } else { 0.0 };
                let c = *self.cum_gsq.last().unwrap();
                self.grid_mid.push(gm);
                self.cum_gsq.push(c + gr * gr);
                g += 1;
            }
            // same-second later tick: batch keeps the LAST tick of the second in
            // grid_mid — overwrite the current second's slot and adjust cum_gsq.
            if gsec_new == filled {
                let m = self.grid_mid.len();
                if m >= 2 {
                    let a_ = self.grid_mid[m - 2];
                    let gr = if a_ > 0.0 && mid > 0.0 { mid.ln() - a_.ln() } else { 0.0 };
                    self.grid_mid[m - 1] = mid;
                    self.cum_gsq[m - 1] = self.cum_gsq[m - 2] + gr * gr;
                }else {
                    self.grid_mid[0] = mid;
                }
            }
        }

        // [39] bipower cum_pair (batch: cum_pair[k] for k>=2)
        if i < 2 {
            self.cum_pair.push(0.0);
        } else {
            let m2 = self.mids[i - 2];
            let m1 = self.mids[i - 1];
            let r1 = if m1 > 0.0 && mid > 0.0 { (mid.ln() - m1.ln()).abs() } else { 0.0 };
            let r0 = if m2 > 0.0 && m1 > 0.0 { (m1.ln() - m2.ln()).abs() } else { 0.0 };
            self.cum_pair.push(self.cum_pair[i - 1] + r1 * r0);
        }

        // [47] kyle: right_cur + xy/xx prefixes (trades already committed <= ts)
        let rc = searchsorted_right_i64(&self.t_ts, ts);
        self.right_cur.push(rc);
        let (xy, xx) = if i == 0 {
            (0.0, 0.0)
        } else {
            let x = self.cum_signed[rc] - self.cum_signed[self.right_cur[i - 1]];
            let y = self.log_mid[i] - self.log_mid[i - 1];
            (x * y, x * x)
        };
        self.kyle_cxy.push(self.kyle_cxy[i] + xy);
        self.kyle_cxx.push(self.kyle_cxx[i] + xx);

        // [22] new tick's intensity slot; trades with t_ts == ts map to it (batch
        // searchsorted_right includes equals). Rolling windows extend lazily at the
        // NEXT push (see the block at the top of this fn).
        self.tick_intensity.push(0.0);
        for &tt in &eq_pending {
            let _ = tt;
            self.tick_intensity[i] += 1.0;
        }

        // [33] effective-spread EMA (batch recurrence over all depth ticks)
        let a = QUEUE_DECAY_ALPHA;
        if !self.t_ts.is_empty() {
            let r = searchsorted_right_i64(&self.t_ts, ts);
            let nt = self.t_ts.len();
            let lt = if r == 0 { 0 } else { (r - 1).min(nt - 1) };
            let sp = (a0p - b0p).max(1e-9);
            let m = if b0p > 0.0 && a0p > 0.0 { 0.5 * (b0p + a0p) } else { 0.0 };
            let ratio = if r > 0 { (self.t_px[lt] - m).abs() / sp } else { 0.0 };
            self.eff_acc = a * ratio + (1.0 - a) * self.eff_acc;
        } else {
            self.eff_acc = (1.0 - a) * self.eff_acc;
        }
        self.eff_ema.push(self.eff_acc);

        // [55] eth-corr prefixes + eth_per_tick (eth already committed <= ts)
        let ept = {
            let r = searchsorted_right_i64(&self.e_ts, ts);
            if r > 0 { self.e_px[r - 1] } else { 0.0 }
        };
        self.eth_per_tick.push(ept);
        let (x55, y55) = if i == 0 {
            (0.0, 0.0)
        } else {
            let (ea, eb) = (self.eth_per_tick[i - 1], ept);
            let er = if ea > 0.0 && eb > 0.0 { (eb / ea).ln() } else { 0.0 };
            let (ma, mb) = (self.mids[i - 1], mid);
            let br = if ma > 0.0 && mb > 0.0 { (mb / ma).ln() } else { 0.0 };
            (br, er) // x = btc(=own mid) ret, y = eth ret
        };
        self.c55_x.push(self.c55_x[i] + x55);
        self.c55_y.push(self.c55_y[i] + y55);
        self.c55_xx.push(self.c55_xx[i] + x55 * x55);
        self.c55_yy.push(self.c55_yy[i] + y55 * y55);
        self.c55_xy.push(self.c55_xy[i] + x55 * y55);

        // remember prev row
        for k in 0..5 {
            self.prev_bq5[k] = bids[k].1;
            self.prev_aq5[k] = asks[k].1;
        }
        self.prev_bq0 = b0q;
        self.prev_aq0 = a0q;
        self.prev_bp0 = b0p;
        self.prev_ap0 = a0p;
    }

    // ---------------------------------------------------------------- decision kernel
    /// Features at tick `idx` — per-sample bodies copied verbatim from the batch
    /// fill_* loops, reading the incrementally-maintained arrays. Input scope =
    /// deployed h150 (cols 17-19/30/50-53 stay 0). f32 rounding points identical.
    pub fn compute64(&self, idx: usize) -> [f32; NUM_FEATURES] {
        let mut f = [0f32; NUM_FEATURES];
        let n = self.ts.len();
        if idx >= n {
            return f;
        }
        let sample_ts = self.ts[idx];
        let mid_i = self.mids[idx];

        // ---- compute_lob_features per-sample ----
        f[0] = self.ofi[idx] as f32;
        f[1] = self.imb[idx] as f32;
        if idx >= 5 {
            f[2] = (self.imb[idx] - self.imb[idx - 5]) as f32;
        }
        f[3] = self.spread[idx] as f32;
        f[4] = if self.av5[idx] > 0.0 { (self.bv5[idx] / self.av5[idx]) as f32 } else { 10.0 };
        f[5] = if self.large[idx] { 1.0 } else { 0.0 };
        if idx >= 50 {
            let prev = self.mids[idx - 50];
            if prev > 0.0 {
                f[12] = ((mid_i - prev) / prev) as f32;
            }
        }
        let vol_len = self.vol_all.len();
        if idx >= 10 && vol_len > 0 {
            let j = (idx - 10).min(vol_len - 1);
            f[10] = self.vol_all[j] as f32;
        }
        {
            let target = sample_ts - 60_000;
            let lo = searchsorted_left_i64(&self.ts, target);
            let hi = idx + 1;
            let count = hi.saturating_sub(lo);
            if count > 0 {
                let vwap = (self.cum_mid[hi] - self.cum_mid[lo]) / (count as f64);
                if vwap > 0.0 {
                    f[11] = ((mid_i - vwap) / vwap) as f32;
                }
            }
        }

        // ---- fill_trade_features [6,7,8,9] ----
        if !self.t_ts.is_empty() {
            let right = searchsorted_right_i64(&self.t_ts, sample_ts);
            let left_5s = searchsorted_left_i64(&self.t_ts, sample_ts - 5_000);
            let left_1s = searchsorted_left_i64(&self.t_ts, sample_ts - 1_000);
            let left_30s = searchsorted_left_i64(&self.t_ts, sample_ts - 30_000);
            let buys5 = self.cum_buy[right] - self.cum_buy[left_5s];
            let sells5 = self.cum_sell[right] - self.cum_sell[left_5s];
            let tot5 = buys5 + sells5;
            f[6] = if tot5 > 0.0 { ((buys5 - sells5) / tot5) as f32 } else { 0.0 };
            f[7] = (right as f64 - left_1s as f64) as f32;
            f[8] = if self.cum_large[right] - self.cum_large[left_5s] > 0.0 { 1.0 } else { 0.0 };
            f[9] = ((self.cum_buy[right] - self.cum_buy[left_30s])
                - (self.cum_sell[right] - self.cum_sell[left_30s])) as f32;
        }

        // ---- fill_microstructure_depth ----
        if idx >= 25 && f[5] > 0.0 {
            let prev = self.mids[idx - 25];
            if (mid_i - prev).abs() < 0.10 {
                f[20] = 1.0;
            }
        }
        let vm_len = self.vol_mean_all.len();
        if idx >= 40 && vm_len > 0 && vol_len > 0 {
            let adj_vr = (idx - 40).min(vm_len - 1);
            let adj_v = (idx - 10).min(vol_len - 1);
            let vm = self.vol_mean_all[adj_vr];
            f[21] = if vm > 0.0 { (self.vol_all[adj_v] / (vm + 1e-10)) as f32 } else { 1.0 };
        }
        let hurst_len = self.hurst.len();
        if idx >= 100 && hurst_len > 0 {
            let adj = (idx - 100).min(hurst_len - 1);
            f[23] = self.hurst[adj] as f32;
        } else {
            f[23] = 0.5;
        }
        if idx >= 1 {
            let tick = 0.10;
            let bj = (self.bp0[idx] - self.bp0[idx - 1]).abs() / tick;
            let aj = (self.ap0[idx] - self.ap0[idx - 1]).abs() / tick;
            f[24] = (bj.max(aj) - 1.0).max(0.0) as f32;
        }
        let cw_len = self.bc_win.len();
        if idx >= 10 && cw_len > 0 {
            let adj = (idx - 10).min(cw_len - 1);
            f[25] = (self.ac_win[adj] - self.bc_win[adj]) as f32;
        }
        if idx >= 10 && !self.ofi_1s.is_empty() {
            let a = (idx - 10).min(self.ofi_1s.len() - 1);
            f[26] = self.ofi_1s[a] as f32;
        }
        if idx >= 50 && !self.ofi_5s.is_empty() {
            let a = (idx - 50).min(self.ofi_5s.len() - 1);
            f[27] = self.ofi_5s[a] as f32;
        }
        if idx >= 300 && !self.ofi_30s.is_empty() {
            let a = (idx - 300).min(self.ofi_30s.len() - 1);
            f[28] = self.ofi_30s[a] as f32;
        }
        if idx >= 300 && (f[26] as f64) * (f[28] as f64) < 0.0 {
            f[29] = f[26] - f[28];
        }
        f[31] = (self.ask_ema[idx] - self.bid_ema[idx]) as f32;
        f[32] = self.top3[idx] as f32;

        // ---- fill_microstructure_trades [22, 33] ----
        // batch guards: n >= 40 && ci_len >= 30 (ci_len = n-9 for the day) — with the
        // day-anchored state ci_len>=30 <=> n>=39, subsumed by n>=40.
        let ci_len_b = n.saturating_sub(9);        // batch ci_len for the same array
        let im_len_b = ci_len_b.saturating_sub(29); // batch im_len
        if n >= 40 && ci_len_b >= 30 && idx >= 40 && !self.curr_int.is_empty() && !self.int_mean_all.is_empty() {
            let adj_ci = (idx - 10).min(ci_len_b - 1).min(self.curr_int.len() - 1);
            let adj_im = (idx - 40).min(im_len_b - 1).min(self.int_mean_all.len() - 1);
            let im = self.int_mean_all[adj_im];
            f[22] = if im > 0.0 { (self.curr_int[adj_ci] / (im + 1e-10)) as f32 } else { 1.0 };
        }
        f[33] = self.eff_ema[idx] as f32;

        // ---- fill_horizon_features [34..39] ----
        {
            let now = sample_ts;
            let ts0 = self.ts[0];
            let mid_at = |t_ms: i64| -> f64 {
                let p = self.ts.partition_point(|&x| x <= t_ms);
                if p == 0 { 0.0 } else { self.mids[p - 1] }
            };
            let n_grid = self.grid_mid.len();
            let gn = self.g0 + n_grid as i64 - 1;
            let gidx = |t_ms: i64| -> usize {
                let s = t_ms / 1000;
                if s <= self.g0 { 0 } else if s >= gn { n_grid - 1 } else { (s - self.g0) as usize }
            };
            if now - 30_000 >= ts0 {
                let past = mid_at(now - 30_000);
                if past > 0.0 && mid_i > 0.0 {
                    f[34] = ((mid_i - past) / past) as f32;
                }
            }
            if now - 60_000 >= ts0 {
                let past = mid_at(now - 60_000);
                if past > 0.0 && mid_i > 0.0 {
                    f[35] = ((mid_i - past) / past) as f32;
                }
            }
            if now - 120_000 >= ts0 {
                let past = mid_at(now - 120_000);
                if past > 0.0 && mid_i > 0.0 {
                    f[36] = ((mid_i - past) / past) as f32;
                }
            }
            let gh = gidx(now);
            if now - 60_000 >= ts0 {
                let rv = self.cum_gsq[gh] - self.cum_gsq[gidx(now - 60_000)];
                f[37] = rv.max(0.0).sqrt() as f32;
            }
            if now - 120_000 >= ts0 {
                let rv = self.cum_gsq[gh] - self.cum_gsq[gidx(now - 120_000)];
                f[38] = rv.max(0.0).sqrt() as f32;
            }
            if idx >= 1200 {
                let bv = self.cum_pair[idx] - self.cum_pair[idx - 1200 + 1];
                f[39] = (std::f64::consts::FRAC_PI_2 * bv) as f32;
            }
        }

        // ---- fill_horizon_features_b [40..44] ----
        if idx >= 600 {
            f[40] = (self.cum_ofi_b[idx] - self.cum_ofi_b[idx - 600]) as f32;
        }
        if idx >= 1200 {
            f[41] = (self.cum_ofi_b[idx] - self.cum_ofi_b[idx - 1200]) as f32;
        }
        if !self.t_ts.is_empty() {
            let lo = searchsorted_left_i64(&self.t_ts, sample_ts - 60_000);
            let hi = searchsorted_right_i64(&self.t_ts, sample_ts);
            let signed = self.cum_signed[hi] - self.cum_signed[lo];
            let total = self.cum_abs[hi] - self.cum_abs[lo];
            f[42] = if total > 0.0 { (signed / total) as f32 } else { 0.0 };
        }
        if sample_ts > 0 {
            const FUNDING_PERIOD_MS: i64 = 8 * 3600 * 1000;
            let rem = sample_ts.rem_euclid(FUNDING_PERIOD_MS);
            f[43] = if rem == 0 { 0.0 } else { ((FUNDING_PERIOD_MS - rem) as f64 / 60_000.0) as f32 };
        }
        // [44] funding basis: the day-anchor file has mark_price=0 -> guard fails -> 0.
        // (Batch with the anchor single row: r=1>0, mark=0.0, `mark > 0` false -> 0.)

        // ---- fill_horizon_features_c [45..49] ----
        {
            let b0 = self.bp0[idx];
            let a0 = self.ap0[idx];
            let bq0 = self.bq0[idx];
            let aq0 = self.aq0[idx];
            let tot = bq0 + aq0;
            let spread = a0 - b0;
            if tot > 0.0 && spread > 1e-12 {
                let microprice = (aq0 * b0 + bq0 * a0) / tot;
                f[45] = ((microprice - mid_i) / spread) as f32;
            }
            if idx >= 30 {
                f[46] = (self.cum_ofi5[idx] - self.cum_ofi5[idx - 30]) as f32;
            }
            if idx >= 600 && !self.t_ts.is_empty() {
                let num = self.kyle_cxy[idx] - self.kyle_cxy[idx - 600];
                let den = self.kyle_cxx[idx] - self.kyle_cxx[idx - 600];
                if den > 1e-18 {
                    f[47] = (num / den) as f32;
                }
            }
            if !self.t_ts.is_empty() {
                let mut sum_abs_net = 0.0f64;
                let mut sum_total = 0.0f64;
                for k in 0..6i64 {
                    let hi_ts = sample_ts - k * 10_000;
                    let lo_ts = sample_ts - (k + 1) * 10_000;
                    let hi_idx = searchsorted_right_i64(&self.t_ts, hi_ts);
                    let lo_idx = searchsorted_right_i64(&self.t_ts, lo_ts);
                    let net = self.cum_signed[hi_idx] - self.cum_signed[lo_idx];
                    let tot_ = self.cum_abs[hi_idx] - self.cum_abs[lo_idx];
                    sum_abs_net += net.abs();
                    sum_total += tot_;
                }
                if sum_total > 0.0 {
                    f[48] = (sum_abs_net / sum_total) as f32;
                }
                if idx >= 300 {
                    let num = self.cum_cancel_c[idx] - self.cum_cancel_c[idx - 300];
                    let lo = searchsorted_left_i64(&self.t_ts, sample_ts - 30_000);
                    let hi = searchsorted_right_i64(&self.t_ts, sample_ts);
                    let den = self.cum_abs[hi] - self.cum_abs[lo];
                    if den > 0.0 {
                        f[49] = (num / den) as f32;
                    }
                }
            }
        }

        // ---- fill_horizon_features_d [55] (eth) — [54] overwritten below ----
        if !self.e_ts.is_empty() && idx >= 300 {
            let ti = idx;
            let lo = idx - 300;
            let w = 300f64;
            let sx = self.c55_x[ti] - self.c55_x[lo];
            let sy = self.c55_y[ti] - self.c55_y[lo];
            let sxx = self.c55_xx[ti] - self.c55_xx[lo];
            let syy = self.c55_yy[ti] - self.c55_yy[lo];
            let sxy = self.c55_xy[ti] - self.c55_xy[lo];
            let num = w * sxy - sx * sy;
            let den = (w * sxx - sx * sx) * (w * syy - sy * sy);
            if den > 1e-24 {
                f[55] = (num / den.sqrt()) as f32;
            }
        }

        // ---- fill_funding_features [13] — day-anchor single row => rate const ----
        if let Some(rate) = self.anchor_rate {
            f[13] = rate as f32;
        }

        // ---- fill_eth_features [14,15,16,54] ----
        if !self.e_ts.is_empty() {
            let right = searchsorted_right_i64(&self.e_ts, sample_ts);
            let left_1s = searchsorted_left_i64(&self.e_ts, sample_ts - 1_000);
            let bz = self.e_cum_buy[right] - self.e_cum_buy[left_1s];
            let sz = self.e_cum_sell[right] - self.e_cum_sell[left_1s];
            let tot = bz + sz;
            let flow_imb = if tot > 0.0 { (bz - sz) / tot } else { 0.0 };
            let last_price_at = |t: i64| -> f64 {
                let j = searchsorted_right_i64(&self.e_ts, t);
                if j > 0 { self.e_px[j - 1] } else { 0.0 }
            };
            let logret = |p_now: f64, p_prev: f64| -> f64 {
                if p_now > 0.0 && p_prev > 0.0 { (p_now / p_prev).ln() } else { 0.0 }
            };
            let p_now = last_price_at(sample_ts);
            f[14] = logret(p_now, last_price_at(sample_ts - 1_000)) as f32;
            f[16] = logret(p_now, last_price_at(sample_ts - 2_000)) as f32;
            f[54] = logret(p_now, last_price_at(sample_ts - 5_000)) as f32;
            f[15] = flow_imb as f32;
        }

        // ---- fill_deep_book [61,62,63] ----
        f[61] = self.obi20[idx] as f32;
        f[62] = self.obi1[idx] as f32;
        f[63] = self.obi10[idx] as f32;

        // ---- fill_liquidation_features [56,57,58] ----
        if !self.l_ts.is_empty() {
            let r = searchsorted_right_i64(&self.l_ts, sample_ts);
            let l5 = searchsorted_left_i64(&self.l_ts, sample_ts - 5_000);
            let l30 = searchsorted_left_i64(&self.l_ts, sample_ts - 30_000);
            let l60 = searchsorted_left_i64(&self.l_ts, sample_ts - 60_000);
            let a5 = self.l_can[r] - self.l_can[l5];
            let a30 = self.l_can[r] - self.l_can[l30];
            f[56] = if a5 > 0.0 { ((self.l_csn[r] - self.l_csn[l5]) / a5) as f32 } else { 0.0 };
            f[57] = if a30 > 0.0 { ((self.l_csn[r] - self.l_csn[l30]) / a30) as f32 } else { 0.0 };
            f[58] = ((self.l_can[r] - self.l_can[l60]).max(0.0) + 1.0).ln() as f32;
        }

        // ---- fill_oi_features [59,60] ----
        if !self.oi_ts.is_empty() {
            let r = searchsorted_right_i64(&self.oi_ts, sample_ts);
            if r > 0 {
                let now = self.oi_v[r - 1];
                if now > 0.0 {
                    let j30 = searchsorted_right_i64(&self.oi_ts, sample_ts - 30_000);
                    let j300 = searchsorted_right_i64(&self.oi_ts, sample_ts - 300_000);
                    if j30 > 0 {
                        f[59] = ((now - self.oi_v[j30 - 1]) / now) as f32;
                    }
                    if j300 > 0 {
                        f[60] = ((now - self.oi_v[j300 - 1]) / now) as f32;
                    }
                }
            }
        }

        f
    }
}
