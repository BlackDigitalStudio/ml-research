//! Rust port of `src/live_sim.py::simulate_trade` — forward trade simulator.
//!
//! Parity contract: byte-exact TradeOutcome (net_pnl_pct, gross_pnl_pct,
//! exit_reason, duration_ticks, partial_filled, trailing_step_reached) when
//! called with identical inputs. See `tests/parity_rust_live_sim.py`.

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SimDirection {
    Long,
    Short,
}

#[derive(Clone, Debug)]
pub struct LiveSimConfig {
    pub tp_pct: f64,
    pub sl_pct: f64,
    pub timeout_ticks: i64,
    pub commission_win_pct: f64,
    pub commission_loss_pct: f64,
    pub partial_tp_progress: f64,
    pub trailing_step1_progress: f64,
    pub trailing_step1_sl_floor_pct: f64,
    pub trailing_step1_sl_ratio: f64,
    pub trailing_step2_progress: f64,
    pub trailing_step2_sl_ratio: f64,
    pub fast_fill_adverse_ms: f64,
    pub fast_fill_threshold_ms: f64,
    pub fast_fill_sl_multiplier: f64,
    pub timeout_limit_ticks: i64,
    pub partial_enabled: bool,
    pub trailing_enabled: bool,
}

impl Default for LiveSimConfig {
    fn default() -> Self {
        LiveSimConfig {
            tp_pct: 0.20,
            sl_pct: 0.20,
            timeout_ticks: 600,
            commission_win_pct: 0.04,
            commission_loss_pct: 0.07,
            partial_tp_progress: 0.50,
            trailing_step1_progress: 0.50,
            trailing_step1_sl_floor_pct: 0.08,
            trailing_step1_sl_ratio: 0.30,
            trailing_step2_progress: 0.75,
            trailing_step2_sl_ratio: 0.50,
            fast_fill_adverse_ms: 50.0,
            fast_fill_threshold_ms: 100.0,
            fast_fill_sl_multiplier: 0.5,
            timeout_limit_ticks: 20,
            partial_enabled: true,
            trailing_enabled: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TradeOutcome {
    pub net_pnl_pct: f64,
    pub gross_pnl_pct: f64,
    pub exit_reason: ExitReason,
    pub duration_ticks: i64,
    pub partial_filled: bool,
    pub trailing_step_reached: i32,
}

/// Must serialize to the same strings used in Python (REASONS tuple in live_sim.py).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExitReason {
    TpHit,
    SlHit,
    TrailingSl1,
    TrailingSl2,
    PartialPlusTp,
    PartialPlusTrailingSl1,
    PartialPlusTrailingSl2,
    TimeoutLimit,
    TimeoutMarket,
    FastFillAdverse,
    FastFillSl,
    NoForwardData,
}

impl ExitReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExitReason::TpHit => "tp_hit",
            ExitReason::SlHit => "sl_hit",
            ExitReason::TrailingSl1 => "trailing_sl_1",
            ExitReason::TrailingSl2 => "trailing_sl_2",
            ExitReason::PartialPlusTp => "partial_plus_tp",
            ExitReason::PartialPlusTrailingSl1 => "partial_plus_trailing_sl_1",
            ExitReason::PartialPlusTrailingSl2 => "partial_plus_trailing_sl_2",
            ExitReason::TimeoutLimit => "timeout_limit",
            ExitReason::TimeoutMarket => "timeout_market",
            ExitReason::FastFillAdverse => "fast_fill_adverse",
            ExitReason::FastFillSl => "fast_fill_sl",
            ExitReason::NoForwardData => "no_forward_data",
        }
    }

    /// u8 id matching the index in Python live_sim.py REASONS tuple — stable
    /// wire-level code used by the batch sim binary.
    pub fn id(&self) -> u8 {
        match self {
            ExitReason::TpHit => 0,
            ExitReason::SlHit => 1,
            ExitReason::TrailingSl1 => 2,
            ExitReason::TrailingSl2 => 3,
            ExitReason::PartialPlusTp => 4,
            ExitReason::PartialPlusTrailingSl1 => 5,
            ExitReason::PartialPlusTrailingSl2 => 6,
            ExitReason::TimeoutLimit => 7,
            ExitReason::TimeoutMarket => 8,
            ExitReason::FastFillAdverse => 9,
            ExitReason::FastFillSl => 10,
            ExitReason::NoForwardData => 11,
        }
    }
}

/// Fold a simulation into a TradeOutcome with commissions applied.
fn make_outcome(
    direction: SimDirection,
    entry_px: f64,
    exit_px_remainder: f64,
    partial_px: Option<f64>,
    reason: ExitReason,
    duration_ticks: i64,
    partial_filled: bool,
    trailing_step_reached: i32,
    cfg: &LiveSimConfig,
) -> TradeOutcome {
    let sign = if matches!(direction, SimDirection::Long) {
        1.0
    } else {
        -1.0
    };
    let gross_pct = if partial_filled && partial_px.is_some() {
        let p = partial_px.unwrap();
        let a = sign * (p - entry_px) / entry_px * 100.0;
        let b = sign * (exit_px_remainder - entry_px) / entry_px * 100.0;
        0.5 * a + 0.5 * b
    } else {
        sign * (exit_px_remainder - entry_px) / entry_px * 100.0
    };

    let is_taker = matches!(
        reason,
        ExitReason::SlHit
            | ExitReason::TrailingSl1
            | ExitReason::TrailingSl2
            | ExitReason::PartialPlusTrailingSl1
            | ExitReason::PartialPlusTrailingSl2
            | ExitReason::FastFillSl
            | ExitReason::FastFillAdverse
            | ExitReason::TimeoutMarket
    );
    let commission_pct = if is_taker {
        cfg.commission_loss_pct
    } else {
        cfg.commission_win_pct
    };
    let net_pct = gross_pct - commission_pct;

    TradeOutcome {
        net_pnl_pct: net_pct,
        gross_pnl_pct: gross_pct,
        exit_reason: reason,
        duration_ticks,
        partial_filled,
        trailing_step_reached,
    }
}

/// Forward-simulate a single trade. Mirrors Python semantics exactly.
pub fn simulate_trade(
    direction: SimDirection,
    entry_px: f64,
    mid_path: &[f64],
    cfg: &LiveSimConfig,
    fill_latency_ms: f64,
) -> TradeOutcome {
    if entry_px <= 0.0 {
        return TradeOutcome {
            net_pnl_pct: 0.0,
            gross_pnl_pct: 0.0,
            exit_reason: ExitReason::NoForwardData,
            duration_ticks: 0,
            partial_filled: false,
            trailing_step_reached: 0,
        };
    }
    if mid_path.is_empty() {
        return TradeOutcome {
            net_pnl_pct: 0.0,
            gross_pnl_pct: 0.0,
            exit_reason: ExitReason::NoForwardData,
            duration_ticks: 0,
            partial_filled: false,
            trailing_step_reached: 0,
        };
    }

    let mut sl_pct_eff = cfg.sl_pct;
    if fill_latency_ms < cfg.fast_fill_adverse_ms {
        let exit_px = mid_path[0];
        return make_outcome(
            direction,
            entry_px,
            exit_px,
            None,
            ExitReason::FastFillAdverse,
            1,
            false,
            0,
            cfg,
        );
    }
    if fill_latency_ms < cfg.fast_fill_threshold_ms {
        sl_pct_eff = cfg.sl_pct * cfg.fast_fill_sl_multiplier;
    }

    let sign = if matches!(direction, SimDirection::Long) {
        1.0
    } else {
        -1.0
    };
    let tp_dist = entry_px * cfg.tp_pct / 100.0;
    let sl_dist = entry_px * sl_pct_eff / 100.0;
    let tp_px = entry_px + sign * tp_dist;
    let mut sl_px = entry_px - sign * sl_dist;

    let max_mon_ticks = (mid_path.len() as i64 - cfg.timeout_limit_ticks).max(0);
    let timeout_ticks = cfg.timeout_ticks.min(max_mon_ticks);
    if timeout_ticks <= 0 {
        let exit_px = mid_path[0];
        return make_outcome(
            direction,
            entry_px,
            exit_px,
            None,
            ExitReason::TimeoutMarket,
            1,
            false,
            0,
            cfg,
        );
    }

    let mut trailing_step: i32 = 0;
    let mut partial_filled = false;
    let mut partial_px: Option<f64> = None;

    let tt = timeout_ticks as usize;
    for t in 0..tt {
        let px = mid_path[t];
        let (hit_sl, hit_tp, progress);
        match direction {
            SimDirection::Long => {
                hit_sl = px <= sl_px;
                hit_tp = px >= tp_px;
                progress = if tp_dist > 0.0 { (px - entry_px) / tp_dist } else { 0.0 };
            }
            SimDirection::Short => {
                hit_sl = px >= sl_px;
                hit_tp = px <= tp_px;
                progress = if tp_dist > 0.0 { (entry_px - px) / tp_dist } else { 0.0 };
            }
        }
        if hit_sl {
            let reason = match (trailing_step, partial_filled) {
                (0, _) => ExitReason::SlHit,
                (1, false) => ExitReason::TrailingSl1,
                (1, true) => ExitReason::PartialPlusTrailingSl1,
                (_, false) => ExitReason::TrailingSl2,
                (_, true) => ExitReason::PartialPlusTrailingSl2,
            };
            return make_outcome(
                direction,
                entry_px,
                sl_px,
                partial_px,
                reason,
                (t + 1) as i64,
                partial_filled,
                trailing_step,
                cfg,
            );
        }
        if hit_tp {
            let reason = if partial_filled {
                ExitReason::PartialPlusTp
            } else {
                ExitReason::TpHit
            };
            return make_outcome(
                direction,
                entry_px,
                tp_px,
                partial_px,
                reason,
                (t + 1) as i64,
                partial_filled,
                trailing_step,
                cfg,
            );
        }
        // progress checks, largest-first so a single tick can jump both.
        if cfg.trailing_enabled
            && progress >= cfg.trailing_step2_progress
            && trailing_step < 2
        {
            let sl_offset = tp_dist * cfg.trailing_step2_sl_ratio;
            sl_px = entry_px + sign * sl_offset;
            trailing_step = 2;
        } else if progress >= cfg.trailing_step1_progress && trailing_step < 1 {
            if cfg.partial_enabled && !partial_filled {
                partial_px = Some(px);
                partial_filled = true;
            }
            if cfg.trailing_enabled {
                let min_offset = entry_px * (cfg.trailing_step1_sl_floor_pct / 100.0);
                let ratio_offset = tp_dist * cfg.trailing_step1_sl_ratio;
                let sl_offset = min_offset.max(ratio_offset);
                sl_px = entry_px + sign * sl_offset;
                trailing_step = 1;
            }
        }
    }

    // --- Timeout branch ---
    let timeout_mid = mid_path[tt - 1];
    let limit_window_start = tt;
    let limit_window_end =
        ((tt as i64 + cfg.timeout_limit_ticks) as usize).min(mid_path.len());

    let mut filled = false;
    let mut fill_tick = limit_window_end;
    if limit_window_end > limit_window_start {
        let window = &mid_path[limit_window_start..limit_window_end];
        for (i, &m) in window.iter().enumerate() {
            let hit = match direction {
                SimDirection::Long => m >= timeout_mid,
                SimDirection::Short => m <= timeout_mid,
            };
            if hit {
                filled = true;
                fill_tick = limit_window_start + i;
                break;
            }
        }
    }

    if filled {
        return make_outcome(
            direction,
            entry_px,
            timeout_mid,
            partial_px,
            ExitReason::TimeoutLimit,
            (fill_tick + 1) as i64,
            partial_filled,
            trailing_step,
            cfg,
        );
    }
    let fallback_tick = if limit_window_end > limit_window_start {
        limit_window_end - 1
    } else {
        tt - 1
    };
    let fallback_px = mid_path[fallback_tick];
    make_outcome(
        direction,
        entry_px,
        fallback_px,
        partial_px,
        ExitReason::TimeoutMarket,
        (fallback_tick + 1) as i64,
        partial_filled,
        trailing_step,
        cfg,
    )
}

/// Pick the best-PnL direction for training labels.
/// Returns (label, target_pnl_pct) where label matches src.model constants:
///   0 = UP, 1 = DOWN, 2 = FLAT.
pub fn label_from_outcomes(long_o: &TradeOutcome, short_o: &TradeOutcome) -> (u8, f64) {
    const UP: u8 = 0;
    const DOWN: u8 = 1;
    const FLAT: u8 = 2;
    let lp = long_o.net_pnl_pct;
    let sp = short_o.net_pnl_pct;
    if lp <= 0.0 && sp <= 0.0 {
        return (FLAT, lp.max(sp));
    }
    if lp >= sp {
        (UP, lp)
    } else {
        (DOWN, sp)
    }
}
