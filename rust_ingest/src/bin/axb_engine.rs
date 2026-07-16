//! axb_engine — sub-millisecond live decision engine for the h150 deploy.
//!
//! Bit-parity contract with the offline validation (recorder-EV, anchored policy):
//! - book = MirrorBook reconstruction from @depth@100ms diffs (chronos OrderBookV2
//!   semantics: REST seed limit=100, cap 100 levels, skip u<=last, no pu chain,
//!   reconcile every 900s reseed >=2 top-20 findings, reseed on reconnect);
//! - features = features_incr::FeatState (BYTE-EXACT vs feature_builder, see
//!   fb_incr_harness), day-anchored, reset at UTC midnight (== per-day sim files);
//! - scoring = gbt (BYTE-EXACT vs xgboost, see score_harness) with boot-solved
//!   base-margin bits + per-seed mu/sd/sA/sBg npys;
//! - funding = DAY-ANCHOR (col13 const per day, col44 = 0);
//! - decisions on the 3s exchange-ts grid anchored at calendar UTC midnight,
//!   decision tick = last book tick <= grid point, np.unique dedupe;
//! - tau = causal_rolling day-level (np.quantile 'linear' port), seeded from the
//!   anchored recorder score distribution (boot artifact).
//!
//! Hot path budget (measured on the harness): ~70us per decision. Order execution
//! stays in Python (axb_exec.py) behind a Unix socket — the engine only sends
//! {"side","score"} and logs the reply.
//!
//! Boot artifacts (WORKDIR/boot, produced by axb_boot.py):
//!   A{0..3}.json Bg{0..3}.json  mu{s}.npy sd{s}.npy sA{s}.npy sBg{s}.npy
//!   base_A{s}.npy base_Bg{s}.npy  tau_seed.npy  anchor.json {"day","rate"}
use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use ndarray::Array1;
use ndarray_npy::{read_npy, write_npy};
use scalper_ingest::features_incr::FeatState;
use scalper_ingest::gbt::Gbt;

const NS: i64 = 1_000_000_000;
const DAY_NS: i64 = 86400 * NS;
const GRID_STEP_NS: i64 = 3 * NS;
const BUDGETS: [f64; 4] = [5.0, 10.0, 20.0, 40.0];
const WPD: f64 = 28800.0;
const KDAYS: usize = 30;
const WARMUP_S: i64 = 400;
const STALE_S: f64 = 10.0;
const OI_POLL_S: u64 = 15;
const RECONCILE_S: u64 = 900;
const SEED_LIMIT: u32 = 100;
const MAX_LEVELS: usize = 100;
const LV: usize = 20;
const REST_BASE: &str = "https://fapi.binance.com";

// ---------------------------------------------------------------- events
enum Ev {
    Diff { sym: u8, u: i64, ets_ms: i64, b: Vec<(f64, f64)>, a: Vec<(f64, f64)> },
    BookSnap { sym: u8, uid: i64, b: Vec<(f64, f64)>, a: Vec<(f64, f64)>, reconcile: bool },
    Trade { eth: bool, ts_ms: i64, px: f64, qty: f64, is_sell: bool },
    Liq { ts_ms: i64, sn: f64, an: f64 },
    Mark { ets_ms: i64, rate: f64, mark: f64 },
    Oi { ts_ms: i64, v: f64 },
}

// ---------------------------------------------------------------- mirror book
struct MirrorBook {
    bids: std::collections::BTreeMap<ordered_float::OrderedFloat<f64>, f64>,
    asks: std::collections::BTreeMap<ordered_float::OrderedFloat<f64>, f64>,
    last_uid: i64,
    synced: bool,
}

impl MirrorBook {
    fn new() -> Self {
        Self { bids: Default::default(), asks: Default::default(), last_uid: -1, synced: false }
    }
    fn load(&mut self, uid: i64, b: &[(f64, f64)], a: &[(f64, f64)]) {
        self.bids.clear();
        self.asks.clear();
        for &(p, q) in b {
            if q > 0.0 {
                self.bids.insert(p.into(), q);
            }
        }
        for &(p, q) in a {
            if q > 0.0 {
                self.asks.insert(p.into(), q);
            }
        }
        self.last_uid = uid;
        self.synced = true;
        self.prune();
    }
    fn prune(&mut self) {
        while self.bids.len() > MAX_LEVELS {
            let k = *self.bids.keys().next().unwrap(); // lowest bid = worst
            self.bids.remove(&k);
        }
        while self.asks.len() > MAX_LEVELS {
            let k = *self.asks.keys().next_back().unwrap(); // highest ask = worst
            self.asks.remove(&k);
        }
    }
    fn apply(&mut self, u: i64, b: &[(f64, f64)], a: &[(f64, f64)]) -> bool {
        if !self.synced || u <= self.last_uid {
            return self.synced && false;
        }
        for &(p, q) in b {
            if q == 0.0 {
                self.bids.remove(&p.into());
            } else {
                self.bids.insert(p.into(), q);
            }
        }
        for &(p, q) in a {
            if q == 0.0 {
                self.asks.remove(&p.into());
            } else {
                self.asks.insert(p.into(), q);
            }
        }
        self.last_uid = u;
        self.prune();
        true
    }
    fn top20(&self, bids_out: &mut [(f64, f64); LV], asks_out: &mut [(f64, f64); LV]) {
        for k in 0..LV {
            bids_out[k] = (0.0, 0.0);
            asks_out[k] = (0.0, 0.0);
        }
        for (k, (p, q)) in self.bids.iter().rev().take(LV).enumerate() {
            bids_out[k] = (p.into_inner(), *q);
        }
        for (k, (p, q)) in self.asks.iter().take(LV).enumerate() {
            asks_out[k] = (p.into_inner(), *q);
        }
    }
    fn l1_mid(&self) -> f64 {
        match (self.bids.keys().next_back(), self.asks.keys().next()) {
            (Some(b), Some(a)) => 0.5 * (b.into_inner() + a.into_inner()),
            _ => 0.0,
        }
    }
    /// top-20 both sides vs REST snapshot, tol 0; findings count (recorder parity).
    fn reconcile_findings(&self, b: &[(f64, f64)], a: &[(f64, f64)]) -> usize {
        let rb: std::collections::HashMap<u64, f64> =
            b.iter().take(LV).map(|&(p, q)| (p.to_bits(), q)).collect();
        let ra: std::collections::HashMap<u64, f64> =
            a.iter().take(LV).map(|&(p, q)| (p.to_bits(), q)).collect();
        let lb: Vec<(f64, f64)> =
            self.bids.iter().rev().take(LV).map(|(p, q)| (p.into_inner(), *q)).collect();
        let la: Vec<(f64, f64)> =
            self.asks.iter().take(LV).map(|(p, q)| (p.into_inner(), *q)).collect();
        let mut n = 0usize;
        for (local, rest) in [(&lb, &rb), (&la, &ra)] {
            for &(p, q) in local.iter() {
                match rest.get(&p.to_bits()) {
                    Some(&qr) if qr == q => {}
                    _ => n += 1,
                }
            }
            let lset: std::collections::HashSet<u64> =
                local.iter().map(|&(p, _)| p.to_bits()).collect();
            n += rest.keys().filter(|k| !lset.contains(k)).count();
        }
        n
    }
}

// ---------------------------------------------------------------- tau (causal day-level)
struct Tau {
    buf: Vec<f64>,
    pending: Vec<f64>,
    frozen: bool,
    day: String,
    taus: [f64; 4],
    cap: usize,
    work: PathBuf,
}

fn np_quantile_linear(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let pos = q * (n as f64 - 1.0);
    let i = pos.floor() as usize;
    let frac = pos - i as f64;
    if i + 1 >= n {
        sorted[n - 1]
    } else {
        sorted[i] + (sorted[i + 1] - sorted[i]) * frac
    }
}

impl Tau {
    fn new(work: &PathBuf, boot: &PathBuf, day: &str) -> Result<Self> {
        let cap = (KDAYS as f64 * WPD) as usize;
        // FREEZE_TAU=1 -> FIXQ policy (HD5-DEPLOY): taus fixed at the boot-seed
        // quantiles for the whole run; recalibration = refresh RECEV_DIR + restart.
        // Frozen mode always derives from tau_seed.npy (never state_buf.npy) so a
        // restart is idempotent and rolling state cannot leak into the threshold.
        let frozen = std::env::var("FREEZE_TAU").map(|v| v == "1").unwrap_or(false);
        let (buf, pending, saved_day): (Vec<f64>, Vec<f64>, String) = {
            let b = work.join("state_buf.npy");
            if frozen {
                let sv: Array1<f64> = read_npy(boot.join("tau_seed.npy")).context("tau_seed")?;
                (sv.to_vec(), Vec::new(), String::new())
            } else if b.exists() {
                let bv: Array1<f64> = read_npy(&b).context("state_buf")?;
                let pv: Array1<f64> = read_npy(work.join("state_pending.npy")).unwrap_or_else(|_| Array1::zeros(0));
                let d = std::fs::read_to_string(work.join("state_day.txt")).unwrap_or_default();
                (bv.to_vec(), pv.to_vec(), d.trim().to_string())
            } else {
                let sv: Array1<f64> = read_npy(boot.join("tau_seed.npy")).context("tau_seed")?;
                (sv.to_vec(), Vec::new(), String::new())
            }
        };
        let pending = if saved_day == day { pending } else { Vec::new() };
        let mut t = Self { buf, pending, day: day.to_string(), taus: [0.0; 4], cap, work: work.clone(), frozen };
        if t.buf.len() > t.cap {
            t.buf = t.buf.split_off(t.buf.len() - t.cap);
        }
        t.recompute();
        eprintln!("tau seeded: buf={} pending={} taus={:?} frozen={}", t.buf.len(), t.pending.len(), t.taus, t.frozen);
        Ok(t)
    }
    fn recompute(&mut self) {
        let mut s = self.buf.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (i, t) in BUDGETS.iter().enumerate() {
            self.taus[i] = np_quantile_linear(&s, (1.0 - t / WPD).max(0.0));
        }
    }
    fn observe(&mut self, day: &str, score: f64) -> [f64; 4] {
        if day != self.day {
            self.buf.extend(self.pending.drain(..));
            if self.buf.len() > self.cap {
                self.buf = self.buf.split_off(self.buf.len() - self.cap);
            }
            self.day = day.to_string();
            if !self.frozen {
                self.recompute();
            }
            eprintln!("UTC day roll -> {} buf={} taus={:?} frozen={}", day, self.buf.len(), self.taus, self.frozen);
        }
        self.pending.push(score);
        self.taus
    }
    fn save(&self) {
        let _ = write_npy(self.work.join("state_buf.npy"), &Array1::from_vec(self.buf.clone()));
        let _ = write_npy(self.work.join("state_pending.npy"), &Array1::from_vec(self.pending.clone()));
        let _ = std::fs::write(self.work.join("state_day.txt"), &self.day);
    }
}

// ---------------------------------------------------------------- scoring bundle
struct Seed {
    a: Gbt,
    bg: Gbt,
    mu: Vec<f64>,
    sd: Vec<f64>,
    s_a: Vec<f64>,
    s_bg: Vec<f64>,
}

fn load_seeds(boot: &PathBuf) -> Result<Vec<Seed>> {
    let mut out = Vec::new();
    for s in 0..4 {
        let mut a = Gbt::load_json(&boot.join(format!("A{s}.json")))?;
        let mut bg = Gbt::load_json(&boot.join(format!("Bg{s}.json")))?;
        let ba: Array1<f32> = read_npy(boot.join(format!("base_A{s}.npy")))?;
        let bb: Array1<f32> = read_npy(boot.join(format!("base_Bg{s}.npy")))?;
        a.set_base_margin(ba[0]);
        bg.set_base_margin(bb[0]);
        let mu: Array1<f64> = read_npy(boot.join(format!("mu{s}.npy")))?;
        let sd: Array1<f64> = read_npy(boot.join(format!("sd{s}.npy")))?;
        let s_a: Array1<f64> = read_npy(boot.join(format!("sA{s}.npy")))?;
        let s_bg: Array1<f64> = read_npy(boot.join(format!("sBg{s}.npy")))?;
        out.push(Seed { a, bg, mu: mu.to_vec(), sd: sd.to_vec(), s_a: s_a.to_vec(), s_bg: s_bg.to_vec() });
    }
    Ok(out)
}

fn cdf(refs: &[f64], v: f64) -> f64 {
    refs.partition_point(|&x| x <= v) as f64 / refs.len().max(1) as f64
}

fn day_of_ms(ts_ms: i64) -> String {
    let days = ts_ms / 86_400_000;
    let (mut y, mut rem) = (1970i64, days);
    loop {
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let dy = if leap { 366 } else { 365 };
        if rem < dy {
            break;
        }
        rem -= dy;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let ml = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 0usize;
    while rem >= ml[m] {
        rem -= ml[m];
        m += 1;
    }
    format!("{:04}{:02}{:02}", y, m + 1, rem + 1)
}

// ---------------------------------------------------------------- exec bridge
struct ExecBridge {
    path: PathBuf,
}
impl ExecBridge {
    fn trade(&self, side_long: bool, score: f64) -> bool {
        use std::io::{BufRead, BufReader};
        use std::os::unix::net::UnixStream;
        let mut s = match UnixStream::connect(&self.path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("exec socket connect failed: {e}");
                return false;
            }
        };
        let _ = s.set_read_timeout(Some(std::time::Duration::from_millis(500)));
        let msg = format!("{{\"side\":\"{}\",\"score\":{score}}}\n", if side_long { "long" } else { "short" });
        if s.write_all(msg.as_bytes()).is_err() {
            return false;
        }
        let mut line = String::new();
        if BufReader::new(&s).read_line(&mut line).is_err() {
            return false;
        }
        line.contains("true")
    }
}

// ---------------------------------------------------------------- WS + REST tasks
fn fetch_depth_snapshot(sym: &str) -> Result<(i64, Vec<(f64, f64)>, Vec<(f64, f64)>)> {
    let url = format!("{REST_BASE}/fapi/v1/depth?symbol={}&limit={SEED_LIMIT}", sym.to_uppercase());
    let body: serde_json::Value = ureq::get(&url).timeout(std::time::Duration::from_secs(10)).call()?.into_json()?;
    let uid = body["lastUpdateId"].as_i64().context("lastUpdateId")?;
    let side = |k: &str| -> Vec<(f64, f64)> {
        body[k].as_array().map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let p: f64 = e[0].as_str()?.parse().ok()?;
                    let q: f64 = e[1].as_str()?.parse().ok()?;
                    Some((p, q))
                })
                .collect()
        }).unwrap_or_default()
    };
    Ok((uid, side("bids"), side("asks")))
}

fn parse_levels(v: &serde_json::Value) -> Vec<(f64, f64)> {
    v.as_array().map(|arr| {
        arr.iter()
            .filter_map(|e| {
                let p: f64 = e[0].as_str()?.parse().ok()?;
                let q: f64 = e[1].as_str()?.parse().ok()?;
                Some((p, q))
            })
            .collect()
    }).unwrap_or_default()
}

async fn ws_task(
    path: &'static str,
    streams: Vec<String>,
    tx: tokio::sync::mpsc::UnboundedSender<Ev>,
    doge_sym: String,
    btc_sym: String,
    eth_sym: String,
) {
    let url = format!("wss://fstream.binance.com/{path}/stream?streams={}", streams.join("/"));
    loop {
        match tokio_tungstenite::connect_async(&url).await {
            Ok((mut ws, _)) => {
                eprintln!("WS connected: /{path}");
                // (re)seed mirror books carried by this connection
                for s in &streams {
                    let base = s.split('@').next().unwrap_or("").to_string();
                    if (base == doge_sym || base == btc_sym) && s.contains("@depth@") {
                        let sym_id = if base == doge_sym { 0u8 } else { 1u8 };
                        let txc = tx.clone();
                        let b2 = base.clone();
                        tokio::task::spawn_blocking(move || {
                            match fetch_depth_snapshot(&b2) {
                                Ok((uid, b, a)) => {
                                    let _ = txc.send(Ev::BookSnap { sym: sym_id, uid, b, a, reconcile: false });
                                }
                                Err(e) => eprintln!("seed {b2} failed: {e}"),
                            }
                        });
                    }
                }
                while let Some(msg) = ws.next().await {
                    match msg {
                        Ok(tokio_tungstenite::tungstenite::Message::Text(raw)) => {
                            let v: serde_json::Value = match serde_json::from_str(&raw) {
                                Ok(v) => v,
                                Err(_) => continue,
                            };
                            let d = v.get("data").unwrap_or(&v);
                            match d.get("e").and_then(|e| e.as_str()).unwrap_or("") {
                                "depthUpdate" => {
                                    let s = d["s"].as_str().unwrap_or("").to_lowercase();
                                    let sym = if s == doge_sym { 0u8 } else if s == btc_sym { 1u8 } else { continue };
                                    let _ = tx.send(Ev::Diff {
                                        sym,
                                        u: d["u"].as_i64().unwrap_or(0),
                                        ets_ms: d["E"].as_i64().unwrap_or(0),
                                        b: parse_levels(&d["b"]),
                                        a: parse_levels(&d["a"]),
                                    });
                                }
                                "aggTrade" => {
                                    let s = d["s"].as_str().unwrap_or("").to_lowercase();
                                    let _ = tx.send(Ev::Trade {
                                        eth: s == eth_sym,
                                        ts_ms: d["T"].as_i64().unwrap_or(0),
                                        px: d["p"].as_str().and_then(|x| x.parse().ok()).unwrap_or(0.0),
                                        qty: d["q"].as_str().and_then(|x| x.parse().ok()).unwrap_or(0.0),
                                        is_sell: d["m"].as_bool().unwrap_or(false),
                                    });
                                }
                                "forceOrder" => {
                                    let o = &d["o"];
                                    let qty: f64 = o["q"].as_str().and_then(|x| x.parse().ok()).unwrap_or(0.0);
                                    let px: f64 = o["p"].as_str().and_then(|x| x.parse().ok()).unwrap_or(0.0);
                                    let side_buy = o["S"].as_str().unwrap_or("") == "BUY";
                                    let notional = qty.abs() * px;
                                    let _ = tx.send(Ev::Liq {
                                        ts_ms: o["T"].as_i64().unwrap_or(0),
                                        sn: if side_buy { notional } else { -notional },
                                        an: notional,
                                    });
                                }
                                "markPriceUpdate" => {
                                    let _ = tx.send(Ev::Mark {
                                        ets_ms: d["E"].as_i64().unwrap_or(0),
                                        rate: d["r"].as_str().and_then(|x| x.parse().ok()).unwrap_or(0.0),
                                        mark: d["p"].as_str().and_then(|x| x.parse().ok()).unwrap_or(0.0),
                                    });
                                }
                                _ => {}
                            }
                        }
                        Ok(tokio_tungstenite::tungstenite::Message::Ping(p)) => {
                            let _ = ws.send(tokio_tungstenite::tungstenite::Message::Pong(p)).await;
                        }
                        Ok(_) => {}
                        Err(e) => {
                            eprintln!("WS /{path} error: {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => eprintln!("WS /{path} connect failed: {e}"),
        }
        eprintln!("WS /{path} reconnect in 2s");
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
}

// ---------------------------------------------------------------- main
fn main() -> Result<()> {
    let work = PathBuf::from(std::env::var("WORKDIR").unwrap_or_else(|_| "/home/delmi/axb_h150".into()));
    let boot = work.join("boot");
    let mode = std::env::var("MODE").unwrap_or_else(|_| "shadow".into());
    let sym = std::env::var("SIGNAL_SYM").unwrap_or_else(|_| "dogeusdt".into());
    let btc_sym = "btcusdt".to_string();
    let eth_sym = "ethusdt".to_string();
    // Self-referential instance (SIGNAL_SYM == btc lead symbol, i.e. the BTC deploy):
    // subscribe the depth stream ONCE and feed the btc_lead mid series from the same
    // MirrorBook after each apply — identical semantics to the separate-book branch
    // (offline builders read the same single stream for both roles).
    let self_lead = sym == btc_sym;
    let trade_budget: f64 = std::env::var("TRADE_BUDGET").ok().and_then(|v| v.parse().ok()).unwrap_or(5.0);
    // FUNDING_MODE: "anchor" (default, deployed DOGE/XRP policy: col13 day-frozen,
    // col44=0) | "true" (live markPrice rows -> col13/col44, batch semantics; the
    // BTC x true-funding policy class, HD3 rev10).
    let funding_true = std::env::var("FUNDING_MODE").map(|v| v == "true").unwrap_or(false);
    std::fs::create_dir_all(work.join("decisions"))?;
    std::fs::create_dir_all(work.join("features"))?;

    let seeds = load_seeds(&boot)?;
    eprintln!("engine: {} seeds loaded, MODE={mode}", seeds.len());
    // funding anchor from boot (mid-day restart); day rolls update it from the stream
    let anchor_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(boot.join("anchor.json")).unwrap_or_else(|_| "{}".into()))
            .unwrap_or(serde_json::Value::Null);
    let mut anchor_day = anchor_json["day"].as_str().unwrap_or("").to_string();
    let mut anchor_rate = anchor_json["rate"].as_f64();
    eprintln!("funding anchor: day={anchor_day} rate={anchor_rate:?}");

    let today = day_of_ms(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis() as i64);
    let mut tau = Tau::new(&work, &boot, &today)?;

    let exec = ExecBridge { path: work.join("exec.sock") };
    let live = mode == "live";

    // ---- async IO -> engine channel ----
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Ev>();
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    {
        let pub_streams = if self_lead {
            vec![format!("{sym}@depth@100ms")]
        } else {
            vec![format!("{sym}@depth@100ms"), format!("{btc_sym}@depth@100ms")]
        };
        let mkt_streams = vec![
            format!("{sym}@aggTrade"),
            format!("{sym}@forceOrder"),
            format!("{sym}@markPrice@1s"),
            format!("{eth_sym}@aggTrade"),
        ];
        let (t1, t2, t3, t4) = (tx.clone(), tx.clone(), tx.clone(), tx.clone());
        let (s1, s2) = (sym.clone(), sym.clone());
        let (b1, b2) = (btc_sym.clone(), btc_sym.clone());
        let (e1, e2) = (eth_sym.clone(), eth_sym.clone());
        rt.spawn(async move { ws_task("public", pub_streams, t1, s1, b1, e1).await });
        rt.spawn(async move { ws_task("market", mkt_streams, t2, s2, b2, e2).await });
        // OI poll (15s, local receive time == recorder derivatives_poll)
        let oi_sym = sym.to_uppercase();
        rt.spawn(async move {
            loop {
                let symc = oi_sym.clone();
                let r = tokio::task::spawn_blocking(move || -> Result<f64> {
                    let url = format!("{REST_BASE}/fapi/v1/openInterest?symbol={symc}");
                    let v: serde_json::Value = ureq::get(&url).timeout(std::time::Duration::from_secs(8)).call()?.into_json()?;
                    Ok(v["openInterest"].as_str().and_then(|x| x.parse().ok()).context("oi")?)
                })
                .await;
                if let Ok(Ok(v)) = r {
                    let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
                    let _ = t3.send(Ev::Oi { ts_ms: now_ms, v });
                }
                tokio::time::sleep(std::time::Duration::from_secs(OI_POLL_S)).await;
            }
        });
        // reconcile every 900s (both symbols)
        let (rs1, rb1) = (sym.clone(), btc_sym.clone());
        rt.spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(RECONCILE_S)).await;
                for (sid, s) in [(0u8, rs1.clone()), (1u8, rb1.clone())] {
                    let txc = t4.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        if let Ok((uid, b, a)) = fetch_depth_snapshot(&s) {
                            let _ = txc.send(Ev::BookSnap { sym: sid, uid, b, a, reconcile: true });
                        }
                    })
                    .await;
                }
            }
        });
    }

    // ---- engine state ----
    let mut doge = MirrorBook::new();
    let mut btc = MirrorBook::new();
    let mut st = FeatState::new();
    st.anchor_rate = anchor_rate;
    st.funding_true = funding_true;
    if funding_true {
        eprintln!("FUNDING_MODE=true: col13/col44 from live markPrice rows");
    }
    let mut btc_ts: Vec<i64> = Vec::new();
    let mut btc_mid: Vec<f64> = Vec::new();
    // pending stream queues (commit contract: events with ts <= tick ts before the tick)
    let mut q_tr: VecDeque<(i64, f64, f64, bool)> = VecDeque::new();
    let mut q_eth: VecDeque<(i64, f64, f64, bool)> = VecDeque::new();
    let mut q_liq: VecDeque<(i64, f64, f64)> = VecDeque::new();
    let mut q_oi: VecDeque<(i64, f64)> = VecDeque::new();
    let mut grid_day = String::new();
    let mut next_g: i64 = 0;
    let mut last_dec_tick: i64 = -1;
    let mut last_book_wall = Instant::now();
    // Mid-day process start: the day-anchored state is missing the day's earlier
    // history, so gate decisions on a 400s span (Python-engine convention) until the
    // next midnight reset — from then on the state IS the sim's day state and the
    // sim-equivalent n>=50 gate applies (sim ends start at W-1=49).
    let mut partial_day = true;
    let mut n_dec: u64 = 0;
    let mut bids20 = [(0f64, 0f64); LV];
    let mut asks20 = [(0f64, 0f64); LV];

    eprintln!("engine loop started");
    while let Some(ev) = rx.blocking_recv() {
        match ev {
            Ev::BookSnap { sym: s, uid, b, a, reconcile } => {
                let book = if s == 0 { &mut doge } else { &mut btc };
                if reconcile && book.synced {
                    let n = book.reconcile_findings(&b, &a);
                    if n >= 2 {
                        eprintln!("reconcile sym{s} drift findings={n} — reseeding");
                        book.load(uid, &b, &a);
                    }
                } else {
                    book.load(uid, &b, &a);
                    eprintln!("MirrorBook sym{s} seeded: uid={uid}");
                }
            }
            Ev::Trade { eth, ts_ms, px, qty, is_sell } => {
                if eth {
                    q_eth.push_back((ts_ms, px, qty, is_sell));
                } else {
                    q_tr.push_back((ts_ms, px, qty, is_sell));
                }
            }
            Ev::Liq { ts_ms, sn, an } => q_liq.push_back((ts_ms, sn, an)),
            Ev::Oi { ts_ms, v } => q_oi.push_back((ts_ms, v)),
            Ev::Mark { ets_ms, rate, mark } => {
                if funding_true {
                    // TRUE mode: every markPrice row feeds col13/col44 (batch
                    // at-or-before semantics inside compute64). Day roll clears
                    // the rows with the rest of the day-anchored state.
                    st.push_funding(ets_ms, rate, mark);
                } else {
                    let d = day_of_ms(ets_ms);
                    if d != anchor_day {
                        anchor_day = d.clone();
                        anchor_rate = Some(rate);
                        st.anchor_rate = Some(rate);
                        eprintln!("funding day-anchor {d} = {rate}");
                    }
                }
            }
            Ev::Diff { sym: s, u, ets_ms, b, a } => {
                if s == 1 {
                    if btc.apply(u, &b, &a) {
                        let m = btc.l1_mid();
                        if m > 0.0 {
                            btc_ts.push(ets_ms * 1_000_000);
                            btc_mid.push(m);
                        }
                    }
                    continue;
                }
                if !doge.apply(u, &b, &a) {
                    continue;
                }
                if self_lead {
                    let m = doge.l1_mid();
                    if m > 0.0 {
                        btc_ts.push(ets_ms * 1_000_000);
                        btc_mid.push(m);
                    }
                }
                last_book_wall = Instant::now();
                let tick_ns = ets_ms * 1_000_000;
                // UTC day roll: reset day-anchored feature state (== per-day sim files)
                let d = day_of_ms(ets_ms);
                if d != grid_day {
                    let was_running = !grid_day.is_empty();
                    grid_day = d.clone();
                    next_g = (tick_ns / DAY_NS) * DAY_NS;
                    if st.n_ticks() > 0 {
                        st = FeatState::new();
                        st.anchor_rate = anchor_rate;
                        st.funding_true = funding_true;
                        btc_ts.clear();
                        btc_mid.clear();
                        last_dec_tick = -1;
                        eprintln!("day roll {d}: FeatState reset, grid anchor = midnight");
                    }
                    if was_running {
                        partial_day = false; // from midnight on, state == sim day state
                    }
                }
                // commit pending events with ts <= tick, then the tick itself
                while let Some(&(t, px, qty, is)) = q_tr.front() {
                    if t <= ets_ms {
                        st.push_trade(t, px, qty, is);
                        q_tr.pop_front();
                    } else {
                        break;
                    }
                }
                while let Some(&(t, px, qty, is)) = q_eth.front() {
                    if t <= ets_ms {
                        st.push_eth(t, px, qty, is);
                        q_eth.pop_front();
                    } else {
                        break;
                    }
                }
                while let Some(&(t, sn, an)) = q_liq.front() {
                    if t <= ets_ms {
                        st.push_liq(t, sn, an);
                        q_liq.pop_front();
                    } else {
                        break;
                    }
                }
                while let Some(&(t, v)) = q_oi.front() {
                    if t <= ets_ms {
                        st.push_oi(t, v);
                        q_oi.pop_front();
                    } else {
                        break;
                    }
                }
                doge.top20(&mut bids20, &mut asks20);
                st.push_book(ets_ms, &bids20, &asks20);

                // ---- grid decision: this tick crossed one or more grid points? ----
                if tick_ns < next_g {
                    continue;
                }
                // catch-up to the most recent passed grid point (np.unique dedupe below)
                let anchor = (tick_ns / DAY_NS) * DAY_NS;
                let k = (tick_ns - anchor) / GRID_STEP_NS;
                let g = anchor + k * GRID_STEP_NS;
                let g_eff = if g >= next_g { g } else { next_g };
                next_g = g_eff + GRID_STEP_NS;
                // gates: sim-equivalent n>=50 after a midnight reset; 400s span on the
                // partial start-up day (state lacks earlier history); staleness always.
                let n = st.n_ticks();
                if n < 50 {
                    continue;
                }
                if partial_day && st.ts[n - 1] - st.ts[0] < WARMUP_S * 1000 {
                    continue;
                }
                if last_book_wall.elapsed().as_secs_f64() > STALE_S {
                    continue;
                }
                // decision tick = last tick <= g_eff
                let idx = st.ts.partition_point(|&x| x <= g_eff / 1_000_000).saturating_sub(1);
                if st.ts[idx] as i64 == last_dec_tick {
                    continue; // np.unique(ends)
                }
                last_dec_tick = st.ts[idx];
                let t0 = Instant::now();
                let x64 = st.compute64(idx);
                let dtd = st.ts[idx] * 1_000_000; // ns
                // feat71
                let mut row = [0f64; 71];
                for c in 0..64 {
                    row[c] = x64[c] as f64;
                }
                if !btc_ts.is_empty() {
                    let nb = btc_ts.len();
                    let i_now = btc_ts.partition_point(|&x| x <= dtd).saturating_sub(1).min(nb - 1);
                    let b_now = btc_mid[i_now];
                    for (k2, wd) in [(0usize, 5i64), (1, 30), (2, 60)] {
                        let j = btc_ts.partition_point(|&x| x <= dtd - wd * NS).saturating_sub(1).min(nb - 1);
                        let a_ = btc_mid[j];
                        row[64 + k2] = if a_ > 0.0 && b_now > 0.0 { (b_now / a_).ln() * 1e4 } else { 0.0 };
                    }
                }
                let h = ((dtd as f64 / NS as f64) % 86400.0) / 3600.0;
                let hf = h % 8.0;
                let pi = std::f64::consts::PI;
                row[67] = (2.0 * pi * h / 24.0).sin();
                row[68] = (2.0 * pi * h / 24.0).cos();
                row[69] = (2.0 * pi * hf / 8.0).sin();
                row[70] = (2.0 * pi * hf / 8.0).cos();
                let mut x71 = [0f32; 71];
                for c in 0..71 {
                    x71[c] = row[c] as f32;
                }
                // score
                let mut fn_row = [0f32; 71];
                let mut sc = 0f64;
                let mut pa_sum = 0f64;
                let mut pb_sum = 0f64;
                for sd_ in &seeds {
                    for c in 0..71 {
                        fn_row[c] = ((x71[c] as f64 - sd_.mu[c]) / sd_.sd[c]) as f32;
                    }
                    let pa = sd_.a.predict_prob(&fn_row);
                    let pb = sd_.bg.predict_prob(&fn_row);
                    sc += cdf(&sd_.s_a, pa as f64) * cdf(&sd_.s_bg, (pb - 0.5f32).abs() as f64);
                    pa_sum += pa as f64;
                    pb_sum += pb as f64;
                }
                sc /= 4.0;
                let pa_m = pa_sum / 4.0;
                let pb_m = pb_sum / 4.0;
                let side_long = pb_m >= 0.5;
                let taus = tau.observe(&grid_day, sc);
                let takes: Vec<bool> = taus.iter().map(|&t| sc >= t).collect();
                let mut executed = false;
                let ti = BUDGETS.iter().position(|&b| b == trade_budget).unwrap_or(0);
                if live && takes[ti] {
                    executed = exec.trade(side_long, sc);
                }
                let lat_ms = t0.elapsed().as_secs_f64() * 1000.0;
                n_dec += 1;
                // decision log (schema-compatible with the Python engine)
                let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis() as i64;
                let rec = serde_json::json!({
                    "ts": iso8601_ms(now_ms),
                    "book_ts_us": st.ts[idx] * 1000,
                    "grid_ts_us": g_eff / 1000,
                    "bid": st_bp0(&st, idx), "ask": st_ap0(&st, idx),
                    "pA": (pa_m * 1e6).round() / 1e6, "pBg": (pb_m * 1e6).round() / 1e6,
                    "score": (sc * 1e6).round() / 1e6,
                    "side": if side_long { "long" } else { "short" },
                    "tau": {"5": r6(taus[0]), "10": r6(taus[1]), "20": r6(taus[2]), "40": r6(taus[3])},
                    "take5": takes[0], "take10": takes[1], "take20": takes[2], "take40": takes[3],
                    "executed": executed,
                    "lat_ms": (lat_ms * 10.0).round() / 10.0,
                    "nb": st.n_ticks(), "nt": q_len_t(&st), "nl": st.n_liq(),
                    "ne": q_len_e(&st), "nfd": 1, "noi": st.n_oi(),
                });
                let day_file = work.join("decisions").join(format!("{grid_day}.jsonl"));
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&day_file) {
                    let _ = writeln!(f, "{}", rec);
                }
                // feature capture: the engine's own decision inputs, so offline replay
                // is bit-exact by construction (independent of the recorder stream).
                // Record = grid_ts_us i64 LE + 71 x f32 LE (292 B, ~8.4 MB/day).
                let feat_file = work.join("features").join(format!("{grid_day}.bin"));
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&feat_file) {
                    let mut fbuf = Vec::with_capacity(8 + 71 * 4);
                    fbuf.extend_from_slice(&(g_eff / 1000).to_le_bytes());
                    for c in 0..71 {
                        fbuf.extend_from_slice(&x71[c].to_le_bytes());
                    }
                    let _ = f.write_all(&fbuf);
                }
                if takes.iter().any(|&t| t) || n_dec % 100 == 0 {
                    eprintln!("#{n_dec} score={sc:.4} side={} takes={:?} lat={lat_ms:.2}ms",
                        if side_long { "long" } else { "short" },
                        BUDGETS.iter().zip(&takes).filter(|(_, &t)| t).map(|(b, _)| *b as i64).collect::<Vec<_>>());
                }
                if n_dec % 50 == 0 {
                    tau.save();
                }
            }
        }
    }
    Ok(())
}

fn r6(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}
/// "YYYY-MM-DDTHH:MM:SS.mmm+00:00" from epoch ms (matches the Python engine's format).
fn iso8601_ms(ms: i64) -> String {
    let day = day_of_ms(ms);
    let rem = ms.rem_euclid(86_400_000);
    let (h, m2) = (rem / 3_600_000, (rem % 3_600_000) / 60_000);
    let (s, mil) = ((rem % 60_000) / 1000, rem % 1000);
    format!("{}-{}-{}T{:02}:{:02}:{:02}.{:03}+00:00", &day[0..4], &day[4..6], &day[6..8], h, m2, s, mil)
}
fn st_bp0(st: &FeatState, idx: usize) -> f64 {
    st.bp0_at(idx)
}
fn st_ap0(st: &FeatState, idx: usize) -> f64 {
    st.ap0_at(idx)
}
fn q_len_t(st: &FeatState) -> usize {
    st.n_trades()
}
fn q_len_e(st: &FeatState) -> usize {
    st.n_eth()
}
