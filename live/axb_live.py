#!/usr/bin/env python3
"""AxB live engine — real-time deploy of the frozen deploy_robust2 maker model (DOGE, 30s hold).

Modes (env MODE):
  shadow (default) — signal + decision log only, NO orders.
  live             — signal from DOGEUSDT (model's market), EXECUTION on DOGEUSDC (maker fee 0%):
                     entry GTX post-only at touch (window ~12.8s = 120 ticks, else cancel),
                     hold to decision_ts+30s, pegged maker-only reduce-only exit (re-quote on
                     adverse touch move), taker backstop only on EXIT_MAX_S / hard-loss breach.
                     Sizing: SIZE_FRAC of available USDC; budget take-flag = TRADE_BUDGET.
                     Guards: one position at a time, daily loss halt, daily trade-count halt,
                     crash recovery closes any orphan position at startup.

SHADOW path (also active in live mode — the decision log is identical):
Economics of shadow decisions are evaluated offline by the frozen simulator (grid_sim_exitdbg)
on the recorder's captured stream — see live/axb_shadow_eval.py. Parity with the validated
offline pipeline (scripts/subs60_recorder_ev.py) by construction:
  - book = @depth20@100ms (/public routed path) == recorder depth_snapshot view (top-20, 100ms);
  - trades = @aggTrade, liq = @forceOrder (/market routed path — legacy /ws drops these SILENTLY);
  - features: the SAME rust feature_builder binary on a rolling window (all feature windows are
    <= 300s / 1200 ticks, so a slice >= WARMUP_S of history is identical to full-day compute);
  - feat71 = X64 + btc_lead zeros + ToD (the validated recorder-EV config; btc/funding/OI zeros
    are the conservative frozen config — do NOT add live values without re-validating);
  - normalization mu/sd frozen from the bundle (KNORM last train days);
  - score = cdf(pA,sA)*cdf(|pBg-0.5|,sBg); side = pBg>=0.5;
  - threshold: causal_rolling DAY-level parity — tau fixed within a UTC day, buffer extended at
    day roll, seeded from bundle axb_seed + the recorder-EV validated days (_recev_tmp), cap
    KDAYS*WPD; one tau per budget in BUDGETS.
Decision cadence DECIDE_S=10.8s == offline TARGET=8000 samples/day.

State: /home/delmi/axb/state.npz (threshold buffer), decisions appended to
/home/delmi/axb/decisions/YYYYMMDD.jsonl and uploaded to GCS hourly.
Env: MODE=shadow (only mode implemented), BUNDLE_DIR, FB_BIN, WORKDIR.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import websockets
import xgboost as xgb
from google.cloud import storage

log = logging.getLogger("axb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROJ = "project-0998ac51-36ba-445c-bc7"
MKT = "market-data-0998ac51"
# Symbol parameterization (one codebase, one systemd unit per symbol).
SYMK = os.environ.get("SYMK", "DOGE")           # DOGE / BTC
SYM = os.environ.get("SIGNAL_SYM", "dogeusdt")  # signal streams (USDT perp)
EXEC_SYM = os.environ.get("EXEC_SYM", "DOGEUSDC")  # maker fee 0% venue
ETH_SYM = "ethusdt"                             # ETH-lead trades (features 14-16, 55)
BTC_SYM = "btcusdt"                             # BTC-lead book mid (features 64-66)
# h150 deploy bundle = 4-seed ENSEMBLE; recorder ensemble scores seed the threshold.
BUNDLE = os.environ.get("BUNDLE_DIR", f"research_runs/deploy_h150/{SYMK}")
ENSEMBLE_SEEDS = [0, 1, 2, 3]
RECEV_TMP = os.environ.get("RECEV_DIR", f"research_runs/_recev_h150_{SYMK}")  # recorder score seed
SHADOW_GCS = f"research_runs/axb_shadow_h150/{SYMK}"
FB = os.environ.get("FB_BIN", "/home/delmi/axb/feature_builder")
WORK = os.environ.get("WORKDIR", "/home/delmi/axb")
WS_BASE = "wss://fstream.binance.com"

NS = 1_000_000_000
LV = 20
DECIDE_S = float(os.environ.get("DECIDE_S", "3.0"))   # h150 live cadence (offline 3s grid)
WPD = round(86400 / DECIDE_S)                          # 28800
BUDGETS = [5.0, 10.0, 20.0, 40.0]
KDAYS = 30
BUFFER_S = 1200                 # ring buffers (h150: entry 60s + hold 150s + feature window 300s)
WARMUP_S = 400                  # no decisions until this much book history
STALE_S = 10                    # no decisions if the freshest book tick is older than this
OI_POLL_S = 4.0                 # open-interest REST poll cadence (recorder derivatives_poll ~4s)

# ---- execution (MODE=live) ----
MODE = os.environ.get("MODE", "shadow")
TRADE_BUDGET = float(os.environ.get("TRADE_BUDGET", "5"))   # which take-flag fires a trade
SIZE_FRAC = float(os.environ.get("SIZE_FRAC", "1.0"))        # fraction of available USDC as notional
LEVERAGE = 2                    # margin headroom only; notional stays SIZE_FRAC * balance
ENTRY_WIN_S = float(os.environ.get("ENTRY_WIN_S", "60.0"))  # h150 maker entry window from decision
HOLD_S = float(os.environ.get("HOLD_S", "150.0"))           # h150 hold FROM FILL before chasing exit
# Per-symbol price/qty formatting (Binance USDⓈ-M exchangeInfo). DOGEUSDC tick 1e-5, qty step 1;
# BTCUSDC tick 0.1, qty step 1e-3. Verified before enabling a symbol's live mode.
PX_DEC = {"DOGEUSDC": 5, "BTCUSDC": 1}.get(EXEC_SYM, 5)
QTY_DEC = {"DOGEUSDC": 0, "BTCUSDC": 3}.get(EXEC_SYM, 0)
PX_TICK = {"DOGEUSDC": 0.00001, "BTCUSDC": 0.1}.get(EXEC_SYM, 0.00001)
# Policy (user 2026-07-05): maker-only exit, NEVER taker — chase until filled. The backstop
# below is a CATASTROPHIC-only guard (flash-crash account protection), not an exit path:
# 5-min chase fills 98.6-98.8% (measured 20260704), so these should fire ~never.
EXIT_MAX_S = float(os.environ.get("EXIT_MAX_S", "86400"))
HARD_LOSS_BP = float(os.environ.get("HARD_LOSS_BP", "300"))
DAY_LOSS_HALT = 0.05            # halt trading for the UTC day at -5% of day-start balance
DAY_TRADES_HALT = 40            # anomaly brake (t10 selectivity targets ~10/day)
REST_BASE = "https://fapi.binance.com"


# ---------------------------------------------------------------- bundle
class Bundle:
    """4-seed ENSEMBLE. Each seed has its own vol-norm (mu/sd) + CDF refs (sA/sBg); the ensemble
    score = mean over seeds of cdf(pA_s)*cdf(|pBg_s-.5|); side = mean pBg >= 0.5. Byte-mirror of
    recorder_ev_h150.py so live scores match the validated offline recorder-EV."""

    def __init__(self) -> None:
        cl = storage.Client(project=PROJ)
        self.mkt = cl.bucket(MKT)
        self.seeds = []
        for s in ENSEMBLE_SEEDS:
            base = f"{BUNDLE}/seed{s}"
            refs = np.load(io.BytesIO(self.mkt.blob(f"{base}/refs.npz").download_as_bytes()))
            meta = json.loads(self.mkt.blob(f"{base}/meta.json").download_as_bytes())
            knorm = meta["KNORM"]
            gstd = refs["gstd"].astype(np.float64)
            mu = refs["day_mean"].astype(np.float64)[-knorm:].mean(0)
            sd = np.maximum(np.sqrt(np.maximum(refs["day_var"].astype(np.float64)[-knorm:].mean(0), 0)),
                            0.2 * gstd + 1e-9)
            A = xgb.Booster(); Bg = xgb.Booster()
            for nm, m in (("A", A), ("Bg", Bg)):
                p = f"{WORK}/{nm}{s}.json"
                self.mkt.blob(f"{base}/{nm}.json").download_to_filename(p)
                m.load_model(p)
            self.seeds.append({"A": A, "Bg": Bg, "mu": mu, "sd": sd,
                               "sA": refs["sA"], "sBg": refs["sBg"]})
        log.info("ensemble bundle %s loaded (%d seeds, KNORM=%d)", BUNDLE, len(self.seeds), knorm)

    def score(self, x71: np.ndarray) -> tuple[float, float, float]:
        """x71 = raw feat71 (1,71) f32, UN-normalized (each seed applies its own mu/sd)."""
        cdf = lambda v, ref: float(np.searchsorted(ref, v, "right")) / max(len(ref), 1)
        sc = 0.0; pb_sum = 0.0; pa_sum = 0.0
        for sd in self.seeds:
            fn = ((x71 - sd["mu"]) / sd["sd"]).astype(np.float32)
            dm = xgb.DMatrix(fn)
            pa_ = float(sd["A"].predict(dm)[0]); pb = float(sd["Bg"].predict(dm)[0])
            sc += cdf(pa_, sd["sA"]) * cdf(abs(pb - 0.5), sd["sBg"])
            pb_sum += pb; pa_sum += pa_
        n = len(self.seeds)
        return pa_sum / n, pb_sum / n, sc / n


# ---------------------------------------------------------------- threshold (day-level causal_rolling)
class Threshold:
    """Mirror of causal_rolling: tau fixed within a UTC day, buffer extended at day roll."""

    STATE = f"{WORK}/state.npz"

    def __init__(self, bundle: Bundle) -> None:
        self.cap = KDAYS * WPD
        self.day = self._utc_day()
        self.pending: list[float] = []      # scores of the current (incomplete) day
        if os.path.exists(self.STATE):
            z = np.load(self.STATE)
            self.buf = list(z["buf"].astype(np.float64))
            saved_day = str(z["day"])
            if saved_day == self.day:
                self.pending = list(z["pending"].astype(np.float64))
            log.info("threshold state restored: buf=%d pending=%d (saved day %s)",
                     len(self.buf), len(self.pending), saved_day)
        else:
            # Seed tau from the RECORDER ensemble score distribution (matches the live venue; the CL
            # seed ran hot and under-traded — 2026-07-07 diagnosis). Causal roll from here.
            self.buf = []
            for bl in sorted(b.name for b in bundle.mkt.client.list_blobs(bundle.mkt, prefix=f"{RECEV_TMP}/D_")
                             if b.name.endswith(".npz")):
                z = np.load(io.BytesIO(bundle.mkt.blob(bl).download_as_bytes()))
                self.buf.extend(z["score"].astype(np.float64).tolist())
            self.buf = self.buf[-self.cap:]
            log.info("threshold seeded from recorder scores %s -> buf=%d", RECEV_TMP, len(self.buf))
        self.tau = self._taus()

    @staticmethod
    def _utc_day() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    def _taus(self) -> dict[float, float]:
        if not self.buf:
            return {t: 0.0 for t in BUDGETS}
        a = np.asarray(self.buf)
        return {t: float(np.quantile(a, max(0.0, 1.0 - t / WPD))) for t in BUDGETS}

    def observe(self, score: float) -> dict[float, float]:
        """Register a decision score; returns the taus in force for it. Rolls day if needed."""
        d = self._utc_day()
        if d != self.day:
            self.buf.extend(self.pending)
            self.buf = self.buf[-self.cap:]
            self.pending = []
            self.day = d
            self.tau = self._taus()
            log.info("UTC day roll -> %s, buf=%d, tau=%s", d, len(self.buf),
                     {int(k): round(v, 4) for k, v in self.tau.items()})
        self.pending.append(score)
        return self.tau

    def save(self) -> None:
        np.savez(self.STATE, buf=np.asarray(self.buf), pending=np.asarray(self.pending), day=self.day)


# ---------------------------------------------------------------- market state
class Buffers:
    def __init__(self) -> None:
        self.book: deque = deque()     # (ts_us, bids[(p,q)*20], asks[(p,q)*20])
        self.trades: deque = deque()   # (ts_us, id, price, qty, is_buyer_maker)
        self.liq: deque = deque()      # (ts_us, side_lower, qty, price)
        self.eth: deque = deque()      # (ts_us, id, price, qty, is_buyer_maker) — ETH-lead
        self.btc: deque = deque()      # (ts_us, mid) — BTC-lead book mid
        self.funding: deque = deque()  # (ts_us, funding_rate, mark_price)
        self.oi: deque = deque()       # (ts_us, open_interest) — REST poll
        self.last_book_wall = 0.0

    def prune(self) -> None:
        cut = (time.time() - BUFFER_S) * 1e6
        for dq in (self.book, self.trades, self.liq, self.eth, self.btc, self.funding, self.oi):
            while dq and dq[0][0] < cut:
                dq.popleft()

    def warm(self) -> bool:
        return (len(self.book) > 100
                and self.book[-1][0] - self.book[0][0] >= WARMUP_S * 1e6
                and time.time() - self.last_book_wall < STALE_S)


BUF = Buffers()
EXEC: "Executor | None" = None


async def ws_consumer(path: str, streams: list[str]) -> None:
    url = f"{WS_BASE}/{path}/stream?streams={'/'.join(streams)}"
    while True:
        try:
            async with websockets.connect(url, ping_interval=180, ping_timeout=600,
                                          max_size=2 ** 22) as ws:
                log.info("WS connected: /%s (%s)", path, ",".join(streams))
                async for raw in ws:
                    m = json.loads(raw)
                    d = m.get("data", m)
                    e = d.get("e", "")
                    if e == "depthUpdate":
                        ts = int(d["E"]) * 1000
                        bids = [(float(p), float(q)) for p, q in d["b"][:LV]]
                        asks = [(float(p), float(q)) for p, q in d["a"][:LV]]
                        if len(bids) == LV and len(asks) == LV:
                            BUF.book.append((ts, bids, asks))
                            BUF.last_book_wall = time.time()
                    elif e == "aggTrade":
                        rec = (int(d["T"]) * 1000, int(d["a"]), float(d["p"]),
                               float(d["q"]), bool(d["m"]))
                        (BUF.eth if str(d.get("s", "")).lower() == ETH_SYM else BUF.trades).append(rec)
                    elif e == "forceOrder":
                        o = d["o"]
                        BUF.liq.append((int(o["T"]) * 1000, str(o["S"]).lower(),
                                        float(o["q"]), float(o["p"])))
                    elif e == "markPriceUpdate":
                        BUF.funding.append((int(d["E"]) * 1000, float(d.get("r", 0) or 0), float(d["p"])))
                    elif e == "bookTicker":
                        s = str(d.get("s", "")).lower()
                        if s == BTC_SYM:
                            BUF.btc.append((int(d.get("T", d.get("E", 0))) * 1000,
                                            0.5 * (float(d["b"]) + float(d["a"]))))
                        elif EXEC is not None:
                            EXEC.on_book_ticker(d)
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            log.warning("WS /%s dropped: %s — reconnect in 2s", path, ex)
            await asyncio.sleep(2)


# ---------------------------------------------------------------- feature computation
def _write_cl_trades(rows: list, path: str) -> int:
    """CL-format trades parquet from (ts_us, id, price, qty, is_buyer_maker) rows; dedup by id."""
    n = len(rows)
    if not n:
        return 0
    tid = np.fromiter((t[1] for t in rows), np.int64, n)
    _, uidx = np.unique(tid, return_index=True)
    uidx.sort()
    tts = np.fromiter((rows[i][0] for i in uidx), np.int64, len(uidx))
    pq.write_table(pa.table({
        "side": np.array(["sell" if rows[i][4] else "buy" for i in uidx]),
        "amount": np.fromiter((rows[i][3] for i in uidx), np.float64, len(uidx)),
        "price": np.fromiter((rows[i][2] for i in uidx), np.float64, len(uidx)),
        "id": tid[uidx],
        "timestamp": tts * 1000,
        "receipt_timestamp": tts * 1000,
    }).sort_by("timestamp"), path)
    return len(uidx)


def write_window_parquet(book: list, trades: list, liq: list,
                         eth: list, funding: list, oi: list) -> tuple:
    """Frozen buffers -> CL-format parquet in WORK. Returns (n_book, n_trades, n_liq, n_eth, n_fd, n_oi, last_ts_us)."""
    nb = len(book)
    cols: dict[str, np.ndarray] = {}
    ts = np.fromiter((b[0] for b in book), np.int64, nb)
    cols["timestamp"] = ts * 1000                     # us -> ns (CL convention)
    cols["receipt_timestamp"] = ts * 1000
    cols["sequence_number"] = np.arange(nb, dtype=np.int64)
    for i in range(LV):
        cols[f"bid_{i}_price"] = np.fromiter((b[1][i][0] for b in book), np.float64, nb)
        cols[f"bid_{i}_size"] = np.fromiter((b[1][i][1] for b in book), np.float64, nb)
        cols[f"ask_{i}_price"] = np.fromiter((b[2][i][0] for b in book), np.float64, nb)
        cols[f"ask_{i}_size"] = np.fromiter((b[2][i][1] for b in book), np.float64, nb)
    pq.write_table(pa.table(cols), f"{WORK}/book.parquet")

    nt = _write_cl_trades(trades, f"{WORK}/trades.parquet")
    ne = _write_cl_trades(eth, f"{WORK}/eth.parquet")

    nl = len(liq)
    if nl:
        lts = np.fromiter((x[0] for x in liq), np.int64, nl)
        pq.write_table(pa.table({
            "side": np.array([x[1] for x in liq]),
            "quantity": np.fromiter((x[2] for x in liq), np.float64, nl),
            "price": np.fromiter((x[3] for x in liq), np.float64, nl),
            "id": np.arange(nl, dtype=np.int64),
            "status": np.array(["filled"] * nl),
            "timestamp": lts * 1000,
            "receipt_timestamp": lts * 1000,
        }).sort_by("timestamp"), f"{WORK}/liq.parquet")

    nfd = len(funding)
    if nfd:
        fts = np.fromiter((x[0] for x in funding), np.int64, nfd)
        pq.write_table(pa.table({
            "funding_rate": np.fromiter((x[1] for x in funding), np.float64, nfd),
            "mark_price": np.fromiter((x[2] for x in funding), np.float64, nfd),
            "timestamp": fts * 1000,
        }).sort_by("timestamp"), f"{WORK}/fund.parquet")

    noi = len(oi)
    if noi:
        ots = np.fromiter((x[0] for x in oi), np.int64, noi)
        pq.write_table(pa.table({
            "open_interest": np.fromiter((x[1] for x in oi), np.float64, noi),
            "timestamp": ots * 1000,
        }).sort_by("timestamp"), f"{WORK}/oi.parquet")
    return nb, nt, nl, ne, nfd, noi, int(ts[-1])


def compute_features(nb: int, nt: int, nl: int, ne: int, nfd: int, noi: int) -> np.ndarray | None:
    np.save(f"{WORK}/idx.npy", np.array([nb - 1], dtype=np.int64))
    cmd = [FB, "--depth", f"{WORK}/book.parquet", "--indices", f"{WORK}/idx.npy",
           "--out", f"{WORK}/f.npy"]
    if nt:
        cmd += ["--trades", f"{WORK}/trades.parquet"]
    if nl:
        cmd += ["--liquidations", f"{WORK}/liq.parquet"]
    if nfd:
        cmd += ["--funding", f"{WORK}/fund.parquet"]
    if noi:
        cmd += ["--open-interest", f"{WORK}/oi.parquet"]
    if ne:
        cmd += ["--eth", f"{WORK}/eth.parquet"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("FB failed: %s", r.stderr[-300:])
        return None
    return np.load(f"{WORK}/f.npy").astype(np.float64)


def feat71(ts_ns: int, x64: np.ndarray, btc: list) -> np.ndarray:
    # btc_lead cols 64-66: log(mid_now / mid_{5,30,60s ago}) * 1e4 from the BTC book buffer.
    bl = [0.0, 0.0, 0.0]
    if btc:
        bts = np.fromiter((b[0] for b in btc), np.int64, len(btc))
        bmid = np.fromiter((b[1] for b in btc), np.float64, len(btc))
        now = bmid[-1]
        for k, Wd in enumerate((5, 30, 60)):
            j = int(np.searchsorted(bts, ts_ns - int(Wd * NS), "right") - 1)
            if 0 <= j < len(bmid) and bmid[j] > 0 and now > 0:
                bl[k] = float(np.log(now / bmid[j]) * 1e4)
    h = ((ts_ns / NS) % 86400.0) / 3600.0
    hf = h % 8.0
    tod = [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
           np.sin(2 * np.pi * hf / 8), np.cos(2 * np.pi * hf / 8)]
    # f32 round before normalization == offline feat71 (parity)
    return np.concatenate([x64.ravel(), np.asarray(bl), np.asarray(tod)])[None, :].astype(np.float32)


# ---------------------------------------------------------------- execution (MODE=live)
class Rest:
    """Minimal signed Binance USDS-M futures REST client (blocking; call via to_thread)."""

    def __init__(self) -> None:
        import hashlib
        import hmac
        import urllib.parse
        import urllib.request
        self._h = hashlib
        self._hmac = hmac
        self._up = urllib.parse
        self._ur = urllib.request
        cfg = {}
        with open(f"{WORK}/config.env") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, _, v = line.strip().partition("=")
                    cfg[k.strip()] = v.strip()
        self.key = cfg["BINANCE_API_KEY"]
        self.sec = cfg["BINANCE_API_SECRET"].encode()

    def call(self, method: str, path: str, params: dict | None = None, signed: bool = True) -> dict | list:
        params = dict(params or {})
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
        qs = self._up.urlencode(params)
        if signed:
            qs += "&signature=" + self._hmac.new(self.sec, qs.encode(), self._h.sha256).hexdigest()
        url = f"{REST_BASE}{path}?{qs}"
        req = self._ur.Request(url, headers={"X-MBX-APIKEY": self.key}, method=method)
        try:
            with self._ur.urlopen(req, timeout=10) as r:
                return json.loads(r.read())
        except Exception as ex:  # HTTPError carries the Binance error json
            body = ""
            if hasattr(ex, "read"):
                try:
                    body = ex.read().decode()[:400]
                except Exception:
                    pass
            return {"_error": str(ex), "_body": body}


def _err(r) -> str | None:
    return (r.get("_body") or r.get("_error")) if isinstance(r, dict) and ("_error" in r) else None


class Executor:
    """One trade at a time: GTX entry at touch -> hold -> pegged reduce-only maker exit."""

    TICK = 0.00001  # DOGEUSDC price tick

    def __init__(self, rest: Rest) -> None:
        self.rest = rest
        self.busy = False
        self.halted = ""
        self.day = ""
        self.day_pnl = 0.0
        self.day_trades = 0
        self.day_bal0 = 0.0
        self.usdc_bid = 0.0
        self.usdc_ask = 0.0
        self.usdc_ts = 0.0
        os.makedirs(f"{WORK}/trades", exist_ok=True)

    # -- market data (fed by ws_consumer) --
    def on_book_ticker(self, d: dict) -> None:
        self.usdc_bid = float(d["b"])
        self.usdc_ask = float(d["a"])
        self.usdc_ts = time.time()

    def touch_ok(self) -> bool:
        return self.usdc_bid > 0 and self.usdc_ask > self.usdc_bid and time.time() - self.usdc_ts < 5

    # -- day accounting --
    def _roll_day(self) -> None:
        d = datetime.now(timezone.utc).strftime("%Y%m%d")
        if d != self.day:
            self.day = d
            self.day_pnl = 0.0
            self.day_trades = 0
            self.day_bal0 = 0.0
            if self.halted.startswith("day"):
                self.halted = ""

    def _tlog(self, rec: dict) -> None:
        rec["ts"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with open(f"{WORK}/trades/{self.day}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info("TRADE %s", json.dumps(rec))

    # -- REST helpers (blocking; used inside the trade thread) --
    def _avail_usdc(self) -> float:
        r = self.rest.call("GET", "/fapi/v2/balance")
        if isinstance(r, list):
            for b in r:
                if b["asset"] == "USDC":
                    return float(b["availableBalance"])
        return 0.0

    def _order(self, **p) -> dict:
        return self.rest.call("POST", "/fapi/v1/order", p)

    def _get_order(self, oid: int) -> dict:
        return self.rest.call("GET", "/fapi/v1/order", {"symbol": EXEC_SYM, "orderId": oid})

    def _cancel(self, oid: int) -> dict:
        return self.rest.call("DELETE", "/fapi/v1/order", {"symbol": EXEC_SYM, "orderId": oid})

    def _position_amt(self) -> float:
        r = self.rest.call("GET", "/fapi/v2/positionRisk", {"symbol": EXEC_SYM})
        if isinstance(r, list) and r:
            return float(r[0].get("positionAmt", 0))
        return 0.0

    def _place_gtx(self, side: str, qty: float, price: float, reduce_only: bool) -> dict:
        p = {"symbol": EXEC_SYM, "side": side, "type": "LIMIT", "timeInForce": "GTX",
             "quantity": f"{qty:.{QTY_DEC}f}", "price": f"{price:.{PX_DEC}f}"}
        if reduce_only:
            p["reduceOnly"] = "true"
        return self._order(**p)

    def _market_close(self, side: str, qty: float) -> dict:
        return self._order(symbol=EXEC_SYM, side=side, type="MARKET",
                           quantity=f"{qty:.{QTY_DEC}f}", reduceOnly="true")

    def startup_recover(self) -> None:
        """Close any orphan position / cancel stray orders left by a crash."""
        self.rest.call("DELETE", "/fapi/v1/allOpenOrders", {"symbol": EXEC_SYM})
        amt = self._position_amt()
        if amt != 0:
            side = "SELL" if amt > 0 else "BUY"
            r = self._market_close(side, abs(amt))
            self._tlog({"ev": "recover_close", "amt": amt, "resp_err": _err(r)})
        r = self.rest.call("POST", "/fapi/v1/leverage", {"symbol": EXEC_SYM, "leverage": LEVERAGE})
        log.info("executor ready: leverage resp=%s", r if _err(r) else r.get("leverage"))

    # -- the trade lifecycle (runs in a worker thread) --
    def run_trade(self, decision_ts: float, side_long: bool, score: float) -> None:
        try:
            self._trade(decision_ts, side_long, score)
        except Exception:
            log.exception("trade lifecycle crashed — emergency flatten")
            try:
                self.rest.call("DELETE", "/fapi/v1/allOpenOrders", {"symbol": EXEC_SYM})
                amt = self._position_amt()
                if amt != 0:
                    self._market_close("SELL" if amt > 0 else "BUY", abs(amt))
                    self._tlog({"ev": "crash_flatten", "amt": amt})
            except Exception:
                log.exception("emergency flatten failed")
        finally:
            self.busy = False

    def _trade(self, decision_ts: float, side_long: bool, score: float) -> None:
        entry_side = "BUY" if side_long else "SELL"
        exit_side = "SELL" if side_long else "BUY"
        avail = self._avail_usdc()
        if self.day_bal0 <= 0:
            self.day_bal0 = max(avail, 1e-9)
        px = self.usdc_bid if side_long else self.usdc_ask
        raw = avail * SIZE_FRAC / px * (1 - 1e-3)
        step = 10 ** QTY_DEC
        qty = float(np.floor(raw * step)) / step
        if qty * px < 5.0:
            self._tlog({"ev": "skip_small", "avail": avail, "qty": qty})
            return
        # ---- entry: GTX at touch, window ENTRY_WIN_S from decision ----
        r = self._place_gtx(entry_side, qty, px, reduce_only=False)
        if _err(r) or r.get("status") == "EXPIRED":
            px = self.usdc_bid if side_long else self.usdc_ask   # one retry at fresh touch
            r = self._place_gtx(entry_side, qty, px, reduce_only=False)
            if _err(r) or r.get("status") == "EXPIRED":
                self._tlog({"ev": "entry_reject", "err": _err(r) or r.get("status"), "px": px})
                return
        oid = int(r["orderId"])
        self._tlog({"ev": "entry_placed", "oid": oid, "side": entry_side, "qty": qty,
                    "px": px, "score": score})
        filled = 0.0
        entry_px = px
        status = "NEW"
        deadline = decision_ts + ENTRY_WIN_S
        while time.time() < deadline and status not in ("FILLED", "CANCELED", "EXPIRED"):
            time.sleep(0.7)
            o = self._get_order(oid)
            if _err(o):
                continue
            status = o.get("status", "NEW")
            filled = float(o.get("executedQty", 0) or 0)
            if status == "FILLED":
                entry_px = float(o.get("avgPrice") or px)
        if status != "FILLED":
            self._cancel(oid)
            o = self._get_order(oid)
            if not _err(o):
                filled = float(o.get("executedQty", 0) or 0)
                if filled > 0:
                    entry_px = float(o.get("avgPrice") or px)
        if filled == 0:
            self._tlog({"ev": "entry_miss", "oid": oid})
            return
        fill_ts = time.time()   # sim holds `to` ticks FROM FILL (grid_sim: es = k + c.to)
        self._tlog({"ev": "entry_fill", "oid": oid, "filled": filled, "avg_px": entry_px})
        # ---- hold to fill + HOLD_S ----
        time.sleep(max(0.0, fill_ts + HOLD_S - time.time()))
        # ---- pegged maker exit (reduce-only GTX at touch, re-quote on adverse move) ----
        t_exit0 = time.time()
        exit_oid = 0
        last_oid = 0
        exit_px = 0.0
        backstop = ""
        while True:
            now = time.time()
            mid = (self.usdc_bid + self.usdc_ask) / 2
            unreal_bp = ((mid - entry_px) / entry_px if side_long else (entry_px - mid) / entry_px) * 1e4
            if now - t_exit0 > EXIT_MAX_S:
                backstop = "time"
            elif unreal_bp < -HARD_LOSS_BP:
                backstop = "loss"
            if backstop:
                if exit_oid:
                    self._cancel(exit_oid)
                amt = self._position_amt()
                if amt != 0:
                    self._market_close(exit_side, abs(amt))
                break
            want = self.usdc_ask if side_long else self.usdc_bid   # sell at ask / buy at bid
            if exit_oid:
                o = self._get_order(exit_oid)
                st = o.get("status", "")
                if st == "FILLED":
                    exit_px = float(o["avgPrice"])
                    break
                adverse = (want < exit_px - 1e-9) if side_long else (want > exit_px + 1e-9)
                if adverse and st in ("NEW", "PARTIALLY_FILLED"):
                    self._cancel(exit_oid)
                    last_oid = exit_oid
                    exit_oid = 0
            if not exit_oid:
                amt = abs(self._position_amt())
                if amt == 0:
                    # position closed between cancel and check (late fill of the cancelled order)
                    o = self._get_order(last_oid) if last_oid else {}
                    exit_px = float(o.get("avgPrice") or want) if not _err(o) else want
                    break
                r = self._place_gtx(exit_side, amt, want, reduce_only=True)
                if not _err(r) and r.get("status") != "EXPIRED":
                    exit_oid = int(r["orderId"])
                    exit_px = want
            time.sleep(0.7)
        if backstop:
            time.sleep(1.0)
            inc = self.rest.call("GET", "/fapi/v1/userTrades",
                                 {"symbol": EXEC_SYM, "startTime": int(t_exit0 * 1000)})
            exit_px = (float(inc[-1]["price"]) if isinstance(inc, list) and inc else
                       (self.usdc_bid if side_long else self.usdc_ask))
        pnl_bp = ((exit_px - entry_px) / entry_px if side_long else (entry_px - exit_px) / entry_px) * 1e4
        pnl_usd = pnl_bp / 1e4 * entry_px * filled
        if backstop:
            pnl_usd -= 4e-4 * exit_px * filled   # taker fee on the backstop leg
        self.day_pnl += pnl_usd
        self.day_trades += 1
        self._tlog({"ev": "closed", "side": "long" if side_long else "short", "qty": filled,
                    "entry_px": entry_px, "exit_px": exit_px, "pnl_bp": round(pnl_bp, 2),
                    "pnl_usd": round(pnl_usd, 5), "backstop": backstop,
                    "exit_chase_s": round(time.time() - t_exit0, 1),
                    "day_pnl": round(self.day_pnl, 5), "day_trades": self.day_trades})
        if self.day_pnl < -DAY_LOSS_HALT * self.day_bal0:
            self.halted = "day_loss"
            self._tlog({"ev": "halt", "why": "day_loss", "day_pnl": self.day_pnl})
        if self.day_trades >= DAY_TRADES_HALT:
            self.halted = "day_trades"
            self._tlog({"ev": "halt", "why": "day_trades", "n": self.day_trades})

    # -- entry point from the decision loop --
    def maybe_trade(self, side_long: bool, score: float) -> bool:
        self._roll_day()
        if self.busy or self.halted or not self.touch_ok():
            return False
        self.busy = True
        decision_ts = time.time()
        threading.Thread(target=self.run_trade, args=(decision_ts, side_long, score),
                         daemon=True).start()
        return True


# ---------------------------------------------------------------- decision log
class DecisionLog:
    def __init__(self, mkt) -> None:
        self.mkt = mkt
        os.makedirs(f"{WORK}/decisions", exist_ok=True)
        self.last_upload = time.time()

    def append(self, rec: dict) -> None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        with open(f"{WORK}/decisions/{day}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if time.time() - self.last_upload > 3600:
            self.upload()

    def upload(self) -> None:
        try:
            for fn in sorted(os.listdir(f"{WORK}/decisions")):
                self.mkt.blob(f"{SHADOW_GCS}/decisions/{fn}").upload_from_filename(
                    f"{WORK}/decisions/{fn}")
            self.last_upload = time.time()
        except Exception as ex:
            log.warning("decision upload failed: %s", ex)


# ---------------------------------------------------------------- main loop
async def decide_loop(bundle: Bundle, thr: Threshold, dlog: DecisionLog) -> None:
    n = 0
    while True:
        t_next = time.time() + DECIDE_S
        try:
            BUF.prune()
            if BUF.warm():
                t0 = time.time()
                frozen = (list(BUF.book), list(BUF.trades), list(BUF.liq),
                          list(BUF.eth), list(BUF.funding), list(BUF.oi), list(BUF.btc))

                def _work() -> tuple | None:
                    book, trades, liq, eth, funding, oi, btc = frozen
                    nb, nt, nl, ne, nfd, noi, last_us = write_window_parquet(
                        book, trades, liq, eth, funding, oi)
                    x = compute_features(nb, nt, nl, ne, nfd, noi)
                    return None if x is None else (nb, nt, nl, ne, nfd, noi, last_us, x, btc)

                res = await asyncio.to_thread(_work)
                if res is not None:
                    nb, nt, nl, ne, nfd, noi, last_us, x, btc = res
                    ts_ns = last_us * 1000
                    x71 = feat71(ts_ns, x, btc)
                    pa_, pb, sc = bundle.score(x71)
                    taus = thr.observe(sc)
                    takes = {f"take{int(t)}": bool(sc >= taus[t]) for t in BUDGETS}
                    executed = False
                    if EXEC is not None and takes.get(f"take{int(TRADE_BUDGET)}", False):
                        executed = EXEC.maybe_trade(pb >= 0.5, sc)
                    bb = frozen[0][-1]
                    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                           "book_ts_us": last_us, "bid": bb[1][0][0], "ask": bb[2][0][0],
                           "pA": round(pa_, 6), "pBg": round(pb, 6), "score": round(sc, 6),
                           "side": "long" if pb >= 0.5 else "short",
                           "tau": {str(int(t)): round(taus[t], 6) for t in BUDGETS},
                           **takes, "executed": executed,
                           "lat_ms": round((time.time() - t0) * 1000, 1),
                           "nb": nb, "nt": nt, "nl": nl, "ne": ne, "nfd": nfd, "noi": noi}
                    dlog.append(rec)
                    n += 1
                    if any(takes.values()) or n % 100 == 0:
                        log.info("#%d score=%.4f side=%s takes=%s lat=%.0fms",
                                 n, sc, rec["side"],
                                 [int(t) for t in BUDGETS if takes[f"take{int(t)}"]], rec["lat_ms"])
                    if n % 50 == 0:
                        thr.save()
            else:
                log.info("warming up: book=%d span=%.0fs", len(BUF.book),
                         (BUF.book[-1][0] - BUF.book[0][0]) / 1e6 if BUF.book else 0)
        except Exception as ex:
            log.exception("decision failed: %s", ex)
        await asyncio.sleep(max(0.0, t_next - time.time()))


async def oi_poller() -> None:
    """Unsigned open-interest REST poll -> BUF.oi (feature parity with recorder derivatives_poll)."""
    import urllib.request
    url = f"{REST_BASE}/fapi/v1/openInterest?symbol={SYM.upper()}"
    while True:
        try:
            with urllib.request.urlopen(url, timeout=8) as r:
                d = json.loads(r.read())
            BUF.oi.append((int(d["time"]) * 1000, float(d["openInterest"])))
        except Exception as ex:
            log.warning("OI poll failed: %s", ex)
        await asyncio.sleep(OI_POLL_S)


async def main() -> None:
    global EXEC
    os.makedirs(WORK, exist_ok=True)
    bundle = Bundle()
    thr = Threshold(bundle)
    dlog = DecisionLog(bundle.mkt)
    # public: signal depth + BTC-lead bookTicker (+ exec bookTicker in live)
    pub_streams = [f"{SYM}@depth20@100ms", f"{BTC_SYM}@bookTicker"]
    # market: signal trades/liq/funding + ETH-lead trades
    mkt_streams = [f"{SYM}@aggTrade", f"{SYM}@forceOrder", f"{SYM}@markPrice@1s",
                   f"{ETH_SYM}@aggTrade"]
    if MODE == "live":
        EXEC = Executor(Rest())
        await asyncio.to_thread(EXEC.startup_recover)
        pub_streams.append(f"{EXEC_SYM.lower()}@bookTicker")
        log.info("MODE=live: exec %s budget=t%d size_frac=%.2f entry=%.0fs hold=%.0fs",
                 EXEC_SYM, int(TRADE_BUDGET), SIZE_FRAC, ENTRY_WIN_S, HOLD_S)
    else:
        log.info("MODE=shadow: no orders")
    tasks = [
        asyncio.create_task(ws_consumer("public", pub_streams)),
        asyncio.create_task(ws_consumer("market", mkt_streams)),
        asyncio.create_task(oi_poller()),
        asyncio.create_task(decide_loop(bundle, thr, dlog)),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        thr.save()
        dlog.upload()


if __name__ == "__main__":
    asyncio.run(main())
