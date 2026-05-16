-- ============================================================================
-- scalper-bot research ledger — canonical schema (single source of truth)
-- ============================================================================
-- This DDL is the CONTRACT. research.db is a *derived* index built from the
-- append-only JSONL ledgers (experiments.jsonl, hypotheses.jsonl) by
-- research/ledger.py build-db. Never hand-edit research.db — it is rebuilt.
--
-- Design rules (each column maps to a concrete, expensive lesson):
--   * fee_regime is NOT NULL + enum. TAKER<->MAKER confusion produced THREE
--     false-positive findings (DOGE +19.6%->-1.5%, ETH +0.036%->-0.047%,
--     phase56 +1.30%->-1.4%). The single biggest source of wasted compute.
--   * cache_id / data provenance is NOT NULL. Caches die (Contabo lost,
--     v3 1.85M OOM-deleted). A number without its exact data origin is noise.
--   * split_method is NOT NULL + enum. CPCV vs walk-forward vs honest
--     val->test are not comparable; mixing them silently is how the chaos
--     started.
--   * A result is IMMUTABLE. Corrections append a NEW experiment row with
--     supersedes -> old id and status in (refuted, artifact). The lineage
--     IS the audit trail (RESEARCH_LOG section 4 "Resolved confusions").
-- ============================================================================

PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------------------------
-- hypotheses — structured form of RESEARCH_LOG section 7.
-- Append-only event log in hypotheses.jsonl; build-db keeps max(rev) per id.
-- ----------------------------------------------------------------------------
CREATE TABLE hypotheses (
    hypothesis_id     TEXT NOT NULL,          -- "H2", "H5", ...
    rev               INTEGER NOT NULL,       -- monotonic per hypothesis_id
    ts                TEXT NOT NULL,          -- ISO8601 UTC of this revision
    statement         TEXT NOT NULL,          -- the hypothesis itself
    expected_lift     TEXT,                   -- e.g. "+0.02-0.05% EV/tr"
    compute_cost_usd  TEXT,                   -- "$0 eval-only", "$50-100"
    compute_time      TEXT,                   -- "~1h GCP 96vCPU"
    prerequisites     TEXT,                   -- blocking deps (free text)
    priority_rank     INTEGER,                -- cheap-first ordering
    status            TEXT NOT NULL
        CHECK (status IN ('active','testing','confirmed','refuted',
                          'blocked','superseded')),
    result_experiment_id TEXT,                -- FK -> experiments(experiment_id)
    note              TEXT,
    PRIMARY KEY (hypothesis_id, rev)
);

-- ----------------------------------------------------------------------------
-- experiments — one row per (subject x data x methodology) evaluation.
-- A grid sweep is ONE experiment: experiment-level metadata once, the chosen
-- operating point in the result columns, full grid kept as an artifact
-- (artifact_path + artifact_sha256), NOT inlined.
-- ----------------------------------------------------------------------------
CREATE TABLE experiments (
    -- identity / lineage -----------------------------------------------------
    experiment_id     TEXT PRIMARY KEY,       -- "2026-05-16T2210Z_h2_ptts_btc"
    ts                TEXT NOT NULL,          -- ISO8601 UTC
    git_commit        TEXT NOT NULL,          -- repo SHA the run used
    author            TEXT NOT NULL,          -- "claude" | "<human>"
    hypothesis_id     TEXT,                   -- FK -> hypotheses
    status            TEXT NOT NULL
        CHECK (status IN ('confirmed','suspect','refuted',
                          'artifact','exploratory')),
    supersedes        TEXT,                   -- experiment_id this corrects
    note              TEXT,

    -- SUBJECT — what was tested ---------------------------------------------
    setup             TEXT NOT NULL,          -- short human description
    model_family      TEXT NOT NULL,          -- xgb|tcn|mamba|transformer|
                                              -- ensemble|cascade|grid_only|...
    params_json       TEXT NOT NULL,          -- {tp,sl,timeout,kelly,partial,
                                              --  trailing, inner PT/TS,
                                              --  meta_thr,min_prob,spread_bps,
                                              --  fill_prob, ...}

    -- DATA — what it was tested on ------------------------------------------
    data_source       TEXT NOT NULL
        CHECK (data_source IN ('v3_btc','cryptolake','recorder',
                               'volaware','other')),
    cache_id          TEXT NOT NULL,          -- exact cache filename / hash
    symbols_json      TEXT NOT NULL,          -- ["BTCUSDT", ...]
    date_range_start  TEXT,                   -- ISO8601 or NULL if n/a
    date_range_end    TEXT,
    n_samples         INTEGER NOT NULL,
    label_horizon_ticks INTEGER,              -- SIM_HORIZON used

    -- METHODOLOGY — how it was measured (the anti-chaos core) ---------------
    fee_regime        TEXT NOT NULL
        CHECK (fee_regime IN ('TAKER','MAKER_FIRST','MAKER')),
    commission_win_pct  REAL NOT NULL,
    commission_loss_pct REAL NOT NULL,
    split_method      TEXT NOT NULL
        CHECK (split_method IN ('CPCV_6_2','walkforward_7525',
                                'honest_val_test','other')),
    embargo           TEXT,
    purge             TEXT,
    gap_ticks         INTEGER,
    label_def         TEXT NOT NULL,          -- the exact y= rule used

    -- RESULT — owner 7 metrics (Самые важные метрики) ----------------------
    -- shares are fractions in [0,1]; 1..5 mutually exclusive, sum -> 1.0
    pct_full_tp       REAL,                   -- tp_hit
    pct_full_sl       REAL,                   -- sl_hit+fast_fill_adverse+_sl
    pct_timeout       REAL,                   -- timeout_limit+_market+no_fwd
    pct_trailing      REAL,                   -- trailing_sl_1/2 (+partial+)
    pct_partial_only  REAL,                   -- partial_plus_tp
    pnl_gross_pct     REAL,                   -- before commissions+spread, %
    pnl_gross_usd     REAL,                   -- before commissions+spread, $
    pnl_net_pct       REAL,                   -- after commissions+spread, %
    pnl_net_usd       REAL,                   -- after commissions+spread, $

    -- RESULT — frontier metrics (RESEARCH_LOG section 3) -------------------
    ev_per_trade_pct  REAL,                   -- PRIMARY frontier metric
    trades_per_day    REAL,                   -- operating-point anchor
    net_return_pct    REAL,                   -- kelly-scaled net over window
    kelly_frac        REAL,                   -- never compare nets across k
    win_rate_pct      REAL,                   -- direction-aware realised
    base_rate_pct     REAL,                   -- P(pl_long>0), per-symbol
    n_trades          INTEGER NOT NULL,
    sharpe            REAL,
    max_dd_pct        REAL,
    exit_hist_json    TEXT,                   -- {reason: count} all 12

    -- REPRODUCIBILITY -------------------------------------------------------
    artifact_path     TEXT,                   -- gs:// or repo-relative path
    artifact_sha256   TEXT,                   -- hash of the full result blob
    repro_cmd         TEXT NOT NULL           -- exact command to reproduce
);
-- NB: hypothesis_id is a SOFT reference (hypotheses is an append-only event
-- log with a composite PK, so a SQL FK cannot target hypothesis_id alone).
-- ledger.py enforces that every experiment.hypothesis_id exists.

CREATE INDEX ix_exp_hypothesis ON experiments (hypothesis_id);
CREATE INDEX ix_exp_status     ON experiments (status);
CREATE INDEX ix_exp_fee        ON experiments (fee_regime);
CREATE INDEX ix_exp_source     ON experiments (data_source);

-- ----------------------------------------------------------------------------
-- VIEWS
-- ----------------------------------------------------------------------------

-- Current hypotheses: latest revision per id.
CREATE VIEW v_current_hypotheses AS
SELECT h.*
FROM hypotheses h
JOIN (SELECT hypothesis_id, MAX(rev) AS rev
      FROM hypotheses GROUP BY hypothesis_id) m
  ON h.hypothesis_id = m.hypothesis_id AND h.rev = m.rev;

-- Live experiments: those not corrected by a later supersedes-> row.
CREATE VIEW v_live_experiments AS
SELECT e.*
FROM experiments e
WHERE e.experiment_id NOT IN
      (SELECT supersedes FROM experiments WHERE supersedes IS NOT NULL);

-- Frontier: the section-3 table, fee_regime ALWAYS explicit. Only
-- trustworthy rows (not refuted/artifact, real coverage).
CREATE VIEW v_frontier AS
SELECT
    substr(ts,1,10)       AS date,
    setup,
    symbols_json          AS symbols,
    fee_regime,
    ev_per_trade_pct,
    trades_per_day,
    net_return_pct,
    kelly_frac,
    n_trades,
    status
FROM v_live_experiments
WHERE status IN ('confirmed','suspect','exploratory')
  AND n_trades >= 30
ORDER BY ev_per_trade_pct DESC;

-- Artifact chains: the supersede lineage (e.g. DOGE +19.6% -> -1.5%).
-- Every correction in RESEARCH_LOG section 4 should appear here.
CREATE VIEW v_artifact_chains AS
SELECT
    bad.experiment_id     AS refuted_id,
    bad.status            AS refuted_status,
    bad.ev_per_trade_pct  AS claimed_ev,
    bad.fee_regime        AS claimed_fee,
    fix.experiment_id     AS correction_id,
    fix.ev_per_trade_pct  AS corrected_ev,
    fix.fee_regime        AS corrected_fee,
    fix.note              AS reason
FROM experiments fix
JOIN experiments bad ON fix.supersedes = bad.experiment_id;

-- Methodology audit: structurally CONFIRMED-positive must not be TAKER.
-- A positive EV under TAKER labels was wrong all 3 prior times — the
-- ledger refuses to *store* it as confirmed (ledger.py), this view is the
-- defence-in-depth check that nothing slipped through.
CREATE VIEW v_methodology_audit AS
SELECT experiment_id, status, fee_regime, ev_per_trade_pct, note
FROM experiments
WHERE status = 'confirmed'
  AND ev_per_trade_pct > 0
  AND fee_regime = 'TAKER';
