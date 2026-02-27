-- init_db.sql
-- Initialize the ai_model database schema for Trading-TFG.
-- Designed to run as a Docker entrypoint init script
-- (mount into /docker-entrypoint-initdb.d/).

-- ============================================================
-- 1. rates - OHLCV price data per symbol and timeframe
-- ============================================================
CREATE TABLE IF NOT EXISTS rates (
    rate_time   BIGINT       NOT NULL,  -- unix epoch seconds
    symbol      VARCHAR(10)  NOT NULL,  -- e.g. EURUSD
    timeframe   VARCHAR(5)   NOT NULL,  -- e.g. H1, D1, M30
    open        NUMERIC(15,8) NOT NULL,
    high        NUMERIC(15,8) NOT NULL,
    low         NUMERIC(15,8) NOT NULL,
    close       NUMERIC(15,8) NOT NULL,
    volume      NUMERIC(15,2) NOT NULL DEFAULT 0,
    readable_date TEXT,

    PRIMARY KEY (symbol, timeframe, rate_time)
);

CREATE INDEX IF NOT EXISTS idx_rates_time
    ON rates (rate_time);
CREATE INDEX IF NOT EXISTS idx_rates_symbol_tf
    ON rates (symbol, timeframe);


-- ============================================================
-- 2. technical_indicators - computed indicators per candle
-- ============================================================
CREATE TABLE IF NOT EXISTS technical_indicators (
    id           SERIAL PRIMARY KEY,
    symbol       VARCHAR(10)  NOT NULL,
    timeframe    VARCHAR(5)   NOT NULL,
    timestamp    TIMESTAMPTZ  NOT NULL,
    open         NUMERIC(15,8),
    high         NUMERIC(15,8),
    low          NUMERIC(15,8),
    close        NUMERIC(15,8),
    volume       NUMERIC(15,2),
    sma_20       NUMERIC(18,10),
    sma_50       NUMERIC(18,10),
    sma_200      NUMERIC(18,10),
    ema_12       NUMERIC(18,10),
    ema_26       NUMERIC(18,10),
    ema_50       NUMERIC(18,10),
    rsi_14       NUMERIC(18,10),
    atr_14       NUMERIC(18,10),
    bb_upper_20  NUMERIC(18,10),
    bb_middle_20 NUMERIC(18,10),
    bb_lower_20  NUMERIC(18,10),
    macd_line    NUMERIC(18,10),
    macd_signal  NUMERIC(18,10),
    macd_histogram NUMERIC(18,10),
    created_at   TIMESTAMPTZ,
    updated_at   TIMESTAMPTZ,
    stoch_k      NUMERIC(18,10),
    stoch_d      NUMERIC(18,10),
    ob_bullish_high  NUMERIC(15,8),
    ob_bullish_low   NUMERIC(15,8),
    ob_bearish_high  NUMERIC(15,8),
    ob_bearish_low   NUMERIC(15,8)
);

CREATE INDEX IF NOT EXISTS idx_ti_symbol_tf_ts
    ON technical_indicators (symbol, timeframe, timestamp);
CREATE INDEX IF NOT EXISTS idx_ti_timestamp
    ON technical_indicators (timestamp);


-- ============================================================
-- 3. paper_trades - simulated trade records
-- ============================================================
CREATE TABLE IF NOT EXISTS paper_trades (
    id                SERIAL PRIMARY KEY,
    symbol            VARCHAR(10)  NOT NULL,
    direction         VARCHAR(5)   NOT NULL,  -- LONG / SHORT
    entry_time        TIMESTAMPTZ,
    entry_price       NUMERIC(15,8),
    exit_time         TIMESTAMPTZ,
    exit_price        NUMERIC(15,8),
    sl_price          NUMERIC(15,8),
    tp_price          NUMERIC(15,8),
    size              NUMERIC(12,4),
    pnl_pips          NUMERIC(12,4),
    exit_reason       VARCHAR(50),
    entry_signal_data TEXT,               -- JSON blob
    created_at        TIMESTAMPTZ,
    entry_model       VARCHAR(100),
    signal_timeframe  VARCHAR(5)
);

CREATE INDEX IF NOT EXISTS idx_pt_symbol_dir
    ON paper_trades (symbol, direction);
CREATE INDEX IF NOT EXISTS idx_pt_exit_time
    ON paper_trades (exit_time);


-- ============================================================
-- 4. signal_discoveries - pipeline run results
-- ============================================================
CREATE TABLE IF NOT EXISTS signal_discoveries (
    id                       SERIAL PRIMARY KEY,
    discovery_date           DATE,
    symbol                   VARCHAR(10),
    direction                VARCHAR(5),
    lookback_years           NUMERIC(6,2),
    total_candles            INTEGER,
    train_candles            INTEGER,
    val_candles              INTEGER,
    test_candles             INTEGER,
    data_start_date          TIMESTAMPTZ,
    data_end_date            TIMESTAMPTZ,
    phase1_signals_tested    INTEGER,
    phase1_signals_passed    INTEGER,
    phase1_duration_seconds  NUMERIC(12,2),
    phase2_signals_validated INTEGER,
    phase2_duration_seconds  NUMERIC(12,2),
    top_signal_name          VARCHAR(100),
    top_signal_timeframe     VARCHAR(5),
    top_signal_wr            NUMERIC(8,4),
    top_signal_oos_wr        NUMERIC(8,4),
    top_signal_trades        INTEGER,
    top_signal_pvalue        NUMERIC(12,8),
    phase1_results           TEXT,         -- JSON blob
    phase2_results           TEXT,         -- JSON blob
    pipeline_version         VARCHAR(20),
    hostname                 VARCHAR(100),
    created_at               TIMESTAMPTZ,
    phase3_trials_completed  INTEGER,
    phase3_best_trial_number INTEGER,
    phase3_best_pf           NUMERIC(10,4),
    phase3_best_hyperparams  TEXT,         -- JSON blob
    phase3_duration_seconds  NUMERIC(12,2),
    phase4_folds_completed   INTEGER,
    phase4_avg_pf            NUMERIC(10,4),
    phase4_std_pf            NUMERIC(10,4),
    phase4_avg_wr            NUMERIC(8,4),
    phase4_duration_seconds  NUMERIC(12,2),
    phase5_test_pf           NUMERIC(10,4),
    phase5_test_wr           NUMERIC(8,4),
    phase5_test_trades       INTEGER,
    phase5_approved_for_production BOOLEAN,
    phase5_approval_date     TIMESTAMPTZ,
    training_locked_at       TIMESTAMPTZ,
    training_locked_by       VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_sd_symbol_dir
    ON signal_discoveries (symbol, direction);
CREATE INDEX IF NOT EXISTS idx_sd_discovery_date
    ON signal_discoveries (discovery_date);
