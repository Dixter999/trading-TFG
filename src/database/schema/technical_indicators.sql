-- Schema Reference: technical_indicators table
-- Database: ai_model
-- Purpose: Real-time technical indicators for ML model training
--
-- ⚠️  LOOK-AHEAD BIAS WARNING ⚠️
-- This table must ONLY contain real-time calculated indicators.
-- Never backfill with historical data.
--
-- See migration: 004_create_technical_indicators.sql

CREATE TABLE technical_indicators (
    -- Primary Key: Composite (symbol, timeframe, timestamp)
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,

    -- OHLCV Reference Data
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,

    -- Moving Averages (6 columns)
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    ema_12 DOUBLE PRECISION,
    ema_26 DOUBLE PRECISION,
    ema_50 DOUBLE PRECISION,

    -- Momentum Indicators (1 column)
    rsi_14 DOUBLE PRECISION,

    -- Volatility Indicators (4 columns)
    atr_14 DOUBLE PRECISION,
    bb_upper_20 DOUBLE PRECISION,
    bb_middle_20 DOUBLE PRECISION,
    bb_lower_20 DOUBLE PRECISION,

    -- MACD (3 columns)
    macd_line DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_histogram DOUBLE PRECISION,

    -- Stochastic Oscillator (2 columns)
    stoch_k DOUBLE PRECISION,
    stoch_d DOUBLE PRECISION,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (symbol, timeframe, timestamp)
);

-- Indexes
CREATE INDEX idx_technical_indicators_timestamp ON technical_indicators(timestamp DESC);
CREATE INDEX idx_technical_indicators_symbol_timeframe ON technical_indicators(symbol, timeframe);
CREATE INDEX idx_technical_indicators_lookup ON technical_indicators(symbol, timeframe, timestamp DESC);

-- Total Indicator Columns: 17
-- - Moving Averages: 6 (sma_20, sma_50, sma_200, ema_12, ema_26, ema_50)
-- - RSI: 1 (rsi_14)
-- - ATR: 1 (atr_14)
-- - Bollinger Bands: 3 (bb_upper_20, bb_middle_20, bb_lower_20)
-- - MACD: 3 (macd_line, macd_signal, macd_histogram)
-- - Stochastic: 2 (stoch_k, stoch_d)
