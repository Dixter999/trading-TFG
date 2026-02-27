/**
 * Market Data Query Functions
 *
 * Functions for querying OHLCV market data with technical indicators
 * from the Markets and AI Model databases.
 */

import { getMarketsPool, getAiModelPool } from '../connection';
import { executeQuery } from '../utils/query';
import { ValidationError } from '../errors';

/**
 * Technical indicators structure (all optional as LEFT JOIN may return nulls)
 */
export interface TechnicalIndicators {
  sma_20?: number;
  sma_50?: number;
  sma_200?: number;
  ema_12?: number;
  ema_26?: number;
  ema_50?: number;
  rsi_14?: number;
  atr_14?: number;
  bb_upper_20?: number;
  bb_middle_20?: number;
  bb_lower_20?: number;
  macd_line?: number;
  macd_signal?: number;
  macd_histogram?: number;
  stoch_k?: number;
  stoch_d?: number;
  ob_bullish_high?: number;
  ob_bullish_low?: number;
  ob_bearish_high?: number;
  ob_bearish_low?: number;
}

/**
 * Market data with technical indicators
 */
export interface MarketDataWithIndicators {
  rateTime: number; // Unix timestamp (seconds since epoch)
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  indicators: TechnicalIndicators;
}

/**
 * Supported timeframes
 */
export type Timeframe =
  | 'M1'
  | 'M5'
  | 'M15'
  | 'M30'
  | 'H1'
  | 'H2'
  | 'H3'
  | 'H4'
  | 'H6'
  | 'H8'
  | 'H12'
  | 'D1';

/**
 * Supported symbols (FOREX only)
 * Stale symbol cleanup (2026-02-24): GOLD, SILVER, BTCUSD removed
 */
export type Symbol = 'EURUSD' | 'GBPUSD' | 'USDJPY' | 'EURJPY' | 'USDCAD' | 'USDCHF' | 'EURCAD' | 'EURGBP';

/**
 * Constants for validation
 */
const SUPPORTED_SYMBOLS: Symbol[] = [
  'EURUSD',
  'GBPUSD',
  'USDJPY',
  'EURJPY',
  'USDCAD',
  'USDCHF',
  'EURCAD',
  'EURGBP',
];

/**
 * Map API symbols to database table names
 */
function getDbSymbol(symbol: string): string {
  return symbol;
}
const VALID_TIMEFRAMES: Timeframe[] = [
  'M1',
  'M5',
  'M15',
  'M30',
  'H1',
  'H2',
  'H3',
  'H4',
  'H6',
  'H8',
  'H12',
  'D1',
];
const DEFAULT_LIMIT = 1000;
const MAX_LIMIT = 10000; // Increased to support 1 year of H1 data (~8760 candles)

/**
 * Database row structure (snake_case from PostgreSQL)
 */
interface MarketDataRow {
  rate_time: number; // Unix timestamp (BIGINT)
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Technical indicator row from ai_model database
 */
interface TechnicalIndicatorRow {
  timestamp: Date;
  sma_20?: number;
  sma_50?: number;
  sma_200?: number;
  ema_12?: number;
  ema_26?: number;
  ema_50?: number;
  rsi_14?: number;
  atr_14?: number;
  bb_upper_20?: number;
  bb_middle_20?: number;
  bb_lower_20?: number;
  macd_line?: number;
  macd_signal?: number;
  macd_histogram?: number;
  stoch_k?: number;
  stoch_d?: number;
  ob_bullish_high?: number;
  ob_bullish_low?: number;
  ob_bearish_high?: number;
  ob_bearish_low?: number;
}

/**
 * Validate date is valid and not in future
 * Allows a 2-minute buffer to support loading developing candles at current time boundary
 * @throws ValidationError if validation fails
 */
function validateDate(date: Date, fieldName: string): void {
  if (!(date instanceof Date) || isNaN(date.getTime())) {
    throw new ValidationError(`Invalid ${fieldName} date`, {
      field: fieldName,
      value: date,
    });
  }

  // Allow 3-hour buffer: broker stores timestamps in CET/CEST (UTC+1/+2),
  // so the most recent candle epoch can be up to 2h ahead of actual UTC.
  const now = new Date();
  const bufferMs = 3 * 60 * 60 * 1000; // 3 hours in milliseconds
  const maxAllowedTime = new Date(now.getTime() + bufferMs);

  if (date > maxAllowedTime) {
    throw new ValidationError(
      `${fieldName.charAt(0).toUpperCase() + fieldName.slice(1)} date cannot be in the future`,
      {
        field: fieldName,
        value: date,
      }
    );
  }
}

/**
 * Validate input parameters
 * @throws ValidationError if validation fails
 */
function validateInputs(
  symbol: string,
  timeframe: string,
  start: Date,
  end: Date,
  limit: number
): void {
  // Validate symbol
  if (!SUPPORTED_SYMBOLS.includes(symbol as Symbol)) {
    throw new ValidationError(`Invalid symbol. Must be one of: ${SUPPORTED_SYMBOLS.join(', ')}`, {
      field: 'symbol',
      value: symbol,
      expected: SUPPORTED_SYMBOLS.join(', '),
    });
  }

  // Validate timeframe
  if (!VALID_TIMEFRAMES.includes(timeframe as Timeframe)) {
    throw new ValidationError(`Invalid timeframe. Must be one of: ${VALID_TIMEFRAMES.join(', ')}`, {
      field: 'timeframe',
      value: timeframe,
      expected: VALID_TIMEFRAMES.join(', '),
    });
  }

  // Validate dates
  validateDate(start, 'start');
  validateDate(end, 'end');

  // Check end >= start
  if (end < start) {
    throw new ValidationError('End date must be greater than or equal to start date', {
      field: 'end',
      value: end,
      constraint: 'end >= start',
    });
  }

  // Validate limit
  if (limit <= 0) {
    throw new ValidationError('Limit must be greater than 0', {
      field: 'limit',
      value: limit,
      constraint: 'limit > 0',
    });
  }

  if (limit > MAX_LIMIT) {
    throw new ValidationError(`Limit cannot exceed ${MAX_LIMIT}`, {
      field: 'limit',
      value: limit,
      constraint: `limit <= ${MAX_LIMIT}`,
    });
  }
}

/**
 * Get earliest timestamp for a symbol/timeframe
 * Used when start parameter is not provided
 *
 * @param symbol - Trading symbol (currently only 'EURUSD' supported)
 * @param timeframe - Timeframe ('M1', 'M5', 'M15', 'M30', 'H1', 'H4', or 'D1')
 * @returns Earliest timestamp as Date
 * @throws QueryError on database query failure
 */
export async function getEarliestTimestamp(symbol: string, timeframe: Timeframe): Promise<Date> {
  const dbSymbol = getDbSymbol(symbol);
  const tableName = `${dbSymbol.toLowerCase()}_${timeframe.toLowerCase()}_rates`;
  const sql = `SELECT MIN(rate_time) as earliest FROM ${tableName}`;

  const marketsPool = getMarketsPool();
  const rows = await executeQuery<{ earliest: number | null }>(marketsPool, sql, []);

  if (rows.length === 0 || !rows[0] || rows[0].earliest === null) {
    // If no data exists, return a very old date
    return new Date('2000-01-01T00:00:00Z');
  }

  // Convert Unix timestamp to Date
  return new Date(rows[0].earliest * 1000);
}

/**
 * Get latest timestamp for a symbol/timeframe
 * Used when end parameter is not provided
 *
 * @param symbol - Trading symbol (currently only 'EURUSD' supported)
 * @param timeframe - Timeframe ('M1', 'M5', 'M15', 'M30', 'H1', 'H4', or 'D1')
 * @returns Latest timestamp as Date
 * @throws QueryError on database query failure
 */
export async function getLatestTimestamp(symbol: string, timeframe: Timeframe): Promise<Date> {
  const dbSymbol = getDbSymbol(symbol);
  const tableName = `${dbSymbol.toLowerCase()}_${timeframe.toLowerCase()}_rates`;
  const sql = `SELECT MAX(rate_time) as latest FROM ${tableName}`;

  const marketsPool = getMarketsPool();
  const rows = await executeQuery<{ latest: number | null }>(marketsPool, sql, []);

  if (rows.length === 0 || !rows[0] || rows[0].latest === null) {
    // If no data exists, return current date
    return new Date();
  }

  // Convert Unix timestamp to Date
  return new Date(rows[0].latest * 1000);
}

/**
 * Get market data with technical indicators
 *
 * Queries the Markets database for OHLCV data and joins with AI Model database
 * for technical indicators. Uses LEFT JOIN to include rows even if indicators
 * are not yet calculated.
 *
 * @param symbol - Trading symbol (currently only 'EURUSD' supported)
 * @param timeframe - Timeframe ('H1', 'H4', or 'D1')
 * @param start - Start date (inclusive)
 * @param end - End date (inclusive)
 * @param limit - Maximum number of rows to return (default: 1000, max: 5000)
 * @returns Array of market data with indicators, ordered by date DESC (newest first)
 * @throws ValidationError on invalid inputs
 * @throws QueryError on database query failure
 */
export async function getMarketDataWithIndicators(
  symbol: string,
  timeframe: Timeframe,
  start: Date,
  end: Date,
  limit: number = DEFAULT_LIMIT
): Promise<MarketDataWithIndicators[]> {
  // Validate inputs
  validateInputs(symbol, timeframe, start, end, limit);

  // Map API symbol to database table name
  const dbSymbol = getDbSymbol(symbol);

  // Build table name based on symbol and timeframe
  const tableName = `${dbSymbol.toLowerCase()}_${timeframe.toLowerCase()}_rates`;

  // SQL query for OHLCV data from markets database
  // Note: Tables are in public schema (default), not 'markets' schema
  // 'markets' is the database name, schema is 'public'
  // Cast volume to integer (pg driver returns bigint as string)
  // rate_time is Unix timestamp (BIGINT) - convert Date params to Unix timestamps
  const startUnix = Math.floor(start.getTime() / 1000);
  const endUnix = Math.floor(end.getTime() / 1000);

  const marketSql = `
    SELECT
      rate_time::bigint as rate_time,
      open,
      high,
      low,
      close,
      volume::int as volume
    FROM ${tableName}
    WHERE rate_time >= $1 AND rate_time <= $2
    ORDER BY rate_time DESC
    LIMIT $3
  `;

  // Build indicator table name - ALL symbols use per-symbol tables: 'technical_indicator_{symbol}'
  const indicatorTable = `technical_indicator_${dbSymbol.toLowerCase()}`;

  // SQL query for technical indicators from ai_model database
  // CRITICAL: Must filter by symbol to avoid mixing indicators from different symbols
  // NOTE: Order Block columns (ob_*) don't exist in per-symbol tables, removed from query

  const indicatorSql = `
    SELECT
      timestamp,
      sma_20,
      sma_50,
      sma_200,
      ema_12,
      ema_26,
      ema_50,
      rsi_14,
      atr_14,
      bb_upper_20,
      bb_middle_20,
      bb_lower_20,
      macd_line,
      macd_signal,
      macd_histogram,
      stoch_k,
      stoch_d
    FROM ${indicatorTable}
    WHERE timeframe = $1
      AND symbol = $2
      AND timestamp >= $3 AND timestamp <= $4
    ORDER BY timestamp DESC
    LIMIT $5
  `;

  // Execute both queries in parallel
  const marketsPool = getMarketsPool();
  const aiModelPool = getAiModelPool();

  // Fetch market data (required) - pass Unix timestamps
  const marketRows = await executeQuery<MarketDataRow>(marketsPool, marketSql, [
    startUnix,
    endUnix,
    limit,
  ]);

  // Try to fetch indicators but continue if they fail (optional)
  let indicatorRows: TechnicalIndicatorRow[] = [];
  try {
    indicatorRows = await executeQuery<TechnicalIndicatorRow>(aiModelPool, indicatorSql, [
      timeframe,
      symbol.toUpperCase(),
      start,
      end,
      limit,
    ]);
  } catch (error) {
    // Indicators not available - continue without them (graceful degradation)
  }

  // Create a map of indicators by Unix timestamp for fast lookup
  const indicatorMap = new Map<number, TechnicalIndicatorRow>();
  indicatorRows.forEach((row) => {
    const unixTime = Math.floor(new Date(row.timestamp).getTime() / 1000);
    indicatorMap.set(unixTime, row);
  });

  // Merge market data with indicators
  const mergedData = marketRows.map((row) => {
    // Convert rate_time to number for map lookup (pg returns bigint as string)
    const rateTimeNum =
      typeof row.rate_time === 'string' ? parseInt(row.rate_time, 10) : row.rate_time;
    const indicatorRow = indicatorMap.get(rateTimeNum);

    // Filter out timestamp field and convert to indicators object
    let indicatorData: TechnicalIndicators = {};
    if (indicatorRow) {
      const { timestamp: _, ...rest } = indicatorRow;
      indicatorData = rest;
    }

    return {
      rate_time: row.rate_time,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close,
      volume: row.volume,
      indicators: indicatorData,
    };
  });

  // Return merged rows - responseFormatter middleware will handle transformation
  return mergedData as any;
}
