/**
 * Order Blocks Query Functions
 *
 * Functions for querying SMC (Smart Money Concepts) order blocks
 * and their touch events from the AI Model database.
 */

import { getAiModelPool } from '../connection';
import { executeQuery } from '../utils/query';
import { ValidationError } from '../errors';

/**
 * Order Block structure
 */
export interface OrderBlock {
  id: number;
  symbol: string;
  timeframe: string;
  ts: number; // Unix timestamp in milliseconds
  direction: 'BULLISH' | 'BEARISH';
  obLow: number;
  obHigh: number;
  obBodyLow?: number;
  obBodyHigh?: number;
  status: 'ACTIVE' | 'MITIGATED' | 'INVALIDATED';
  mitigationTs?: number;
  mitigationPrice?: number;
  bodyRatio?: number;
  displacementAtr?: number;
}

/**
 * Order Block Touch structure
 */
export interface OrderBlockTouch {
  id: number;
  orderBlockId: number;
  symbol: string;
  timeframe: string;
  ts: number; // Unix timestamp in milliseconds
  touchType: 'FIRST_TOUCH' | 'RETOUCH';
  touchPrice: number;
}

/**
 * Supported timeframes
 */
export type Timeframe = 'M30' | 'H1' | 'H2' | 'H4' | 'H6' | 'H8' | 'H12' | 'D1';

/**
 * Supported symbols
 */
export type Symbol = 'EURUSD' | 'GBPUSD' | 'USDJPY' | 'EURJPY' | 'USDCAD' | 'USDCHF' | 'EURCAD' | 'EURGBP';

/**
 * Constants for validation
 */
const SUPPORTED_SYMBOLS: Symbol[] = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'EURCAD', 'EURGBP'];
const VALID_TIMEFRAMES: Timeframe[] = ['M30', 'H1', 'H2', 'H4', 'H6', 'H8', 'H12', 'D1'];
const DEFAULT_LIMIT = 100;
const MAX_LIMIT = 1000;

/**
 * Database row structure from order_blocks table
 * NOTE: PostgreSQL BIGINT and DECIMAL types come as strings to preserve precision
 */
interface OrderBlockRow {
  id: string; // BIGINT comes as string
  symbol: string;
  timeframe: string;
  ts: string; // BIGINT comes as string
  direction: string;
  ob_low: string; // DECIMAL comes as string
  ob_high: string;
  ob_body_low?: string;
  ob_body_high?: string;
  status: string;
  mitigation_ts?: string;
  mitigation_price?: string;
  body_ratio?: string;
  displacement_atr?: string;
}

/**
 * Database row structure from order_block_touches table
 * NOTE: PostgreSQL BIGINT and DECIMAL types come as strings to preserve precision
 */
interface OrderBlockTouchRow {
  id: string; // BIGINT comes as string
  order_block_id: string; // BIGINT comes as string
  symbol: string;
  timeframe: string;
  ts: string; // BIGINT comes as string
  touch_type: string;
  touch_price: string; // DECIMAL comes as string
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
      value: symbol,
      expected: VALID_TIMEFRAMES.join(', '),
    });
  }

  // Validate dates
  if (!(start instanceof Date) || isNaN(start.getTime())) {
    throw new ValidationError('Invalid start date', {
      field: 'start',
      value: start,
    });
  }

  if (!(end instanceof Date) || isNaN(end.getTime())) {
    throw new ValidationError('Invalid end date', {
      field: 'end',
      value: end,
    });
  }

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
 * Convert database row to OrderBlock interface
 * NOTE: Database timestamp precision varies:
 *   - ts: nanoseconds (divide by 1,000,000 to get milliseconds)
 *   - mitigation_ts: seconds (multiply by 1,000 to get milliseconds)
 * NOTE: All numeric fields come as strings from PostgreSQL and must be parsed
 */
function mapOrderBlockRow(row: OrderBlockRow): OrderBlock {
  return {
    id: parseInt(row.id, 10), // Convert BIGINT string to number
    symbol: row.symbol,
    timeframe: row.timeframe,
    ts: Math.floor(parseInt(row.ts, 10) / 1_000_000), // Convert nanoseconds to milliseconds
    direction: row.direction as 'BULLISH' | 'BEARISH',
    obLow: parseFloat(row.ob_low),
    obHigh: parseFloat(row.ob_high),
    obBodyLow: row.ob_body_low ? parseFloat(row.ob_body_low) : undefined,
    obBodyHigh: row.ob_body_high ? parseFloat(row.ob_body_high) : undefined,
    status: row.status as 'ACTIVE' | 'MITIGATED' | 'INVALIDATED',
    mitigationTs: row.mitigation_ts ? parseInt(row.mitigation_ts, 10) * 1_000 : undefined, // Convert seconds to milliseconds
    mitigationPrice: row.mitigation_price ? parseFloat(row.mitigation_price) : undefined,
    bodyRatio: row.body_ratio ? parseFloat(row.body_ratio) : undefined,
    displacementAtr: row.displacement_atr ? parseFloat(row.displacement_atr) : undefined,
  };
}

/**
 * Convert database row to OrderBlockTouch interface
 * NOTE: Database stores timestamps in nanoseconds, convert to milliseconds for frontend
 * NOTE: All numeric fields come as strings from PostgreSQL and must be parsed
 */
function mapOrderBlockTouchRow(row: OrderBlockTouchRow): OrderBlockTouch {
  return {
    id: parseInt(row.id, 10), // Convert BIGINT string to number
    orderBlockId: parseInt(row.order_block_id, 10), // Convert BIGINT string to number
    symbol: row.symbol,
    timeframe: row.timeframe,
    ts: Math.floor(parseInt(row.ts, 10) / 1_000_000), // Convert nanoseconds to milliseconds
    touchType: row.touch_type as 'FIRST_TOUCH' | 'RETOUCH',
    touchPrice: parseFloat(row.touch_price),
  };
}

/**
 * Get order blocks for a symbol and timeframe within a time range
 *
 * @param symbol - Trading symbol (e.g., 'EURUSD')
 * @param timeframe - Timeframe (e.g., 'H1')
 * @param start - Start date (inclusive)
 * @param end - End date (inclusive)
 * @param limit - Maximum number of order blocks to return (default: 100, max: 1000)
 * @returns Array of order blocks, ordered by timestamp descending (newest first)
 * @throws ValidationError if inputs are invalid
 */
export async function getOrderBlocks(
  symbol: string,
  timeframe: string,
  start: Date,
  end: Date,
  limit: number = DEFAULT_LIMIT
): Promise<OrderBlock[]> {
  // Validate inputs
  validateInputs(symbol, timeframe, start, end, limit);

  // Convert dates to Unix timestamps in nanoseconds (database stores BIGINT in nanoseconds)
  // Date.getTime() returns milliseconds, multiply by 1,000,000 to get nanoseconds
  const startTs = start.getTime() * 1_000_000;
  const endTs = end.getTime() * 1_000_000;

  // Query order blocks table
  const query = `
    SELECT
      id,
      symbol,
      timeframe,
      ts,
      direction,
      ob_low,
      ob_high,
      ob_body_low,
      ob_body_high,
      status,
      mitigation_ts,
      mitigation_price,
      body_ratio,
      displacement_atr
    FROM order_blocks
    WHERE symbol = $1
      AND timeframe = $2
      AND ts >= $3
      AND ts <= $4
    ORDER BY ts DESC
    LIMIT $5
  `;

  const params = [symbol, timeframe, startTs, endTs, limit];

  // Execute query
  const pool = getAiModelPool();
  const rows = await executeQuery<OrderBlockRow>(pool, query, params);

  // Map rows to OrderBlock interface
  return rows.map(mapOrderBlockRow);
}

/**
 * Get order block touches for a symbol and timeframe within a time range
 *
 * @param symbol - Trading symbol (e.g., 'EURUSD')
 * @param timeframe - Timeframe (e.g., 'H1')
 * @param start - Start date (inclusive)
 * @param end - End date (inclusive)
 * @param limit - Maximum number of touches to return (default: 100, max: 1000)
 * @returns Array of order block touches, ordered by timestamp descending (newest first)
 * @throws ValidationError if inputs are invalid
 */
export async function getOrderBlockTouches(
  symbol: string,
  timeframe: string,
  start: Date,
  end: Date,
  limit: number = DEFAULT_LIMIT
): Promise<OrderBlockTouch[]> {
  // Validate inputs
  validateInputs(symbol, timeframe, start, end, limit);

  // Convert dates to Unix timestamps in nanoseconds (database stores BIGINT in nanoseconds)
  // Date.getTime() returns milliseconds, multiply by 1,000,000 to get nanoseconds
  const startTs = start.getTime() * 1_000_000;
  const endTs = end.getTime() * 1_000_000;

  // Query order_block_touches table
  const query = `
    SELECT
      id,
      order_block_id,
      symbol,
      timeframe,
      ts,
      touch_type,
      touch_price
    FROM order_block_touches
    WHERE symbol = $1
      AND timeframe = $2
      AND ts >= $3
      AND ts <= $4
    ORDER BY ts DESC
    LIMIT $5
  `;

  const params = [symbol, timeframe, startTs, endTs, limit];

  // Execute query
  const pool = getAiModelPool();
  const rows = await executeQuery<OrderBlockTouchRow>(pool, query, params);

  // Map rows to OrderBlockTouch interface
  return rows.map(mapOrderBlockTouchRow);
}
