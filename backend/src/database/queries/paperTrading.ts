/**
 * Paper Trading Query Functions
 *
 * Functions for querying paper trading data from the AI Model database:
 * - paper_positions: Open positions
 * - paper_trades: Closed trades history
 * - paper_performance: Performance metrics
 *
 * Issue #435 - Real-time Monitoring Dashboard Backend API
 */

import { getAiModelPool } from '../connection';
import { executeQuery } from '../utils/query';

// ============================================================================
// Types
// ============================================================================

/**
 * Open position data structure
 */
export interface Position {
  id: number;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  lots: number;           // Issue #631: Position size in lots
  margin_used: number;    // Issue #631: Estimated margin used (lots * margin_per_lot)
  entry_price: number;
  current_price: number;
  pnl_pips: number;
  pnl_unit: 'pips';
  sl_pips: number;
  tp_pips: number;
  opened_at: string;
  entry_model: string | null;
  signal_timeframe: string | null;
  ticket: number | null;  // MT5 broker ticket for exact position matching
}

/**
 * Closed trade data structure
 */
export interface Trade {
  id: number;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  size: number | null;
  entry_price: number;
  exit_price: number;
  sl_price: number | null;
  tp_price: number | null;
  pnl_pips: number;
  pnl_unit: 'pips';
  exit_reason: string;
  opened_at: string;
  closed_at: string;
  entry_model: string | null;  // Model used for entry decision
  signal_timeframe: string | null;
}

/**
 * Performance metrics data structure
 */
export interface Performance {
  profit_factor: number;
  win_rate: number;
  max_drawdown_pips: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  total_pnl_pips: number;
  pnl_unit: 'pips';
}

/**
 * System status data structure
 */
export interface Status {
  enabled: boolean;
  last_update: string | null;
  active_symbols: string[];
  open_positions_count: number;
}

/**
 * Filter options for trades query
 */
export interface TradeFilters {
  symbol?: string;
  direction?: 'LONG' | 'SHORT';
  start?: Date;
  end?: Date;
  limit?: number;
}

/**
 * Filter options for performance query
 */
export interface PerformanceFilters {
  symbol?: string;
  start?: Date;
  end?: Date;
}

/**
 * Filter options for decision log query
 */
export interface DecisionLogFilters {
  symbol?: string;
  direction?: 'LONG' | 'SHORT';
  log_type?: string;
  position_id?: string;
  start?: Date;
  end?: Date;
  limit?: number;
}

/**
 * Decision log entry data structure
 */
export interface DecisionLogEntry {
  id: number;
  timestamp: string;
  log_type: string;
  symbol: string | null;
  direction: string | null;
  entry_price: number | null;
  signal_confidence: number | null;
  rejection_reason: string | null;
  context_data: Record<string, unknown> | null;
}

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_LIMIT = 100;
const MAX_LIMIT = 1000;
const DECISION_LOG_MAX_LIMIT = 500;

// ============================================================================
// Database Row Types
// ============================================================================

interface PositionRow {
  id: number;
  symbol: string;
  direction: string;
  entry_time: Date;
  entry_price: string;
  sl_price: string;
  tp_price: string;
  size: string;
  unrealized_pnl: string | null;
  updated_at: Date;
  entry_model: string | null;
  signal_timeframe: string | null;
  ticket: number | null;
}

interface TradeRow {
  id: number;
  symbol: string;
  direction: string;
  entry_time: Date;
  entry_price: string;
  exit_time: Date | null;
  exit_price: string | null;
  sl_price: string;
  tp_price: string;
  size: string | null;
  pnl_pips: string | null;
  exit_reason: string | null;
  entry_model: string | null;  // Model used for entry decision
  signal_timeframe: string | null;
  created_at: Date;
}

// ============================================================================
// Query Functions
// ============================================================================

/**
 * Get all open paper trading positions
 *
 * @returns Array of open positions with calculated PnL
 */
export async function getOpenPositions(): Promise<Position[]> {
  const pool = getAiModelPool();

  // Note: paper_positions should only contain OPEN positions (not closed trades)
  // Issue #631: The table has grown massive - use primary key ordering which is indexed
  // This returns the most recently inserted positions efficiently
  const sql = `
    SELECT
      id,
      symbol,
      direction,
      entry_time,
      entry_price,
      sl_price,
      tp_price,
      size,
      unrealized_pnl,
      updated_at,
      entry_model,
      signal_timeframe,
      ticket
    FROM paper_positions
    ORDER BY id DESC
    LIMIT 20
  `;

  const rows = await executeQuery<PositionRow>(pool, sql);

  // Transform database rows to API response format
  return rows.map((row) => transformPositionRow(row));
}

/**
 * Get paper trading trade history with optional filters
 *
 * @param filters - Optional filters for symbol, direction, date range, limit
 * @returns Array of closed trades
 */
export async function getTrades(filters: TradeFilters = {}): Promise<Trade[]> {
  const pool = getAiModelPool();
  const { symbol, direction, start, end, limit = DEFAULT_LIMIT } = filters;

  // Build dynamic query
  const conditions: string[] = ['exit_time IS NOT NULL']; // Only closed trades
  const params: (string | Date | number)[] = [];
  let paramIndex = 1;

  if (symbol) {
    conditions.push(`symbol = $${paramIndex}`);
    params.push(symbol);
    paramIndex++;
  }

  if (direction) {
    conditions.push(`direction = $${paramIndex}`);
    params.push(direction);
    paramIndex++;
  }

  if (start) {
    conditions.push(`exit_time >= $${paramIndex}`);
    params.push(start);
    paramIndex++;
  }

  if (end) {
    conditions.push(`exit_time <= $${paramIndex}`);
    params.push(end);
    paramIndex++;
  }

  const limitValue = Math.min(limit, MAX_LIMIT);
  params.push(limitValue);

  const sql = `
    SELECT
      id,
      symbol,
      direction,
      entry_time,
      entry_price,
      exit_time,
      exit_price,
      sl_price,
      tp_price,
      size,
      pnl_pips,
      exit_reason,
      entry_model,
      signal_timeframe,
      created_at
    FROM paper_trades
    WHERE ${conditions.join(' AND ')}
    ORDER BY exit_time DESC
    LIMIT $${paramIndex}
  `;

  const rows = await executeQuery<TradeRow>(pool, sql, params);

  return rows.map((row) => transformTradeRow(row));
}

/**
 * Get aggregated performance metrics
 *
 * @param filters - Optional filters for symbol and date range
 * @returns Performance metrics object
 */
export async function getPerformance(filters: PerformanceFilters = {}): Promise<Performance> {
  const pool = getAiModelPool();
  const { symbol, start, end } = filters;

  // Build dynamic query
  const conditions: string[] = ['exit_time IS NOT NULL'];
  const params: (string | Date)[] = [];
  let paramIndex = 1;

  if (symbol) {
    conditions.push(`symbol = $${paramIndex}`);
    params.push(symbol);
    paramIndex++;
  }

  if (start) {
    conditions.push(`exit_time >= $${paramIndex}`);
    params.push(start);
    paramIndex++;
  }

  if (end) {
    conditions.push(`exit_time <= $${paramIndex}`);
    params.push(end);
    paramIndex++;
  }

  const sql = `
    SELECT
      COUNT(*) as total_trades,
      COUNT(*) FILTER (WHERE pnl_pips > 0) as winning_trades,
      COUNT(*) FILTER (WHERE pnl_pips <= 0) as losing_trades,
      COALESCE(SUM(pnl_pips), 0) as total_pnl_pips,
      COALESCE(SUM(pnl_pips) FILTER (WHERE pnl_pips > 0), 0) as gross_profit,
      COALESCE(ABS(SUM(pnl_pips) FILTER (WHERE pnl_pips < 0)), 0) as gross_loss
    FROM paper_trades
    WHERE ${conditions.join(' AND ')}
  `;

  const rows = await executeQuery<{
    total_trades: string;
    winning_trades: string;
    losing_trades: string;
    total_pnl_pips: string;
    gross_profit: string;
    gross_loss: string;
  }>(pool, sql, params);

  const row = rows[0];
  if (!row) {
    return getDefaultPerformance();
  }

  const totalTrades = parseInt(row.total_trades, 10);
  const winningTrades = parseInt(row.winning_trades, 10);
  const losingTrades = parseInt(row.losing_trades, 10);
  const totalPnlPips = parseFloat(row.total_pnl_pips);
  const grossProfit = parseFloat(row.gross_profit);
  const grossLoss = parseFloat(row.gross_loss);

  // Calculate profit factor (avoid division by zero)
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;

  // Calculate win rate
  const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;

  // Get max drawdown (separate query for running drawdown calculation)
  const maxDrawdown = await calculateMaxDrawdown(filters);

  return {
    profit_factor: Math.round(profitFactor * 100) / 100,
    win_rate: Math.round(winRate * 100) / 100,
    max_drawdown_pips: maxDrawdown,
    total_trades: totalTrades,
    winning_trades: winningTrades,
    losing_trades: losingTrades,
    total_pnl_pips: Math.round(totalPnlPips * 100) / 100,
    pnl_unit: 'pips',
  };
}

/**
 * Get paper trading system status
 *
 * @returns System status information
 */
export async function getStatus(): Promise<Status> {
  const pool = getAiModelPool();

  // Get open positions count and active symbols
  const positionsSql = `
    SELECT
      COUNT(*) as open_count,
      array_agg(DISTINCT symbol) as symbols,
      MAX(updated_at) as last_update
    FROM paper_positions
  `;

  const rows = await executeQuery<{
    open_count: string;
    symbols: string[] | null;
    last_update: Date | null;
  }>(pool, positionsSql);

  const row = rows[0];
  const openCount = row ? parseInt(row.open_count, 10) : 0;
  const symbols = row?.symbols?.filter(Boolean) || [];
  const lastUpdate = row?.last_update;

  // Check if paper trading is enabled by checking for recent activity
  // (Within last 24 hours = enabled)
  const recentActivityThreshold = new Date(Date.now() - 24 * 60 * 60 * 1000);
  const enabled = lastUpdate ? new Date(lastUpdate) > recentActivityThreshold : false;

  return {
    enabled,
    last_update: lastUpdate ? lastUpdate.toISOString() : null,
    active_symbols: symbols,
    open_positions_count: openCount,
  };
}

/**
 * Get decision log entries with optional filters
 *
 * @param filters - Optional filters for symbol, direction, log_type, date range, limit
 * @returns Array of decision log entries
 */
export async function getDecisionLog(filters: DecisionLogFilters = {}): Promise<DecisionLogEntry[]> {
  const pool = getAiModelPool();
  const { symbol, direction, log_type, position_id, start, end, limit = DEFAULT_LIMIT } = filters;

  const conditions: string[] = [];
  const params: (string | Date | number)[] = [];
  let paramIndex = 1;

  if (symbol) {
    conditions.push(`symbol = $${paramIndex}`);
    params.push(symbol);
    paramIndex++;
  }

  if (direction) {
    conditions.push(`direction = $${paramIndex}`);
    params.push(direction.toLowerCase());
    paramIndex++;
  }

  if (log_type) {
    conditions.push(`log_type = $${paramIndex}`);
    params.push(log_type);
    paramIndex++;
  }

  if (position_id) {
    conditions.push(`context_data->>'position_id' = $${paramIndex}`);
    params.push(position_id);
    paramIndex++;
  }

  if (start) {
    conditions.push(`timestamp >= $${paramIndex}`);
    params.push(start);
    paramIndex++;
  }

  if (end) {
    conditions.push(`timestamp <= $${paramIndex}`);
    params.push(end);
    paramIndex++;
  }

  const limitValue = Math.min(limit, DECISION_LOG_MAX_LIMIT);
  params.push(limitValue);

  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';

  const sql = `
    SELECT
      id,
      timestamp,
      log_type,
      symbol,
      direction,
      entry_price,
      signal_confidence,
      rejection_reason,
      context_data
    FROM paper_decision_log
    ${whereClause}
    ORDER BY timestamp DESC
    LIMIT $${paramIndex}
  `;

  interface DecisionLogRow {
    id: number;
    timestamp: Date;
    log_type: string;
    symbol: string | null;
    direction: string | null;
    entry_price: string | null;
    signal_confidence: string | null;
    rejection_reason: string | null;
    context_data: Record<string, unknown> | null;
  }

  const rows = await executeQuery<DecisionLogRow>(pool, sql, params);

  return rows.map((row) => ({
    id: row.id,
    timestamp: row.timestamp.toISOString(),
    log_type: row.log_type,
    symbol: row.symbol,
    direction: row.direction ? row.direction.toUpperCase() : null,
    entry_price: row.entry_price ? parseFloat(row.entry_price) : null,
    signal_confidence: row.signal_confidence ? parseFloat(row.signal_confidence) : null,
    rejection_reason: row.rejection_reason,
    context_data: row.context_data,
  }));
}

// ============================================================================
// Helper Functions
// ============================================================================

// Issue #631: Margin per lot estimates by symbol (matches balance_allocator.py)
const MARGIN_PER_LOT: Record<string, number> = {
  EURUSD: 3666.67,
  GBPUSD: 3666.67,
  USDJPY: 3333.33,
  USDCHF: 3333.33,
  EURJPY: 5500.00,
  EURGBP: 5000.00,
  EURCAD: 5000.00,
  USDCAD: 5000.00,
};
const DEFAULT_MARGIN_PER_LOT = 5000;

/**
 * Transform database position row to API response format
 */
function transformPositionRow(row: PositionRow): Position {
  const entryPrice = parseFloat(row.entry_price);
  const slPrice = parseFloat(row.sl_price);
  const tpPrice = parseFloat(row.tp_price);
  const lots = parseFloat(row.size) || 0;  // Issue #631: Get lot size from database

  // Calculate pip values based on direction
  // For forex pairs, 1 pip = 0.0001 (except JPY pairs where 1 pip = 0.01)
  const pipMultiplier = row.symbol.includes('JPY') ? 100 : 10000;
  // Issue #571: Use case-insensitive comparison - database stores 'long'/'short' (lowercase)
  const isLong = row.direction.toUpperCase() === 'LONG';

  // SL and TP in pips (distance from entry)
  const slPips = Math.abs((entryPrice - slPrice) * pipMultiplier);
  const tpPips = Math.abs((tpPrice - entryPrice) * pipMultiplier);

  // Current PnL from unrealized_pnl column
  const pnlPips = row.unrealized_pnl ? parseFloat(row.unrealized_pnl) : 0;

  // Estimate current price from entry and PnL
  const currentPrice = isLong
    ? entryPrice + pnlPips / pipMultiplier
    : entryPrice - pnlPips / pipMultiplier;

  // Issue #631: Calculate margin used based on lots and symbol
  const marginPerLot = MARGIN_PER_LOT[row.symbol.toUpperCase()] || DEFAULT_MARGIN_PER_LOT;
  const marginUsed = lots * marginPerLot;

  return {
    id: row.id,
    symbol: row.symbol,
    // Normalize direction to uppercase for API consistency
    direction: row.direction.toUpperCase() as 'LONG' | 'SHORT',
    lots: Math.round(lots * 100) / 100,  // Issue #631: Include lot size
    margin_used: Math.round(marginUsed * 100) / 100,  // Issue #631: Include margin used
    entry_price: entryPrice,
    current_price: Math.round(currentPrice * pipMultiplier) / pipMultiplier,
    pnl_pips: Math.round(pnlPips * 100) / 100,
    pnl_unit: 'pips',
    sl_pips: Math.round(slPips * 100) / 100,
    tp_pips: Math.round(tpPips * 100) / 100,
    opened_at: row.entry_time.toISOString(),
    entry_model: row.entry_model,
    signal_timeframe: row.signal_timeframe,
    ticket: row.ticket,
  };
}

/**
 * Transform database trade row to API response format
 */
function transformTradeRow(row: TradeRow): Trade {
  return {
    id: row.id,
    symbol: row.symbol,
    // Normalize direction to uppercase for API consistency
    direction: row.direction.toUpperCase() as 'LONG' | 'SHORT',
    size: row.size ? Math.round(parseFloat(row.size) * 100) / 100 : null,
    entry_price: parseFloat(row.entry_price),
    exit_price: row.exit_price ? parseFloat(row.exit_price) : 0,
    sl_price: row.sl_price ? parseFloat(row.sl_price) : null,
    tp_price: row.tp_price ? parseFloat(row.tp_price) : null,
    pnl_pips: row.pnl_pips ? parseFloat(row.pnl_pips) : 0,
    pnl_unit: 'pips',
    exit_reason: row.exit_reason || 'UNKNOWN',
    opened_at: row.entry_time.toISOString(),
    closed_at: row.exit_time ? row.exit_time.toISOString() : '',
    entry_model: row.entry_model,  // Model used for entry decision
    signal_timeframe: row.signal_timeframe,
  };
}

/**
 * Get default performance metrics (when no trades exist)
 */
function getDefaultPerformance(): Performance {
  return {
    profit_factor: 0,
    win_rate: 0,
    max_drawdown_pips: 0,
    total_trades: 0,
    winning_trades: 0,
    losing_trades: 0,
    total_pnl_pips: 0,
    pnl_unit: 'pips',
  };
}

/**
 * Live trading performance by symbol and direction
 */
export interface LiveTradingPerformance {
  symbol: string;
  direction: string;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_pnl_pips: number;
  avg_pnl_pips: number;
  profit_factor: number;
  first_trade: string | null;
  last_trade: string | null;
}

/**
 * Get live trading performance grouped by symbol and direction
 *
 * @returns Array of performance metrics per symbol/direction
 */
export async function getLiveTradingPerformance(): Promise<LiveTradingPerformance[]> {
  const pool = getAiModelPool();

  const sql = `
    SELECT
      symbol,
      direction,
      COUNT(*) as total_trades,
      COUNT(*) FILTER (WHERE pnl_pips > 0) as winning_trades,
      COUNT(*) FILTER (WHERE pnl_pips <= 0) as losing_trades,
      COALESCE(SUM(pnl_pips), 0) as total_pnl_pips,
      COALESCE(AVG(pnl_pips), 0) as avg_pnl_pips,
      COALESCE(SUM(pnl_pips) FILTER (WHERE pnl_pips > 0), 0) as gross_profit,
      COALESCE(ABS(SUM(pnl_pips) FILTER (WHERE pnl_pips < 0)), 0) as gross_loss,
      MIN(exit_time) as first_trade,
      MAX(exit_time) as last_trade
    FROM paper_trades
    WHERE exit_time IS NOT NULL
    GROUP BY symbol, direction
    ORDER BY symbol, direction
  `;

  const rows = await executeQuery<{
    symbol: string;
    direction: string;
    total_trades: string;
    winning_trades: string;
    losing_trades: string;
    total_pnl_pips: string;
    avg_pnl_pips: string;
    gross_profit: string;
    gross_loss: string;
    first_trade: Date | null;
    last_trade: Date | null;
  }>(pool, sql);

  return rows.map((row) => {
    const totalTrades = parseInt(row.total_trades, 10);
    const winningTrades = parseInt(row.winning_trades, 10);
    const grossProfit = parseFloat(row.gross_profit);
    const grossLoss = parseFloat(row.gross_loss);
    // Calculate profit factor (handle division by zero)
    let profitFactor: number;
    if (grossLoss === 0) {
      profitFactor = grossProfit > 0 ? 999.99 : 0; // Use large number instead of Infinity for JSON compatibility
    } else {
      profitFactor = grossProfit / grossLoss;
    }

    return {
      symbol: row.symbol,
      direction: row.direction.toLowerCase(),
      total_trades: totalTrades,
      winning_trades: winningTrades,
      losing_trades: parseInt(row.losing_trades, 10),
      win_rate: totalTrades > 0 ? Math.round((winningTrades / totalTrades) * 1000) / 10 : 0,
      total_pnl_pips: Math.round(parseFloat(row.total_pnl_pips) * 100) / 100,
      avg_pnl_pips: Math.round(parseFloat(row.avg_pnl_pips) * 100) / 100,
      profit_factor: Math.min(Math.round(profitFactor * 100) / 100, 999.99),
      first_trade: row.first_trade ? row.first_trade.toISOString() : null,
      last_trade: row.last_trade ? row.last_trade.toISOString() : null,
    };
  });
}

/**
 * Calculate maximum drawdown from trade history
 */
async function calculateMaxDrawdown(filters: PerformanceFilters): Promise<number> {
  const pool = getAiModelPool();
  const { symbol, start, end } = filters;

  // Build dynamic query
  const conditions: string[] = ['exit_time IS NOT NULL'];
  const params: (string | Date)[] = [];
  let paramIndex = 1;

  if (symbol) {
    conditions.push(`symbol = $${paramIndex}`);
    params.push(symbol);
    paramIndex++;
  }

  if (start) {
    conditions.push(`exit_time >= $${paramIndex}`);
    params.push(start);
    paramIndex++;
  }

  if (end) {
    conditions.push(`exit_time <= $${paramIndex}`);
    params.push(end);
    paramIndex++;
  }

  // Calculate running equity and max drawdown using chained CTEs
  // (PostgreSQL doesn't allow nested window functions)
  const sql = `
    WITH cumulative AS (
      SELECT
        exit_time,
        pnl_pips,
        SUM(pnl_pips) OVER (ORDER BY exit_time) as cumulative_pnl
      FROM paper_trades
      WHERE ${conditions.join(' AND ')}
    ),
    running_equity AS (
      SELECT
        exit_time,
        pnl_pips,
        cumulative_pnl,
        MAX(cumulative_pnl) OVER (ORDER BY exit_time ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as peak
      FROM cumulative
    )
    SELECT COALESCE(MAX(peak - cumulative_pnl), 0) as max_drawdown
    FROM running_equity
  `;

  const rows = await executeQuery<{ max_drawdown: string }>(pool, sql, params);

  return rows[0] ? parseFloat(rows[0].max_drawdown) : 0;
}

// ============================================================================
// Signal Weights (read from Python-computed signal_kelly_weights table)
// ============================================================================

export interface SignalWeight {
  key: string;
  wins: number;
  losses: number;
  avg_win: number;
  avg_loss: number;
  consecutive_losses: number;
  weight: number;
  raw_kelly?: number;
  decay_factor?: number;
  blend_phase?: string;
  win_rate?: number;
  reward_ratio?: number;
}

/**
 * Read Kelly weights from signal_kelly_weights table (Python is single source of truth).
 * Falls back to paper_trades computation if table doesn't exist or is empty.
 */
export async function getSignalWeights(): Promise<Record<string, SignalWeight>> {
  const pool = getAiModelPool();

  const sql = `
    SELECT signal_key, wins, losses, avg_win_pips, avg_loss_pips,
           consecutive_losses, raw_kelly, decay_factor, blend_phase,
           final_weight, win_rate, reward_ratio
    FROM signal_kelly_weights
    ORDER BY signal_key
  `;

  try {
    const rows = await executeQuery<{
      signal_key: string;
      wins: number;
      losses: number;
      avg_win_pips: string;
      avg_loss_pips: string;
      consecutive_losses: number;
      raw_kelly: string;
      decay_factor: string;
      blend_phase: string;
      final_weight: string;
      win_rate: string | null;
      reward_ratio: string | null;
    }>(pool, sql);

    if (rows.length === 0) {
      // Table empty — Python hasn't populated yet, return empty
      return {};
    }

    const result: Record<string, SignalWeight> = {};
    for (const row of rows) {
      result[row.signal_key] = {
        key: row.signal_key,
        wins: row.wins,
        losses: row.losses,
        avg_win: Math.round(parseFloat(row.avg_win_pips) * 100) / 100,
        avg_loss: Math.round(parseFloat(row.avg_loss_pips) * 100) / 100,
        consecutive_losses: row.consecutive_losses,
        weight: Math.round(parseFloat(row.final_weight) * 1000) / 1000,
        raw_kelly: Math.round(parseFloat(row.raw_kelly) * 10000) / 10000,
        decay_factor: Math.round(parseFloat(row.decay_factor) * 1000) / 1000,
        blend_phase: row.blend_phase,
        win_rate: row.win_rate ? Math.round(parseFloat(row.win_rate) * 10000) / 10000 : undefined,
        reward_ratio: row.reward_ratio ? Math.round(parseFloat(row.reward_ratio) * 10000) / 10000 : undefined,
      };
    }
    return result;
  } catch {
    // Table may not exist yet — return empty
    return {};
  }
}
