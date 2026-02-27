/**
 * Signal Preview Query Functions
 *
 * Functions for querying signal preview snapshots from the AI Model database.
 * Table: signal_preview_snapshots
 *
 * This module provides:
 * - getSignalPreviewSnapshots(): Raw database query
 * - formatSignalPreviewResponse(): Formatted API response with grouping
 */

import { getAiModelPool } from '../connection';
import { executeQuery } from '../utils/query';

// ============================================================================
// Types
// ============================================================================

/**
 * Signal condition that was evaluated
 */
export interface SignalCondition {
  name: string;
  met: boolean;
  current: string;
  required: string;
}

/**
 * Fresh indicator values from technical_indicator_{symbol} tables
 */
export interface FreshIndicators {
  timestamp: string;
  stoch_k: number | null;
  stoch_d: number | null;
  rsi_14: number | null;
  sma_20: number | null;
  sma_50: number | null;
  sma_200: number | null;
  ema_12: number | null;
  ema_26: number | null;
  macd_line: number | null;
  macd_signal: number | null;
  macd_histogram: number | null;
  bb_upper_20: number | null;
  bb_lower_20: number | null;
  close: number | null;
}

/**
 * Model consensus information
 */
export interface ModelConsensus {
  agreement: string;        // "25/30" format
  models_agree: number;
  total_models: number;
  action: string;
  confidence: number;
}

/**
 * Raw signal preview snapshot from database
 */
export interface SignalPreviewSnapshot {
  id: number;
  symbol: string;
  direction: 'long' | 'short';
  signal_name: string;
  timeframe: string;
  confidence: number;
  next_candle_close: Date;
  conditions: string;       // JSON string (aliased from conditions_met)
  indicator_values: string; // JSON string
  timestamp: Date;
  status: string;            // READY/LOCKED/BLOCKED/CONDITIONS_MET/APPROACHING
  blocked_reason: string;    // e.g., "position_exists", "signal_locked"
}

/**
 * Formatted signal for API response
 */
export interface FormattedSignal {
  symbol: string;
  direction: 'long' | 'short';
  signal_name: string;
  timeframe: string;
  confidence: number;
  conditions: SignalCondition[];
  model_consensus?: ModelConsensus;
  status: string;            // READY/LOCKED/BLOCKED/CONDITIONS_MET/APPROACHING
  blocked_reason: string;    // e.g., "position_exists", "signal_locked"
}

/**
 * Candle group containing signals that close at the same time
 */
export interface CandleGroup {
  close_time: string;           // ISO timestamp
  seconds_until: number;
  timeframes: string[];
  signal_count: number;
  high_confidence_count: number;
  signals: FormattedSignal[];
}

/**
 * Account configuration for dynamic position sizing (Issue #631)
 */
export interface AccountConfig {
  balance_usd: number;           // Account balance in USD
  currency: string;              // Account currency (PLN, USD, etc.)
  exchange_rate_to_usd: number;  // Exchange rate to USD
  max_concurrent: number;        // Max concurrent positions allowed
}

/**
 * API response structure for signal preview endpoint
 */
export interface SignalPreviewResponse {
  data: {
    candle_groups: CandleGroup[];
    last_update: string;
    fresh_indicators: Record<string, FreshIndicators>;  // Key: "{SYMBOL}_{TIMEFRAME}"
  };
  metadata: {
    total_signals: number;
    total_high_confidence: number;
    next_close: string | null;
    timestamp: string;
    indicators_timestamp: string | null;
    account: AccountConfig;  // Issue #631: Account config for dynamic position sizing
  };
}

// ============================================================================
// Constants
// ============================================================================

const HIGH_CONFIDENCE_THRESHOLD = 0.8;

// Status sort priority (lower = higher priority, READY first)
const STATUS_PRIORITY: Record<string, number> = {
  READY: 1,
  LOCKED: 2,
  BLOCKED: 3,
  CONDITIONS_MET: 4,
  APPROACHING: 5,
};

// Issue #631: Account configuration for dynamic position sizing
// Values from config/paper_trading.yaml
const ACCOUNT_CONFIG: AccountConfig = {
  balance_usd: 13954,           // 49,532 PLN * 0.2817 = $13,954 USD
  currency: 'PLN',
  exchange_rate_to_usd: 0.2817,
  max_concurrent: 12,
};

// Valid symbols for indicator tables (lowercase for table names)
const VALID_SYMBOLS = ['eurusd', 'gbpusd', 'usdjpy', 'usdchf', 'eurjpy', 'eurgbp', 'eurcad', 'usdcad', 'xagusd', 'xauusd'];

// ============================================================================
// Query Functions
// ============================================================================

/**
 * Get fresh indicator values from technical_indicator_{symbol} tables
 *
 * Queries each symbol's indicator table for the most recent values per timeframe.
 * This provides real-time indicator data vs the snapshot data in signal_preview_snapshots.
 *
 * @param symbolTimeframes - Array of {symbol, timeframe} combinations to query
 * @returns Map of "{SYMBOL}_{TIMEFRAME}" -> FreshIndicators
 */
export async function getFreshIndicators(
  symbolTimeframes: Array<{ symbol: string; timeframe: string }>
): Promise<Record<string, FreshIndicators>> {
  const pool = getAiModelPool();
  const result: Record<string, FreshIndicators> = {};

  // Group by symbol for batch efficiency
  const bySymbol = new Map<string, string[]>();
  for (const { symbol, timeframe } of symbolTimeframes) {
    const symbolLower = symbol.toLowerCase();
    if (!VALID_SYMBOLS.includes(symbolLower)) continue;

    if (!bySymbol.has(symbolLower)) {
      bySymbol.set(symbolLower, []);
    }
    bySymbol.get(symbolLower)!.push(timeframe.toUpperCase());
  }

  // Query each symbol's table
  for (const [symbol, timeframes] of bySymbol.entries()) {
    try {
      // Use DISTINCT ON to get most recent row per timeframe
      const sql = `
        SELECT DISTINCT ON (timeframe)
          timeframe,
          timestamp,
          stoch_k,
          stoch_d,
          rsi_14,
          sma_20,
          sma_50,
          sma_200,
          ema_12,
          ema_26,
          macd_line,
          macd_signal,
          macd_histogram,
          bb_upper_20,
          bb_lower_20,
          close
        FROM technical_indicator_${symbol}
        WHERE timeframe = ANY($1)
        ORDER BY timeframe, timestamp DESC
      `;

      const rows = await executeQuery<{
        timeframe: string;
        timestamp: Date;
        stoch_k: number | null;
        stoch_d: number | null;
        rsi_14: number | null;
        sma_20: number | null;
        sma_50: number | null;
        sma_200: number | null;
        ema_12: number | null;
        ema_26: number | null;
        macd_line: number | null;
        macd_signal: number | null;
        macd_histogram: number | null;
        bb_upper_20: number | null;
        bb_lower_20: number | null;
        close: number | null;
      }>(pool, sql, [timeframes]);

      for (const row of rows) {
        const key = `${symbol.toUpperCase()}_${row.timeframe}`;
        result[key] = {
          timestamp: row.timestamp.toISOString(),
          stoch_k: row.stoch_k,
          stoch_d: row.stoch_d,
          rsi_14: row.rsi_14,
          sma_20: row.sma_20,
          sma_50: row.sma_50,
          sma_200: row.sma_200,
          ema_12: row.ema_12,
          ema_26: row.ema_26,
          macd_line: row.macd_line,
          macd_signal: row.macd_signal,
          macd_histogram: row.macd_histogram,
          bb_upper_20: row.bb_upper_20,
          bb_lower_20: row.bb_lower_20,
          close: row.close,
        };
      }
    } catch (error) {
      // Log warning but continue - graceful degradation
      console.warn(`Failed to query fresh indicators for ${symbol}:`, error);
    }
  }

  return result;
}

/**
 * Get signal preview snapshots from database
 *
 * Queries the signal_preview_snapshots table for records within the last 2 minutes,
 * ordered by next_candle_close and confidence.
 *
 * @returns Array of signal preview snapshots
 * @throws Error on database failure
 */
export async function getSignalPreviewSnapshots(): Promise<SignalPreviewSnapshot[]> {
  const pool = getAiModelPool();

  // CRITICAL FIX (Issue #631): Filter by approved_models to show ONLY 100 approved signals
  // Previously returned 398 snapshots (all signal_preview_snapshots)
  // Now returns ~100 (only signals that exist in approved_models table)
  // NOTE: Must also match on timeframe because same signal can be approved for multiple TFs
  // NOTE: Use DISTINCT ON to deduplicate multiple snapshots per signal (keeps most recent)
  const sql = `
    SELECT * FROM (
      SELECT DISTINCT ON (sp.symbol, sp.direction, sp.signal_name, sp.timeframe)
        sp.id,
        sp.symbol,
        sp.direction,
        sp.signal_name,
        sp.timeframe,
        sp.confidence,
        sp.next_candle_close,
        sp.conditions_met as conditions,
        sp.indicator_values,
        sp.timestamp,
        COALESCE(sp.status, 'APPROACHING') as status,
        COALESCE(sp.blocked_reason, '') as blocked_reason
      FROM signal_preview_snapshots sp
      INNER JOIN approved_models am
        ON LOWER(sp.symbol) = LOWER(am.symbol)
        AND LOWER(sp.direction) = LOWER(am.direction)
        AND sp.signal_name = am.signal_name
        AND LOWER(sp.timeframe) = LOWER(am.timeframe)
      WHERE sp.timestamp >= NOW() - INTERVAL '5 minutes'
        AND sp.conditions_met::text LIKE '[%'
      ORDER BY sp.symbol, sp.direction, sp.signal_name, sp.timeframe, sp.timestamp DESC
    ) deduped
    ORDER BY next_candle_close, confidence DESC
  `;

  const rows = await executeQuery<SignalPreviewSnapshot>(pool, sql);
  return rows;
}

/**
 * Format signal preview snapshots into API response structure
 *
 * Groups signals by next_candle_close time, calculates seconds_until,
 * and aggregates metadata.
 *
 * @returns Formatted SignalPreviewResponse
 */
export async function formatSignalPreviewResponse(): Promise<SignalPreviewResponse> {
  const snapshots = await getSignalPreviewSnapshots();
  const now = new Date();

  // Handle empty result
  if (snapshots.length === 0) {
    return {
      data: {
        candle_groups: [],
        last_update: now.toISOString(),
        fresh_indicators: {},
      },
      metadata: {
        total_signals: 0,
        total_high_confidence: 0,
        next_close: null,
        timestamp: now.toISOString(),
        indicators_timestamp: null,
        account: ACCOUNT_CONFIG,  // Issue #631: Include account config
      },
    };
  }

  // Collect unique symbol/timeframe combinations for fresh indicator query
  const symbolTimeframes: Array<{ symbol: string; timeframe: string }> = [];
  const seenKeys = new Set<string>();
  for (const snapshot of snapshots) {
    const key = `${snapshot.symbol}_${snapshot.timeframe}`;
    if (!seenKeys.has(key)) {
      seenKeys.add(key);
      symbolTimeframes.push({ symbol: snapshot.symbol, timeframe: snapshot.timeframe });
    }
  }

  // Fetch fresh indicators in parallel with response building
  const freshIndicatorsPromise = getFreshIndicators(symbolTimeframes);

  // Group signals by next_candle_close
  const groupMap = new Map<string, SignalPreviewSnapshot[]>();

  for (const snapshot of snapshots) {
    const closeTime = snapshot.next_candle_close.toISOString();
    if (!groupMap.has(closeTime)) {
      groupMap.set(closeTime, []);
    }
    groupMap.get(closeTime)!.push(snapshot);
  }

  // Build candle groups
  const candle_groups: CandleGroup[] = [];
  let totalHighConfidence = 0;

  for (const [closeTime, groupSnapshots] of groupMap.entries()) {
    const closeDate = new Date(closeTime);
    const secondsUntil = Math.max(0, Math.floor((closeDate.getTime() - now.getTime()) / 1000));

    // Collect unique timeframes
    const timeframeSet = new Set<string>();
    const signals: FormattedSignal[] = [];
    let highConfidenceCount = 0;

    for (const snapshot of groupSnapshots) {
      timeframeSet.add(snapshot.timeframe);

      // Parse conditions JSON - convert from object format {name: boolean} to array format
      let conditions: SignalCondition[] = [];
      try {
        const conditionsData = typeof snapshot.conditions === 'string'
          ? JSON.parse(snapshot.conditions)
          : snapshot.conditions;

        // Handle object format: {"SMA20 < SMA50": false, "Stoch_K > -100": true}
        if (conditionsData && typeof conditionsData === 'object' && !Array.isArray(conditionsData)) {
          conditions = Object.entries(conditionsData).map(([name, met]) => ({
            name,
            met: Boolean(met),
            current: '',  // Not stored in DB
            required: '', // Not stored in DB
          }));
        } else if (Array.isArray(conditionsData)) {
          // Handle array format if already in correct format
          conditions = conditionsData as SignalCondition[];
        }
      } catch {
        conditions = [];
      }

      // Note: model_consensus is not stored in database yet
      // It will be added when Python evaluator saves data with that field
      const modelConsensus: ModelConsensus | undefined = undefined;

      // Count high confidence signals
      if (snapshot.confidence >= HIGH_CONFIDENCE_THRESHOLD) {
        highConfidenceCount++;
        totalHighConfidence++;
      }

      signals.push({
        symbol: snapshot.symbol,
        direction: snapshot.direction,
        signal_name: snapshot.signal_name,
        timeframe: snapshot.timeframe,
        confidence: snapshot.confidence,
        conditions,
        model_consensus: modelConsensus,
        status: snapshot.status || 'APPROACHING',
        blocked_reason: snapshot.blocked_reason || '',
      });
    }

    // Sort signals by status priority (READY first), then confidence descending
    signals.sort((a, b) => {
      const priorityDiff = (STATUS_PRIORITY[a.status] ?? 5) - (STATUS_PRIORITY[b.status] ?? 5);
      if (priorityDiff !== 0) return priorityDiff;
      return b.confidence - a.confidence;
    });

    candle_groups.push({
      close_time: closeTime,
      seconds_until: secondsUntil,
      timeframes: Array.from(timeframeSet),
      signal_count: signals.length,
      high_confidence_count: highConfidenceCount,
      signals,
    });
  }

  // Sort candle groups by close time (ascending)
  candle_groups.sort((a, b) => new Date(a.close_time).getTime() - new Date(b.close_time).getTime());

  // Get last update from most recent snapshot
  const lastUpdate = snapshots.reduce((latest, s) => {
    return s.timestamp > latest ? s.timestamp : latest;
  }, snapshots[0]!.timestamp);

  // Await fresh indicators
  const freshIndicators = await freshIndicatorsPromise;

  // Get the most recent indicator timestamp
  let indicatorsTimestamp: string | null = null;
  for (const indicators of Object.values(freshIndicators)) {
    if (!indicatorsTimestamp || indicators.timestamp > indicatorsTimestamp) {
      indicatorsTimestamp = indicators.timestamp;
    }
  }

  return {
    data: {
      candle_groups,
      last_update: lastUpdate.toISOString(),
      fresh_indicators: freshIndicators,
    },
    metadata: {
      total_signals: snapshots.length,
      total_high_confidence: totalHighConfidence,
      next_close: candle_groups.length > 0 ? candle_groups[0]!.close_time : null,
      timestamp: now.toISOString(),
      indicators_timestamp: indicatorsTimestamp,
      account: ACCOUNT_CONFIG,  // Issue #631: Include account config
    },
  };
}
