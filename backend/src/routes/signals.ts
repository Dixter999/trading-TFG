/**
 * Signals Routes (Issue #631)
 *
 * API routes for trading signals and approved models:
 * - GET /api/signals/approved-models - Get approved models with PF-based lot suggestions
 *
 * These endpoints power the enhanced trading dashboard and provide
 * signal recommendations based on Phase 5 performance metrics.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { getAiModelPool } from '../database/connection';

const router = Router();

// ============================================================================
// Types
// ============================================================================

interface RequestWithId extends Request {
  id?: string;
}

interface ApiResponse<T> {
  data: T;
  metadata: {
    count?: number;
    timestamp: string;
    requestId?: string;
  };
}

interface ApprovedModelRow {
  symbol: string;
  signal_name: string;
  direction: string;
  timeframe: string;
  phase5_pf: number | null;
  phase5_wr: number | null;
}

interface ApprovedModelResponse {
  symbol: string;
  signalName: string;
  direction: string;
  timeframe: string;
  phase5Pf: number | null;
  phase5Wr: number | null;
  suggestedLots: string;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Calculate suggested lot size based on Profit Factor
 *
 * @param pf - Phase 5 Profit Factor
 * @returns Suggested lot size string (e.g., "0.5L", "0.3L", "0.2L", "0.1L")
 *
 * Rules:
 * - PF >= 3.0 -> 0.5L (high confidence)
 * - PF >= 2.0 -> 0.3L (good confidence)
 * - PF >= 1.5 -> 0.2L (moderate confidence)
 * - PF < 1.5  -> 0.1L (low confidence)
 */
function getVolumeFromPF(pf: number | null): string {
  if (pf === null || pf === undefined) {
    return '0.1L';
  }

  if (pf >= 3.0) {
    return '0.5L';
  } else if (pf >= 2.0) {
    return '0.3L';
  } else if (pf >= 1.5) {
    return '0.2L';
  } else {
    return '0.1L';
  }
}

/**
 * Transform database row to API response format
 */
function transformModelRow(row: ApprovedModelRow): ApprovedModelResponse {
  return {
    symbol: row.symbol,
    signalName: row.signal_name,
    direction: row.direction,
    timeframe: row.timeframe,
    phase5Pf: row.phase5_pf,
    phase5Wr: row.phase5_wr,
    suggestedLots: getVolumeFromPF(row.phase5_pf),
  };
}

/**
 * Format standard API response
 */
function formatResponse<T>(data: T, req: RequestWithId, count?: number): ApiResponse<T> {
  return {
    data,
    metadata: {
      count,
      timestamp: new Date().toISOString(),
      requestId: req.id,
    },
  };
}

// ============================================================================
// Route Handlers
// ============================================================================

/**
 * GET /api/signals/approved-models
 *
 * Get all approved models with PF-based lot suggestions.
 *
 * Returns approved models sorted by Phase 5 Profit Factor descending, including:
 * - symbol: Trading symbol (e.g., EURUSD)
 * - signalName: Name of the signal
 * - direction: LONG or SHORT
 * - timeframe: Trading timeframe (e.g., H4, H1)
 * - phase5Pf: Phase 5 Profit Factor
 * - phase5Wr: Phase 5 Win Rate
 * - suggestedLots: Recommended lot size based on PF
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/signals/approved-models',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const pool = getAiModelPool();

      const query = `
        SELECT symbol, signal_name, direction, timeframe, phase5_pf, phase5_wr
        FROM approved_models
        ORDER BY phase5_pf DESC
      `;

      const result = await pool.query(query);
      const models = (result.rows as ApprovedModelRow[]).map(transformModelRow);

      res.json(formatResponse(models, req, models.length));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/signals/lifecycle-summary
 *
 * Returns per-signal lifecycle details from signal_lifecycle table.
 * Each row includes: signal_id, symbol, direction, signal_name, timeframe,
 * lifecycle_state, phase5_pf, phase5_wr, last_state_change, state_history.
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/signals/lifecycle-summary',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const pool = getAiModelPool();

      const result = await pool.query(`
        SELECT
          signal_id,
          UPPER(symbol) AS symbol,
          UPPER(direction) AS direction,
          signal_name,
          UPPER(timeframe) AS timeframe,
          lifecycle_state,
          phase5_pf,
          phase5_wr,
          last_state_change,
          updated_at
        FROM signal_lifecycle
        ORDER BY lifecycle_state ASC, UPPER(symbol), signal_name
      `);

      const signals = result.rows.map((row: Record<string, unknown>) => ({
        signalId: row['signal_id'],
        symbol: row['symbol'],
        direction: row['direction'],
        signalName: row['signal_name'],
        timeframe: row['timeframe'],
        state: row['lifecycle_state'],
        phase5Pf: row['phase5_pf'] != null ? Number(row['phase5_pf']) : null,
        phase5Wr: row['phase5_wr'] != null ? Number(row['phase5_wr']) : null,
        lastStateChange: row['last_state_change'],
        updatedAt: row['updated_at'],
      }));

      res.json(formatResponse(signals, req, signals.length));
    } catch (error) {
      next(error);
    }
  }
);

export default router;
