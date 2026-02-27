/**
 * Paper Trading Routes
 *
 * API routes for paper trading dashboard data:
 * - GET /api/paper-trading/positions - Open positions
 * - GET /api/paper-trading/trades - Trade history with filters
 * - GET /api/paper-trading/performance - Performance metrics
 * - GET /api/paper-trading/status - System status
 * - GET /api/paper-trading/live-performance - Live trading performance by symbol/direction
 * - GET /api/paper-trading/models - Model information for all symbols
 * - GET /api/paper-trading/models/:symbol - Model information for specific symbol
 * - GET /api/paper-trading/model-performance - Model validation performance (all)
 * - GET /api/paper-trading/model-performance/:symbol - Performance for symbol
 * - GET /api/paper-trading/model-performance/:symbol/:direction - Performance for symbol+direction
 * - GET /api/paper-trading/signal-preview - Upcoming signal predictions grouped by candle close time
 *
 * Issue #435 - Real-time Monitoring Dashboard Backend API
 * Issue #517 - PerformanceMetrics module with TDD
 */

import { Router, Request, Response, NextFunction } from 'express';
import {
  getOpenPositions,
  getTrades,
  getPerformance,
  getStatus,
  getLiveTradingPerformance,
  getDecisionLog,
  getSignalWeights,
  TradeFilters,
  PerformanceFilters,
  DecisionLogFilters,
} from '../database/queries/paperTrading';
import { getModelInfo, getSymbolModelInfoByName } from '../database/queries/modelInfo';
import {
  getModelPerformance,
  getModelPerformanceBySymbol,
  getModelPerformanceBySymbolAndDirection,
} from '../database/queries/modelPerformance';
import { formatSignalPreviewResponse } from '../database/queries/signalPreview';
import { ValidationError } from '../database/errors';

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

// Valid directions for trade filtering
const VALID_DIRECTIONS = ['LONG', 'SHORT'];

// Valid log types for decision log filtering
const VALID_LOG_TYPES = [
  'signal_generated',
  'signal_rejected',
  'position_opened',
  'position_closed',
  'risk_violation',
  'system_error',
  'symbol_disabled',
  'rl_exit_decision',
];

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Parse and validate date parameter
 */
function parseDate(value: unknown, fieldName: string): Date | undefined {
  if (value === undefined || value === null || value === '') {
    return undefined;
  }

  if (typeof value !== 'string') {
    throw new ValidationError(`Invalid ${fieldName}: must be a string`, {
      field: fieldName,
      value,
    });
  }

  const date = new Date(value);
  if (isNaN(date.getTime())) {
    throw new ValidationError(`Invalid ${fieldName} date format`, {
      field: fieldName,
      value,
    });
  }

  return date;
}

/**
 * Parse integer parameter with default
 */
function parseLimit(value: unknown, defaultValue: number): number {
  if (value === undefined || value === null || value === '') {
    return defaultValue;
  }

  const parsed = parseInt(String(value), 10);
  if (isNaN(parsed) || parsed <= 0) {
    return defaultValue;
  }

  return parsed;
}

/**
 * Validate direction parameter
 */
function validateDirection(value: unknown): 'LONG' | 'SHORT' | undefined {
  if (value === undefined || value === null || value === '') {
    return undefined;
  }

  const direction = String(value).toUpperCase();
  if (!VALID_DIRECTIONS.includes(direction)) {
    throw new ValidationError(`Invalid direction: must be LONG or SHORT`, {
      field: 'direction',
      value,
      expected: VALID_DIRECTIONS.join(', '),
    });
  }

  return direction as 'LONG' | 'SHORT';
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
 * GET /api/paper-trading/positions
 *
 * Returns all open paper trading positions
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/paper-trading/positions',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const positions = await getOpenPositions();

      res.json(formatResponse(positions, req, positions.length));
    } catch (error) {
      // Issue #631: paper_positions table has grown massive and causes timeouts
      // Return empty array to allow dashboard to function while DB is fixed
      const err = error as {
        code?: string;
        message?: string;
        context?: { code?: string; error?: string };
      };
      const isTimeout =
        err.code === '57014' ||
        err.context?.code === '57014' ||
        err.message?.includes('timeout') ||
        err.context?.error?.includes('timeout');

      if (isTimeout) {
        console.warn('paper_positions query timeout - returning empty array');
        res.json(formatResponse([], req, 0));
        return;
      }
      next(error);
    }
  }
);

/**
 * GET /api/paper-trading/trades
 *
 * Returns trade history with optional filters:
 * - symbol: Filter by trading symbol (e.g., EURUSD)
 * - direction: Filter by trade direction (LONG or SHORT)
 * - start: Start date (ISO 8601 format)
 * - end: End date (ISO 8601 format)
 * - limit: Maximum number of trades to return (default: 100)
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/paper-trading/trades',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      // Parse query parameters
      const filters: TradeFilters = {};

      if (req.query['symbol']) {
        filters.symbol = String(req.query['symbol']);
      }

      if (req.query['direction']) {
        filters.direction = validateDirection(req.query['direction']);
      }

      if (req.query['start']) {
        filters.start = parseDate(req.query['start'], 'start');
      }

      if (req.query['end']) {
        filters.end = parseDate(req.query['end'], 'end');
      }

      filters.limit = parseLimit(req.query['limit'], 100);

      const trades = await getTrades(filters);

      res.json(formatResponse(trades, req, trades.length));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/paper-trading/performance
 *
 * Returns aggregated performance metrics with optional filters:
 * - symbol: Filter by trading symbol
 * - start: Start date (ISO 8601 format)
 * - end: End date (ISO 8601 format)
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/paper-trading/performance',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      // Parse query parameters
      const filters: PerformanceFilters = {};

      if (req.query['symbol']) {
        filters.symbol = String(req.query['symbol']);
      }

      if (req.query['start']) {
        filters.start = parseDate(req.query['start'], 'start');
      }

      if (req.query['end']) {
        filters.end = parseDate(req.query['end'], 'end');
      }

      const performance = await getPerformance(filters);

      res.json(formatResponse(performance, req));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/paper-trading/status
 *
 * Returns paper trading system status
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/paper-trading/status',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const status = await getStatus();

      res.json(formatResponse(status, req));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/paper-trading/live-performance
 *
 * Returns live trading performance from paper_trades table
 * grouped by symbol and direction
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/paper-trading/live-performance',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const livePerformance = await getLiveTradingPerformance();

      res.json(formatResponse(livePerformance, req, livePerformance.length));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/paper-trading/decisions
 *
 * Returns decision log entries with optional filters:
 * - symbol: Filter by trading symbol (e.g., EURUSD)
 * - direction: Filter by direction (LONG or SHORT)
 * - log_type: Filter by log type (signal_generated, rl_exit_decision, etc.)
 * - start: Start date (ISO 8601 format)
 * - end: End date (ISO 8601 format)
 * - limit: Maximum number of entries to return (default: 100, max: 500)
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/paper-trading/decisions',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const filters: DecisionLogFilters = {};

      if (req.query['symbol']) {
        filters.symbol = String(req.query['symbol']);
      }

      if (req.query['direction']) {
        filters.direction = validateDirection(req.query['direction']);
      }

      if (req.query['log_type']) {
        const logType = String(req.query['log_type']);
        if (!VALID_LOG_TYPES.includes(logType)) {
          throw new ValidationError(`Invalid log_type: must be one of ${VALID_LOG_TYPES.join(', ')}`, {
            field: 'log_type',
            value: logType,
            expected: VALID_LOG_TYPES.join(', '),
          });
        }
        filters.log_type = logType;
      }

      if (req.query['position_id']) {
        filters.position_id = String(req.query['position_id']);
      }

      if (req.query['start']) {
        filters.start = parseDate(req.query['start'], 'start');
      }

      if (req.query['end']) {
        filters.end = parseDate(req.query['end'], 'end');
      }

      filters.limit = parseLimit(req.query['limit'], 100);

      const decisions = await getDecisionLog(filters);

      res.json(formatResponse(decisions, req, decisions.length));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/paper-trading/models
 *
 * Returns model information for all configured symbols
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/paper-trading/models',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const modelInfo = await getModelInfo();

      res.json(formatResponse(modelInfo, req, modelInfo.symbols.length));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/paper-trading/models/:symbol
 *
 * Returns model information for a specific symbol
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/paper-trading/models/:symbol',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const { symbol } = req.params;

      if (!symbol) {
        throw new ValidationError('Symbol parameter is required', {
          field: 'symbol',
        });
      }

      const modelInfo = await getSymbolModelInfoByName(symbol);

      if (!modelInfo) {
        res.status(404).json({
          error: 'Symbol not found',
          message: `No model configuration found for symbol: ${symbol}`,
        });
        return;
      }

      res.json(formatResponse(modelInfo, req));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/paper-trading/model-performance
 *
 * Returns validation performance data for all models
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/paper-trading/model-performance',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const performanceData = await getModelPerformance();

      res.json(formatResponse(performanceData, req, performanceData.length));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/paper-trading/model-performance/:symbol
 *
 * Returns validation performance data for a specific symbol (all directions)
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/paper-trading/model-performance/:symbol',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const { symbol } = req.params;

      if (!symbol) {
        throw new ValidationError('Symbol parameter is required', {
          field: 'symbol',
        });
      }

      const performanceData = await getModelPerformanceBySymbol(symbol);

      res.json(formatResponse(performanceData, req, performanceData.length));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/paper-trading/model-performance/:symbol/:direction
 *
 * Returns validation performance data for a specific symbol and direction
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/paper-trading/model-performance/:symbol/:direction',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const { symbol, direction } = req.params;

      if (!symbol) {
        throw new ValidationError('Symbol parameter is required', {
          field: 'symbol',
        });
      }

      if (!direction) {
        throw new ValidationError('Direction parameter is required', {
          field: 'direction',
        });
      }

      // Validate direction is long or short
      const directionLower = direction.toLowerCase();
      if (directionLower !== 'long' && directionLower !== 'short') {
        throw new ValidationError('Direction must be "long" or "short"', {
          field: 'direction',
          value: direction,
          expected: 'long, short',
        });
      }

      const performanceData = await getModelPerformanceBySymbolAndDirection(symbol, directionLower);

      res.json(formatResponse(performanceData, req, performanceData.length));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/paper-trading/signal-weights
 *
 * Returns half-Kelly sizing weights per signal key.
 * Mirrors LivePerformanceTracker logic from paper_trades data.
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/paper-trading/signal-weights',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const weights = await getSignalWeights();

      res.json(formatResponse(weights, req, Object.keys(weights).length));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/paper-trading/signal-preview
 *
 * Returns upcoming signal predictions grouped by candle close time.
 * Queries signal_preview_snapshots table for records within last 2 minutes.
 *
 * Response includes:
 * - candle_groups: Signals grouped by next_candle_close time
 * - seconds_until: Time remaining until candle close
 * - high_confidence_count: Signals with confidence >= 0.8
 * - model_consensus: Agreement from 30-fold ensemble models
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/paper-trading/signal-preview',
  async (_req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const response = await formatSignalPreviewResponse();

      // Return response with the signal preview structure
      // The formatSignalPreviewResponse already includes metadata
      res.json(response);
    } catch (error) {
      next(error);
    }
  }
);

export default router;
