/**
 * Markets Routes
 *
 * Defines API routes for market data endpoints
 */

import { Router } from 'express';
import { getMarketData, getMarketConfig } from '../controllers/marketsController';
import { formatResponse } from '../middleware/responseFormatter';

const router = Router();

/**
 * GET /api/markets/config
 *
 * Returns supported symbols and timeframes
 * This endpoint must be defined BEFORE the parametric route to avoid route conflicts
 *
 * Example:
 * GET /api/markets/config
 * Response: { success: true, data: { symbols: [...], timeframes: [...], defaultSymbol: "EURUSD", defaultTimeframe: "H1" } }
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get('/api/markets/config', getMarketConfig);

/**
 * GET /api/markets/:symbol/:timeframe
 *
 * Query parameters:
 * - start: ISO 8601 date string (optional, defaults to earliest available)
 * - end: ISO 8601 date string (optional, defaults to latest available)
 * - limit: number (optional, default: 1000, max: 10000)
 *
 * Example:
 * GET /api/markets/EURUSD/H1?start=2024-01-01T00:00:00Z&end=2024-01-31T23:59:59Z&limit=1000
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get('/api/markets/:symbol/:timeframe', getMarketData, formatResponse);

export default router;
