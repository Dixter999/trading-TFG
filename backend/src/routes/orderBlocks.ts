/**
 * Order Blocks Routes
 *
 * Defines API routes for SMC order blocks and touch events
 */

import { Router } from 'express';
import { getOrderBlocksData, getOrderBlockTouchesData } from '../controllers/orderBlocksController';

const router = Router();

/**
 * GET /api/ai-model/order-blocks/:symbol/:timeframe
 *
 * Retrieves order blocks for specified symbol and timeframe
 *
 * Query parameters:
 * - start: ISO 8601 date string (required)
 * - end: ISO 8601 date string (required)
 * - limit: number (optional, default: 100, max: 1000)
 *
 * Example:
 * GET /api/ai-model/order-blocks/EURUSD/H1?start=2024-01-01T00:00:00Z&end=2024-01-31T23:59:59Z&limit=100
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get('/api/ai-model/order-blocks/:symbol/:timeframe', getOrderBlocksData);

/**
 * GET /api/ai-model/order-block-touches/:symbol/:timeframe
 *
 * Retrieves order block touch events for specified symbol and timeframe
 *
 * Query parameters:
 * - start: ISO 8601 date string (required)
 * - end: ISO 8601 date string (required)
 * - limit: number (optional, default: 100, max: 1000)
 *
 * Example:
 * GET /api/ai-model/order-block-touches/EURUSD/H1?start=2024-01-01T00:00:00Z&end=2024-01-31T23:59:59Z&limit=100
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get('/api/ai-model/order-block-touches/:symbol/:timeframe', getOrderBlockTouchesData);

export default router;
