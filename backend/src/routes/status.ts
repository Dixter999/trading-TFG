/**
 * Service Status Routes (TFG)
 *
 * Provides endpoints to check database connectivity status.
 * Simplified from production — no Docker container management.
 */

import { Router, Request, Response } from 'express';
import { getMarketsPool, getAiModelPool } from '../database/connection';

const router = Router();

/**
 * Check if data exists in markets database
 */
async function checkUpdaterStatus(): Promise<{ status: string; lastUpdate: Date | null }> {
  try {
    const pool = getMarketsPool();
    const result = await pool.query(`
      SELECT MAX(rate_time) as last_update
      FROM eurusd_h1_rates
    `);

    const lastUpdate = result.rows[0]?.last_update;
    if (!lastUpdate) {
      return { status: 'no_data', lastUpdate: null };
    }

    return { status: 'active', lastUpdate: new Date(lastUpdate) };
  } catch (error) {
    return { status: 'error', lastUpdate: null };
  }
}

/**
 * Check if indicators are present in ai_model database
 */
async function checkIndicatorStatus(): Promise<{
  status: string;
  lastUpdate: Date | null;
  count: number;
}> {
  try {
    const pool = getAiModelPool();
    const result = await pool.query(`
      SELECT MAX(timestamp) as last_update, COUNT(*) as total_count
      FROM technical_indicators
      WHERE symbol = 'EURUSD'
    `);

    const lastUpdate = result.rows[0]?.last_update;
    const count = parseInt(result.rows[0]?.total_count || '0');

    if (count === 0) {
      return { status: 'no_data', lastUpdate: null, count: 0 };
    }

    return { status: 'active', lastUpdate: new Date(lastUpdate), count };
  } catch (error) {
    return { status: 'error', lastUpdate: null, count: 0 };
  }
}

/**
 * GET /api/status/services
 * Returns data pipeline status (database connectivity in TFG)
 */
router.get('/api/status/services', async (_req: Request, res: Response) => {
  try {
    const [marketData, indicators] = await Promise.all([
      checkUpdaterStatus(),
      checkIndicatorStatus(),
    ]);

    res.json({
      services: {
        syncer: {
          name: 'Market Data',
          description: 'CSV → PostgreSQL (markets)',
          ...marketData,
        },
        indicators: {
          name: 'Technical Indicators',
          description: 'Markets DB → Technical Indicators',
          ...indicators,
        },
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    res.status(500).json({
      error: {
        code: 'SERVICE_STATUS_ERROR',
        message: 'Failed to retrieve service status',
      },
    });
  }
});

export default router;
