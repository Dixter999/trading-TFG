/**
 * Clusters Routes
 *
 * Defines API routes for K-Means cluster data endpoints
 */

import { Router } from 'express';
import { getClusters } from '../controllers/clustersController';

const router = Router();

/**
 * GET /api/clusters/:symbol/:timeframe
 *
 * Query parameters:
 * - start: ISO 8601 date string (optional)
 * - end: ISO 8601 date string (optional)
 * - limit: number (optional, default: 1000)
 *
 * Example:
 * GET /api/clusters/EURUSD/H1?start=2024-01-01T00:00:00Z&end=2024-01-31T23:59:59Z&limit=1000
 *
 * Returns:
 * - clusterInfo: Configuration for the cluster model (nClusters, windowSize, featureMode)
 * - assignments: Array of timestamp -> cluster_id mappings
 * - performance: Array of cluster performance metrics (sharpeRatio, winRate, meanReturn)
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get('/api/clusters/:symbol/:timeframe', getClusters);

export default router;
