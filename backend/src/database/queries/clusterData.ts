/**
 * Cluster Data Queries
 *
 * Queries for fetching K-Means cluster assignments and performance data
 * from the AI Model database.
 */

import { getAiModelPool } from '../connection';
import { logger } from '../../config/logging';

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
 * Cluster assignment record from database
 */
export interface ClusterAssignmentRow {
  timestamp: Date;
  cluster_id: number;
  is_outlier: boolean;
  distance_to_centroid: number | null;
}

/**
 * Pattern cluster config from database
 */
export interface PatternClusterRow {
  id: number;
  symbol: string;
  timeframe: string;
  n_clusters: number;
  window_size: number;
  feature_mode: string;
}

/**
 * Cluster performance record from database
 */
export interface ClusterPerformanceRow {
  cluster_id: number;
  count: number;
  sharpe_ratio: number | null;
  win_rate: number | null;
  mean_return: number | null;
}

/**
 * Get pattern cluster configuration for a symbol/timeframe
 */
export async function getPatternClusterConfig(
  symbol: string,
  timeframe: Timeframe
): Promise<PatternClusterRow | null> {
  const pool = getAiModelPool();

  const query = `
    SELECT id, symbol, timeframe, n_clusters, window_size, feature_mode
    FROM pattern_clusters
    WHERE symbol = $1 AND timeframe = $2
    LIMIT 1
  `;

  try {
    const result = await pool.query(query, [symbol.toUpperCase(), timeframe]);

    if (result.rows.length === 0) {
      logger.warn('No pattern cluster config found', { symbol, timeframe });
      return null;
    }

    return result.rows[0] as PatternClusterRow;
  } catch (error) {
    logger.error('Error fetching pattern cluster config', {
      error: error instanceof Error ? error.message : String(error),
      symbol,
      timeframe,
    });
    throw error;
  }
}

/**
 * Get cluster assignments for a symbol/timeframe within date range
 */
export async function getClusterAssignments(
  clusterConfigId: number,
  start?: Date,
  end?: Date,
  limit: number = 1000
): Promise<ClusterAssignmentRow[]> {
  const pool = getAiModelPool();

  let query = `
    SELECT timestamp, cluster_id, is_outlier, distance_to_centroid
    FROM cluster_assignments
    WHERE cluster_config_id = $1
  `;

  const params: (number | Date | string)[] = [clusterConfigId];
  let paramIndex = 2;

  if (start) {
    query += ` AND timestamp >= $${paramIndex}`;
    params.push(start);
    paramIndex++;
  }

  if (end) {
    query += ` AND timestamp <= $${paramIndex}`;
    params.push(end);
    paramIndex++;
  }

  query += ` ORDER BY timestamp DESC LIMIT $${paramIndex}`;
  params.push(limit);

  try {
    const result = await pool.query(query, params);

    logger.info('Fetched cluster assignments', {
      clusterConfigId,
      count: result.rows.length,
      start: start?.toISOString(),
      end: end?.toISOString(),
    });

    return result.rows as ClusterAssignmentRow[];
  } catch (error) {
    logger.error('Error fetching cluster assignments', {
      error: error instanceof Error ? error.message : String(error),
      clusterConfigId,
    });
    throw error;
  }
}

/**
 * Get cluster performance metrics for a cluster config
 */
export async function getClusterPerformance(
  clusterConfigId: number
): Promise<ClusterPerformanceRow[]> {
  const pool = getAiModelPool();

  const query = `
    SELECT cluster_id, count, sharpe_ratio, win_rate, mean_return
    FROM cluster_performance
    WHERE cluster_config_id = $1
    ORDER BY cluster_id
  `;

  try {
    const result = await pool.query(query, [clusterConfigId]);

    logger.info('Fetched cluster performance', {
      clusterConfigId,
      count: result.rows.length,
    });

    return result.rows as ClusterPerformanceRow[];
  } catch (error) {
    logger.error('Error fetching cluster performance', {
      error: error instanceof Error ? error.message : String(error),
      clusterConfigId,
    });
    throw error;
  }
}

/**
 * Get all cluster data (config, assignments, performance) for a symbol/timeframe
 */
export async function getClusterData(
  symbol: string,
  timeframe: Timeframe,
  start?: Date,
  end?: Date,
  limit: number = 1000
): Promise<{
  clusterInfo: PatternClusterRow | null;
  assignments: ClusterAssignmentRow[];
  performance: ClusterPerformanceRow[];
}> {
  // First get the cluster config
  const clusterInfo = await getPatternClusterConfig(symbol, timeframe);

  if (!clusterInfo) {
    return {
      clusterInfo: null,
      assignments: [],
      performance: [],
    };
  }

  // Fetch assignments and performance in parallel
  const [assignments, performance] = await Promise.all([
    getClusterAssignments(clusterInfo.id, start, end, limit),
    getClusterPerformance(clusterInfo.id),
  ]);

  return {
    clusterInfo,
    assignments,
    performance,
  };
}
