/**
 * Clusters Controller
 *
 * Handles requests for K-Means cluster data from AI Model database
 */

import { Request, Response, NextFunction } from 'express';
import { getClusterData, Timeframe } from '../database/queries/clusterData';
import { ValidationError } from '../database/errors';
import { randomBytes } from 'crypto';

// Extend Request type to include id
interface RequestWithId extends Request {
  id?: string;
}

/**
 * Generate unique request ID
 */
function generateRequestId(): string {
  return `req_${randomBytes(8).toString('hex')}`;
}

/**
 * Parse and validate query parameters for cluster data
 */
function parseClusterQueryParams(query: any): {
  start?: Date;
  end?: Date;
  limit: number;
} {
  let start: Date | undefined;
  let end: Date | undefined;

  // Parse start date (optional)
  if (query.start) {
    start = new Date(query.start);
    if (isNaN(start.getTime())) {
      throw new ValidationError('Invalid date format for start parameter', {
        field: 'start',
        value: query.start,
      });
    }
  }

  // Parse end date (optional)
  if (query.end) {
    end = new Date(query.end);
    if (isNaN(end.getTime())) {
      throw new ValidationError('Invalid date format for end parameter', {
        field: 'end',
        value: query.end,
      });
    }
  }

  // Parse limit (default 1000)
  let limit = 1000;
  if (query.limit !== undefined) {
    limit = parseInt(query.limit, 10);
    if (isNaN(limit)) {
      throw new ValidationError('Invalid limit parameter - must be a number', {
        field: 'limit',
        value: query.limit,
      });
    }
  }

  return { start, end, limit };
}

/**
 * GET /api/clusters/:symbol/:timeframe
 *
 * Retrieves K-Means cluster data for specified symbol and timeframe
 *
 * @param req Express request with params (symbol, timeframe) and optional query (start, end, limit)
 * @param res Express response
 * @param next Express next function for error handling
 */
export async function getClusters(
  req: RequestWithId,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    // Generate and attach request ID if not present
    if (!req.id) {
      req.id = generateRequestId();
    }

    // Extract path parameters
    const symbol = req.params['symbol'] || '';
    const timeframe = req.params['timeframe'] || '';

    // Validate timeframe
    const validTimeframes: Timeframe[] = [
      'M1',
      'M5',
      'M15',
      'M30',
      'H1',
      'H2',
      'H3',
      'H4',
      'H6',
      'H8',
      'H12',
      'D1',
    ];
    if (!validTimeframes.includes(timeframe as Timeframe)) {
      throw new ValidationError('Invalid timeframe', {
        field: 'timeframe',
        value: timeframe,
        validValues: validTimeframes,
      });
    }

    // Parse query parameters
    const { start, end, limit } = parseClusterQueryParams(req.query);

    // Query database for cluster data
    const data = await getClusterData(symbol, timeframe as Timeframe, start, end, limit);

    // Transform data to match frontend expectations
    const clusterData = {
      clusterInfo: data.clusterInfo
        ? {
            id: data.clusterInfo.id,
            symbol: data.clusterInfo.symbol,
            timeframe: data.clusterInfo.timeframe,
            nClusters: data.clusterInfo.n_clusters,
            windowSize: data.clusterInfo.window_size,
            featureMode: data.clusterInfo.feature_mode,
          }
        : null,
      assignments: data.assignments.map((a) => ({
        timestamp:
          a.timestamp instanceof Date ? a.timestamp.getTime() : new Date(a.timestamp).getTime(),
        clusterId: a.cluster_id,
        isOutlier: a.is_outlier,
        distanceToCentroid: a.distance_to_centroid,
      })),
      performance: data.performance.map((p) => ({
        clusterId: p.cluster_id,
        count: p.count,
        sharpeRatio: p.sharpe_ratio,
        winRate: p.win_rate,
        meanReturn: p.mean_return,
      })),
    };

    // Wrap response with data and metadata to match frontend expectations
    const response = {
      data: clusterData,
      metadata: {
        count: data.assignments.length,
        requestId: req.id || generateRequestId(),
        timestamp: new Date().toISOString(),
      },
    };

    // Send response
    res.json(response);
  } catch (error) {
    // Pass errors to error handling middleware
    next(error);
  }
}
