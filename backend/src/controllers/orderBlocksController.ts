/**
 * Order Blocks Controller
 *
 * Handles requests for SMC order blocks and touch events
 */

import { Request, Response, NextFunction } from 'express';
import { getOrderBlocks, getOrderBlockTouches, Timeframe } from '../database/queries/orderBlocks';
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
 * Parse and validate query parameters
 */
function parseQueryParams(query: any): {
  start: Date;
  end: Date;
  limit: number;
} {
  // Parse and validate start date (required)
  const start = new Date(query.start);
  if (isNaN(start.getTime())) {
    throw new ValidationError('Invalid date format for start parameter', {
      field: 'start',
      value: query.start,
    });
  }

  // Parse and validate end date (required)
  const end = new Date(query.end);
  if (isNaN(end.getTime())) {
    throw new ValidationError('Invalid date format for end parameter', {
      field: 'end',
      value: query.end,
    });
  }

  // Parse and validate limit (optional, default: 100)
  let limit = 100;
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
 * GET /api/ai-model/order-blocks/:symbol/:timeframe
 *
 * Retrieves order blocks for specified symbol and timeframe
 *
 * @param req Express request with params (symbol, timeframe) and query (start, end, limit)
 *            - start: Start date (required, ISO 8601)
 *            - end: End date (required, ISO 8601)
 *            - limit: Result limit (optional, default: 100, max: 1000)
 * @param res Express response
 * @param next Express next function for error handling
 */
export async function getOrderBlocksData(
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

    // Parse and validate query parameters
    const { start, end, limit } = parseQueryParams(req.query);

    // Query database for order blocks
    const data = await getOrderBlocks(symbol, timeframe as Timeframe, start, end, limit);

    // Send response directly (order blocks have their own structure, not OHLCV)
    res.json({
      data,
      metadata: {
        count: data.length,
        timestamp: new Date().toISOString(),
        requestId: req.id,
      },
    });
  } catch (error) {
    // Pass errors to error handling middleware
    next(error);
  }
}

/**
 * GET /api/ai-model/order-block-touches/:symbol/:timeframe
 *
 * Retrieves order block touch events for specified symbol and timeframe
 *
 * @param req Express request with params (symbol, timeframe) and query (start, end, limit)
 *            - start: Start date (required, ISO 8601)
 *            - end: End date (required, ISO 8601)
 *            - limit: Result limit (optional, default: 100, max: 1000)
 * @param res Express response
 * @param next Express next function for error handling
 */
export async function getOrderBlockTouchesData(
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

    // Parse and validate query parameters
    const { start, end, limit } = parseQueryParams(req.query);

    // Query database for order block touches
    const data = await getOrderBlockTouches(symbol, timeframe as Timeframe, start, end, limit);

    // Send response directly (order block touches have their own structure, not OHLCV)
    res.json({
      data,
      metadata: {
        count: data.length,
        timestamp: new Date().toISOString(),
        requestId: req.id,
      },
    });
  } catch (error) {
    // Pass errors to error handling middleware
    next(error);
  }
}
