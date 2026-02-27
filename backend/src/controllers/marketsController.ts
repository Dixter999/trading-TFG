/**
 * Markets Controller
 *
 * Handles requests for market data with technical indicators
 */

import { Request, Response, NextFunction } from 'express';
import {
  getMarketDataWithIndicators,
  getEarliestTimestamp,
  getLatestTimestamp,
  Timeframe,
} from '../database/queries/marketData';
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
 * Start and end are now optional - will be filled with earliest/latest timestamps
 */
async function parseQueryParams(
  query: any,
  symbol: string,
  timeframe: Timeframe
): Promise<{
  start: Date;
  end: Date;
  limit: number;
}> {
  // Parse and validate start date (optional)
  let start: Date;
  if (query.start) {
    start = new Date(query.start);
    if (isNaN(start.getTime())) {
      throw new ValidationError('Invalid date format for start parameter', {
        field: 'start',
        value: query.start,
      });
    }
  } else {
    // Get earliest timestamp from database when start is not provided
    start = await getEarliestTimestamp(symbol, timeframe);
  }

  // Parse and validate end date (optional)
  let end: Date;
  if (query.end) {
    end = new Date(query.end);
    if (isNaN(end.getTime())) {
      throw new ValidationError('Invalid date format for end parameter', {
        field: 'end',
        value: query.end,
      });
    }
  } else {
    // Get latest timestamp from database when end is not provided
    end = await getLatestTimestamp(symbol, timeframe);
  }

  // Parse and validate limit
  let limit = 1000; // default
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
 * GET /api/markets/config
 *
 * Retrieves supported symbols and timeframes for the trading system
 * This allows the frontend to dynamically adapt to backend configuration changes
 *
 * @param req Express request
 * @param res Express response
 * @param next Express next function for error handling
 */
export async function getMarketConfig(
  req: RequestWithId,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    // Generate and attach request ID if not present
    if (!req.id) {
      req.id = generateRequestId();
    }

    // Return supported symbols and timeframes
    const config = {
      symbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'EURCAD', 'EURGBP'] as const,
      timeframes: [
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
      ] as const,
      defaultTimeframe: 'H1' as const,
      defaultSymbol: 'EURUSD' as const,
    };

    // Send config directly as JSON response (not using formatResponse middleware)
    res.json({
      success: true,
      data: config,
    });
  } catch (error) {
    // Pass errors to error handling middleware
    next(error);
  }
}

/**
 * GET /api/markets/:symbol/:timeframe
 *
 * Retrieves market data with technical indicators for specified symbol and timeframe
 *
 * @param req Express request with params (symbol, timeframe) and optional query (start, end, limit)
 *            - start: Optional start date (defaults to earliest available)
 *            - end: Optional end date (defaults to latest available)
 *            - limit: Optional result limit (defaults to 1000)
 * @param res Express response
 * @param next Express next function for error handling
 */
export async function getMarketData(
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

    // Parse and validate query parameters (now async, fetches earliest/latest if needed)
    const { start, end, limit } = await parseQueryParams(req.query, symbol, timeframe as Timeframe);

    // Query database for market data with indicators
    const data = await getMarketDataWithIndicators(
      symbol,
      timeframe as Timeframe,
      start,
      end,
      limit
    );

    // Store data in res.locals for response formatter middleware
    res.locals['data'] = data;

    // Call next() to let response formatter handle the response
    next();
  } catch (error) {
    // Pass errors to error handling middleware
    next(error);
  }
}
