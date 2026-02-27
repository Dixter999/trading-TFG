import { Request, Response, NextFunction } from 'express';

// Extend Request type to include optional id field
interface RequestWithId extends Request {
  id?: string;
}

// Define the database row structure
interface DatabaseRow {
  rate_time: number; // Unix timestamp (seconds since epoch)
  open: string | number;
  high: string | number;
  low: string | number;
  close: string | number;
  volume: number;
  [key: string]: string | number | null; // Allow for indicator fields
}

// Define the formatted response structure
interface FormattedData {
  timestamp: number; // Unix timestamp (milliseconds since epoch)
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  indicators: Record<string, number | null>;
}

interface ApiResponse {
  data: FormattedData[];
  metadata: {
    count: number;
    timestamp: string;
    requestId: string | undefined;
  };
}

// List of OHLCV fields that should not be included in indicators
const OHLCV_FIELDS = ['rate_time', 'open', 'high', 'low', 'close', 'volume'];

/**
 * Convert string numeric values to numbers
 */
function toNumber(value: string | number | null): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === 'number') {
    return value;
  }
  const parsed = parseFloat(value);
  return isNaN(parsed) ? null : parsed;
}

/**
 * Transform database row to API response format
 */
function transformRow(row: DatabaseRow): FormattedData {
  // Check if row already has indicators field (from query with merged indicators)
  let indicators: Record<string, number | null> = {};

  if ('indicators' in row && typeof row['indicators'] === 'object' && row['indicators'] !== null) {
    // Row already has indicators field from query - use it directly
    indicators = row['indicators'] as Record<string, number | null>;
  } else {
    // Extract indicator fields (anything not in OHLCV_FIELDS)
    for (const [key, value] of Object.entries(row)) {
      if (!OHLCV_FIELDS.includes(key) && key !== 'indicators') {
        indicators[key] = toNumber(value);
      }
    }
  }

  // Convert Unix seconds to milliseconds for JavaScript Date compatibility
  const rateTimeSeconds =
    typeof row.rate_time === 'string' ? parseInt(row.rate_time, 10) : row.rate_time;

  return {
    timestamp: rateTimeSeconds * 1000, // Convert seconds to milliseconds
    open: toNumber(row.open) as number,
    high: toNumber(row.high) as number,
    low: toNumber(row.low) as number,
    close: toNumber(row.close) as number,
    volume: row.volume,
    indicators,
  };
}

/**
 * Response formatter middleware
 * Transforms database results to match PRD specification
 * Adds metadata (count, timestamp, requestId)
 */
export function formatResponse(req: RequestWithId, res: Response, next: NextFunction): void {
  // Check if there's data to format in res.locals
  if (!res.locals['data']) {
    return next();
  }

  const dbResults = res.locals['data'] as DatabaseRow[];
  const formattedData = dbResults.map(transformRow);

  const response: ApiResponse = {
    data: formattedData,
    metadata: {
      count: formattedData.length,
      timestamp: new Date().toISOString(),
      requestId: req.id,
    },
  };

  res.json(response);
}
