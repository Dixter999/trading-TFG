import { Request, Response, NextFunction } from 'express';
import { z } from 'zod';

// Validation constants - exported for reusability in other modules
// Stale symbol cleanup (2026-02-24): GOLD, SILVER, BTCUSD removed - FOREX ONLY system
export const VALID_SYMBOLS = [
  'EURUSD',
  'GBPUSD',
  'USDJPY',
  'EURJPY',
  'USDCAD',
  'USDCHF',
  'EURCAD',
  'EURGBP',
] as const;
export const VALID_TIMEFRAMES = [
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
] as const;
export const MIN_LIMIT = 1;
export const MAX_LIMIT = 5000;
export const DEFAULT_LIMIT = 1000;

// Zod schema for route parameters
const paramsSchema = z.object({
  symbol: z.enum(VALID_SYMBOLS, {
    message: `Invalid symbol. Must be one of: ${VALID_SYMBOLS.join(', ')}`,
  }),
  timeframe: z.enum(VALID_TIMEFRAMES, {
    message: `Invalid timeframe. Must be one of: ${VALID_TIMEFRAMES.join(', ')}`,
  }),
});

// Custom date validator that checks for ISO 8601 format and not in future
const dateSchema = z
  .string()
  .datetime({ message: 'Must be a valid ISO 8601 date' })
  .refine(
    (dateStr) => {
      const date = new Date(dateStr);
      const now = new Date();
      return date <= now;
    },
    { message: 'Date must not be in the future' }
  )
  .optional();

// Zod schema for query parameters
const querySchema = z
  .object({
    start: dateSchema,
    end: dateSchema,
    limit: z
      .string()
      .optional()
      .transform((val) => (val ? parseInt(val, 10) : DEFAULT_LIMIT))
      .refine((val) => !isNaN(val) && val >= MIN_LIMIT && val <= MAX_LIMIT, {
        message: `Limit must be a number between ${MIN_LIMIT} and ${MAX_LIMIT}`,
      }),
  })
  .refine(
    (data) => {
      // If both start and end are provided, validate that end >= start
      if (data.start && data.end) {
        const startDate = new Date(data.start);
        const endDate = new Date(data.end);
        return endDate >= startDate;
      }
      return true;
    },
    {
      message: 'end date must be greater than or equal to start date',
      path: ['end'],
    }
  );

/**
 * Helper function to build error response details for validation errors
 */
function buildErrorDetails(
  field: string | number | symbol | undefined,
  invalidValue: string | undefined
): Record<string, unknown> {
  if (field === 'limit') {
    return { min: MIN_LIMIT, max: MAX_LIMIT };
  }
  if (field === 'symbol') {
    return { symbol: invalidValue, valid: VALID_SYMBOLS };
  }
  if (field === 'timeframe') {
    return { timeframe: invalidValue, valid: VALID_TIMEFRAMES };
  }
  return {};
}

/**
 * Helper function to send validation error response
 */
function sendValidationError(
  res: Response,
  message: string,
  details: Record<string, unknown>
): void {
  res.status(400).json({
    error: {
      code: 'VALIDATION_ERROR',
      message,
      details,
    },
  });
}

/**
 * Validation middleware for market data requests
 * Validates route parameters (symbol, timeframe) and query parameters (start, end, limit)
 */
export function validateMarketDataRequest(req: Request, res: Response, next: NextFunction): void {
  try {
    // Validate route parameters
    const paramsResult = paramsSchema.safeParse(req.params);
    if (!paramsResult.success) {
      const firstError = paramsResult.error.issues[0];
      const field = firstError?.path[0];
      const invalidValue = req.params[field as string];
      const message = firstError?.message || 'Invalid parameters';
      const details = buildErrorDetails(field, invalidValue);

      sendValidationError(res, message, details);
      return;
    }

    // Validate query parameters
    const queryResult = querySchema.safeParse(req.query);
    if (!queryResult.success) {
      const firstError = queryResult.error.issues[0];
      const field = firstError?.path[0];
      const message = firstError?.message || 'Invalid query parameters';
      const details = buildErrorDetails(field, undefined);

      sendValidationError(res, message, details);
      return;
    }

    // All validations passed
    next();
  } catch (error) {
    // Handle unexpected errors
    sendValidationError(res, 'Validation failed', {});
  }
}
