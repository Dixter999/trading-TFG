/**
 * Global Error Handler Middleware
 *
 * Express error handler middleware that:
 * - Maps database errors to appropriate HTTP status codes
 * - Formats error responses consistently
 * - Logs errors with stack traces
 * - Sanitizes errors in production (hides internal details)
 * - Returns user-friendly error messages
 *
 * Usage:
 * app.use(errorHandler); // Must be last middleware
 */

import { Request, Response, NextFunction } from 'express';
import { DatabaseError } from '../database/errors/DatabaseError';
import { mapErrorToHttpStatus } from '../utils/httpStatusMapper';
import { logger } from '../config/logging';

/**
 * Global error handler middleware
 * Handles all errors thrown in the application and formats responses
 *
 * @param error - Error object thrown by application or middleware
 * @param req - Express request object
 * @param res - Express response object
 * @param next - Express next function (unused but required by Express)
 */
export function errorHandler(error: Error, req: Request, res: Response, _next: NextFunction): void {
  // Log error with full context for debugging
  logError(error, req);

  // Map error to HTTP status and response format
  const { statusCode, response } = mapErrorToHttpStatus(error);

  // Send formatted error response
  res.status(statusCode).json({ error: response });
}

/**
 * Log error with request context and stack trace
 *
 * @param error - Error to log
 * @param req - Request context
 */
function logError(error: Error, req: Request): void {
  const isProduction = process.env['NODE_ENV'] === 'production';

  // Build structured log metadata
  const logData: Record<string, any> = {
    error: {
      name: error.name,
      message: error.message,
    },
    request: {
      method: req.method,
      path: req.path,
      query: req.query,
    },
  };

  // Include stack trace in development
  if (!isProduction && error.stack) {
    logData['error'].stack = error.stack;
  }

  // Include error context for DatabaseError types
  if (error instanceof DatabaseError && error.context) {
    logData['error'].context = error.context;
  }

  // Log using centralized structured logger
  logger.error(`Request error: ${error.message}`, logData);
}

/**
 * Default export for convenience
 */
export default errorHandler;
