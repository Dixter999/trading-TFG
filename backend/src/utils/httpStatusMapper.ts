/**
 * HTTP Status Mapper Utility
 *
 * Maps database errors to appropriate HTTP status codes
 * Provides consistent error code naming and user-friendly messages
 */

import { DatabaseError } from '../database/errors/DatabaseError';
import { ValidationError } from '../database/errors/ValidationError';
import { QueryError } from '../database/errors/QueryError';
import { ConnectionError } from '../database/errors/ConnectionError';

/**
 * Error response structure
 */
export interface ErrorResponse {
  code: string;
  message: string;
  details?: Record<string, any>;
}

/**
 * HTTP status mapping result
 */
export interface HttpErrorMapping {
  statusCode: number;
  response: ErrorResponse;
}

/**
 * Map database error to HTTP status code and error response
 *
 * @param error - Error instance to map
 * @returns HTTP status code and formatted error response
 */
export function mapErrorToHttpStatus(error: Error): HttpErrorMapping {
  // ValidationError → 400 Bad Request
  if (error instanceof ValidationError) {
    return {
      statusCode: 400,
      response: {
        code: 'VALIDATION_ERROR',
        message: error.message,
        details: sanitizeContext(error.context),
      },
    };
  }

  // ConnectionError → 503 Service Unavailable
  if (error instanceof ConnectionError) {
    return {
      statusCode: 503,
      response: {
        code: 'CONNECTION_ERROR',
        message: 'Database service temporarily unavailable',
        // Don't expose connection details in response
      },
    };
  }

  // QueryError → 500 Internal Server Error
  if (error instanceof QueryError) {
    return {
      statusCode: 500,
      response: {
        code: 'QUERY_ERROR',
        message: 'An error occurred while processing your request',
        // Don't expose query details in response (security)
      },
    };
  }

  // Generic DatabaseError → 500 Internal Server Error
  if (error instanceof DatabaseError) {
    return {
      statusCode: 500,
      response: {
        code: 'DATABASE_ERROR',
        message: 'An error occurred while accessing the database',
      },
    };
  }

  // Unknown error → 500 Internal Server Error
  return {
    statusCode: 500,
    response: {
      code: 'INTERNAL_ERROR',
      message: 'An unexpected error occurred',
    },
  };
}

/**
 * Sanitize error context before sending to client
 * Removes sensitive fields like timestamps, internal IDs, etc.
 *
 * @param context - Error context object
 * @returns Sanitized context safe for client consumption
 */
function sanitizeContext(context?: Record<string, any>): Record<string, any> | undefined {
  if (!context) {
    return undefined;
  }

  // Create shallow copy
  const sanitized: Record<string, any> = {};

  // Fields to exclude from client response
  const excludedFields = ['timestamp', 'stack', 'query', 'params', 'host', 'port', 'user'];

  // Copy only safe fields
  for (const [key, value] of Object.entries(context)) {
    if (!excludedFields.includes(key)) {
      sanitized[key] = value;
    }
  }

  return Object.keys(sanitized).length > 0 ? sanitized : undefined;
}

/**
 * Get user-friendly error message based on error type
 * Used for production environments where detailed errors should be hidden
 *
 * @param error - Error instance
 * @returns User-friendly message
 */
export function getUserFriendlyMessage(error: Error): string {
  if (error instanceof ValidationError) {
    return error.message; // Validation messages are safe to expose
  }

  if (error instanceof ConnectionError) {
    return 'Database service temporarily unavailable';
  }

  if (error instanceof QueryError) {
    return 'An error occurred while processing your request';
  }

  if (error instanceof DatabaseError) {
    return 'An error occurred while accessing the database';
  }

  return 'An unexpected error occurred';
}

/**
 * Check if error details should be exposed based on environment
 *
 * @returns true if in development mode, false otherwise
 */
export function shouldExposeErrorDetails(): boolean {
  return process.env['NODE_ENV'] !== 'production';
}
