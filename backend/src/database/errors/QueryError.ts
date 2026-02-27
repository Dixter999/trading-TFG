/**
 * Query Error Class
 *
 * Error thrown when SQL query execution fails.
 * Used for syntax errors, constraint violations, timeout errors, etc.
 */

import { DatabaseError } from './DatabaseError';

export interface QueryErrorContext {
  query?: string;
  params?: any[];
  duration?: number;
  error?: string;
  code?: string;
  sanitize?: boolean;
  timestamp?: number;
  [key: string]: any;
}

export class QueryError extends DatabaseError {
  constructor(message: string, context?: QueryErrorContext) {
    super(message, context);
    this.name = 'QueryError';

    // Maintains proper stack trace for where our error was thrown (only available on V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }
}
