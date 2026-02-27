/**
 * Connection Error Class
 *
 * Error thrown when database connection fails or encounters issues.
 * Used for connection timeout, authentication failures, network issues, etc.
 */

import { DatabaseError } from './DatabaseError';

export interface ConnectionErrorContext {
  host?: string;
  port?: number;
  database?: string;
  user?: string;
  attempt?: number;
  timestamp?: number;
  [key: string]: any;
}

export class ConnectionError extends DatabaseError {
  constructor(message: string, context?: ConnectionErrorContext) {
    super(message, context);
    this.name = 'ConnectionError';

    // Maintains proper stack trace for where our error was thrown (only available on V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }
}
