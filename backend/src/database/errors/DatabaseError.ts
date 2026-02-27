/**
 * Base Database Error Class
 *
 * Base error class for all database-related errors with context and logging support.
 * Automatically adds timestamps to context for debugging and audit trails.
 */

export interface ErrorContext {
  timestamp?: number;
  [key: string]: any;
}

export class DatabaseError extends Error {
  public readonly context?: ErrorContext;

  constructor(message: string, context?: Record<string, any>) {
    super(message);
    this.name = 'DatabaseError';

    // Add timestamp to context if context is provided
    if (context) {
      this.context = {
        ...context,
        timestamp: Date.now(),
      };
    }

    // Maintains proper stack trace for where our error was thrown (only available on V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Serialize error to JSON for structured logging
   */
  toJSON(): Record<string, any> {
    return {
      name: this.name,
      message: this.message,
      context: this.context,
      stack: this.stack,
    };
  }

  /**
   * Convert error to string representation
   */
  override toString(): string {
    let str = `${this.name}: ${this.message}`;
    if (this.context) {
      str += `\nContext: ${JSON.stringify(this.context, null, 2)}`;
    }
    return str;
  }
}
