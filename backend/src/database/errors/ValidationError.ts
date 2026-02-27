/**
 * Validation Error Class
 *
 * Error thrown when input validation fails before query execution.
 * Used for parameter type checking, format validation, constraint checking, etc.
 */

import { DatabaseError } from './DatabaseError';

export interface ValidationFieldError {
  field: string;
  message: string;
  value?: any;
  constraint?: string;
}

export interface ValidationErrorContext {
  field?: string;
  value?: any;
  constraint?: string;
  expected?: string;
  errors?: ValidationFieldError[];
  timestamp?: number;
  [key: string]: any;
}

export class ValidationError extends DatabaseError {
  constructor(message: string, context?: ValidationErrorContext) {
    super(message, context);
    this.name = 'ValidationError';

    // Maintains proper stack trace for where our error was thrown (only available on V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }
}
