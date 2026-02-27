/**
 * Error Handler Middleware Tests
 * TDD approach: Write failing tests first, then implement
 *
 * Tests cover:
 * - ValidationError → 400 Bad Request
 * - QueryError → 500 Internal Server Error
 * - ConnectionError → 503 Service Unavailable
 * - Generic errors → 500 Internal Server Error
 * - Error response format consistency
 * - Stack trace logging (development vs production)
 */

import { Request, Response, NextFunction } from 'express';
import { ValidationError } from '../../database/errors/ValidationError';
import { QueryError } from '../../database/errors/QueryError';
import { ConnectionError } from '../../database/errors/ConnectionError';
import { errorHandler } from '../../middleware/errorHandler';

describe('errorHandler middleware', () => {
  let mockRequest: Partial<Request>;
  let mockResponse: Partial<Response>;
  let mockNext: NextFunction;
  let jsonMock: jest.Mock;
  let statusMock: jest.Mock;

  beforeEach(() => {
    // Mock Express request
    mockRequest = {
      method: 'GET',
      path: '/api/markets/EURUSD/H1',
      query: {},
    };

    // Mock Express response with chainable methods
    jsonMock = jest.fn();
    statusMock = jest.fn().mockReturnValue({ json: jsonMock });
    mockResponse = {
      status: statusMock,
      json: jsonMock,
    };

    // Mock next function
    mockNext = jest.fn();

    // Clear console spies
    jest.spyOn(console, 'error').mockImplementation();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('ValidationError handling', () => {
    it('should return 400 Bad Request for ValidationError', () => {
      const error = new ValidationError('Invalid timeframe parameter', {
        field: 'timeframe',
        value: 'H2',
        expected: 'H1, H4, or D1',
      });

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(400);
      expect(jsonMock).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Invalid timeframe parameter',
          details: {
            field: 'timeframe',
            value: 'H2',
            expected: 'H1, H4, or D1',
          },
        },
      });
    });

    it('should include multiple validation errors in details', () => {
      const error = new ValidationError('Multiple validation errors', {
        errors: [
          { field: 'start', message: 'Must be valid ISO 8601 date' },
          { field: 'end', message: 'Must be greater than start date' },
        ],
      });

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(400);
      expect(jsonMock).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Multiple validation errors',
          details: {
            errors: [
              { field: 'start', message: 'Must be valid ISO 8601 date' },
              { field: 'end', message: 'Must be greater than start date' },
            ],
          },
        },
      });
    });

    it('should sanitize context to exclude timestamp before sending response', () => {
      const error = new ValidationError('Invalid input', {
        field: 'symbol',
        timestamp: 1234567890,
      });

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      const responseBody = jsonMock.mock.calls[0][0];
      expect(responseBody.error.details).not.toHaveProperty('timestamp');
      expect(responseBody.error.details).toHaveProperty('field', 'symbol');
    });
  });

  describe('QueryError handling', () => {
    it('should return 500 Internal Server Error for QueryError', () => {
      const error = new QueryError('Database query failed', {
        query: 'SELECT * FROM market_data WHERE symbol = $1',
        params: ['EURUSD'],
        error: 'syntax error at or near "SELECTT"',
      });

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(500);
      expect(jsonMock).toHaveBeenCalledWith({
        error: {
          code: 'QUERY_ERROR',
          message: 'An error occurred while processing your request',
        },
      });
    });

    it('should not expose query details in production', () => {
      const originalEnv = process.env['NODE_ENV'];
      process.env['NODE_ENV'] = 'production';

      const error = new QueryError('SQL injection attempt', {
        query: 'SELECT * FROM users WHERE password = $1',
        params: ['sensitive_data'],
      });

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      const responseBody = jsonMock.mock.calls[0][0];
      expect(responseBody.error).not.toHaveProperty('details');
      expect(responseBody.error.message).toBe('An error occurred while processing your request');

      process.env['NODE_ENV'] = originalEnv;
    });

    it('should log query error with full context', () => {
      const consoleErrorSpy = jest.spyOn(console, 'error');

      const error = new QueryError('Query timeout', {
        query: 'SELECT * FROM large_table',
        duration: 30000,
      });

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(consoleErrorSpy).toHaveBeenCalled();
      if (consoleErrorSpy.mock.calls[0]) {
        const loggedMessage = consoleErrorSpy.mock.calls[0][0];
        expect(loggedMessage).toContain('QueryError');
        expect(loggedMessage).toContain('Query timeout');
      }
    });
  });

  describe('ConnectionError handling', () => {
    it('should return 503 Service Unavailable for ConnectionError', () => {
      const error = new ConnectionError('Failed to connect to database', {
        host: 'localhost',
        port: 5432,
        database: 'markets',
        attempt: 3,
      });

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(503);
      expect(jsonMock).toHaveBeenCalledWith({
        error: {
          code: 'CONNECTION_ERROR',
          message: 'Database service temporarily unavailable',
        },
      });
    });

    it('should not expose database connection details in response', () => {
      const error = new ConnectionError('Connection timeout', {
        host: 'localhost',
        port: 5432,
        user: 'admin',
      });

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      const responseBody = jsonMock.mock.calls[0][0];
      expect(responseBody.error).not.toHaveProperty('details');
      expect(responseBody.error.message).not.toContain('localhost');
      expect(responseBody.error.message).not.toContain('admin');
    });
  });

  describe('Generic error handling', () => {
    it('should return 500 for unknown error types', () => {
      const error = new Error('Something unexpected happened');

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(500);
      expect(jsonMock).toHaveBeenCalledWith({
        error: {
          code: 'INTERNAL_ERROR',
          message: 'An unexpected error occurred',
        },
      });
    });

    it('should handle errors without message', () => {
      const error = new Error();

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(500);
      expect(jsonMock).toHaveBeenCalledWith({
        error: {
          code: 'INTERNAL_ERROR',
          message: 'An unexpected error occurred',
        },
      });
    });

    it('should handle non-Error objects', () => {
      const error = { message: 'String error' };

      errorHandler(error as Error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(500);
      expect(jsonMock).toHaveBeenCalled();
    });
  });

  describe('Error logging', () => {
    it('should log all errors with stack trace', () => {
      const consoleErrorSpy = jest.spyOn(console, 'error');

      const error = new Error('Test error');
      error.stack = 'Error: Test error\n    at test.ts:10:5';

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(consoleErrorSpy).toHaveBeenCalled();
      const logCall = consoleErrorSpy.mock.calls[0];
      if (logCall) {
        expect(logCall[0]).toContain('Error');
      }
    });

    it('should include request context in logs', () => {
      const consoleErrorSpy = jest.spyOn(console, 'error');

      mockRequest.method = 'POST';
      Object.defineProperty(mockRequest, 'path', {
        value: '/api/markets/EURUSD/H1',
        writable: true,
      });

      const error = new Error('Request failed');

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(consoleErrorSpy).toHaveBeenCalled();
    });
  });

  describe('Development vs Production behavior', () => {
    it('should include stack trace in development mode', () => {
      const originalEnv = process.env['NODE_ENV'];
      process.env['NODE_ENV'] = 'development';

      const error = new Error('Dev error');
      error.stack = 'Error: Dev error\n    at test:1:1';

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      const consoleErrorSpy = jest.spyOn(console, 'error');
      expect(consoleErrorSpy).toHaveBeenCalled();

      process.env['NODE_ENV'] = originalEnv;
    });

    it('should sanitize errors in production mode', () => {
      const originalEnv = process.env['NODE_ENV'];
      process.env['NODE_ENV'] = 'production';

      const error = new Error('Internal database credentials exposed');

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      const responseBody = jsonMock.mock.calls[0][0];
      expect(responseBody.error.message).toBe('An unexpected error occurred');
      expect(responseBody.error.message).not.toContain('credentials');

      process.env['NODE_ENV'] = originalEnv;
    });
  });

  describe('Error response format consistency', () => {
    it('should always return error object with code and message', () => {
      const errors = [
        new ValidationError('Validation failed'),
        new QueryError('Query failed'),
        new ConnectionError('Connection failed'),
        new Error('Generic error'),
      ];

      errors.forEach((error) => {
        jsonMock.mockClear();
        statusMock.mockClear();

        errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

        const responseBody = jsonMock.mock.calls[0][0];
        expect(responseBody).toHaveProperty('error');
        expect(responseBody.error).toHaveProperty('code');
        expect(responseBody.error).toHaveProperty('message');
        expect(typeof responseBody.error.code).toBe('string');
        expect(typeof responseBody.error.message).toBe('string');
      });
    });

    it('should use consistent error code naming convention', () => {
      const testCases = [
        { error: new ValidationError('test'), expectedCode: 'VALIDATION_ERROR' },
        { error: new QueryError('test'), expectedCode: 'QUERY_ERROR' },
        { error: new ConnectionError('test'), expectedCode: 'CONNECTION_ERROR' },
        { error: new Error('test'), expectedCode: 'INTERNAL_ERROR' },
      ];

      testCases.forEach(({ error, expectedCode }) => {
        jsonMock.mockClear();
        statusMock.mockClear();

        errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

        const responseBody = jsonMock.mock.calls[0][0];
        expect(responseBody.error.code).toBe(expectedCode);
      });
    });
  });

  describe('Edge cases', () => {
    it('should handle error with circular references in context', () => {
      const context: any = { field: 'test' };
      context.circular = context;

      const error = new ValidationError('Circular reference', context);

      expect(() => {
        errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);
      }).not.toThrow();
    });

    it('should handle very long error messages', () => {
      const longMessage = 'a'.repeat(10000);
      const error = new Error(longMessage);

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(500);
      expect(jsonMock).toHaveBeenCalled();
    });

    it('should handle error with undefined context', () => {
      const error = new ValidationError('No context');

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(400);
      expect(jsonMock).toHaveBeenCalled();
    });
  });
});
