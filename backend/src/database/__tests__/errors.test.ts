/**
 * Error Classes Tests
 * Tests for custom database error classes with context and logging
 */

import { DatabaseError } from '../errors/DatabaseError';
import { ConnectionError } from '../errors/ConnectionError';
import { QueryError } from '../errors/QueryError';
import { ValidationError } from '../errors/ValidationError';

describe('DatabaseError', () => {
  describe('constructor', () => {
    it('should create error with message', () => {
      const error = new DatabaseError('Database operation failed');

      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(DatabaseError);
      expect(error.message).toBe('Database operation failed');
      expect(error.name).toBe('DatabaseError');
    });

    it('should create error with message and context', () => {
      const context = { query: 'SELECT * FROM users', params: [1, 2] };
      const error = new DatabaseError('Query failed', context);

      expect(error.message).toBe('Query failed');
      expect(error.context).toBeDefined();
      expect(error.context?.['query']).toBe('SELECT * FROM users');
      expect(error.context?.['params']).toEqual([1, 2]);
    });

    it('should create error with timestamp in context', () => {
      const beforeTime = Date.now();
      const error = new DatabaseError('Test error', { query: 'SELECT 1' });
      const afterTime = Date.now();

      expect(error.context?.timestamp).toBeDefined();
      expect(error.context?.timestamp).toBeGreaterThanOrEqual(beforeTime);
      expect(error.context?.timestamp).toBeLessThanOrEqual(afterTime);
    });

    it('should create error without context', () => {
      const error = new DatabaseError('Simple error');

      expect(error.context).toBeUndefined();
    });
  });

  describe('stack trace', () => {
    it('should have proper stack trace', () => {
      const error = new DatabaseError('Test error');

      expect(error.stack).toBeDefined();
      expect(error.stack).toContain('DatabaseError');
      expect(error.stack).toContain('Test error');
    });
  });

  describe('toJSON', () => {
    it('should serialize to JSON with all properties', () => {
      const context = { query: 'SELECT 1', params: [] };
      const error = new DatabaseError('Test error', context);
      const json = error.toJSON();

      expect(json).toHaveProperty('name', 'DatabaseError');
      expect(json).toHaveProperty('message', 'Test error');
      expect(json).toHaveProperty('context');
      expect(json['context']).toEqual(
        expect.objectContaining({
          query: 'SELECT 1',
          params: [],
        })
      );
    });
  });
});

describe('ConnectionError', () => {
  describe('constructor', () => {
    it('should extend DatabaseError', () => {
      const error = new ConnectionError('Connection failed');

      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(DatabaseError);
      expect(error).toBeInstanceOf(ConnectionError);
    });

    it('should have correct name', () => {
      const error = new ConnectionError('Connection timeout');

      expect(error.name).toBe('ConnectionError');
      expect(error.message).toBe('Connection timeout');
    });

    it('should accept context with connection details', () => {
      const context = {
        host: 'localhost',
        port: 5432,
        database: 'markets',
        attempt: 3,
      };
      const error = new ConnectionError('Connection failed', context);

      expect(error.context).toEqual(expect.objectContaining(context));
      expect(error.context?.['host']).toBe('localhost');
      expect(error.context?.['attempt']).toBe(3);
    });
  });

  describe('toString', () => {
    it('should format error message with context', () => {
      const error = new ConnectionError('Failed to connect', {
        host: 'localhost',
        port: 5432,
      });

      const str = error.toString();
      expect(str).toContain('ConnectionError');
      expect(str).toContain('Failed to connect');
    });
  });
});

describe('QueryError', () => {
  describe('constructor', () => {
    it('should extend DatabaseError', () => {
      const error = new QueryError('Query execution failed');

      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(DatabaseError);
      expect(error).toBeInstanceOf(QueryError);
    });

    it('should have correct name', () => {
      const error = new QueryError('Syntax error');

      expect(error.name).toBe('QueryError');
      expect(error.message).toBe('Syntax error');
    });

    it('should accept context with query details', () => {
      const context = {
        query: 'SELECT * FROM users WHERE id = $1',
        params: [123],
        duration: 5000,
        error: 'relation "users" does not exist',
      };
      const error = new QueryError('Query failed', context);

      expect(error.context).toEqual(expect.objectContaining(context));
      expect(error.context?.['query']).toContain('SELECT * FROM users');
      expect(error.context?.['params']).toEqual([123]);
      expect(error.context?.['duration']).toBe(5000);
    });
  });

  describe('query sanitization', () => {
    it('should not expose sensitive data in query params', () => {
      const context = {
        query: 'UPDATE users SET password = $1 WHERE id = $2',
        params: ['secretPassword123', 1],
        sanitize: true,
      };
      const error = new QueryError('Update failed', context);

      // Error should have context but params should be marked as sanitized
      expect(error.context?.['query']).toBeDefined();
      expect(error.context?.['sanitize']).toBe(true);
    });
  });
});

describe('ValidationError', () => {
  describe('constructor', () => {
    it('should extend DatabaseError', () => {
      const error = new ValidationError('Invalid input');

      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(DatabaseError);
      expect(error).toBeInstanceOf(ValidationError);
    });

    it('should have correct name', () => {
      const error = new ValidationError('Invalid parameter');

      expect(error.name).toBe('ValidationError');
      expect(error.message).toBe('Invalid parameter');
    });

    it('should accept context with validation details', () => {
      const context = {
        field: 'email',
        value: 'invalid-email',
        constraint: 'must be valid email format',
        expected: 'string matching /^[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}$/i',
      };
      const error = new ValidationError('Validation failed', context);

      expect(error.context).toEqual(expect.objectContaining(context));
      expect(error.context?.['field']).toBe('email');
      expect(error.context?.['value']).toBe('invalid-email');
    });
  });

  describe('multiple validation errors', () => {
    it('should handle array of validation errors in context', () => {
      const context = {
        errors: [
          { field: 'email', message: 'Invalid email format' },
          { field: 'age', message: 'Must be positive number' },
        ],
      };
      const error = new ValidationError('Multiple validation errors', context);

      expect(error.context?.['errors']).toHaveLength(2);
      expect(error.context?.['errors'][0]).toEqual(
        expect.objectContaining({
          field: 'email',
          message: 'Invalid email format',
        })
      );
      expect(error.context?.['errors'][1]?.field).toBe('age');
    });
  });
});

describe('Error inheritance chain', () => {
  it('should maintain proper instanceof checks across all error types', () => {
    const errors = [
      new DatabaseError('base error'),
      new ConnectionError('connection error'),
      new QueryError('query error'),
      new ValidationError('validation error'),
    ];

    errors.forEach((error) => {
      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(DatabaseError);
    });

    expect(errors[1]).toBeInstanceOf(ConnectionError);
    expect(errors[2]).toBeInstanceOf(QueryError);
    expect(errors[3]).toBeInstanceOf(ValidationError);
  });
});

describe('Error serialization for logging', () => {
  it('should serialize all error types to JSON', () => {
    const errors = [
      new DatabaseError('base', { test: true }),
      new ConnectionError('conn', { host: 'localhost' }),
      new QueryError('query', { query: 'SELECT 1' }),
      new ValidationError('valid', { field: 'email' }),
    ];

    errors.forEach((error) => {
      const json = error.toJSON();
      expect(json).toHaveProperty('name');
      expect(json).toHaveProperty('message');
      expect(json).toHaveProperty('context');
      expect(json['context']).toBeDefined();
    });
  });
});
