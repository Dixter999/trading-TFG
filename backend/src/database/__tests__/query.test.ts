/**
 * Query Utilities Tests
 * Tests for executeQuery and executeTransaction with mocked pg.Pool
 */

import { Pool, PoolClient, QueryResult } from 'pg';
import { executeQuery, executeTransaction } from '../utils/query';
import { QueryError, ConnectionError } from '../errors';

// Mock pg module
jest.mock('pg', () => {
  const mPool = {
    query: jest.fn(),
    connect: jest.fn(),
    end: jest.fn(),
  };
  return { Pool: jest.fn(() => mPool) };
});

describe('executeQuery', () => {
  let pool: Pool;
  let mockQuery: jest.Mock;

  beforeEach(() => {
    pool = new Pool();
    mockQuery = (pool.query as jest.Mock).mockReset();
  });

  describe('successful queries', () => {
    it('should execute simple SELECT query and return rows', async () => {
      const mockResult: QueryResult = {
        rows: [
          { id: 1, name: 'Alice' },
          { id: 2, name: 'Bob' },
        ],
        command: 'SELECT',
        rowCount: 2,
        oid: 0,
        fields: [],
      };
      mockQuery.mockResolvedValue(mockResult);

      const result = await executeQuery(pool, 'SELECT * FROM users', []);

      expect(result).toEqual(mockResult.rows);
      expect(mockQuery).toHaveBeenCalledWith('SELECT * FROM users', []);
      expect(mockQuery).toHaveBeenCalledTimes(1);
    });

    it('should execute parameterized query', async () => {
      const mockResult: QueryResult = {
        rows: [{ id: 1, name: 'Alice', email: 'alice@example.com' }],
        command: 'SELECT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockQuery.mockResolvedValue(mockResult);

      const result = await executeQuery(pool, 'SELECT * FROM users WHERE id = $1', [1]);

      expect(result).toEqual(mockResult.rows);
      expect(mockQuery).toHaveBeenCalledWith('SELECT * FROM users WHERE id = $1', [1]);
    });

    it('should return empty array for no results', async () => {
      const mockResult: QueryResult = {
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      };
      mockQuery.mockResolvedValue(mockResult);

      const result = await executeQuery(pool, 'SELECT * FROM users WHERE id = $1', [999]);

      expect(result).toEqual([]);
      expect(result).toHaveLength(0);
    });

    it('should execute INSERT query and return inserted row', async () => {
      const mockResult: QueryResult = {
        rows: [{ id: 3, name: 'Charlie', email: 'charlie@example.com' }],
        command: 'INSERT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockQuery.mockResolvedValue(mockResult);

      const result = await executeQuery(
        pool,
        'INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *',
        ['Charlie', 'charlie@example.com']
      );

      expect(result).toEqual(mockResult.rows);
      expect(result[0]).toHaveProperty('id');
      expect(result[0]?.['name']).toBe('Charlie');
    });

    it('should execute UPDATE query', async () => {
      const mockResult: QueryResult = {
        rows: [{ id: 1, name: 'Alice Updated' }],
        command: 'UPDATE',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockQuery.mockResolvedValue(mockResult);

      const result = await executeQuery(
        pool,
        'UPDATE users SET name = $1 WHERE id = $2 RETURNING *',
        ['Alice Updated', 1]
      );

      expect(result[0]?.['name']).toBe('Alice Updated');
    });

    it('should execute DELETE query', async () => {
      const mockResult: QueryResult = {
        rows: [],
        command: 'DELETE',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockQuery.mockResolvedValue(mockResult);

      const result = await executeQuery(pool, 'DELETE FROM users WHERE id = $1', [1]);

      expect(result).toEqual([]);
    });
  });

  describe('error handling', () => {
    it('should throw QueryError on syntax error', async () => {
      const sqlError = new Error('syntax error at or near "FORM"');
      (sqlError as any).code = '42601';
      mockQuery.mockRejectedValue(sqlError);

      await expect(executeQuery(pool, 'SELECT * FORM users', [])).rejects.toThrow(QueryError);

      await expect(executeQuery(pool, 'SELECT * FORM users', [])).rejects.toThrow(
        'Query execution failed'
      );
    });

    it('should throw QueryError on constraint violation', async () => {
      const constraintError = new Error('duplicate key value violates unique constraint');
      (constraintError as any).code = '23505';
      mockQuery.mockRejectedValue(constraintError);

      await expect(
        executeQuery(pool, 'INSERT INTO users (email) VALUES ($1)', ['duplicate@example.com'])
      ).rejects.toThrow(QueryError);
    });

    it('should throw ConnectionError on connection timeout', async () => {
      const timeoutError = new Error('connection timeout');
      (timeoutError as any).code = 'ETIMEDOUT';
      mockQuery.mockRejectedValue(timeoutError);

      await expect(executeQuery(pool, 'SELECT 1', [])).rejects.toThrow(ConnectionError);
    });

    it('should throw ConnectionError on network error', async () => {
      const networkError = new Error('ECONNREFUSED');
      (networkError as any).code = 'ECONNREFUSED';
      mockQuery.mockRejectedValue(networkError);

      await expect(executeQuery(pool, 'SELECT 1', [])).rejects.toThrow(ConnectionError);
    });

    it('should include query context in error', async () => {
      const error = new Error('table "nonexistent" does not exist');
      (error as any).code = '42P01';
      mockQuery.mockRejectedValue(error);

      try {
        await executeQuery(pool, 'SELECT * FROM nonexistent', []);
        fail('Should have thrown error');
      } catch (e) {
        expect(e).toBeInstanceOf(QueryError);
        const queryError = e as QueryError;
        expect(queryError.context?.['query']).toBe('SELECT * FROM nonexistent');
        expect(queryError.context?.['params']).toEqual([]);
        expect(queryError.context?.['code']).toBe('42P01');
      }
    });

    it('should measure and include query duration in error context', async () => {
      const error = new Error('query timeout');
      mockQuery.mockImplementation(() => {
        return new Promise((_, reject) => {
          setTimeout(() => reject(error), 100);
        });
      });

      try {
        await executeQuery(pool, 'SELECT pg_sleep(10)', []);
        fail('Should have thrown error');
      } catch (e) {
        expect(e).toBeInstanceOf(QueryError);
        const queryError = e as QueryError;
        expect(queryError.context?.['duration']).toBeGreaterThan(0);
      }
    });
  });

  describe('query timeout', () => {
    it('should timeout long-running queries', async () => {
      mockQuery.mockImplementation(() => {
        return new Promise((resolve) => {
          setTimeout(
            () => resolve({ rows: [], command: 'SELECT', rowCount: 0, oid: 0, fields: [] }),
            10000
          );
        });
      });

      await expect(
        executeQuery(pool, 'SELECT pg_sleep(100)', [], { timeout: 100 })
      ).rejects.toThrow('Query timeout');
    }, 1000);

    it('should not timeout fast queries', async () => {
      const mockResult: QueryResult = {
        rows: [{ result: 1 }],
        command: 'SELECT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockQuery.mockResolvedValue(mockResult);

      const result = await executeQuery(pool, 'SELECT 1', [], { timeout: 5000 });

      expect(result).toEqual(mockResult.rows);
    });
  });

  describe('result type validation', () => {
    it('should validate result rows are array', async () => {
      const invalidResult = { rows: null } as any;
      mockQuery.mockResolvedValue(invalidResult);

      const result = await executeQuery(pool, 'SELECT 1', []);

      expect(Array.isArray(result)).toBe(true);
      expect(result).toEqual([]);
    });

    it('should handle undefined rows', async () => {
      const invalidResult = {} as any;
      mockQuery.mockResolvedValue(invalidResult);

      const result = await executeQuery(pool, 'SELECT 1', []);

      expect(result).toEqual([]);
    });
  });

  describe('parameter handling', () => {
    it('should handle null parameters', async () => {
      const mockResult: QueryResult = {
        rows: [{ id: 1, value: null }],
        command: 'SELECT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockQuery.mockResolvedValue(mockResult);

      const result = await executeQuery(pool, 'SELECT * FROM data WHERE value = $1', [null]);

      expect(result[0]?.['value']).toBeNull();
      expect(mockQuery).toHaveBeenCalledWith('SELECT * FROM data WHERE value = $1', [null]);
    });

    it('should handle undefined by converting to null', async () => {
      const mockResult: QueryResult = {
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      };
      mockQuery.mockResolvedValue(mockResult);

      await executeQuery(pool, 'SELECT * FROM data WHERE value = $1', [undefined]);

      expect(mockQuery).toHaveBeenCalledWith('SELECT * FROM data WHERE value = $1', [null]);
    });

    it('should handle complex parameter types', async () => {
      const mockResult: QueryResult = {
        rows: [{ data: { key: 'value' } }],
        command: 'SELECT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockQuery.mockResolvedValue(mockResult);

      const jsonData = { key: 'value', nested: { prop: 123 } };
      await executeQuery(pool, 'INSERT INTO documents (data) VALUES ($1)', [jsonData]);

      expect(mockQuery).toHaveBeenCalledWith('INSERT INTO documents (data) VALUES ($1)', [
        jsonData,
      ]);
    });
  });
});

describe('executeTransaction', () => {
  let pool: Pool;
  let mockConnect: jest.Mock;
  let mockClient: PoolClient;

  beforeEach(() => {
    pool = new Pool();
    mockClient = {
      query: jest.fn(),
      release: jest.fn(),
    } as any;
    mockConnect = (pool.connect as jest.Mock).mockReset().mockResolvedValue(mockClient);
  });

  describe('successful transactions', () => {
    it('should execute multiple queries in transaction', async () => {
      const mockResults: QueryResult[] = [
        { rows: [{ id: 1 }], command: 'INSERT', rowCount: 1, oid: 0, fields: [] },
        { rows: [{ id: 2 }], command: 'INSERT', rowCount: 1, oid: 0, fields: [] },
      ];

      (mockClient.query as jest.Mock)
        .mockResolvedValueOnce({ rows: [], command: 'BEGIN', rowCount: 0, oid: 0, fields: [] })
        .mockResolvedValueOnce(mockResults[0])
        .mockResolvedValueOnce(mockResults[1])
        .mockResolvedValueOnce({ rows: [], command: 'COMMIT', rowCount: 0, oid: 0, fields: [] });

      const queries = [
        { sql: 'INSERT INTO users (name) VALUES ($1) RETURNING id', params: ['Alice'] },
        { sql: 'INSERT INTO users (name) VALUES ($1) RETURNING id', params: ['Bob'] },
      ];

      const results = await executeTransaction(pool, queries);

      expect(results).toHaveLength(2);
      expect(results[0]).toEqual(mockResults[0]?.rows);
      expect(results[1]).toEqual(mockResults[1]?.rows);
      expect(mockClient.query).toHaveBeenCalledWith('BEGIN');
      expect(mockClient.query).toHaveBeenCalledWith('COMMIT');
      expect(mockClient.release).toHaveBeenCalled();
    });

    it('should execute single query in transaction', async () => {
      const mockResult: QueryResult = {
        rows: [{ id: 1, name: 'Test' }],
        command: 'INSERT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };

      (mockClient.query as jest.Mock)
        .mockResolvedValueOnce({ rows: [], command: 'BEGIN', rowCount: 0, oid: 0, fields: [] })
        .mockResolvedValueOnce(mockResult)
        .mockResolvedValueOnce({ rows: [], command: 'COMMIT', rowCount: 0, oid: 0, fields: [] });

      const queries = [
        { sql: 'INSERT INTO users (name) VALUES ($1) RETURNING *', params: ['Test'] },
      ];

      const results = await executeTransaction(pool, queries);

      expect(results).toHaveLength(1);
      expect(results[0]).toEqual(mockResult.rows);
    });
  });

  describe('transaction rollback', () => {
    it('should rollback on query error', async () => {
      const error = new Error('constraint violation');
      (error as any).code = '23505';

      (mockClient.query as jest.Mock)
        .mockResolvedValueOnce({ rows: [], command: 'BEGIN', rowCount: 0, oid: 0, fields: [] })
        .mockResolvedValueOnce({
          rows: [{ id: 1 }],
          command: 'INSERT',
          rowCount: 1,
          oid: 0,
          fields: [],
        })
        .mockRejectedValueOnce(error)
        .mockResolvedValueOnce({ rows: [], command: 'ROLLBACK', rowCount: 0, oid: 0, fields: [] });

      const queries = [
        { sql: 'INSERT INTO users (name) VALUES ($1)', params: ['Alice'] },
        { sql: 'INSERT INTO users (email) VALUES ($1)', params: ['duplicate@example.com'] },
      ];

      await expect(executeTransaction(pool, queries)).rejects.toThrow(QueryError);

      expect(mockClient.query).toHaveBeenCalledWith('ROLLBACK');
      expect(mockClient.release).toHaveBeenCalled();
    });

    it('should rollback and release client even if rollback fails', async () => {
      const queryError = new Error('constraint violation');
      const rollbackError = new Error('rollback failed');

      (mockClient.query as jest.Mock)
        .mockResolvedValueOnce({ rows: [], command: 'BEGIN', rowCount: 0, oid: 0, fields: [] })
        .mockRejectedValueOnce(queryError)
        .mockRejectedValueOnce(rollbackError);

      const queries = [{ sql: 'INSERT INTO users (name) VALUES ($1)', params: ['Test'] }];

      await expect(executeTransaction(pool, queries)).rejects.toThrow();

      expect(mockClient.release).toHaveBeenCalled();
    });
  });

  describe('error handling', () => {
    it('should throw ConnectionError if cannot get client', async () => {
      mockConnect.mockRejectedValue(new Error('connection pool exhausted'));

      const queries = [{ sql: 'SELECT 1', params: [] }];

      await expect(executeTransaction(pool, queries)).rejects.toThrow(ConnectionError);
    });

    it('should include transaction context in errors', async () => {
      const error = new Error('deadlock detected');
      (error as any).code = '40P01';

      (mockClient.query as jest.Mock)
        .mockResolvedValueOnce({ rows: [], command: 'BEGIN', rowCount: 0, oid: 0, fields: [] })
        .mockRejectedValueOnce(error);

      const queries = [{ sql: 'UPDATE accounts SET balance = balance + $1', params: [100] }];

      try {
        await executeTransaction(pool, queries);
        fail('Should have thrown error');
      } catch (e) {
        expect(e).toBeInstanceOf(QueryError);
        const queryError = e as QueryError;
        expect(queryError.context?.['transaction']).toBe(true);
        expect(queryError.context?.['queryCount']).toBe(1);
      }
    });
  });

  describe('empty transactions', () => {
    it('should handle empty query array', async () => {
      const queries: Array<{ sql: string; params?: any[] }> = [];

      const results = await executeTransaction(pool, queries);

      expect(results).toEqual([]);
      expect(mockConnect).not.toHaveBeenCalled();
    });
  });
});
