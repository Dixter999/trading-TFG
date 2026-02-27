/**
 * Transaction Support Utilities
 *
 * Utilities for executing multiple queries within a database transaction
 * with automatic rollback on errors.
 */

import { Pool, PoolClient } from 'pg';
import { QueryError, ConnectionError } from '../errors';

/**
 * Query definition for transaction
 */
export interface TransactionQuery {
  sql: string;
  params?: any[];
}

/**
 * Execute multiple queries within a transaction
 *
 * Automatically begins a transaction, executes all queries in sequence,
 * and commits. If any query fails, the transaction is rolled back.
 *
 * @param pool - PostgreSQL connection pool
 * @param queries - Array of queries to execute in transaction
 * @returns Promise resolving to array of result sets (one per query)
 * @throws ConnectionError if cannot acquire client from pool
 * @throws QueryError if any query fails (transaction will be rolled back)
 */
export async function executeTransaction<T = any>(
  pool: Pool,
  queries: TransactionQuery[]
): Promise<T[][]> {
  // Handle empty query array
  if (queries.length === 0) {
    return [];
  }

  let client: PoolClient | null = null;

  try {
    // Acquire client from pool
    client = await pool.connect();
  } catch (error: any) {
    throw new ConnectionError('Failed to acquire database client', {
      error: error.message,
      code: error.code,
    });
  }

  const results: T[][] = [];
  const startTime = Date.now();

  try {
    // Begin transaction
    await client.query('BEGIN');

    // Execute each query in sequence
    for (let i = 0; i < queries.length; i++) {
      const query = queries[i];
      if (!query) continue;

      const { sql, params = [] } = query;

      // Sanitize parameters: convert undefined to null
      const sanitizedParams = params.map((param) => (param === undefined ? null : param));

      const result = await client.query(sql, sanitizedParams);
      results.push((result.rows || []) as T[]);
    }

    // Commit transaction
    await client.query('COMMIT');

    return results;
  } catch (error: any) {
    const duration = Date.now() - startTime;

    // Attempt to rollback transaction
    try {
      await client.query('ROLLBACK');
    } catch (rollbackError: any) {
      // Rollback failed, but we'll still throw the original error
      // The rollback error is secondary and should be logged by the caller if needed
    }

    // Create error with transaction context
    const errorContext = {
      transaction: true,
      queryCount: queries.length,
      completedQueries: results.length,
      duration,
      code: error.code,
      error: error.message,
    };

    throw new QueryError('Transaction failed', errorContext);
  } finally {
    // Always release client back to pool
    if (client) {
      client.release();
    }
  }
}
