/**
 * Query Execution Utilities
 *
 * Generic utilities for executing SQL queries with error handling,
 * timeout support, and result validation.
 */

import { Pool, QueryResult } from 'pg';
import { QueryError, ConnectionError } from '../errors';

// Re-export transaction utilities
export { executeTransaction, TransactionQuery } from './transactions';

/**
 * Query options for execution
 */
export interface QueryOptions {
  timeout?: number; // Query timeout in milliseconds
}

/**
 * Execute a SQL query with error handling and timeout support
 *
 * @param pool - PostgreSQL connection pool
 * @param sql - SQL query string
 * @param params - Query parameters (optional)
 * @param options - Query options (optional)
 * @returns Promise resolving to array of result rows
 * @throws QueryError on query execution failures
 * @throws ConnectionError on connection/network failures
 */
export async function executeQuery<T = any>(
  pool: Pool,
  sql: string,
  params: any[] = [],
  options: QueryOptions = {}
): Promise<T[]> {
  const startTime = Date.now();
  const { timeout } = options;

  // Sanitize parameters: convert undefined to null
  const sanitizedParams = params.map((param) => (param === undefined ? null : param));

  try {
    let queryPromise: Promise<QueryResult>;

    // Execute query
    queryPromise = pool.query(sql, sanitizedParams);

    // Apply timeout if specified
    if (timeout) {
      queryPromise = Promise.race([
        queryPromise,
        new Promise<QueryResult>((_, reject) => {
          setTimeout(() => {
            reject(new Error('Query timeout'));
          }, timeout);
        }),
      ]);
    }

    const result = await queryPromise;

    // Validate result has rows array
    if (!result || !Array.isArray(result.rows)) {
      return [];
    }

    return result.rows as T[];
  } catch (error: any) {
    const duration = Date.now() - startTime;

    // Check if this is a query timeout (not a connection issue)
    const isQueryTimeout = error.message === 'Query timeout';

    // Determine error type based on error code
    const isConnectionError =
      !isQueryTimeout &&
      (error.code === 'ETIMEDOUT' ||
        error.code === 'ECONNREFUSED' ||
        error.code === 'ECONNRESET' ||
        error.code === 'ENOTFOUND' ||
        error.message?.includes('connection'));

    const errorContext = {
      query: sql,
      params: sanitizedParams,
      duration,
      code: error.code,
      error: error.message,
    };

    if (isConnectionError) {
      throw new ConnectionError('Database connection failed', errorContext);
    }

    throw new QueryError(isQueryTimeout ? 'Query timeout' : 'Query execution failed', errorContext);
  }
}
