/**
 * Database Connection Pool Management
 *
 * Manages PostgreSQL connection pools for both Markets and AI Model databases.
 * Implements:
 * - Connection pooling with pg.Pool
 * - Health checks
 * - Exponential backoff retry logic (1s, 2s, 5s, 10s, 30s)
 * - Graceful shutdown
 * - Structured logging
 */

import { Pool } from 'pg';
import {
  getDatabaseConfig,
  getMarketsConnectionString,
  getAiModelConnectionString,
  DatabasePoolConfig,
} from './config';
import { logger } from '../config/logging';

// Singleton pool instances
let marketsPool: Pool | null = null;
let aiModelPool: Pool | null = null;

// Retry configuration
const MAX_RETRY_ATTEMPTS = 5;
const RETRY_DELAYS = [1000, 2000, 5000, 10000, 30000]; // milliseconds

/**
 * Sleep for specified milliseconds
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Log helper function using centralized logger
 */
function log(
  level: 'info' | 'error' | 'warn',
  message: string,
  meta?: Record<string, unknown>
): void {
  logger[level](message, meta);
}

/**
 * Test connection to database pool
 * @param pool - Pool instance to test
 * @param poolName - Name for logging
 * @returns true if connection successful
 */
async function testConnection(pool: Pool, poolName: string): Promise<boolean> {
  try {
    const result = await pool.query('SELECT version()');
    const firstRow = result.rows[0] as Record<string, unknown> | undefined;
    const version =
      firstRow && typeof firstRow === 'object' && 'version' in firstRow
        ? String(firstRow['version'])
        : 'unknown';
    log('info', `${poolName} connection test successful`, {
      version,
    });
    return true;
  } catch (error) {
    log('error', `${poolName} connection test failed`, {
      error: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
}

/**
 * Initialize database pool with retry logic
 * @param config - Pool configuration
 * @param poolName - Name for logging
 * @returns Initialized pool
 * @throws Error if connection fails after max retries
 */
async function initializePoolWithRetry(
  config: DatabasePoolConfig,
  poolName: string
): Promise<Pool> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < MAX_RETRY_ATTEMPTS; attempt++) {
    try {
      log('info', `Initializing ${poolName} (attempt ${attempt + 1}/${MAX_RETRY_ATTEMPTS})`);

      const pool = new Pool(config);

      // Test connection
      const isHealthy = await testConnection(pool, poolName);

      if (isHealthy) {
        log('info', `${poolName} initialized successfully`);
        return pool;
      }

      // If connection test failed, close pool and retry
      await pool.end();
      throw new Error(`Connection test failed for ${poolName}`);
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      log('warn', `${poolName} initialization attempt ${attempt + 1} failed`, {
        error: lastError.message,
        nextRetryIn: attempt < MAX_RETRY_ATTEMPTS - 1 ? `${RETRY_DELAYS[attempt]}ms` : 'none',
      });

      // Wait before retry (except on last attempt)
      if (attempt < MAX_RETRY_ATTEMPTS - 1) {
        await sleep(RETRY_DELAYS[attempt]!);
      }
    }
  }

  // All retry attempts failed
  const error = new Error(
    `Failed to initialize database pools after ${MAX_RETRY_ATTEMPTS} attempts: ${lastError?.message || 'Unknown error'}`
  );
  log('error', 'Database initialization failed', {
    attempts: MAX_RETRY_ATTEMPTS,
    lastError: lastError?.message,
  });
  throw error;
}

/**
 * Initialize both database pools
 * Must be called before using getMarketsPool() or getAiModelPool()
 * @throws Error if initialization fails
 */
export async function initializePools(): Promise<void> {
  // Only initialize if not already initialized
  if (marketsPool && aiModelPool) {
    log('info', 'Database pools already initialized');
    return;
  }

  const config = getDatabaseConfig();

  log('info', 'Starting database pool initialization', {
    marketsConnectionString: getMarketsConnectionString(),
    aiModelConnectionString: getAiModelConnectionString(),
  });

  try {
    // Initialize Markets pool
    if (!marketsPool) {
      marketsPool = await initializePoolWithRetry(config.markets, 'Markets DB');
    }

    // Initialize AI Model pool
    if (!aiModelPool) {
      aiModelPool = await initializePoolWithRetry(config.aiModel, 'AI Model DB');
    }

    log('info', 'All database pools initialized successfully', {
      marketsPools: { max: config.markets.max, idle: config.markets.idleTimeoutMillis },
      aiModelPools: { max: config.aiModel.max, idle: config.aiModel.idleTimeoutMillis },
    });
  } catch (error) {
    // Clean up any partially initialized pools
    await closeAllPools();
    throw error;
  }
}

/**
 * Get Markets database pool
 * @returns Markets pool instance
 * @throws Error if pool not initialized
 */
export function getMarketsPool(): Pool {
  if (!marketsPool) {
    throw new Error('Markets database pool not initialized. Call initializePools() first.');
  }
  return marketsPool;
}

/**
 * Get AI Model database pool
 * @returns AI Model pool instance
 * @throws Error if pool not initialized
 */
export function getAiModelPool(): Pool {
  if (!aiModelPool) {
    throw new Error('AI Model database pool not initialized. Call initializePools() first.');
  }
  return aiModelPool;
}

/**
 * Health check for Markets database
 * @returns true if healthy, false otherwise
 */
export async function healthCheckMarkets(): Promise<boolean> {
  try {
    const pool = getMarketsPool();
    const startTime = Date.now();
    const result = await pool.query('SELECT 1 as result');
    const duration = Date.now() - startTime;

    log('info', 'Markets DB health check passed', { duration });
    return result.rows.length > 0;
  } catch (error) {
    log('error', 'Markets DB health check failed', {
      error: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
}

/**
 * Health check for AI Model database
 * @returns true if healthy, false otherwise
 */
export async function healthCheckAiModel(): Promise<boolean> {
  try {
    const pool = getAiModelPool();
    const startTime = Date.now();
    const result = await pool.query('SELECT 1 as result');
    const duration = Date.now() - startTime;

    log('info', 'AI Model DB health check passed', { duration });
    return result.rows.length > 0;
  } catch (error) {
    log('error', 'AI Model DB health check failed', {
      error: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
}

/**
 * Close all database pools gracefully
 * Should be called on application shutdown
 */
export async function closeAllPools(): Promise<void> {
  log('info', 'Closing all database pools');

  const closePromises: Promise<void>[] = [];

  if (marketsPool) {
    closePromises.push(
      marketsPool
        .end()
        .then(() => {
          log('info', 'Markets DB pool closed');
          marketsPool = null;
        })
        .catch((error) => {
          log('error', 'Error closing Markets DB pool', {
            error: error instanceof Error ? error.message : String(error),
          });
        })
    );
  }

  if (aiModelPool) {
    closePromises.push(
      aiModelPool
        .end()
        .then(() => {
          log('info', 'AI Model DB pool closed');
          aiModelPool = null;
        })
        .catch((error) => {
          log('error', 'Error closing AI Model DB pool', {
            error: error instanceof Error ? error.message : String(error),
          });
        })
    );
  }

  await Promise.all(closePromises);
  log('info', 'All database pools closed');
}

/**
 * Register shutdown handlers for graceful cleanup
 */
export function registerShutdownHandlers(): void {
  const shutdown = async (signal: string): Promise<void> => {
    log('info', `Received ${signal}, closing database connections`);
    await closeAllPools();
    process.exit(0);
  };

  process.on('SIGTERM', () => {
    void shutdown('SIGTERM');
  });
  process.on('SIGINT', () => {
    void shutdown('SIGINT');
  });
}
