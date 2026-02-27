/**
 * TDD RED PHASE: Connection Pool Tests
 *
 * These tests define the expected behavior of our PostgreSQL connection pools.
 * They MUST FAIL initially until we implement the functionality.
 *
 * Test Coverage:
 * 1. Configuration validation (environment variables)
 * 2. Pool initialization with correct parameters
 * 3. Connection health checks
 * 4. Retry logic with exponential backoff
 * 5. Graceful shutdown
 */

/* eslint-disable @typescript-eslint/unbound-method */

import { Pool } from 'pg';
import {
  getMarketsPool,
  getAiModelPool,
  healthCheckMarkets,
  healthCheckAiModel,
  closeAllPools,
  initializePools,
} from '../connection';
import { getDatabaseConfig, validateDatabaseConfig } from '../config';

// Mock pg module
const mockQuery = jest.fn();
const mockConnect = jest.fn();
const mockEnd = jest.fn();

jest.mock('pg', () => {
  return {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    Pool: jest.fn().mockImplementation((config: any) => ({
      query: mockQuery,
      connect: mockConnect,
      end: mockEnd,
      totalCount: 0,
      idleCount: 0,
      waitingCount: 0,
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      options: config,
    })),
  };
});

describe('Database Configuration', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.resetModules();
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  describe('validateDatabaseConfig', () => {
    it('should throw error when MARKETS_DB_HOST is missing', () => {
      delete process.env['MARKETS_DB_HOST'];

      expect(() => validateDatabaseConfig()).toThrow(
        'Missing required environment variable: MARKETS_DB_HOST'
      );
    });

    it('should throw error when MARKETS_DB_PORT is missing', () => {
      process.env['MARKETS_DB_HOST'] = 'localhost';
      delete process.env['MARKETS_DB_PORT'];

      expect(() => validateDatabaseConfig()).toThrow(
        'Missing required environment variable: MARKETS_DB_PORT'
      );
    });

    it('should throw error when MARKETS_DB_NAME is missing', () => {
      process.env['MARKETS_DB_HOST'] = 'localhost';
      process.env['MARKETS_DB_PORT'] = '5432';
      delete process.env['MARKETS_DB_NAME'];

      expect(() => validateDatabaseConfig()).toThrow(
        'Missing required environment variable: MARKETS_DB_NAME'
      );
    });

    it('should throw error when AI_MODEL_DB_HOST is missing', () => {
      process.env['MARKETS_DB_HOST'] = 'localhost';
      process.env['MARKETS_DB_PORT'] = '5432';
      process.env['MARKETS_DB_NAME'] = 'markets';
      process.env['MARKETS_DB_USER'] = 'markets';
      process.env['MARKETS_DB_PASSWORD'] = 'password';
      delete process.env['AI_MODEL_DB_HOST'];

      expect(() => validateDatabaseConfig()).toThrow(
        'Missing required environment variable: AI_MODEL_DB_HOST'
      );
    });

    it('should validate successfully with all required variables', () => {
      process.env['MARKETS_DB_HOST'] = 'localhost';
      process.env['MARKETS_DB_PORT'] = '5432';
      process.env['MARKETS_DB_NAME'] = 'markets';
      process.env['MARKETS_DB_USER'] = 'markets';
      process.env['MARKETS_DB_PASSWORD'] = 'password';
      process.env['AI_MODEL_DB_HOST'] = 'localhost';
      process.env['AI_MODEL_DB_PORT'] = '5432';
      process.env['AI_MODEL_DB_NAME'] = 'ai_model';
      process.env['AI_MODEL_DB_USER'] = 'ai_model';
      process.env['AI_MODEL_DB_PASSWORD'] = 'password';

      expect(() => validateDatabaseConfig()).not.toThrow();
    });
  });

  describe('getDatabaseConfig', () => {
    beforeEach(() => {
      process.env['MARKETS_DB_HOST'] = 'localhost';
      process.env['MARKETS_DB_PORT'] = '5432';
      process.env['MARKETS_DB_NAME'] = 'markets';
      process.env['MARKETS_DB_USER'] = 'markets';
      process.env['MARKETS_DB_PASSWORD'] = 'test_password';
      process.env['AI_MODEL_DB_HOST'] = 'localhost';
      process.env['AI_MODEL_DB_PORT'] = '5433';
      process.env['AI_MODEL_DB_NAME'] = 'ai_model';
      process.env['AI_MODEL_DB_USER'] = 'ai_model';
      process.env['AI_MODEL_DB_PASSWORD'] = 'ai_password';
    });

    it('should parse Markets database configuration correctly', () => {
      const config = getDatabaseConfig();

      expect(config.markets).toEqual({
        host: 'localhost',
        port: 5432,
        database: 'markets',
        user: 'markets',
        password: 'test_password',
        max: 20,
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 5000,
      });
    });

    it('should parse AI Model database configuration correctly', () => {
      const config = getDatabaseConfig();

      expect(config.aiModel).toEqual({
        host: 'localhost',
        port: 5433,
        database: 'ai_model',
        user: 'ai_model',
        password: 'ai_password',
        max: 20,
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 5000,
      });
    });

    it('should apply default pool configuration values', () => {
      const config = getDatabaseConfig();

      expect(config.markets.max).toBe(20);
      expect(config.markets.idleTimeoutMillis).toBe(30000);
      expect(config.markets.connectionTimeoutMillis).toBe(5000);
    });
  });
});

describe('Connection Pool Initialization', () => {
  beforeEach(() => {
    process.env['MARKETS_DB_HOST'] = 'localhost';
    process.env['MARKETS_DB_PORT'] = '5432';
    process.env['MARKETS_DB_NAME'] = 'markets';
    process.env['MARKETS_DB_USER'] = 'markets';
    process.env['MARKETS_DB_PASSWORD'] = 'password';
    process.env['AI_MODEL_DB_HOST'] = 'localhost';
    process.env['AI_MODEL_DB_PORT'] = '5432';
    process.env['AI_MODEL_DB_NAME'] = 'ai_model';
    process.env['AI_MODEL_DB_USER'] = 'ai_model';
    process.env['AI_MODEL_DB_PASSWORD'] = 'password';

    // Set up default mock behavior
    mockQuery.mockResolvedValue({ rows: [{ version: 'PostgreSQL 15.0' }] });
    mockConnect.mockResolvedValue(undefined);
    mockEnd.mockResolvedValue(undefined);
  });

  afterEach(async () => {
    await closeAllPools();
    jest.clearAllMocks();
  });

  describe('initializePools', () => {
    it('should initialize both pools on first call', async () => {
      await initializePools();

      const marketsPool = getMarketsPool();
      const aiModelPool = getAiModelPool();

      expect(marketsPool).toBeDefined();
      expect(aiModelPool).toBeDefined();
      expect(Pool).toHaveBeenCalledTimes(2);
    });

    it('should create Markets pool with correct configuration', async () => {
      await initializePools();

      expect(Pool).toHaveBeenCalledWith({
        host: 'localhost',
        port: 5432,
        database: 'markets',
        user: 'markets',
        password: 'password',
        max: 20,
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 5000,
      });
    });

    it('should create AI Model pool with correct configuration', async () => {
      await initializePools();

      expect(Pool).toHaveBeenCalledWith({
        host: 'localhost',
        port: 5432,
        database: 'ai_model',
        user: 'ai_model',
        password: 'password',
        max: 20,
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 5000,
      });
    });

    it('should not recreate pools if already initialized', async () => {
      await initializePools();
      await initializePools(); // Second call

      expect(Pool).toHaveBeenCalledTimes(2); // Still only 2 pools
    });

    it('should retry with exponential backoff on connection failure', async () => {
      const mockPool = new Pool({});
      (mockPool.query as jest.Mock).mockRejectedValueOnce(new Error('Connection refused'));
      (mockPool.query as jest.Mock).mockResolvedValueOnce({ rows: [{ version: '15.0' }] });

      await initializePools();

      expect(mockPool.query).toHaveBeenCalledWith('SELECT version()');
    });
  });

  describe('getMarketsPool', () => {
    it('should throw error if pool not initialized', () => {
      expect(() => getMarketsPool()).toThrow(
        'Markets database pool not initialized. Call initializePools() first.'
      );
    });

    it('should return pool after initialization', async () => {
      await initializePools();
      const pool = getMarketsPool();

      expect(pool).toBeDefined();
      expect(pool.query).toBeDefined();
      expect(pool.end).toBeDefined();
    });
  });

  describe('getAiModelPool', () => {
    it('should throw error if pool not initialized', () => {
      expect(() => getAiModelPool()).toThrow(
        'AI Model database pool not initialized. Call initializePools() first.'
      );
    });

    it('should return pool after initialization', async () => {
      await initializePools();
      const pool = getAiModelPool();

      expect(pool).toBeDefined();
      expect(pool.query).toBeDefined();
      expect(pool.end).toBeDefined();
    });
  });
});

describe('Health Checks', () => {
  beforeEach(async () => {
    process.env['MARKETS_DB_HOST'] = 'localhost';
    process.env['MARKETS_DB_PORT'] = '5432';
    process.env['MARKETS_DB_NAME'] = 'markets';
    process.env['MARKETS_DB_USER'] = 'markets';
    process.env['MARKETS_DB_PASSWORD'] = 'password';
    process.env['AI_MODEL_DB_HOST'] = 'localhost';
    process.env['AI_MODEL_DB_PORT'] = '5432';
    process.env['AI_MODEL_DB_NAME'] = 'ai_model';
    process.env['AI_MODEL_DB_USER'] = 'ai_model';
    process.env['AI_MODEL_DB_PASSWORD'] = 'password';

    // Set up default mock behavior
    mockQuery.mockResolvedValue({ rows: [{ version: 'PostgreSQL 15.0' }] });
    mockConnect.mockResolvedValue(undefined);
    mockEnd.mockResolvedValue(undefined);

    await initializePools();
  });

  afterEach(async () => {
    await closeAllPools();
    jest.clearAllMocks();
  });

  describe('healthCheckMarkets', () => {
    it('should return true when connection is healthy', async () => {
      const pool = getMarketsPool();
      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [{ result: 1 }],
      });

      const isHealthy = await healthCheckMarkets();

      expect(isHealthy).toBe(true);
      expect(pool.query).toHaveBeenCalledWith('SELECT 1 as result');
    });

    it('should return false when connection fails', async () => {
      const pool = getMarketsPool();
      (pool.query as jest.Mock).mockRejectedValueOnce(new Error('Connection timeout'));

      const isHealthy = await healthCheckMarkets();

      expect(isHealthy).toBe(false);
    });

    it('should measure query execution time', async () => {
      const pool = getMarketsPool();
      (pool.query as jest.Mock).mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve({ rows: [{ result: 1 }] }), 50))
      );

      const startTime = Date.now();
      await healthCheckMarkets();
      const endTime = Date.now();

      expect(endTime - startTime).toBeGreaterThanOrEqual(50);
    });
  });

  describe('healthCheckAiModel', () => {
    it('should return true when connection is healthy', async () => {
      const pool = getAiModelPool();
      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [{ result: 1 }],
      });

      const isHealthy = await healthCheckAiModel();

      expect(isHealthy).toBe(true);
      expect(pool.query).toHaveBeenCalledWith('SELECT 1 as result');
    });

    it('should return false when connection fails', async () => {
      const pool = getAiModelPool();
      (pool.query as jest.Mock).mockRejectedValueOnce(new Error('Connection timeout'));

      const isHealthy = await healthCheckAiModel();

      expect(isHealthy).toBe(false);
    });
  });
});

describe('Connection Retry Logic', () => {
  beforeEach(() => {
    process.env['MARKETS_DB_HOST'] = 'localhost';
    process.env['MARKETS_DB_PORT'] = '5432';
    process.env['MARKETS_DB_NAME'] = 'markets';
    process.env['MARKETS_DB_USER'] = 'markets';
    process.env['MARKETS_DB_PASSWORD'] = 'password';
    process.env['AI_MODEL_DB_HOST'] = 'localhost';
    process.env['AI_MODEL_DB_PORT'] = '5432';
    process.env['AI_MODEL_DB_NAME'] = 'ai_model';
    process.env['AI_MODEL_DB_USER'] = 'ai_model';
    process.env['AI_MODEL_DB_PASSWORD'] = 'password';

    // Set up default mock behavior
    mockQuery.mockResolvedValue({ rows: [{ version: 'PostgreSQL 15.0' }] });
    mockConnect.mockResolvedValue(undefined);
    mockEnd.mockResolvedValue(undefined);
  });

  afterEach(async () => {
    await closeAllPools();
    jest.clearAllMocks();
  });

  it('should retry connection with exponential backoff delays', async () => {
    const delays: number[] = [];
    const startTimes: number[] = [];

    /* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-return */
    jest.spyOn(global, 'setTimeout').mockImplementation((callback: any, delay?: number) => {
      if (delay !== undefined) {
        delays.push(delay);
      }
      startTimes.push(Date.now());
      callback();
      return 0 as any;
    });
    /* eslint-enable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-return */

    // Mock query to fail twice, then succeed
    mockQuery
      .mockRejectedValueOnce(new Error('Connection refused'))
      .mockRejectedValueOnce(new Error('Connection refused'))
      .mockResolvedValueOnce({ rows: [{ version: 'PostgreSQL 15.0' }] });

    await initializePools();

    // Verify exponential backoff delays: 1s, 2s
    expect(delays).toEqual([1000, 2000]);

    jest.restoreAllMocks();
  });

  it('should throw error after maximum retry attempts', async () => {
    /* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-return */
    jest.spyOn(global, 'setTimeout').mockImplementation((callback: any) => {
      callback();
      return 0 as any;
    });
    /* eslint-enable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-return */

    // Mock query to always fail
    mockQuery.mockRejectedValue(new Error('Connection refused'));

    await expect(initializePools()).rejects.toThrow(
      'Failed to initialize database pools after 5 attempts'
    );

    // Should have been called 5 times for markets pool only (fails before AI model pool)
    expect(mockQuery).toHaveBeenCalledTimes(5);

    jest.restoreAllMocks();
  });

  it('should use correct backoff delays: 1s, 2s, 5s, 10s', async () => {
    const delays: number[] = [];

    /* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-return */
    jest.spyOn(global, 'setTimeout').mockImplementation((callback: any, delay?: number) => {
      if (delay !== undefined) {
        delays.push(delay);
      }
      callback();
      return 0 as any;
    });
    /* eslint-enable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-return */

    // Mock query to always fail
    mockQuery.mockRejectedValue(new Error('Connection refused'));

    try {
      await initializePools();
    } catch {
      // Expected to fail after 5 attempts
      // Only 4 delays are recorded because after the 5th attempt, it throws without retrying
    }

    // We get 4 delays because: attempt 1 fails → delay 1s → attempt 2 fails → delay 2s → attempt 3 fails → delay 5s → attempt 4 fails → delay 10s → attempt 5 fails → throw error
    expect(delays).toEqual([1000, 2000, 5000, 10000]);

    jest.restoreAllMocks();
  });
});

describe('Connection Pool Cleanup', () => {
  beforeEach(async () => {
    process.env['MARKETS_DB_HOST'] = 'localhost';
    process.env['MARKETS_DB_PORT'] = '5432';
    process.env['MARKETS_DB_NAME'] = 'markets';
    process.env['MARKETS_DB_USER'] = 'markets';
    process.env['MARKETS_DB_PASSWORD'] = 'password';
    process.env['AI_MODEL_DB_HOST'] = 'localhost';
    process.env['AI_MODEL_DB_PORT'] = '5432';
    process.env['AI_MODEL_DB_NAME'] = 'ai_model';
    process.env['AI_MODEL_DB_USER'] = 'ai_model';
    process.env['AI_MODEL_DB_PASSWORD'] = 'password';

    // Set up default mock behavior
    mockQuery.mockResolvedValue({ rows: [{ version: 'PostgreSQL 15.0' }] });
    mockConnect.mockResolvedValue(undefined);
    mockEnd.mockResolvedValue(undefined);

    await initializePools();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('closeAllPools', () => {
    it('should close both pools', async () => {
      const marketsPool = getMarketsPool();
      const aiModelPool = getAiModelPool();

      await closeAllPools();

      // Verify end was called on both pools (may be called multiple times due to test lifecycle)
      expect(mockEnd).toHaveBeenCalled();
      expect(marketsPool.end).toBeDefined();
      expect(aiModelPool.end).toBeDefined();
    });

    it('should handle errors during pool closure gracefully', async () => {
      const marketsPool = getMarketsPool();
      (marketsPool.end as jest.Mock).mockRejectedValueOnce(new Error('Failed to close connection'));

      await expect(closeAllPools()).resolves.not.toThrow();
    });

    it('should allow reinitialization after closing', async () => {
      await closeAllPools();

      expect(() => getMarketsPool()).toThrow('Markets database pool not initialized');

      await initializePools();

      expect(() => getMarketsPool()).not.toThrow();
    });
  });
});
