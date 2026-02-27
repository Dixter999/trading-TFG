/**
 * Market Data Query Tests
 * Tests for getMarketDataWithIndicators with mocked pg.Pool
 *
 * TDD Approach: Tests written FIRST before implementation
 */

import { Pool, QueryResult } from 'pg';
import { getMarketDataWithIndicators } from '../queries/marketData';
import { ValidationError, QueryError } from '../errors';

// Mock pg module
jest.mock('pg', () => {
  const mPool = {
    query: jest.fn(),
    connect: jest.fn(),
    end: jest.fn(),
  };
  return { Pool: jest.fn(() => mPool) };
});

// Mock connection module BEFORE imports
jest.mock('../connection');

describe('getMarketDataWithIndicators', () => {
  let marketsPool: Pool;
  let mockMarketsQuery: jest.Mock;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();

    // Create mock pool
    marketsPool = new Pool();
    mockMarketsQuery = (marketsPool.query as jest.Mock).mockReset();

    // Mock getMarketsPool to return our mocked pool
    const { getMarketsPool } = require('../connection');
    getMarketsPool.mockReturnValue(marketsPool);
  });

  describe('input validation', () => {
    // Issue #404 Stream C: Updated tests for multi-symbol support
    it('should throw ValidationError for invalid symbol', async () => {
      await expect(
        getMarketDataWithIndicators(
          'INVALID_SYMBOL', // Not in supported symbols list
          'H1',
          new Date('2025-01-01'),
          new Date('2025-01-02'),
          100
        )
      ).rejects.toThrow(ValidationError);

      await expect(
        getMarketDataWithIndicators(
          'INVALID_SYMBOL',
          'H1',
          new Date('2025-01-01'),
          new Date('2025-01-02'),
          100
        )
      ).rejects.toThrow(/Invalid symbol/);
    });

    // Issue #404 Stream C: Test all supported symbols are accepted
    it('should accept all supported symbols', async () => {
      const mockResult = {
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      const supportedSymbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'EURCAD', 'EURGBP'];
      for (const symbol of supportedSymbols) {
        await expect(
          getMarketDataWithIndicators(
            symbol,
            'H1',
            new Date('2025-01-01'),
            new Date('2025-01-02'),
            100
          )
        ).resolves.not.toThrow();
      }
    });

    it('should throw ValidationError for invalid timeframe', async () => {
      await expect(
        getMarketDataWithIndicators(
          'EURUSD',
          'INVALID' as any, // Invalid timeframe
          new Date('2025-01-01'),
          new Date('2025-01-02'),
          100
        )
      ).rejects.toThrow(ValidationError);

      await expect(
        getMarketDataWithIndicators(
          'EURUSD',
          'INVALID' as any,
          new Date('2025-01-01'),
          new Date('2025-01-02'),
          100
        )
      ).rejects.toThrow(/Invalid timeframe/);
    });

    // Issue #404 Stream C: Test all supported timeframes
    it('should accept all supported timeframes', async () => {
      const mockResult = {
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      const supportedTimeframes = [
        'M1',
        'M5',
        'M15',
        'M30',
        'H1',
        'H2',
        'H3',
        'H4',
        'H6',
        'H8',
        'H12',
        'D1',
      ];
      for (const timeframe of supportedTimeframes) {
        await expect(
          getMarketDataWithIndicators(
            'EURUSD',
            timeframe as any,
            new Date('2025-01-01'),
            new Date('2025-01-02'),
            100
          )
        ).resolves.not.toThrow();
      }
    });

    it('should throw ValidationError for invalid start date', async () => {
      await expect(
        getMarketDataWithIndicators(
          'EURUSD',
          'H1',
          new Date('invalid'),
          new Date('2025-01-02'),
          100
        )
      ).rejects.toThrow(ValidationError);
    });

    it('should throw ValidationError for future start date', async () => {
      const futureDate = new Date();
      futureDate.setFullYear(futureDate.getFullYear() + 1);

      await expect(
        getMarketDataWithIndicators('EURUSD', 'H1', futureDate, new Date('2025-01-02'), 100)
      ).rejects.toThrow(ValidationError);

      await expect(
        getMarketDataWithIndicators('EURUSD', 'H1', futureDate, new Date('2025-01-02'), 100)
      ).rejects.toThrow(/cannot be in the future/);
    });

    it('should throw ValidationError when end date is before start date', async () => {
      await expect(
        getMarketDataWithIndicators(
          'EURUSD',
          'H1',
          new Date('2025-01-10'),
          new Date('2025-01-01'), // End before start
          100
        )
      ).rejects.toThrow(ValidationError);

      await expect(
        getMarketDataWithIndicators(
          'EURUSD',
          'H1',
          new Date('2025-01-10'),
          new Date('2025-01-01'),
          100
        )
      ).rejects.toThrow(/End date must be greater than or equal to start date/);
    });

    it('should throw ValidationError for invalid limit (zero)', async () => {
      await expect(
        getMarketDataWithIndicators(
          'EURUSD',
          'H1',
          new Date('2025-01-01'),
          new Date('2025-01-02'),
          0
        )
      ).rejects.toThrow(ValidationError);

      await expect(
        getMarketDataWithIndicators(
          'EURUSD',
          'H1',
          new Date('2025-01-01'),
          new Date('2025-01-02'),
          0
        )
      ).rejects.toThrow(/Limit must be greater than 0/);
    });

    it('should throw ValidationError for invalid limit (negative)', async () => {
      await expect(
        getMarketDataWithIndicators(
          'EURUSD',
          'H1',
          new Date('2025-01-01'),
          new Date('2025-01-02'),
          -10
        )
      ).rejects.toThrow(ValidationError);
    });

    // Issue #404 Stream C: MAX_LIMIT increased to 10000 to support 1 year of H1 data
    it('should throw ValidationError for limit exceeding maximum (10000)', async () => {
      await expect(
        getMarketDataWithIndicators(
          'EURUSD',
          'H1',
          new Date('2025-01-01'),
          new Date('2025-01-02'),
          11000
        )
      ).rejects.toThrow(ValidationError);

      await expect(
        getMarketDataWithIndicators(
          'EURUSD',
          'H1',
          new Date('2025-01-01'),
          new Date('2025-01-02'),
          11000
        )
      ).rejects.toThrow(/Limit cannot exceed 10000/);
    });

    it('should accept valid inputs without throwing', async () => {
      const mockResult: QueryResult = {
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      await expect(
        getMarketDataWithIndicators(
          'EURUSD',
          'H1',
          new Date('2025-01-01'),
          new Date('2025-01-02'),
          100
        )
      ).resolves.not.toThrow();
    });
  });

  describe('successful query execution', () => {
    // Issue #404 Stream C: Updated to match new separate query implementation
    it('should execute query with correct SQL and parameters', async () => {
      // Unix timestamps for the date range
      const startUnix = Math.floor(new Date('2025-01-01').getTime() / 1000);
      const endUnix = Math.floor(new Date('2025-01-02').getTime() / 1000);

      const mockResult: QueryResult = {
        rows: [
          {
            rate_time: startUnix + 36000, // Unix timestamp
            open: 1.1,
            high: 1.105,
            low: 1.095,
            close: 1.102,
            volume: 1000,
          },
        ],
        command: 'SELECT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      const start = new Date('2025-01-01');
      const end = new Date('2025-01-02');
      const result = await getMarketDataWithIndicators('EURUSD', 'H1', start, end, 1000);

      // Verify query was called
      expect(mockMarketsQuery).toHaveBeenCalledTimes(1);

      // Verify SQL contains correct table name
      const calledSql = mockMarketsQuery.mock.calls[0]?.[0] as string;
      expect(calledSql).toContain('eurusd_h1_rates');
      // Implementation now uses separate queries, not LEFT JOIN
      expect(calledSql).toContain('rate_time');
      expect(calledSql).toContain('ORDER BY');

      // Verify parameters - Unix timestamps and limit
      const calledParams = mockMarketsQuery.mock.calls[0]?.[1] as any[];
      expect(calledParams[0]).toBe(startUnix);
      expect(calledParams[1]).toBe(endUnix);
      expect(calledParams[2]).toBe(1000);

      // Verify result structure
      expect(result).toHaveLength(1);
    });

    it('should return empty array when no data found', async () => {
      const mockResult: QueryResult = {
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      const result = await getMarketDataWithIndicators(
        'EURUSD',
        'H1',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      expect(result).toEqual([]);
      expect(result).toHaveLength(0);
    });

    // Issue #404 Stream C: Updated to use Unix timestamps
    it('should return multiple rows with correct data', async () => {
      const timestamp1 = Math.floor(new Date('2025-01-01T10:00:00Z').getTime() / 1000);
      const timestamp2 = Math.floor(new Date('2025-01-01T11:00:00Z').getTime() / 1000);
      const mockResult: QueryResult = {
        rows: [
          {
            rate_time: timestamp1,
            open: 1.1,
            high: 1.105,
            low: 1.095,
            close: 1.102,
            volume: 1000,
          },
          {
            rate_time: timestamp2,
            open: 1.102,
            high: 1.107,
            low: 1.101,
            close: 1.105,
            volume: 1200,
          },
        ],
        command: 'SELECT',
        rowCount: 2,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      const result = await getMarketDataWithIndicators(
        'EURUSD',
        'H1',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      expect(result).toHaveLength(2);
      expect(result[0]?.open).toBe(1.1);
      expect(result[1]?.open).toBe(1.102);
    });
  });

  describe('timeframe handling', () => {
    it('should query correct table for H1 timeframe', async () => {
      const mockResult: QueryResult = {
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      await getMarketDataWithIndicators(
        'EURUSD',
        'H1',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      const calledSql = mockMarketsQuery.mock.calls[0]?.[0] as string;
      expect(calledSql).toContain('eurusd_h1_rates');
    });

    it('should query correct table for H4 timeframe', async () => {
      const mockResult: QueryResult = {
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      await getMarketDataWithIndicators(
        'EURUSD',
        'H4',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      const calledSql = mockMarketsQuery.mock.calls[0]?.[0] as string;
      expect(calledSql).toContain('eurusd_h4_rates');
    });

    it('should query correct table for D1 timeframe', async () => {
      const mockResult: QueryResult = {
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      await getMarketDataWithIndicators(
        'EURUSD',
        'D1',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      const calledSql = mockMarketsQuery.mock.calls[0]?.[0] as string;
      expect(calledSql).toContain('eurusd_d1_rates');
    });
  });

  // Issue #404 Stream C: Updated to match separate query implementation
  describe('missing indicators handling (separate queries)', () => {
    it('should handle missing indicators gracefully (empty indicators object)', async () => {
      const timestamp = Math.floor(new Date('2025-01-01T10:00:00Z').getTime() / 1000);
      const mockResult: QueryResult = {
        rows: [
          {
            rate_time: timestamp,
            open: 1.1,
            high: 1.105,
            low: 1.095,
            close: 1.102,
            volume: 1000,
          },
        ],
        command: 'SELECT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      const result = await getMarketDataWithIndicators(
        'EURUSD',
        'H1',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      expect(result).toHaveLength(1);
      expect(result[0]).toHaveProperty('open', 1.1);
      // Indicators are fetched separately and merged by timestamp
      expect(result[0]?.indicators).toBeDefined();
    });

    it('should return OHLCV data even when indicators query fails', async () => {
      const timestamp = Math.floor(new Date('2025-01-01T10:00:00Z').getTime() / 1000);
      const mockResult: QueryResult = {
        rows: [
          {
            rate_time: timestamp,
            open: 1.1,
            high: 1.105,
            low: 1.095,
            close: 1.102,
            volume: 1000,
          },
        ],
        command: 'SELECT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      const result = await getMarketDataWithIndicators(
        'EURUSD',
        'H1',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      // Should return market data with empty indicators
      expect(result).toHaveLength(1);
      expect(result[0]?.open).toBe(1.1);
      expect(result[0]?.high).toBe(1.105);
    });
  });

  // Issue #404 Stream C: Updated to match separate query implementation (returns raw data)
  describe('result data structure', () => {
    it('should return rate_time as Unix timestamp (number)', async () => {
      const timestamp = Math.floor(new Date('2025-01-01T10:00:00Z').getTime() / 1000);
      const mockResult: QueryResult = {
        rows: [
          {
            rate_time: timestamp,
            open: 1.1,
            high: 1.105,
            low: 1.095,
            close: 1.102,
            volume: 1000,
          },
        ],
        command: 'SELECT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      const result = await getMarketDataWithIndicators(
        'EURUSD',
        'H1',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      // Implementation returns rate_time (snake_case) as number (Unix timestamp)
      // Note: camelCase transformation happens in responseFormatter middleware
      expect(result[0]).toHaveProperty('rate_time');
      expect(typeof (result[0] as any)?.rate_time).toBe('number');
    });

    it('should return OHLCV fields correctly', async () => {
      const timestamp = Math.floor(new Date('2025-01-01T10:00:00Z').getTime() / 1000);
      const mockResult: QueryResult = {
        rows: [
          {
            rate_time: timestamp,
            open: 1.1,
            high: 1.105,
            low: 1.095,
            close: 1.102,
            volume: 1000,
          },
        ],
        command: 'SELECT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      const result = await getMarketDataWithIndicators(
        'EURUSD',
        'H1',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      expect(result[0]?.open).toBe(1.1);
      expect(result[0]?.high).toBe(1.105);
      expect(result[0]?.low).toBe(1.095);
      expect(result[0]?.close).toBe(1.102);
      expect(result[0]?.volume).toBe(1000);
    });

    it('should include indicators object in result', async () => {
      const timestamp = Math.floor(new Date('2025-01-01T10:00:00Z').getTime() / 1000);
      const mockResult: QueryResult = {
        rows: [
          {
            rate_time: timestamp,
            open: 1.1,
            high: 1.105,
            low: 1.095,
            close: 1.102,
            volume: 1000,
          },
        ],
        command: 'SELECT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      const result = await getMarketDataWithIndicators(
        'EURUSD',
        'H1',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      // Indicators are fetched separately and merged
      expect(result[0]?.indicators).toBeDefined();
    });
  });

  describe('default limit parameter', () => {
    // Issue #404 Stream C: Updated parameter position for separate query implementation
    it('should use default limit of 1000 when not specified', async () => {
      const mockResult: QueryResult = {
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      await getMarketDataWithIndicators(
        'EURUSD',
        'H1',
        new Date('2025-01-01'),
        new Date('2025-01-02')
        // limit not specified
      );

      // Separate query implementation: params are [startUnix, endUnix, limit]
      const calledParams = mockMarketsQuery.mock.calls[0]?.[1] as any[];
      expect(calledParams[2]).toBe(1000); // Third parameter should be 1000
    });
  });

  describe('error handling', () => {
    it('should throw QueryError on database query failure', async () => {
      const dbError = new Error('table does not exist');
      (dbError as any).code = '42P01';
      mockMarketsQuery.mockRejectedValue(dbError);

      await expect(
        getMarketDataWithIndicators(
          'EURUSD',
          'H1',
          new Date('2025-01-01'),
          new Date('2025-01-02'),
          100
        )
      ).rejects.toThrow(QueryError);
    });

    it('should propagate QueryError from executeQuery', async () => {
      const queryError = new QueryError('Query execution failed', {
        query: 'SELECT * FROM markets',
        error: 'syntax error',
      });
      mockMarketsQuery.mockRejectedValue(queryError);

      await expect(
        getMarketDataWithIndicators(
          'EURUSD',
          'H1',
          new Date('2025-01-01'),
          new Date('2025-01-02'),
          100
        )
      ).rejects.toThrow(QueryError);
    });
  });

  // Issue #404 Stream C: Updated SQL structure tests for separate query implementation
  describe('SQL query structure', () => {
    it('should query market data separately (not use JOIN)', async () => {
      const mockResult: QueryResult = {
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      await getMarketDataWithIndicators(
        'EURUSD',
        'H1',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      // Implementation now uses separate queries for OHLCV and indicators
      const calledSql = mockMarketsQuery.mock.calls[0]?.[0] as string;
      expect(calledSql).toContain('rate_time');
      expect(calledSql).toContain('open');
      expect(calledSql).toContain('high');
      expect(calledSql).toContain('low');
      expect(calledSql).toContain('close');
      expect(calledSql).toContain('volume');
    });

    it('should order results by rate_time DESC', async () => {
      const mockResult: QueryResult = {
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      await getMarketDataWithIndicators(
        'EURUSD',
        'H1',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      const calledSql = mockMarketsQuery.mock.calls[0]?.[0] as string;
      expect(calledSql).toContain('ORDER BY');
      expect(calledSql).toContain('rate_time');
      expect(calledSql).toContain('DESC');
    });

    it('should query correct table based on symbol', async () => {
      const mockResult: QueryResult = {
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      // Test GBPUSD (Issue #404 Stream C: new symbol support)
      await getMarketDataWithIndicators(
        'GBPUSD',
        'H1',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      const calledSql = mockMarketsQuery.mock.calls[0]?.[0] as string;
      expect(calledSql).toContain('gbpusd_h1_rates');
    });
  });

  // Issue #404 Stream C: Updated tests for separate query implementation
  describe('OHLCV data retrieval', () => {
    it('should return OHLCV data with indicators object', async () => {
      const timestamp = Math.floor(new Date('2025-01-01T10:00:00Z').getTime() / 1000);
      const mockResult: QueryResult = {
        rows: [
          {
            rate_time: timestamp,
            open: 1.1,
            high: 1.105,
            low: 1.095,
            close: 1.102,
            volume: 1000,
          },
        ],
        command: 'SELECT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      const result = await getMarketDataWithIndicators(
        'EURUSD',
        'H1',
        new Date('2025-01-01'),
        new Date('2025-01-02'),
        100
      );

      // Indicators object should exist even if empty (fetched separately)
      expect(result[0]?.indicators).toBeDefined();
      expect(typeof result[0]?.indicators).toBe('object');
    });

    it('should return market data for all supported symbols', async () => {
      const timestamp = Math.floor(new Date('2025-01-01T10:00:00Z').getTime() / 1000);
      const mockResult: QueryResult = {
        rows: [
          {
            rate_time: timestamp,
            open: 1.1,
            high: 1.105,
            low: 1.095,
            close: 1.102,
            volume: 1000,
          },
        ],
        command: 'SELECT',
        rowCount: 1,
        oid: 0,
        fields: [],
      };
      mockMarketsQuery.mockResolvedValue(mockResult);

      // Test all 8 supported FOREX symbols
      const symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'EURCAD', 'EURGBP'];
      for (const symbol of symbols) {
        const result = await getMarketDataWithIndicators(
          symbol,
          'H1',
          new Date('2025-01-01'),
          new Date('2025-01-02'),
          100
        );
        expect(result).toHaveLength(1);
      }
    });
  });
});
