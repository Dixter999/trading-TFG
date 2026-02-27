/**
 * Model Info Database Query Tests
 *
 * Tests for reading trained model data from the training_jobs PostgreSQL table
 * Issue #583 - Stream C: Frontend Database-Driven Model Metrics
 *
 * Test-Driven Development approach:
 * 1. RED - Write failing tests first
 * 2. GREEN - Implement minimal code to pass
 * 3. REFACTOR - Improve code while tests stay green
 */

import { Pool } from 'pg';
import { executeQuery } from '../../utils/query';

// Mock the database utilities
jest.mock('../../utils/query');
jest.mock('../../connection', () => ({
  getAiModelPool: jest.fn(() => ({} as Pool)),
}));

const mockExecuteQuery = executeQuery as jest.MockedFunction<typeof executeQuery>;

// Import after mocking
import { getTrainedModelsFromDB, TrainedModelInfo } from '../modelInfo';

describe('getTrainedModelsFromDB', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should return models from database when rows exist', async () => {
    // Arrange
    const mockRows = [
      {
        symbol: 'gbpusd',
        direction: 'long',
        signal_name: 'SMA50_200_RSI_Stoch_long',
        timeframe: 'H2',
        profit_factor: '3.66',
        win_rate: '0.623',
        phase5_passed: true,
        model_path: 'models/hybrid_v4/gbpusd_long',
        completed_at: new Date('2026-01-21T19:00:00Z'),
        trades: 390,
      },
      {
        symbol: 'eurusd',
        direction: 'short',
        signal_name: 'RSI_overbought_short',
        timeframe: 'H4',
        profit_factor: '1.85',
        win_rate: '0.571',
        phase5_passed: true,
        model_path: 'models/hybrid_v4/eurusd_short',
        completed_at: new Date('2026-01-20T10:00:00Z'),
        trades: 210,
      },
    ];

    mockExecuteQuery.mockResolvedValueOnce(mockRows);

    // Act
    const result = await getTrainedModelsFromDB();

    // Assert
    expect(result).toHaveLength(2);
    expect(result[0]!.symbol).toBe('GBPUSD');
    expect(result[0]!.direction).toBe('long');
    expect(result[0]!.signal).toBe('SMA50_200_RSI_Stoch_long');
    expect(result[0]!.timeframe).toBe('H2');
    expect(result[0]!.profitFactor).toBe(3.66);
    expect(result[0]!.oosWinRate).toBeCloseTo(62.3, 0);
    expect(result[0]!.phase5Validated).toBe(true);
    expect(result[0]!.phase5TestProfitFactor).toBe(3.66);
    expect(result[0]!.phase5TestWinRate).toBeCloseTo(62.3, 0);
    expect(result[0]!.phase5TestTrades).toBe(390);
    expect(result[0]!.completedAt).toBe('2026-01-21T19:00:00.000Z');

    expect(result[1]!.symbol).toBe('EURUSD');
    expect(result[1]!.direction).toBe('short');
    expect(result[1]!.profitFactor).toBe(1.85);
  });

  it('should map DB column names to TypeScript interface correctly', async () => {
    // Arrange
    const mockRows = [
      {
        symbol: 'usdjpy',
        direction: 'long',
        signal_name: 'Stoch_RSI_long_15_30',
        timeframe: 'H1',
        profit_factor: '2.50',
        win_rate: '0.589',
        phase5_passed: true,
        model_path: 'models/hybrid_v4/usdjpy_long',
        completed_at: new Date('2026-01-22T08:00:00Z'),
        trades: 275,
      },
    ];

    mockExecuteQuery.mockResolvedValueOnce(mockRows);

    // Act
    const result = await getTrainedModelsFromDB();

    // Assert - verify every field of TrainedModelInfo is populated
    const model = result[0]!;
    expect(model).toEqual<TrainedModelInfo>({
      symbol: 'USDJPY',
      direction: 'long',
      signal: 'Stoch_RSI_long_15_30',
      timeframe: 'H1',
      foldCount: 30,
      targetFolds: 30,
      oosWinRate: 58.9,
      profitFactor: 2.50,
      pValue: 0.0001,
      phase3Trials: 50,
      phase5Validated: true,
      phase5TestWinRate: 58.9,
      phase5TestProfitFactor: 2.50,
      phase5TestTrades: 275,
      completedAt: '2026-01-22T08:00:00.000Z',
    });
  });

  it('should handle empty result set by returning empty array', async () => {
    // Arrange
    mockExecuteQuery.mockResolvedValueOnce([]);

    // Act
    const result = await getTrainedModelsFromDB();

    // Assert
    expect(result).toEqual([]);
  });

  it('should handle null profit_factor and win_rate gracefully', async () => {
    // Arrange - DB rows with null values (job completed but metrics not yet populated)
    const mockRows = [
      {
        symbol: 'gbpusd',
        direction: 'short',
        signal_name: 'SMA20_50_Stoch_BB_short',
        timeframe: 'H4',
        profit_factor: null,
        win_rate: null,
        phase5_passed: null,
        model_path: null,
        completed_at: null,
        trades: null,
      },
    ];

    mockExecuteQuery.mockResolvedValueOnce(mockRows);

    // Act
    const result = await getTrainedModelsFromDB();

    // Assert - should still return mapped model with 0/false defaults
    expect(result).toHaveLength(1);
    expect(result[0]!.profitFactor).toBe(0);
    expect(result[0]!.oosWinRate).toBe(0);
    expect(result[0]!.phase5Validated).toBe(false);
    expect(result[0]!.phase5TestTrades).toBe(0);
    expect(result[0]!.completedAt).toBe('');
  });

  it('should fall back to TRAINED_MODELS when database query fails', async () => {
    // Arrange - simulate DB error
    mockExecuteQuery.mockRejectedValueOnce(new Error('Connection refused'));

    // Act
    const result = await getTrainedModelsFromDB();

    // Assert - should return fallback data (TRAINED_MODELS array)
    // The fallback array has at least 1 entry (hardcoded models)
    expect(result.length).toBeGreaterThan(0);
    // Verify it's actual TrainedModelInfo data
    expect(result[0]).toHaveProperty('symbol');
    expect(result[0]).toHaveProperty('direction');
    expect(result[0]).toHaveProperty('signal');
    expect(result[0]).toHaveProperty('profitFactor');
    expect(result[0]).toHaveProperty('phase5Validated');
  });

  it('should query training_jobs with correct SQL', async () => {
    // Arrange
    mockExecuteQuery.mockResolvedValueOnce([]);

    // Act
    await getTrainedModelsFromDB();

    // Assert - verify the query was called with expected SQL
    expect(mockExecuteQuery).toHaveBeenCalledTimes(1);
    const [, sql] = mockExecuteQuery.mock.calls[0]!;
    expect(sql).toContain('training_jobs');
    expect(sql).toContain("status = 'completed'");
    expect(sql).toContain('profit_factor IS NOT NULL');
    expect(sql).toContain('ORDER BY');
  });

  it('should uppercase the symbol from DB', async () => {
    // Arrange - DB stores lowercase symbols
    const mockRows = [
      {
        symbol: 'eurcad',
        direction: 'long',
        signal_name: 'SMA20_200_BB_long',
        timeframe: 'H1',
        profit_factor: '1.42',
        win_rate: '0.55',
        phase5_passed: false,
        model_path: null,
        completed_at: new Date('2026-01-25T12:00:00Z'),
        trades: 150,
      },
    ];

    mockExecuteQuery.mockResolvedValueOnce(mockRows);

    // Act
    const result = await getTrainedModelsFromDB();

    // Assert
    expect(result[0]!.symbol).toBe('EURCAD');
  });
});
