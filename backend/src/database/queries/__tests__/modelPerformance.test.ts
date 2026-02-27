/**
 * Model Performance Query Tests
 *
 * Tests for reading and serving model validation performance data
 * Issue #517 - PerformanceMetrics module with TDD
 *
 * Test-Driven Development approach:
 * 1. RED - Write failing tests first
 * 2. GREEN - Implement minimal code to pass
 * 3. REFACTOR - Improve code while tests stay green
 */

import * as fs from 'fs';
import {
  getModelPerformance,
  getModelPerformanceBySymbol,
  getModelPerformanceBySymbolAndDirection,
  ModelPerformance,
} from '../modelPerformance';

// Mock fs module
jest.mock('fs');
const mockFs = fs as jest.Mocked<typeof fs>;

describe('ModelPerformance Query Functions', () => {
  // Sample performance data matching actual JSON structure
  const sampleEurusdLongPerformance: ModelPerformance = {
    symbol: 'EURUSD',
    direction: 'long',
    fold: 7,
    validation_period: '2025-10-01 to 2026-01-04',
    total_trades: 28,
    win_rate: 42.857142857142854,
    avg_win: 38.016666666666666,
    avg_loss: -47.85625,
    risk_reward_ratio: 0.7943929302163597,
    expectancy: -11.05357142857143,
    profit_factor: 0.5957946976622698,
    total_pnl: -309.5,
    total_pnl_after_costs: -365.5,
    max_drawdown_pips: 307.8,
    max_drawdown_percent: 3.095,
    sharpe_ratio: -0.24689017415058326,
    is_profitable: false,
    status: 'unprofitable',
  };

  const sampleEurusdShortPerformance: ModelPerformance = {
    symbol: 'EURUSD',
    direction: 'short',
    fold: 28,
    validation_period: '2025-10-01 to 2026-01-04',
    total_trades: 35,
    win_rate: 48.5,
    avg_win: 42.0,
    avg_loss: -51.2,
    risk_reward_ratio: 0.82,
    expectancy: -8.5,
    profit_factor: 0.68,
    total_pnl: -280.0,
    total_pnl_after_costs: -350.0,
    max_drawdown_pips: 295.0,
    max_drawdown_percent: 2.95,
    sharpe_ratio: -0.2,
    is_profitable: false,
    status: 'unprofitable',
  };

  const sampleUsdjpyLongPerformance: ModelPerformance = {
    symbol: 'USDJPY',
    direction: 'long',
    fold: 5,
    validation_period: '2025-10-01 to 2026-01-04',
    total_trades: 42,
    win_rate: 55.5,
    avg_win: 45.0,
    avg_loss: -38.0,
    risk_reward_ratio: 1.18,
    expectancy: 5.2,
    profit_factor: 1.25,
    total_pnl: 218.4,
    total_pnl_after_costs: 134.4,
    max_drawdown_pips: 180.0,
    max_drawdown_percent: 1.8,
    sharpe_ratio: 0.45,
    is_profitable: true,
    status: 'profitable',
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getModelPerformance', () => {
    it('should return all model performance data when files exist', () => {
      // Arrange
      const mockFiles = [
        'EURUSD_long_fold007_performance.json',
        'EURUSD_short_fold028_performance.json',
        'USDJPY_long_fold005_performance.json',
      ];

      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue(mockFiles as any);
      mockFs.readFileSync
        .mockReturnValueOnce(JSON.stringify(sampleEurusdLongPerformance))
        .mockReturnValueOnce(JSON.stringify(sampleEurusdShortPerformance))
        .mockReturnValueOnce(JSON.stringify(sampleUsdjpyLongPerformance));

      // Act
      const result = getModelPerformance();

      // Assert
      expect(result).toHaveLength(3);
      expect(result[0]).toEqual(sampleEurusdLongPerformance);
      expect(result[1]).toEqual(sampleEurusdShortPerformance);
      expect(result[2]).toEqual(sampleUsdjpyLongPerformance);
    });

    it('should return empty array when results directory does not exist', () => {
      // Arrange
      mockFs.existsSync.mockReturnValue(false);

      // Act
      const result = getModelPerformance();

      // Assert
      expect(result).toEqual([]);
      expect(mockFs.readdirSync).not.toHaveBeenCalled();
    });

    it('should return empty array when no JSON files exist', () => {
      // Arrange
      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue([]);

      // Act
      const result = getModelPerformance();

      // Assert
      expect(result).toEqual([]);
    });

    it('should sort results by symbol then direction', () => {
      // Arrange
      const mockFiles = [
        'USDJPY_long_fold005_performance.json',
        'EURUSD_short_fold028_performance.json',
        'EURUSD_long_fold007_performance.json',
      ];

      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue(mockFiles as any);
      mockFs.readFileSync
        .mockReturnValueOnce(JSON.stringify(sampleUsdjpyLongPerformance))
        .mockReturnValueOnce(JSON.stringify(sampleEurusdShortPerformance))
        .mockReturnValueOnce(JSON.stringify(sampleEurusdLongPerformance));

      // Act
      const result = getModelPerformance();

      // Assert
      expect(result).toHaveLength(3);
      expect(result[0]!.symbol).toBe('EURUSD');
      expect(result[0]!.direction).toBe('long');
      expect(result[1]!.symbol).toBe('EURUSD');
      expect(result[1]!.direction).toBe('short');
      expect(result[2]!.symbol).toBe('USDJPY');
      expect(result[2]!.direction).toBe('long');
    });

    it('should handle invalid JSON gracefully', () => {
      // Arrange
      const mockFiles = ['EURUSD_long_fold007_performance.json', 'invalid_file.json'];

      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue(mockFiles as any);
      mockFs.readFileSync
        .mockReturnValueOnce(JSON.stringify(sampleEurusdLongPerformance))
        .mockReturnValueOnce('invalid json{');

      // Act
      const result = getModelPerformance();

      // Assert - should only return valid entries
      expect(result).toHaveLength(1);
      expect(result[0]).toEqual(sampleEurusdLongPerformance);
    });
  });

  describe('getModelPerformanceBySymbol', () => {
    it('should return performance data for specific symbol', () => {
      // Arrange
      const mockFiles = [
        'EURUSD_long_fold007_performance.json',
        'EURUSD_short_fold028_performance.json',
        'USDJPY_long_fold005_performance.json',
      ];

      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue(mockFiles as any);
      mockFs.readFileSync
        .mockReturnValueOnce(JSON.stringify(sampleEurusdLongPerformance))
        .mockReturnValueOnce(JSON.stringify(sampleEurusdShortPerformance))
        .mockReturnValueOnce(JSON.stringify(sampleUsdjpyLongPerformance));

      // Act
      const result = getModelPerformanceBySymbol('EURUSD');

      // Assert
      expect(result).toHaveLength(2);
      expect(result[0]!.symbol).toBe('EURUSD');
      expect(result[0]!.direction).toBe('long');
      expect(result[1]!.symbol).toBe('EURUSD');
      expect(result[1]!.direction).toBe('short');
    });

    it('should return empty array for unknown symbol', () => {
      // Arrange
      const mockFiles = ['EURUSD_long_fold007_performance.json'];

      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue(mockFiles as any);
      mockFs.readFileSync.mockReturnValueOnce(JSON.stringify(sampleEurusdLongPerformance));

      // Act
      const result = getModelPerformanceBySymbol('GBPUSD');

      // Assert
      expect(result).toEqual([]);
    });

    it('should be case-insensitive for symbol matching', () => {
      // Arrange
      const mockFiles = ['EURUSD_long_fold007_performance.json'];

      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue(mockFiles as any);
      mockFs.readFileSync.mockReturnValueOnce(JSON.stringify(sampleEurusdLongPerformance));

      // Act
      const result = getModelPerformanceBySymbol('eurusd');

      // Assert
      expect(result).toHaveLength(1);
      expect(result[0]!.symbol).toBe('EURUSD');
    });
  });

  describe('getModelPerformanceBySymbolAndDirection', () => {
    it('should return performance data for specific symbol and direction', () => {
      // Arrange
      const mockFiles = [
        'EURUSD_long_fold007_performance.json',
        'EURUSD_short_fold028_performance.json',
      ];

      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue(mockFiles as any);

      // Mock readFileSync to return correct data based on file name
      mockFs.readFileSync.mockImplementation((filePath: any) => {
        if (filePath.includes('EURUSD_long')) {
          return JSON.stringify(sampleEurusdLongPerformance);
        } else if (filePath.includes('EURUSD_short')) {
          return JSON.stringify(sampleEurusdShortPerformance);
        }
        return '{}';
      });

      // Act
      const result = getModelPerformanceBySymbolAndDirection('EURUSD', 'long');

      // Assert
      expect(result).toHaveLength(1);
      expect(result[0]!.symbol).toBe('EURUSD');
      expect(result[0]!.direction).toBe('long');
      expect(result[0]!.fold).toBe(7);
    });

    it('should return empty array for unknown symbol/direction combination', () => {
      // Arrange
      const mockFiles = ['EURUSD_long_fold007_performance.json'];

      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue(mockFiles as any);
      mockFs.readFileSync.mockImplementation((filePath: any) => {
        if (filePath.includes('EURUSD_long')) {
          return JSON.stringify(sampleEurusdLongPerformance);
        }
        return '{}';
      });

      // Act
      const result = getModelPerformanceBySymbolAndDirection('EURUSD', 'short');

      // Assert
      expect(result).toEqual([]);
    });

    it('should be case-insensitive for symbol and direction matching', () => {
      // Arrange
      const mockFiles = ['EURUSD_long_fold007_performance.json'];

      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue(mockFiles as any);
      mockFs.readFileSync.mockReturnValueOnce(JSON.stringify(sampleEurusdLongPerformance));

      // Act
      const result = getModelPerformanceBySymbolAndDirection('eurusd', 'LONG');

      // Assert
      expect(result).toHaveLength(1);
      expect(result[0]!.symbol).toBe('EURUSD');
      expect(result[0]!.direction).toBe('long');
    });
  });
});
