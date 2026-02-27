/**
 * Signal Preview Database Query Tests
 *
 * Tests for querying signal preview snapshots from the ai_model database.
 * Table: signal_preview_snapshots
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
import { getSignalPreviewSnapshots, SignalPreviewSnapshot } from '../signalPreview';

describe('getSignalPreviewSnapshots', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should return signal snapshots from database when rows exist', async () => {
    // Arrange
    const mockRows: SignalPreviewSnapshot[] = [
      {
        id: 1,
        symbol: 'EURUSD',
        direction: 'long',
        signal_name: 'SMA50_200_RSI_long',
        timeframe: 'H1',
        confidence: 0.85,
        next_candle_close: new Date('2026-02-11T14:00:00Z'),
        conditions: JSON.stringify([
          { name: 'SMA50 > SMA200', met: true, current: '1.0850', required: 'above' },
          { name: 'RSI < 70', met: true, current: '55', required: '< 70' },
        ]),
        model_consensus: JSON.stringify({
          agreement: '25/30',
          models_agree: 25,
          total_models: 30,
          action: 'long',
          confidence: 0.83,
        }),
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
      {
        id: 2,
        symbol: 'GBPUSD',
        direction: 'short',
        signal_name: 'RSI_overbought_short',
        timeframe: 'H4',
        confidence: 0.72,
        next_candle_close: new Date('2026-02-11T16:00:00Z'),
        conditions: JSON.stringify([
          { name: 'RSI > 70', met: true, current: '78', required: '> 70' },
        ]),
        model_consensus: null,
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
    ];

    mockExecuteQuery.mockResolvedValueOnce(mockRows);

    // Act
    const result = await getSignalPreviewSnapshots();

    // Assert
    expect(result).toHaveLength(2);
    expect(result[0]!.symbol).toBe('EURUSD');
    expect(result[0]!.direction).toBe('long');
    expect(result[0]!.signal_name).toBe('SMA50_200_RSI_long');
    expect(result[0]!.confidence).toBe(0.85);
    expect(result[1]!.symbol).toBe('GBPUSD');
    expect(result[1]!.direction).toBe('short');
  });

  it('should query with correct SQL (2-minute window)', async () => {
    // Arrange
    mockExecuteQuery.mockResolvedValueOnce([]);

    // Act
    await getSignalPreviewSnapshots();

    // Assert
    expect(mockExecuteQuery).toHaveBeenCalledTimes(1);
    const [, sql] = mockExecuteQuery.mock.calls[0]!;
    expect(sql).toContain('signal_preview_snapshots');
    expect(sql).toContain("NOW() - INTERVAL '2 minutes'");
    expect(sql).toContain('ORDER BY');
    expect(sql).toContain('next_candle_close');
    expect(sql).toContain('confidence DESC');
  });

  it('should handle empty result set by returning empty array', async () => {
    // Arrange
    mockExecuteQuery.mockResolvedValueOnce([]);

    // Act
    const result = await getSignalPreviewSnapshots();

    // Assert
    expect(result).toEqual([]);
  });

  it('should handle database errors gracefully', async () => {
    // Arrange
    mockExecuteQuery.mockRejectedValueOnce(new Error('Connection refused'));

    // Act & Assert
    await expect(getSignalPreviewSnapshots()).rejects.toThrow('Connection refused');
  });

  it('should handle null model_consensus gracefully', async () => {
    // Arrange
    const mockRows: SignalPreviewSnapshot[] = [
      {
        id: 1,
        symbol: 'USDJPY',
        direction: 'long',
        signal_name: 'Stoch_cross_long',
        timeframe: 'H2',
        confidence: 0.65,
        next_candle_close: new Date('2026-02-11T14:00:00Z'),
        conditions: JSON.stringify([{ name: 'Stoch cross up', met: true, current: '25', required: '< 30' }]),
        model_consensus: null,
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
    ];

    mockExecuteQuery.mockResolvedValueOnce(mockRows);

    // Act
    const result = await getSignalPreviewSnapshots();

    // Assert
    expect(result).toHaveLength(1);
    expect(result[0]!.model_consensus).toBeNull();
  });
});

describe('formatSignalPreviewResponse', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Mock Date.now for consistent testing
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2026-02-11T13:59:00Z'));
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should group signals by next_candle_close time', async () => {
    // Arrange
    const mockRows: SignalPreviewSnapshot[] = [
      {
        id: 1,
        symbol: 'EURUSD',
        direction: 'long',
        signal_name: 'Signal1',
        timeframe: 'H1',
        confidence: 0.85,
        next_candle_close: new Date('2026-02-11T14:00:00Z'),
        conditions: JSON.stringify([]),
        model_consensus: null,
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
      {
        id: 2,
        symbol: 'GBPUSD',
        direction: 'short',
        signal_name: 'Signal2',
        timeframe: 'H1',
        confidence: 0.75,
        next_candle_close: new Date('2026-02-11T14:00:00Z'),
        conditions: JSON.stringify([]),
        model_consensus: null,
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
      {
        id: 3,
        symbol: 'USDJPY',
        direction: 'long',
        signal_name: 'Signal3',
        timeframe: 'H4',
        confidence: 0.70,
        next_candle_close: new Date('2026-02-11T16:00:00Z'),
        conditions: JSON.stringify([]),
        model_consensus: null,
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
    ];

    mockExecuteQuery.mockResolvedValueOnce(mockRows);

    // Import the function we need to test
    const { formatSignalPreviewResponse } = await import('../signalPreview');

    // Act
    const response = await formatSignalPreviewResponse();

    // Assert
    expect(response.data.candle_groups).toHaveLength(2);
    expect(response.data.candle_groups[0]!.signals).toHaveLength(2);
    expect(response.data.candle_groups[1]!.signals).toHaveLength(1);
  });

  it('should calculate seconds_until for each candle group', async () => {
    // Arrange
    const mockRows: SignalPreviewSnapshot[] = [
      {
        id: 1,
        symbol: 'EURUSD',
        direction: 'long',
        signal_name: 'Signal1',
        timeframe: 'H1',
        confidence: 0.85,
        next_candle_close: new Date('2026-02-11T14:00:00Z'),
        conditions: JSON.stringify([]),
        model_consensus: null,
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
    ];

    mockExecuteQuery.mockResolvedValueOnce(mockRows);

    const { formatSignalPreviewResponse } = await import('../signalPreview');

    // Act
    const response = await formatSignalPreviewResponse();

    // Assert - 14:00:00 - 13:59:00 = 60 seconds
    expect(response.data.candle_groups[0]!.seconds_until).toBe(60);
  });

  it('should count high confidence signals (>= 0.8)', async () => {
    // Arrange
    const mockRows: SignalPreviewSnapshot[] = [
      {
        id: 1,
        symbol: 'EURUSD',
        direction: 'long',
        signal_name: 'Signal1',
        timeframe: 'H1',
        confidence: 0.85,
        next_candle_close: new Date('2026-02-11T14:00:00Z'),
        conditions: JSON.stringify([]),
        model_consensus: null,
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
      {
        id: 2,
        symbol: 'GBPUSD',
        direction: 'short',
        signal_name: 'Signal2',
        timeframe: 'H1',
        confidence: 0.65,
        next_candle_close: new Date('2026-02-11T14:00:00Z'),
        conditions: JSON.stringify([]),
        model_consensus: null,
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
    ];

    mockExecuteQuery.mockResolvedValueOnce(mockRows);

    const { formatSignalPreviewResponse } = await import('../signalPreview');

    // Act
    const response = await formatSignalPreviewResponse();

    // Assert
    expect(response.data.candle_groups[0]!.high_confidence_count).toBe(1);
    expect(response.metadata.total_high_confidence).toBe(1);
  });

  it('should return correct metadata', async () => {
    // Arrange
    const mockRows: SignalPreviewSnapshot[] = [
      {
        id: 1,
        symbol: 'EURUSD',
        direction: 'long',
        signal_name: 'Signal1',
        timeframe: 'H1',
        confidence: 0.85,
        next_candle_close: new Date('2026-02-11T14:00:00Z'),
        conditions: JSON.stringify([]),
        model_consensus: null,
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
    ];

    mockExecuteQuery.mockResolvedValueOnce(mockRows);

    const { formatSignalPreviewResponse } = await import('../signalPreview');

    // Act
    const response = await formatSignalPreviewResponse();

    // Assert
    expect(response.metadata.total_signals).toBe(1);
    expect(response.metadata.next_close).toBe('2026-02-11T14:00:00.000Z');
    expect(response.metadata.timestamp).toBeDefined();
  });

  it('should handle empty snapshots gracefully', async () => {
    // Arrange
    mockExecuteQuery.mockResolvedValueOnce([]);

    const { formatSignalPreviewResponse } = await import('../signalPreview');

    // Act
    const response = await formatSignalPreviewResponse();

    // Assert
    expect(response.data.candle_groups).toEqual([]);
    expect(response.metadata.total_signals).toBe(0);
    expect(response.metadata.total_high_confidence).toBe(0);
    expect(response.metadata.next_close).toBeNull();
  });

  it('should parse conditions JSON correctly', async () => {
    // Arrange
    const mockRows: SignalPreviewSnapshot[] = [
      {
        id: 1,
        symbol: 'EURUSD',
        direction: 'long',
        signal_name: 'Signal1',
        timeframe: 'H1',
        confidence: 0.85,
        next_candle_close: new Date('2026-02-11T14:00:00Z'),
        conditions: JSON.stringify([
          { name: 'SMA50 > SMA200', met: true, current: '1.0850', required: 'above' },
        ]),
        model_consensus: JSON.stringify({
          agreement: '25/30',
          models_agree: 25,
          total_models: 30,
          action: 'long',
          confidence: 0.83,
        }),
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
    ];

    mockExecuteQuery.mockResolvedValueOnce(mockRows);

    const { formatSignalPreviewResponse } = await import('../signalPreview');

    // Act
    const response = await formatSignalPreviewResponse();

    // Assert
    const signal = response.data.candle_groups[0]!.signals[0]!;
    expect(signal.conditions).toHaveLength(1);
    expect(signal.conditions[0]!.name).toBe('SMA50 > SMA200');
    expect(signal.conditions[0]!.met).toBe(true);
    expect(signal.model_consensus).toBeDefined();
    expect(signal.model_consensus!.agreement).toBe('25/30');
  });

  it('should collect unique timeframes for each candle group', async () => {
    // Arrange
    const mockRows: SignalPreviewSnapshot[] = [
      {
        id: 1,
        symbol: 'EURUSD',
        direction: 'long',
        signal_name: 'Signal1',
        timeframe: 'H1',
        confidence: 0.85,
        next_candle_close: new Date('2026-02-11T14:00:00Z'),
        conditions: JSON.stringify([]),
        model_consensus: null,
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
      {
        id: 2,
        symbol: 'GBPUSD',
        direction: 'short',
        signal_name: 'Signal2',
        timeframe: 'H1',
        confidence: 0.75,
        next_candle_close: new Date('2026-02-11T14:00:00Z'),
        conditions: JSON.stringify([]),
        model_consensus: null,
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
      {
        id: 3,
        symbol: 'USDJPY',
        direction: 'long',
        signal_name: 'Signal3',
        timeframe: 'H2',
        confidence: 0.70,
        next_candle_close: new Date('2026-02-11T14:00:00Z'),
        conditions: JSON.stringify([]),
        model_consensus: null,
        timestamp: new Date('2026-02-11T13:58:00Z'),
      },
    ];

    mockExecuteQuery.mockResolvedValueOnce(mockRows);

    const { formatSignalPreviewResponse } = await import('../signalPreview');

    // Act
    const response = await formatSignalPreviewResponse();

    // Assert
    expect(response.data.candle_groups[0]!.timeframes).toContain('H1');
    expect(response.data.candle_groups[0]!.timeframes).toContain('H2');
    expect(response.data.candle_groups[0]!.timeframes).toHaveLength(2);
  });
});
