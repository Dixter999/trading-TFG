/**
 * Signal Preview Route Tests
 *
 * Tests for GET /api/paper-trading/signal-preview endpoint.
 *
 * Test-Driven Development approach:
 * 1. RED - Write failing tests first
 * 2. GREEN - Implement minimal code to pass
 * 3. REFACTOR - Improve code while tests stay green
 */

import request from 'supertest';
import express, { Express } from 'express';
import paperTradingRouter from '../paperTrading';
import * as signalPreviewQueries from '../../database/queries/signalPreview';

// Mock the query module
jest.mock('../../database/queries/signalPreview');
const mockQueries = signalPreviewQueries as jest.Mocked<typeof signalPreviewQueries>;

// Mock other paper trading dependencies to avoid errors
jest.mock('../../database/queries/paperTrading', () => ({
  getOpenPositions: jest.fn().mockResolvedValue([]),
  getTrades: jest.fn().mockResolvedValue([]),
  getPerformance: jest.fn().mockResolvedValue({
    profit_factor: 0,
    win_rate: 0,
    max_drawdown_pips: 0,
    total_trades: 0,
    winning_trades: 0,
    losing_trades: 0,
    total_pnl_pips: 0,
    pnl_unit: 'pips',
  }),
  getStatus: jest.fn().mockResolvedValue({
    enabled: true,
    last_update: null,
    active_symbols: [],
    open_positions_count: 0,
  }),
}));

jest.mock('../../database/queries/modelInfo', () => ({
  getModelInfo: jest.fn().mockResolvedValue({ symbols: [] }),
  getSymbolModelInfoByName: jest.fn().mockResolvedValue(null),
}));

jest.mock('../../database/queries/modelPerformance', () => ({
  getModelPerformance: jest.fn().mockReturnValue([]),
  getModelPerformanceBySymbol: jest.fn().mockReturnValue([]),
  getModelPerformanceBySymbolAndDirection: jest.fn().mockReturnValue([]),
}));

// Create test app
function createTestApp(): Express {
  const app = express();
  app.use(express.json());
  app.use(paperTradingRouter);
  return app;
}

describe('GET /api/paper-trading/signal-preview', () => {
  let app: Express;

  beforeEach(() => {
    jest.clearAllMocks();
    app = createTestApp();
  });

  it('should return signal preview data with candle groups', async () => {
    // Arrange
    const mockResponse: signalPreviewQueries.SignalPreviewResponse = {
      data: {
        candle_groups: [
          {
            close_time: '2026-02-11T14:00:00.000Z',
            seconds_until: 60,
            timeframes: ['H1', 'H2'],
            signal_count: 2,
            high_confidence_count: 1,
            signals: [
              {
                symbol: 'EURUSD',
                direction: 'long',
                signal_name: 'SMA50_200_RSI_long',
                timeframe: 'H1',
                confidence: 0.85,
                conditions: [
                  { name: 'SMA50 > SMA200', met: true, current: '1.0850', required: 'above' },
                ],
                model_consensus: {
                  agreement: '25/30',
                  models_agree: 25,
                  total_models: 30,
                  action: 'long',
                  confidence: 0.83,
                },
              },
              {
                symbol: 'GBPUSD',
                direction: 'short',
                signal_name: 'RSI_overbought',
                timeframe: 'H2',
                confidence: 0.72,
                conditions: [
                  { name: 'RSI > 70', met: true, current: '78', required: '> 70' },
                ],
              },
            ],
          },
        ],
        last_update: '2026-02-11T13:58:00.000Z',
      },
      metadata: {
        total_signals: 2,
        total_high_confidence: 1,
        next_close: '2026-02-11T14:00:00.000Z',
        timestamp: '2026-02-11T13:59:00.000Z',
      },
    };

    mockQueries.formatSignalPreviewResponse.mockResolvedValue(mockResponse);

    // Act
    const response = await request(app).get('/api/paper-trading/signal-preview');

    // Assert
    expect(response.status).toBe(200);
    expect(response.body.data.candle_groups).toHaveLength(1);
    expect(response.body.data.candle_groups[0].signals).toHaveLength(2);
    expect(response.body.metadata.total_signals).toBe(2);
    expect(response.body.metadata.total_high_confidence).toBe(1);
  });

  it('should return empty array when no signals available', async () => {
    // Arrange
    const mockResponse: signalPreviewQueries.SignalPreviewResponse = {
      data: {
        candle_groups: [],
        last_update: '2026-02-11T13:59:00.000Z',
      },
      metadata: {
        total_signals: 0,
        total_high_confidence: 0,
        next_close: null,
        timestamp: '2026-02-11T13:59:00.000Z',
      },
    };

    mockQueries.formatSignalPreviewResponse.mockResolvedValue(mockResponse);

    // Act
    const response = await request(app).get('/api/paper-trading/signal-preview');

    // Assert
    expect(response.status).toBe(200);
    expect(response.body.data.candle_groups).toEqual([]);
    expect(response.body.metadata.total_signals).toBe(0);
    expect(response.body.metadata.next_close).toBeNull();
  });

  it('should handle database errors gracefully', async () => {
    // Arrange
    mockQueries.formatSignalPreviewResponse.mockRejectedValue(new Error('Database connection failed'));

    // Act
    const response = await request(app).get('/api/paper-trading/signal-preview');

    // Assert
    expect(response.status).toBe(500);
  });

  it('should include metadata.timestamp in response', async () => {
    // Arrange
    const mockResponse: signalPreviewQueries.SignalPreviewResponse = {
      data: {
        candle_groups: [],
        last_update: '2026-02-11T13:59:00.000Z',
      },
      metadata: {
        total_signals: 0,
        total_high_confidence: 0,
        next_close: null,
        timestamp: '2026-02-11T13:59:00.000Z',
      },
    };

    mockQueries.formatSignalPreviewResponse.mockResolvedValue(mockResponse);

    // Act
    const response = await request(app).get('/api/paper-trading/signal-preview');

    // Assert
    expect(response.status).toBe(200);
    expect(response.body.metadata.timestamp).toBeDefined();
  });

  it('should return response under 500ms', async () => {
    // Arrange
    const mockResponse: signalPreviewQueries.SignalPreviewResponse = {
      data: {
        candle_groups: [],
        last_update: '2026-02-11T13:59:00.000Z',
      },
      metadata: {
        total_signals: 0,
        total_high_confidence: 0,
        next_close: null,
        timestamp: '2026-02-11T13:59:00.000Z',
      },
    };

    mockQueries.formatSignalPreviewResponse.mockResolvedValue(mockResponse);

    // Act
    const startTime = Date.now();
    const response = await request(app).get('/api/paper-trading/signal-preview');
    const duration = Date.now() - startTime;

    // Assert
    expect(response.status).toBe(200);
    expect(duration).toBeLessThan(500);
  });

  it('should return correct content type', async () => {
    // Arrange
    const mockResponse: signalPreviewQueries.SignalPreviewResponse = {
      data: {
        candle_groups: [],
        last_update: '2026-02-11T13:59:00.000Z',
      },
      metadata: {
        total_signals: 0,
        total_high_confidence: 0,
        next_close: null,
        timestamp: '2026-02-11T13:59:00.000Z',
      },
    };

    mockQueries.formatSignalPreviewResponse.mockResolvedValue(mockResponse);

    // Act
    const response = await request(app).get('/api/paper-trading/signal-preview');

    // Assert
    expect(response.headers['content-type']).toMatch(/application\/json/);
  });
});
