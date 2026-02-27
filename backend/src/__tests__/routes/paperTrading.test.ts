/**
 * Paper Trading Routes Tests
 *
 * TDD: RED-GREEN-REFACTOR
 * Integration tests for paper trading API endpoints
 *
 * Issue #435 - Real-time Monitoring Dashboard Backend API
 */

import request from 'supertest';
import { app } from '../../index';
import * as paperTradingQueries from '../../database/queries/paperTrading';
import { QueryError } from '../../database/errors';

// Mock the database layer
jest.mock('../../database/queries/paperTrading');

describe('Paper Trading Routes', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // ============================================================================
  // GET /api/paper-trading/positions
  // ============================================================================
  describe('GET /api/paper-trading/positions', () => {
    it('should return 200 with open positions', async () => {
      // Arrange
      const mockPositions = [
        {
          id: 1,
          symbol: 'EURUSD',
          direction: 'LONG',
          entry_price: 1.085,
          current_price: 1.0875,
          pnl_pips: 25,
          sl_pips: 30,
          tp_pips: 60,
          opened_at: '2024-01-15T10:00:00Z',
        },
        {
          id: 2,
          symbol: 'GBPUSD',
          direction: 'SHORT',
          entry_price: 1.265,
          current_price: 1.263,
          pnl_pips: 20,
          sl_pips: 25,
          tp_pips: 50,
          opened_at: '2024-01-15T11:00:00Z',
        },
      ];

      (paperTradingQueries.getOpenPositions as jest.Mock).mockResolvedValue(mockPositions);

      // Act
      const response = await request(app).get('/api/paper-trading/positions');

      // Assert
      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('data');
      expect(response.body.data).toHaveLength(2);
      expect(response.body.data[0]).toHaveProperty('symbol', 'EURUSD');
      expect(response.body.data[0]).toHaveProperty('direction', 'LONG');
      expect(response.body.data[0]).toHaveProperty('pnl_pips', 25);
    });

    it('should return 200 with empty array when no open positions', async () => {
      // Arrange
      (paperTradingQueries.getOpenPositions as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/paper-trading/positions');

      // Assert
      expect(response.status).toBe(200);
      expect(response.body.data).toEqual([]);
    });

    it('should return 500 for database errors', async () => {
      // Arrange
      const queryError = new QueryError('Database query failed', new Error('Connection timeout'));
      (paperTradingQueries.getOpenPositions as jest.Mock).mockRejectedValue(queryError);

      // Act
      const response = await request(app).get('/api/paper-trading/positions');

      // Assert
      expect(response.status).toBe(500);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error.code).toBe('QUERY_ERROR');
    });
  });

  // ============================================================================
  // GET /api/paper-trading/trades
  // ============================================================================
  describe('GET /api/paper-trading/trades', () => {
    it('should return 200 with trade history', async () => {
      // Arrange
      const mockTrades = [
        {
          id: 1,
          symbol: 'EURUSD',
          direction: 'LONG',
          entry_price: 1.08,
          exit_price: 1.085,
          pnl_pips: 50,
          exit_reason: 'TP_HIT',
          opened_at: '2024-01-14T10:00:00Z',
          closed_at: '2024-01-14T14:00:00Z',
        },
      ];

      (paperTradingQueries.getTrades as jest.Mock).mockResolvedValue(mockTrades);

      // Act
      const response = await request(app).get('/api/paper-trading/trades');

      // Assert
      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('data');
      expect(response.body.data).toHaveLength(1);
      expect(response.body.data[0]).toHaveProperty('exit_reason', 'TP_HIT');
    });

    it('should filter trades by symbol', async () => {
      // Arrange
      const mockTrades = [
        {
          id: 1,
          symbol: 'EURUSD',
          direction: 'LONG',
          entry_price: 1.08,
          exit_price: 1.085,
          pnl_pips: 50,
          exit_reason: 'TP_HIT',
          opened_at: '2024-01-14T10:00:00Z',
          closed_at: '2024-01-14T14:00:00Z',
        },
      ];

      (paperTradingQueries.getTrades as jest.Mock).mockResolvedValue(mockTrades);

      // Act
      const response = await request(app)
        .get('/api/paper-trading/trades')
        .query({ symbol: 'EURUSD' });

      // Assert
      expect(response.status).toBe(200);
      expect(paperTradingQueries.getTrades).toHaveBeenCalledWith(
        expect.objectContaining({ symbol: 'EURUSD' })
      );
    });

    it('should filter trades by date range', async () => {
      // Arrange
      (paperTradingQueries.getTrades as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/paper-trading/trades').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.status).toBe(200);
      expect(paperTradingQueries.getTrades).toHaveBeenCalledWith(
        expect.objectContaining({
          start: expect.any(Date),
          end: expect.any(Date),
        })
      );
    });

    it('should filter trades by direction', async () => {
      // Arrange
      (paperTradingQueries.getTrades as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app)
        .get('/api/paper-trading/trades')
        .query({ direction: 'LONG' });

      // Assert
      expect(response.status).toBe(200);
      expect(paperTradingQueries.getTrades).toHaveBeenCalledWith(
        expect.objectContaining({ direction: 'LONG' })
      );
    });

    it('should support limit parameter', async () => {
      // Arrange
      (paperTradingQueries.getTrades as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/paper-trading/trades').query({ limit: 50 });

      // Assert
      expect(response.status).toBe(200);
      expect(paperTradingQueries.getTrades).toHaveBeenCalledWith(
        expect.objectContaining({ limit: 50 })
      );
    });

    it('should return 400 for invalid date format', async () => {
      // Act
      const response = await request(app).get('/api/paper-trading/trades').query({
        start: 'invalid-date',
      });

      // Assert
      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error.message).toMatch(/date/i);
    });

    it('should return 400 for invalid direction', async () => {
      // Act
      const response = await request(app).get('/api/paper-trading/trades').query({
        direction: 'INVALID',
      });

      // Assert
      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error.message).toMatch(/direction/i);
    });

    it('should return 500 for database errors', async () => {
      // Arrange
      const queryError = new QueryError('Database query failed', new Error('Connection timeout'));
      (paperTradingQueries.getTrades as jest.Mock).mockRejectedValue(queryError);

      // Act
      const response = await request(app).get('/api/paper-trading/trades');

      // Assert
      expect(response.status).toBe(500);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error.code).toBe('QUERY_ERROR');
    });
  });

  // ============================================================================
  // GET /api/paper-trading/performance
  // ============================================================================
  describe('GET /api/paper-trading/performance', () => {
    it('should return 200 with performance metrics', async () => {
      // Arrange
      const mockPerformance = {
        profit_factor: 1.85,
        win_rate: 62.5,
        max_drawdown_pips: 150,
        total_trades: 40,
        winning_trades: 25,
        losing_trades: 15,
        total_pnl_pips: 850,
      };

      (paperTradingQueries.getPerformance as jest.Mock).mockResolvedValue(mockPerformance);

      // Act
      const response = await request(app).get('/api/paper-trading/performance');

      // Assert
      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('data');
      expect(response.body.data).toHaveProperty('profit_factor', 1.85);
      expect(response.body.data).toHaveProperty('win_rate', 62.5);
      expect(response.body.data).toHaveProperty('total_trades', 40);
      expect(response.body.data).toHaveProperty('total_pnl_pips', 850);
    });

    it('should filter performance by symbol', async () => {
      // Arrange
      const mockPerformance = {
        profit_factor: 2.0,
        win_rate: 65,
        max_drawdown_pips: 100,
        total_trades: 20,
        winning_trades: 13,
        losing_trades: 7,
        total_pnl_pips: 500,
      };

      (paperTradingQueries.getPerformance as jest.Mock).mockResolvedValue(mockPerformance);

      // Act
      const response = await request(app)
        .get('/api/paper-trading/performance')
        .query({ symbol: 'EURUSD' });

      // Assert
      expect(response.status).toBe(200);
      expect(paperTradingQueries.getPerformance).toHaveBeenCalledWith(
        expect.objectContaining({ symbol: 'EURUSD' })
      );
    });

    it('should filter performance by period', async () => {
      // Arrange
      const mockPerformance = {
        profit_factor: 1.5,
        win_rate: 55,
        max_drawdown_pips: 200,
        total_trades: 100,
        winning_trades: 55,
        losing_trades: 45,
        total_pnl_pips: 1500,
      };

      (paperTradingQueries.getPerformance as jest.Mock).mockResolvedValue(mockPerformance);

      // Act
      const response = await request(app).get('/api/paper-trading/performance').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.status).toBe(200);
      expect(paperTradingQueries.getPerformance).toHaveBeenCalledWith(
        expect.objectContaining({
          start: expect.any(Date),
          end: expect.any(Date),
        })
      );
    });

    it('should return default values when no trades exist', async () => {
      // Arrange
      const mockPerformance = {
        profit_factor: 0,
        win_rate: 0,
        max_drawdown_pips: 0,
        total_trades: 0,
        winning_trades: 0,
        losing_trades: 0,
        total_pnl_pips: 0,
      };

      (paperTradingQueries.getPerformance as jest.Mock).mockResolvedValue(mockPerformance);

      // Act
      const response = await request(app).get('/api/paper-trading/performance');

      // Assert
      expect(response.status).toBe(200);
      expect(response.body.data.total_trades).toBe(0);
      expect(response.body.data.win_rate).toBe(0);
    });

    it('should return 500 for database errors', async () => {
      // Arrange
      const queryError = new QueryError('Database query failed', new Error('Connection timeout'));
      (paperTradingQueries.getPerformance as jest.Mock).mockRejectedValue(queryError);

      // Act
      const response = await request(app).get('/api/paper-trading/performance');

      // Assert
      expect(response.status).toBe(500);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error.code).toBe('QUERY_ERROR');
    });
  });

  // ============================================================================
  // GET /api/paper-trading/status
  // ============================================================================
  describe('GET /api/paper-trading/status', () => {
    it('should return 200 with system status', async () => {
      // Arrange
      const mockStatus = {
        enabled: true,
        last_update: '2024-01-15T15:30:00Z',
        active_symbols: ['EURUSD', 'GBPUSD', 'USDJPY'],
        open_positions_count: 2,
      };

      (paperTradingQueries.getStatus as jest.Mock).mockResolvedValue(mockStatus);

      // Act
      const response = await request(app).get('/api/paper-trading/status');

      // Assert
      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('data');
      expect(response.body.data).toHaveProperty('enabled', true);
      expect(response.body.data).toHaveProperty('active_symbols');
      expect(response.body.data.active_symbols).toContain('EURUSD');
      expect(response.body.data).toHaveProperty('open_positions_count', 2);
    });

    it('should return status when system is disabled', async () => {
      // Arrange
      const mockStatus = {
        enabled: false,
        last_update: null,
        active_symbols: [],
        open_positions_count: 0,
      };

      (paperTradingQueries.getStatus as jest.Mock).mockResolvedValue(mockStatus);

      // Act
      const response = await request(app).get('/api/paper-trading/status');

      // Assert
      expect(response.status).toBe(200);
      expect(response.body.data.enabled).toBe(false);
      expect(response.body.data.active_symbols).toEqual([]);
    });

    it('should return 500 for database errors', async () => {
      // Arrange
      const queryError = new QueryError('Database query failed', new Error('Connection timeout'));
      (paperTradingQueries.getStatus as jest.Mock).mockRejectedValue(queryError);

      // Act
      const response = await request(app).get('/api/paper-trading/status');

      // Assert
      expect(response.status).toBe(500);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error.code).toBe('QUERY_ERROR');
    });
  });

  // ============================================================================
  // Response Format Validation
  // ============================================================================
  describe('Response Format', () => {
    it('should include metadata in positions response', async () => {
      // Arrange
      (paperTradingQueries.getOpenPositions as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/paper-trading/positions');

      // Assert
      expect(response.body).toHaveProperty('metadata');
      expect(response.body.metadata).toHaveProperty('timestamp');
      expect(response.body.metadata).toHaveProperty('count');
    });

    it('should include metadata in trades response', async () => {
      // Arrange
      (paperTradingQueries.getTrades as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/paper-trading/trades');

      // Assert
      expect(response.body).toHaveProperty('metadata');
      expect(response.body.metadata).toHaveProperty('timestamp');
      expect(response.body.metadata).toHaveProperty('count');
    });

    it('should include CORS headers', async () => {
      // Arrange
      (paperTradingQueries.getOpenPositions as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/paper-trading/positions');

      // Assert
      expect(response.headers).toHaveProperty('access-control-allow-origin');
    });
  });
});
