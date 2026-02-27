/**
 * Markets Route Tests
 *
 * TDD: RED-GREEN-REFACTOR
 * Integration tests for the markets route
 */

import request from 'supertest';
import { app } from '../../index';
import * as marketDataQueries from '../../database/queries/marketData';
import { ValidationError, QueryError } from '../../database/errors';

// Mock the database layer
jest.mock('../../database/queries/marketData');

describe('Markets Routes', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('GET /api/markets/:symbol/:timeframe', () => {
    it('should return 200 with market data for valid request', async () => {
      // Arrange
      const mockData = [
        {
          rateTime: new Date('2024-01-15T12:00:00Z'),
          open: 1.085,
          high: 1.0875,
          low: 1.084,
          close: 1.0865,
          volume: 1500,
          indicators: {
            sma_20: 1.0855,
            rsi_14: 55.3,
          },
        },
      ];

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue(mockData);

      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 1000,
      });

      // Assert
      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('data');
      expect(response.body).toHaveProperty('metadata');
      expect(response.body.data).toHaveLength(1);
      expect(response.body.metadata).toHaveProperty('count', 1);
      expect(response.body.metadata).toHaveProperty('timestamp');
      expect(response.body.metadata).toHaveProperty('requestId');
    });

    it('should return 200 with empty array when no data found', async () => {
      // Arrange
      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.status).toBe(200);
      expect(response.body.data).toEqual([]);
      expect(response.body.metadata.count).toBe(0);
    });

    it('should handle H4 timeframe', async () => {
      // Arrange
      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/markets/EURUSD/H4').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.status).toBe(200);
      expect(marketDataQueries.getMarketDataWithIndicators).toHaveBeenCalledWith(
        'EURUSD',
        'H4',
        expect.any(Date),
        expect.any(Date),
        expect.any(Number)
      );
    });

    it('should handle D1 timeframe', async () => {
      // Arrange
      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/markets/EURUSD/D1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.status).toBe(200);
      expect(marketDataQueries.getMarketDataWithIndicators).toHaveBeenCalledWith(
        'EURUSD',
        'D1',
        expect.any(Date),
        expect.any(Date),
        expect.any(Number)
      );
    });

    it('should use default limit when not provided', async () => {
      // Arrange
      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.status).toBe(200);
      expect(marketDataQueries.getMarketDataWithIndicators).toHaveBeenCalledWith(
        'EURUSD',
        'H1',
        expect.any(Date),
        expect.any(Date),
        1000 // default limit
      );
    });

    it('should respect custom limit parameter', async () => {
      // Arrange
      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 500,
      });

      // Assert
      expect(response.status).toBe(200);
      expect(marketDataQueries.getMarketDataWithIndicators).toHaveBeenCalledWith(
        'EURUSD',
        'H1',
        expect.any(Date),
        expect.any(Date),
        500
      );
    });
  });

  describe('GET /api/markets/:symbol/:timeframe - Error Handling', () => {
    it('should return 400 for missing start parameter', async () => {
      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error.message).toMatch(/start/i);
    });

    it('should return 400 for missing end parameter', async () => {
      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
      });

      // Assert
      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error.message).toMatch(/end/i);
    });

    it('should return 400 for invalid date format', async () => {
      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: 'invalid-date',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error.message).toMatch(/date/i);
    });

    it('should return 400 for invalid limit', async () => {
      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 'not-a-number',
      });

      // Assert
      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error.message).toMatch(/limit/i);
    });

    it('should return 400 for ValidationError from database layer', async () => {
      // Arrange
      const validationError = new ValidationError('Invalid timeframe', {
        field: 'timeframe',
        value: 'H2',
      });

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockRejectedValue(
        validationError
      );

      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('should return 500 for QueryError from database layer', async () => {
      // Arrange
      const queryError = new QueryError('Database query failed', new Error('Connection timeout'));

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockRejectedValue(queryError);

      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.status).toBe(500);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error.code).toBe('QUERY_ERROR');
    });

    it('should return 500 for unexpected errors', async () => {
      // Arrange
      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockRejectedValue(
        new Error('Unexpected error')
      );

      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.status).toBe(500);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error.code).toBe('INTERNAL_ERROR');
    });
  });

  describe('GET /api/markets/:symbol/:timeframe - Response Format', () => {
    it('should include requestId in metadata', async () => {
      // Arrange
      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.body.metadata.requestId).toMatch(/^req_/);
    });

    it('should include ISO 8601 timestamp in metadata', async () => {
      // Arrange
      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.body.metadata.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
    });

    it('should include correct count in metadata', async () => {
      // Arrange
      const mockData = Array(5).fill({
        rateTime: new Date('2024-01-15T12:00:00Z'),
        open: 1.085,
        high: 1.0875,
        low: 1.084,
        close: 1.0865,
        volume: 1500,
        indicators: {},
      });

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue(mockData);

      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.body.metadata.count).toBe(5);
      expect(response.body.data).toHaveLength(5);
    });
  });

  describe('GET /api/markets/:symbol/:timeframe - CORS', () => {
    it('should include CORS headers', async () => {
      // Arrange
      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

      // Act
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
      });

      // Assert
      expect(response.headers).toHaveProperty('access-control-allow-origin');
    });
  });
});
