/**
 * Tests for Signals API Routes (Issue #631)
 *
 * TDD tests for the approved models signals endpoint with PF-based lot suggestions.
 */

import request from 'supertest';
import { app } from '../../index';
import { getAiModelPool } from '../../database/connection';

// Mock the database connection for testing
jest.mock('../../database/connection', () => ({
  getAiModelPool: jest.fn(),
}));

const mockQuery = jest.fn();
(getAiModelPool as jest.Mock).mockReturnValue({
  query: mockQuery,
});

describe('Signals API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('GET /api/signals/approved-models', () => {
    it('should return approved models with PF-based lot suggestions', async () => {
      // Arrange: Mock database response
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            symbol: 'EURUSD',
            signal_name: 'RSI_Oversold',
            direction: 'LONG',
            timeframe: 'H4',
            phase5_pf: 3.5,
            phase5_wr: 0.65,
          },
          {
            symbol: 'GBPUSD',
            signal_name: 'MACD_Cross',
            direction: 'SHORT',
            timeframe: 'H1',
            phase5_pf: 2.2,
            phase5_wr: 0.58,
          },
        ],
      });

      // Act
      const response = await request(app).get('/api/signals/approved-models').expect(200);

      // Assert: Check response structure
      expect(response.body).toHaveProperty('data');
      expect(response.body).toHaveProperty('metadata');
      expect(Array.isArray(response.body.data)).toBe(true);
      expect(response.body.data.length).toBe(2);

      // Assert: First model has expected fields
      const firstModel = response.body.data[0];
      expect(firstModel).toHaveProperty('symbol', 'EURUSD');
      expect(firstModel).toHaveProperty('signalName', 'RSI_Oversold');
      expect(firstModel).toHaveProperty('direction', 'LONG');
      expect(firstModel).toHaveProperty('timeframe', 'H4');
      expect(firstModel).toHaveProperty('phase5Pf', 3.5);
      expect(firstModel).toHaveProperty('phase5Wr', 0.65);
      expect(firstModel).toHaveProperty('suggestedLots');
    });

    it('should calculate suggestedLots as 0.5L for PF >= 3.0', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            symbol: 'EURUSD',
            signal_name: 'Test',
            direction: 'LONG',
            timeframe: 'H4',
            phase5_pf: 3.0,
            phase5_wr: 0.65,
          },
        ],
      });

      const response = await request(app).get('/api/signals/approved-models').expect(200);

      expect(response.body.data[0].suggestedLots).toBe('0.5L');
    });

    it('should calculate suggestedLots as 0.3L for PF >= 2.0 but < 3.0', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            symbol: 'GBPUSD',
            signal_name: 'Test',
            direction: 'SHORT',
            timeframe: 'H1',
            phase5_pf: 2.5,
            phase5_wr: 0.60,
          },
        ],
      });

      const response = await request(app).get('/api/signals/approved-models').expect(200);

      expect(response.body.data[0].suggestedLots).toBe('0.3L');
    });

    it('should calculate suggestedLots as 0.2L for PF >= 1.5 but < 2.0', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            symbol: 'USDJPY',
            signal_name: 'Test',
            direction: 'LONG',
            timeframe: 'M30',
            phase5_pf: 1.7,
            phase5_wr: 0.55,
          },
        ],
      });

      const response = await request(app).get('/api/signals/approved-models').expect(200);

      expect(response.body.data[0].suggestedLots).toBe('0.2L');
    });

    it('should calculate suggestedLots as 0.1L for PF < 1.5', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            symbol: 'EURJPY',
            signal_name: 'Test',
            direction: 'SHORT',
            timeframe: 'D1',
            phase5_pf: 1.2,
            phase5_wr: 0.52,
          },
        ],
      });

      const response = await request(app).get('/api/signals/approved-models').expect(200);

      expect(response.body.data[0].suggestedLots).toBe('0.1L');
    });

    it('should return empty array when no approved models exist', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [],
      });

      const response = await request(app).get('/api/signals/approved-models').expect(200);

      expect(response.body.data).toEqual([]);
      expect(response.body.metadata.count).toBe(0);
    });

    it('should include metadata with count and timestamp', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            symbol: 'EURUSD',
            signal_name: 'Test',
            direction: 'LONG',
            timeframe: 'H4',
            phase5_pf: 2.0,
            phase5_wr: 0.60,
          },
        ],
      });

      const response = await request(app).get('/api/signals/approved-models').expect(200);

      expect(response.body.metadata).toHaveProperty('count', 1);
      expect(response.body.metadata).toHaveProperty('timestamp');
      expect(new Date(response.body.metadata.timestamp)).toBeInstanceOf(Date);
    });

    it('should return models sorted by PF descending', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            symbol: 'EURUSD',
            signal_name: 'High_PF',
            direction: 'LONG',
            timeframe: 'H4',
            phase5_pf: 4.0,
            phase5_wr: 0.70,
          },
          {
            symbol: 'GBPUSD',
            signal_name: 'Med_PF',
            direction: 'SHORT',
            timeframe: 'H1',
            phase5_pf: 2.0,
            phase5_wr: 0.55,
          },
          {
            symbol: 'USDJPY',
            signal_name: 'Low_PF',
            direction: 'LONG',
            timeframe: 'M30',
            phase5_pf: 1.3,
            phase5_wr: 0.50,
          },
        ],
      });

      const response = await request(app).get('/api/signals/approved-models').expect(200);

      // Verify the query was called with ORDER BY clause
      expect(mockQuery).toHaveBeenCalledWith(expect.stringContaining('ORDER BY phase5_pf DESC'));

      // Verify response order
      const pfs = response.body.data.map((m: { phase5Pf: number }) => m.phase5Pf);
      expect(pfs).toEqual([4.0, 2.0, 1.3]);
    });

    it('should handle database errors gracefully', async () => {
      mockQuery.mockRejectedValueOnce(new Error('Database connection failed'));

      const response = await request(app).get('/api/signals/approved-models').expect(500);

      expect(response.body).toHaveProperty('error');
    });

    it('should handle null PF values', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            symbol: 'AUDUSD',
            signal_name: 'NullPF',
            direction: 'LONG',
            timeframe: 'H4',
            phase5_pf: null,
            phase5_wr: 0.55,
          },
        ],
      });

      const response = await request(app).get('/api/signals/approved-models').expect(200);

      // null PF should be treated as < 1.5, so 0.1L
      expect(response.body.data[0].suggestedLots).toBe('0.1L');
      expect(response.body.data[0].phase5Pf).toBeNull();
    });
  });
});
