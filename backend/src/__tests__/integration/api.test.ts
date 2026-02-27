/**
 * API Integration Tests
 *
 * Tests the /api/markets endpoint with real PostgreSQL database
 * NO MOCKS - Uses actual database from docker-compose
 */

/* eslint-disable @typescript-eslint/no-unsafe-member-access */
/* eslint-disable @typescript-eslint/no-unsafe-call */

import request from 'supertest';
import { app } from '../../index';
import { initializePools, closeAllPools } from '../../database/connection';
import { MarketDataWithIndicators } from '../../database/queries/marketData';

describe('API Integration Tests - /api/markets', () => {
  beforeAll(async () => {
    // Initialize database pools with real databases
    await initializePools();
  }, 30000);

  afterAll(async () => {
    await closeAllPools();
  });

  describe('GET /api/markets/:symbol/:timeframe - Success Cases', () => {
    it('should return 200 status for valid request', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 100,
      });

      expect(response.status).toBe(200);
    });

    it('should return JSON content type', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 100,
      });

      expect(response.headers['content-type']).toMatch(/application\/json/);
    });

    it('should return market data with correct structure', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 10,
      });

      expect(response.body).toHaveProperty('data');
      expect(Array.isArray(response.body.data)).toBe(true);

      if (response.body.data.length > 0) {
        const firstItem = response.body.data[0] as MarketDataWithIndicators;
        expect(firstItem).toHaveProperty('rateTime');
        expect(firstItem).toHaveProperty('open');
        expect(firstItem).toHaveProperty('high');
        expect(firstItem).toHaveProperty('low');
        expect(firstItem).toHaveProperty('close');
        expect(firstItem).toHaveProperty('volume');
        expect(firstItem).toHaveProperty('indicators');
      }
    });

    it('should respect limit parameter', async () => {
      const limit = 50;
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-12-31T23:59:59Z',
        limit,
      });

      expect(response.status).toBe(200);
      expect(response.body.data.length).toBeLessThanOrEqual(limit);
    });

    it('should support H1 timeframe', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 10,
      });

      expect(response.status).toBe(200);
    });

    it('should support H4 timeframe', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H4').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 10,
      });

      expect(response.status).toBe(200);
    });

    it('should support D1 timeframe', async () => {
      const response = await request(app).get('/api/markets/EURUSD/D1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 10,
      });

      expect(response.status).toBe(200);
    });

    it('should respond in less than 500ms for 100 records', async () => {
      const startTime = Date.now();
      await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 100,
      });
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(500);
    });
  });

  describe('GET /api/markets/:symbol/:timeframe - Error Cases', () => {
    it('should return 400 for invalid symbol', async () => {
      const response = await request(app).get('/api/markets/INVALID/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 100,
      });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('should return 400 for invalid timeframe', async () => {
      const response = await request(app).get('/api/markets/EURUSD/INVALID').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 100,
      });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('should return 400 for missing start parameter', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        end: '2024-01-31T23:59:59Z',
        limit: 100,
      });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('should return 400 for missing end parameter', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        limit: 100,
      });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('should return 400 for invalid date format', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: 'invalid-date',
        end: '2024-01-31T23:59:59Z',
        limit: 100,
      });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('should return 400 for invalid limit (non-numeric)', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 'invalid',
      });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('should return 400 for limit exceeding maximum (5000)', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 10000,
      });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('should return 400 when end date is before start date', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-12-31T23:59:59Z',
        end: '2024-01-01T00:00:00Z',
        limit: 100,
      });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });
  });

  describe('GET /api/markets/:symbol/:timeframe - Data Validation', () => {
    it('should return OHLCV data as numbers', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 1,
      });

      expect(response.status).toBe(200);

      if (response.body.data.length > 0) {
        const item = response.body.data[0] as MarketDataWithIndicators;
        expect(typeof item.open).toBe('number');
        expect(typeof item.high).toBe('number');
        expect(typeof item.low).toBe('number');
        expect(typeof item.close).toBe('number');
        expect(typeof item.volume).toBe('number');
      }
    });

    it('should have high >= low for all candles', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 100,
      });

      expect(response.status).toBe(200);

      response.body.data.forEach((item: MarketDataWithIndicators) => {
        expect(item.high).toBeGreaterThanOrEqual(item.low);
      });
    });

    it('should return valid ISO 8601 timestamps', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 10,
      });

      expect(response.status).toBe(200);

      response.body.data.forEach((item: MarketDataWithIndicators) => {
        const date = new Date(item.rateTime);
        expect(date.toString()).not.toBe('Invalid Date');
      });
    });

    it('should return data sorted by date DESC (newest first)', async () => {
      const response = await request(app).get('/api/markets/EURUSD/H1').query({
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z',
        limit: 100,
      });

      expect(response.status).toBe(200);

      const data = response.body.data as MarketDataWithIndicators[];
      if (data.length > 1) {
        for (let i = 0; i < data.length - 1; i++) {
          const current = new Date(data[i]!.rateTime).getTime();
          const next = new Date(data[i + 1]!.rateTime).getTime();
          expect(current).toBeGreaterThanOrEqual(next);
        }
      }
    });
  });
});
