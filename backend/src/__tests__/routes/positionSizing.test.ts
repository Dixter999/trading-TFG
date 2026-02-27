/**
 * Tests for Position Sizing API Routes (Issue #630)
 *
 * TDD tests for the balance-based position sizing endpoints.
 */

import request from 'supertest';
import { app } from '../../index';

describe('Position Sizing API', () => {
  describe('GET /api/position-sizing/allocation', () => {
    it('should return allocation for valid balance', async () => {
      const response = await request(app)
        .get('/api/position-sizing/allocation')
        .query({ balance: 10000 })
        .expect(200);

      expect(response.body).toHaveProperty('data');
      expect(response.body).toHaveProperty('metadata');

      const data = response.body.data;
      expect(data).toHaveProperty('balance', 10000);
      expect(data).toHaveProperty('availableBalance');
      expect(data).toHaveProperty('marginReserve');
      expect(data).toHaveProperty('tradeableSymbols');
      expect(data).toHaveProperty('totalSymbols');
      expect(data).toHaveProperty('maxTotalLots');
      expect(data).toHaveProperty('marginUtilization');
      expect(data).toHaveProperty('avgLotSize');
      expect(data).toHaveProperty('diversificationScore');
      expect(data).toHaveProperty('allocations');
      expect(Array.isArray(data.allocations)).toBe(true);
    });

    it('should calculate available balance as 90% of total', async () => {
      const response = await request(app)
        .get('/api/position-sizing/allocation')
        .query({ balance: 10000 })
        .expect(200);

      const data = response.body.data;
      expect(data.availableBalance).toBe(9000);
      expect(data.marginReserve).toBe(1000);
    });

    it('should include all symbol allocations', async () => {
      const response = await request(app)
        .get('/api/position-sizing/allocation')
        .query({ balance: 10000 })
        .expect(200);

      const data = response.body.data;
      expect(data.allocations.length).toBeGreaterThan(0);
      expect(data.allocations.length).toBe(data.totalSymbols);

      // Check each allocation has required fields
      for (const alloc of data.allocations) {
        expect(alloc).toHaveProperty('symbol');
        expect(alloc).toHaveProperty('maxLots');
        expect(alloc).toHaveProperty('marginRequired');
        expect(alloc).toHaveProperty('leverage');
        expect(alloc).toHaveProperty('status');
        expect(['excellent', 'good', 'warning', 'insufficient']).toContain(alloc.status);
      }
    });

    it('should have tradeable symbols with $10,000 balance', async () => {
      const response = await request(app)
        .get('/api/position-sizing/allocation')
        .query({ balance: 10000 })
        .expect(200);

      const data = response.body.data;
      expect(data.tradeableSymbols).toBeGreaterThan(0);
      expect(data.maxTotalLots).toBeGreaterThan(0);
    });

    it('should have fewer tradeable symbols with very low balance', async () => {
      const response = await request(app)
        .get('/api/position-sizing/allocation')
        .query({ balance: 100 })
        .expect(200);

      const data = response.body.data;
      // Very low balance should have few or no tradeable symbols
      expect(data.tradeableSymbols).toBeLessThanOrEqual(2);
    });

    it('should scale allocations with balance', async () => {
      const smallBalance = await request(app)
        .get('/api/position-sizing/allocation')
        .query({ balance: 1000 })
        .expect(200);

      const largeBalance = await request(app)
        .get('/api/position-sizing/allocation')
        .query({ balance: 50000 })
        .expect(200);

      expect(largeBalance.body.data.maxTotalLots).toBeGreaterThan(
        smallBalance.body.data.maxTotalLots
      );
      expect(largeBalance.body.data.tradeableSymbols).toBeGreaterThanOrEqual(
        smallBalance.body.data.tradeableSymbols
      );
    });

    it('should return 400 for missing balance', async () => {
      const response = await request(app)
        .get('/api/position-sizing/allocation')
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should return 400 for negative balance', async () => {
      const response = await request(app)
        .get('/api/position-sizing/allocation')
        .query({ balance: -1000 })
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should return 400 for non-numeric balance', async () => {
      const response = await request(app)
        .get('/api/position-sizing/allocation')
        .query({ balance: 'invalid' })
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should have diversification score between 0 and 100', async () => {
      const response = await request(app)
        .get('/api/position-sizing/allocation')
        .query({ balance: 10000 })
        .expect(200);

      const score = response.body.data.diversificationScore;
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(100);
    });
  });

  describe('GET /api/position-sizing/symbols', () => {
    it('should return all symbol configurations', async () => {
      const response = await request(app)
        .get('/api/position-sizing/symbols')
        .expect(200);

      expect(response.body).toHaveProperty('data');
      expect(Array.isArray(response.body.data)).toBe(true);
      expect(response.body.data.length).toBeGreaterThan(0);
    });

    it('should return exactly 8 traded symbols', async () => {
      const response = await request(app)
        .get('/api/position-sizing/symbols')
        .expect(200);

      // Issue #631: Filter to 8 actively traded pairs
      expect(response.body.data.length).toBe(8);
      expect(response.body.metadata.count).toBe(8);
    });

    it('should include only the 8 traded pairs', async () => {
      const response = await request(app)
        .get('/api/position-sizing/symbols')
        .expect(200);

      const symbols = response.body.data.map((s: { symbol: string }) => s.symbol);

      // Major pairs (30x leverage)
      expect(symbols).toContain('EURUSD');
      expect(symbols).toContain('GBPUSD');
      expect(symbols).toContain('USDJPY');
      expect(symbols).toContain('USDCHF');

      // Minor pairs (20x leverage)
      expect(symbols).toContain('EURJPY');
      expect(symbols).toContain('EURGBP');
      expect(symbols).toContain('EURCAD');
      expect(symbols).toContain('USDCAD');

      // Should NOT contain removed symbols
      expect(symbols).not.toContain('NZDUSD');
      expect(symbols).not.toContain('AUDUSD');
      expect(symbols).not.toContain('AUDCAD');
      expect(symbols).not.toContain('AUDNZD');
      expect(symbols).not.toContain('CADJPY');
      expect(symbols).not.toContain('NZDJPY');
      expect(symbols).not.toContain('GBPJPY');
    });

    it('should include required fields for each symbol', async () => {
      const response = await request(app)
        .get('/api/position-sizing/symbols')
        .expect(200);

      for (const symbol of response.body.data) {
        expect(symbol).toHaveProperty('symbol');
        expect(symbol).toHaveProperty('leverage');
        expect(symbol).toHaveProperty('pipValue');
        expect(symbol).toHaveProperty('marginPerLot');
      }
    });

    it('should include major pairs', async () => {
      const response = await request(app)
        .get('/api/position-sizing/symbols')
        .expect(200);

      const symbols = response.body.data.map((s: { symbol: string }) => s.symbol);
      expect(symbols).toContain('EURUSD');
      expect(symbols).toContain('GBPUSD');
      expect(symbols).toContain('USDJPY');
    });

    it('should have correct leverage for major pairs', async () => {
      const response = await request(app)
        .get('/api/position-sizing/symbols')
        .expect(200);

      const eurusd = response.body.data.find((s: { symbol: string }) => s.symbol === 'EURUSD');
      expect(eurusd).toBeDefined();
      expect(eurusd.leverage).toBe(30);
    });
  });

  describe('GET /api/position-sizing/rules', () => {
    it('should return diversification rules', async () => {
      const response = await request(app)
        .get('/api/position-sizing/rules')
        .expect(200);

      expect(response.body).toHaveProperty('data');
      const rules = response.body.data;

      expect(rules).toHaveProperty('maxSingleSymbolAllocation');
      expect(rules).toHaveProperty('maxSingleDirectionAllocation');
      expect(rules).toHaveProperty('minPositionSize');
      expect(rules).toHaveProperty('marginReserveRatio');
    });

    it('should have max single symbol allocation of 25%', async () => {
      const response = await request(app)
        .get('/api/position-sizing/rules')
        .expect(200);

      expect(response.body.data.maxSingleSymbolAllocation).toBe(0.25);
    });

    it('should have min position size of 0.01', async () => {
      const response = await request(app)
        .get('/api/position-sizing/rules')
        .expect(200);

      expect(response.body.data.minPositionSize).toBe(0.01);
    });

    it('should have margin reserve ratio of 10%', async () => {
      const response = await request(app)
        .get('/api/position-sizing/rules')
        .expect(200);

      expect(response.body.data.marginReserveRatio).toBe(0.1);
    });
  });
});
