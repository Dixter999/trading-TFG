/**
 * Model Performance Routes Integration Tests
 *
 * Tests for /api/paper-trading/model-performance endpoints
 * Issue #517 - PerformanceMetrics module with TDD
 */

import request from 'supertest';
import express from 'express';
import paperTradingRouter from '../../routes/paperTrading';

// Create test app
const app = express();
app.use(express.json());
app.use(paperTradingRouter);

// Error handler
app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  res.status(500).json({ error: err.message });
});

describe('Model Performance API Routes', () => {
  describe('GET /api/paper-trading/model-performance', () => {
    it('should return 200 and array of performance data', async () => {
      const response = await request(app)
        .get('/api/paper-trading/model-performance')
        .expect('Content-Type', /json/)
        .expect(200);

      expect(response.body).toHaveProperty('data');
      expect(response.body).toHaveProperty('metadata');
      expect(Array.isArray(response.body.data)).toBe(true);
      expect(response.body.metadata).toHaveProperty('count');
      expect(response.body.metadata).toHaveProperty('timestamp');
    });
  });

  describe('GET /api/paper-trading/model-performance/:symbol', () => {
    it('should return 200 and performance data for valid symbol', async () => {
      const response = await request(app)
        .get('/api/paper-trading/model-performance/EURUSD')
        .expect('Content-Type', /json/)
        .expect(200);

      expect(response.body).toHaveProperty('data');
      expect(Array.isArray(response.body.data)).toBe(true);
      expect(response.body.metadata).toHaveProperty('count');
    });

    it('should return empty array for unknown symbol', async () => {
      const response = await request(app)
        .get('/api/paper-trading/model-performance/UNKNOWN')
        .expect('Content-Type', /json/)
        .expect(200);

      expect(response.body.data).toEqual([]);
      expect(response.body.metadata.count).toBe(0);
    });
  });

  describe('GET /api/paper-trading/model-performance/:symbol/:direction', () => {
    it('should return 200 and performance data for valid symbol and direction', async () => {
      const response = await request(app)
        .get('/api/paper-trading/model-performance/EURUSD/long')
        .expect('Content-Type', /json/)
        .expect(200);

      expect(response.body).toHaveProperty('data');
      expect(Array.isArray(response.body.data)).toBe(true);
      expect(response.body.metadata).toHaveProperty('count');
    });

    it('should accept case-insensitive direction parameter', async () => {
      const response = await request(app)
        .get('/api/paper-trading/model-performance/EURUSD/SHORT')
        .expect('Content-Type', /json/)
        .expect(200);

      expect(response.body).toHaveProperty('data');
      expect(Array.isArray(response.body.data)).toBe(true);
    });

    it('should return 500 for invalid direction', async () => {
      const response = await request(app)
        .get('/api/paper-trading/model-performance/EURUSD/invalid')
        .expect('Content-Type', /json/)
        .expect(500);

      expect(response.body).toHaveProperty('error');
    });

    it('should return empty array for unknown symbol/direction combination', async () => {
      const response = await request(app)
        .get('/api/paper-trading/model-performance/UNKNOWN/long')
        .expect('Content-Type', /json/)
        .expect(200);

      expect(response.body.data).toEqual([]);
      expect(response.body.metadata.count).toBe(0);
    });
  });
});
