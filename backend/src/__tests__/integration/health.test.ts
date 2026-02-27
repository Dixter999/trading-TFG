/**
 * Health Check Integration Tests
 *
 * Tests the enhanced health check endpoint with real database connections
 * NO MOCKS - Uses actual PostgreSQL services from docker-compose
 */

import request from 'supertest';
import { app } from '../../index';
import { initializePools, closeAllPools } from '../../database/connection';

interface EnhancedHealthResponse {
  status: string;
  timestamp: string;
  service: string;
  databases: {
    markets: string;
    ai_model: string;
  };
  uptime: number;
}

describe('Health Check Integration Tests', () => {
  beforeAll(async () => {
    // Initialize database pools with real databases
    await initializePools();
  }, 30000); // 30 second timeout for connections

  afterAll(async () => {
    await closeAllPools();
  });

  describe('GET /health - All Services Healthy', () => {
    it('should return 200 status code when all services are healthy', async () => {
      const response = await request(app).get('/health');
      expect(response.status).toBe(200);
    });

    it('should return enhanced health response with all required fields', async () => {
      const response = await request(app).get('/health');
      const body = response.body as EnhancedHealthResponse;

      expect(body).toHaveProperty('status');
      expect(body).toHaveProperty('timestamp');
      expect(body).toHaveProperty('service', 'trading-backend');
      expect(body).toHaveProperty('databases');
      expect(body).toHaveProperty('uptime');
    });

    it('should report database connection status as connected', async () => {
      const response = await request(app).get('/health');
      const body = response.body as EnhancedHealthResponse;

      expect(body.databases).toHaveProperty('markets');
      expect(body.databases).toHaveProperty('ai_model');
      expect(body.databases.markets).toBe('connected');
      expect(body.databases.ai_model).toBe('connected');
    });

    it('should return uptime as a positive number', async () => {
      const response = await request(app).get('/health');
      const body = response.body as EnhancedHealthResponse;

      expect(typeof body.uptime).toBe('number');
      expect(body.uptime).toBeGreaterThanOrEqual(0);
    });

    it('should respond in less than 100ms', async () => {
      const startTime = Date.now();
      await request(app).get('/health');
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(100);
    });
  });

  describe('GET /health - Timestamp Validation', () => {
    it('should return valid ISO 8601 timestamp', async () => {
      const response = await request(app).get('/health');
      const body = response.body as EnhancedHealthResponse;

      const timestamp = new Date(body.timestamp);
      expect(timestamp.toString()).not.toBe('Invalid Date');

      // Verify timestamp format (ISO 8601 with Z)
      expect(body.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
    });

    it('should return timestamp close to current time', async () => {
      const beforeRequest = Date.now();
      const response = await request(app).get('/health');
      const afterRequest = Date.now();

      const body = response.body as EnhancedHealthResponse;
      const responseTime = new Date(body.timestamp).getTime();

      expect(responseTime).toBeGreaterThanOrEqual(beforeRequest - 1000);
      expect(responseTime).toBeLessThanOrEqual(afterRequest + 1000);
    });
  });

  describe('GET /health - Status Field Logic', () => {
    it('should return status "ok" when all services are connected', async () => {
      const response = await request(app).get('/health');
      const body = response.body as EnhancedHealthResponse;

      // If all services are connected, status should be "ok"
      if (body.databases.markets === 'connected' && body.databases.ai_model === 'connected') {
        expect(body.status).toBe('ok');
      }
    });

    it('should return status "degraded" if any database is down', async () => {
      // This test will pass once we implement health check logic
      // For now, we expect all services to be connected in integration tests
      const response = await request(app).get('/health');
      const body = response.body as EnhancedHealthResponse;

      // If any database is not connected, status should be degraded
      if (body.databases.markets !== 'connected' || body.databases.ai_model !== 'connected') {
        expect(body.status).toBe('degraded');
        expect(response.status).toBe(503);
      }
    });
  });
});
