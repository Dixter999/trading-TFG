import request from 'supertest';
import { app } from '../index';

interface HealthResponse {
  status: string;
  timestamp: string;
  service: string;
}

describe('Health Check Endpoint', () => {
  describe('GET /health', () => {
    it('should return 200 status code', async () => {
      const response = await request(app).get('/health');

      expect(response.status).toBe(200);
    });

    it('should return JSON content type', async () => {
      const response = await request(app).get('/health');

      expect(response.headers['content-type']).toMatch(/application\/json/);
    });

    it('should return status ok in response body', async () => {
      const response = await request(app).get('/health');
      const body = response.body as HealthResponse;

      expect(body).toHaveProperty('status', 'ok');
    });

    it('should return timestamp in response body', async () => {
      const response = await request(app).get('/health');
      const body = response.body as HealthResponse;

      expect(body).toHaveProperty('timestamp');
      expect(typeof body.timestamp).toBe('string');

      // Verify timestamp is a valid ISO 8601 date
      const timestamp = new Date(body.timestamp);
      expect(timestamp.toString()).not.toBe('Invalid Date');
    });

    it('should return service name in response body', async () => {
      const response = await request(app).get('/health');
      const body = response.body as HealthResponse;

      expect(body).toHaveProperty('service');
      expect(typeof body.service).toBe('string');
      expect(body.service.length).toBeGreaterThan(0);
    });
  });
});
