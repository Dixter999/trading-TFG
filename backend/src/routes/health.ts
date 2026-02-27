import { Router, Request, Response } from 'express';
import { healthCheckMarkets, healthCheckAiModel } from '../database/connection';

const router = Router();

// Track process start time for uptime calculation
const processStartTime = Date.now();

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

router.get('/health', async (_req: Request, res: Response<EnhancedHealthResponse>) => {
  try {
    // Check database connections
    const marketsHealthy = await healthCheckMarkets();
    const aiModelHealthy = await healthCheckAiModel();

    // Determine overall status (Issue #330: NATS removed, only check databases)
    const allHealthy = marketsHealthy && aiModelHealthy;
    const status = allHealthy ? 'ok' : 'degraded';
    const httpStatus = allHealthy ? 200 : 503;

    // Calculate uptime in seconds
    const uptime = Math.floor((Date.now() - processStartTime) / 1000);

    const response: EnhancedHealthResponse = {
      status,
      timestamp: new Date().toISOString(),
      service: 'trading-backend',
      databases: {
        markets: marketsHealthy ? 'connected' : 'disconnected',
        ai_model: aiModelHealthy ? 'connected' : 'disconnected',
      },
      uptime,
    };

    res.status(httpStatus).json(response);
  } catch (error) {
    // If health check itself fails, return degraded status
    const uptime = Math.floor((Date.now() - processStartTime) / 1000);

    res.status(503).json({
      status: 'degraded',
      timestamp: new Date().toISOString(),
      service: 'trading-backend',
      databases: {
        markets: 'unknown',
        ai_model: 'unknown',
      },
      uptime,
    });
  }
});

export default router;
