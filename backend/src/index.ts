import express, { Express } from 'express';
import { config } from './config/environment';
import { logger } from './config/logging';
import { setupSecurityMiddleware } from './middleware/security';
import { requestLogger } from './middleware/requestLogger';
import { errorHandler } from './middleware/errorHandler';
import healthRouter from './routes/health';
import marketsRouter from './routes/markets';
import statusRouter from './routes/status';
import clustersRouter from './routes/clusters';
import paperTradingRouter from './routes/paperTrading';
import positionSizingRouter from './routes/positionSizing';
import orderBlocksRouter from './routes/orderBlocks';
import signalsRouter from './routes/signals';
import { initializePools } from './database/connection';

// Create Express app
export const app: Express = express();

// Security middleware
setupSecurityMiddleware(app);

// Request logging middleware
app.use(requestLogger);

// Body parsing middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use(healthRouter);
app.use(marketsRouter);
app.use(statusRouter);
app.use(clustersRouter);
app.use(paperTradingRouter);
app.use(positionSizingRouter);
app.use(orderBlocksRouter);
app.use(signalsRouter);

// Error handler (must be last)
app.use(errorHandler);

// Start server (only if not in test environment)
if (config.nodeEnv !== 'test') {
  // Initialize database pools before starting server
  initializePools()
    .then(() => {
      app.listen(config.port, () => {
        logger.info('Server started', {
          port: config.port,
          nodeEnv: config.nodeEnv,
        });
      });
    })
    .catch((error) => {
      logger.error('Failed to initialize database pools', { error });
      process.exit(1);
    });
}
