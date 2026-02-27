import { Request, Response, NextFunction } from 'express';
import winston from 'winston';

// Extend Request type to include optional id field
interface RequestWithId extends Request {
  id?: string;
}

// Create Winston logger with structured JSON format
export const logger = winston.createLogger({
  format: winston.format.combine(winston.format.timestamp(), winston.format.json()),
  transports: [
    new winston.transports.Console({
      level: process.env['LOG_LEVEL'] || 'info',
    }),
    new winston.transports.File({
      filename: 'logs/error.log',
      level: 'error',
    }),
    new winston.transports.File({
      filename: 'logs/combined.log',
    }),
  ],
});

/**
 * Request logger middleware
 * Logs incoming requests and outgoing responses with timing metrics
 */
export function requestLogger(req: RequestWithId, res: Response, next: NextFunction): void {
  const startTime = Date.now();

  // Log incoming request
  try {
    logger.info({
      type: 'request',
      method: req.method,
      path: req.path,
      query: req.query,
      headers: req.headers,
      requestId: req.id,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    // Don't block request processing if logging fails
    console.error('Failed to log request:', error);
  }

  // Listen for response finish event to log response details
  res.on('finish', () => {
    try {
      const duration = Date.now() - startTime;
      const contentLength = res.getHeader('content-length');
      const bytes = typeof contentLength === 'string' ? parseInt(contentLength, 10) : 0;

      logger.info({
        type: 'response',
        method: req.method,
        path: req.path,
        statusCode: res.statusCode,
        duration,
        bytes: isNaN(bytes) ? 0 : bytes,
        requestId: req.id,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      // Don't throw errors in response logging
      console.error('Failed to log response:', error);
    }
  });

  next();
}
