import { Express } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';

export function setupSecurityMiddleware(app: Express): void {
  // Security headers
  app.use(helmet());

  // CORS configuration
  app.use(cors());

  // Rate limiting - Relaxed for development
  const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 10000, // Limit each IP to 10000 requests per windowMs (development)
    message: 'Too many requests from this IP, please try again later.',
  });
  app.use(limiter);
}
