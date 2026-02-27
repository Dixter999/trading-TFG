import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

interface EnvironmentConfig {
  port: number;
  nodeEnv: string;
  databaseUrlMarkets: string;
  databaseUrlAiModel: string;
  wsPort: number;
  logLevel: string;
}

// TFG: Default DB URLs point to local Docker Compose PostgreSQL service
export const config: EnvironmentConfig = {
  port: parseInt(process.env['PORT'] || '3000', 10),
  nodeEnv: process.env['NODE_ENV'] || 'development',
  databaseUrlMarkets:
    process.env['DATABASE_URL_MARKETS'] ||
    'postgresql://tfg_user:tfg_password@db:5432/markets',
  databaseUrlAiModel:
    process.env['DATABASE_URL_AI_MODEL'] ||
    'postgresql://tfg_user:tfg_password@db:5432/ai_model',
  wsPort: parseInt(process.env['WS_PORT'] || '8080', 10),
  logLevel: process.env['LOG_LEVEL'] || 'info',
};
