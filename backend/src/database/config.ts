/**
 * Database Configuration Module
 *
 * Parses environment variables and provides type-safe database configuration
 * for both Markets (READ-ONLY) and AI Model (READ-WRITE) databases.
 *
 * Configuration:
 * - max: 20 connections per pool
 * - idleTimeoutMillis: 30000 (30 seconds)
 * - connectionTimeoutMillis: 5000 (5 seconds)
 */

import { PoolConfig } from 'pg';

export interface DatabasePoolConfig extends PoolConfig {
  host: string;
  port: number;
  database: string;
  user: string;
  password: string;
  max: number;
  idleTimeoutMillis: number;
  connectionTimeoutMillis: number;
}

export interface DatabaseConfig {
  markets: DatabasePoolConfig;
  aiModel: DatabasePoolConfig;
}

const requiredEnvVars = [
  'MARKETS_DB_HOST',
  'MARKETS_DB_PORT',
  'MARKETS_DB_NAME',
  'MARKETS_DB_USER',
  'MARKETS_DB_PASSWORD',
  'AI_MODEL_DB_HOST',
  'AI_MODEL_DB_PORT',
  'AI_MODEL_DB_NAME',
  'AI_MODEL_DB_USER',
  'AI_MODEL_DB_PASSWORD',
] as const;

/**
 * Validates that all required environment variables are present
 * @throws {Error} If any required environment variable is missing
 */
export function validateDatabaseConfig(): void {
  for (const envVar of requiredEnvVars) {
    if (!process.env[envVar]) {
      throw new Error(`Missing required environment variable: ${envVar}`);
    }
  }
}

/**
 * Retrieves database configuration from environment variables
 * @returns {DatabaseConfig} Configuration for both database pools
 * @throws {Error} If environment variables are invalid
 */
export function getDatabaseConfig(): DatabaseConfig {
  // Validate all required variables are present
  validateDatabaseConfig();

  // Parse Markets database configuration
  const marketsConfig: DatabasePoolConfig = {
    host: process.env['MARKETS_DB_HOST']!,
    port: parseInt(process.env['MARKETS_DB_PORT']!, 10),
    database: process.env['MARKETS_DB_NAME']!,
    user: process.env['MARKETS_DB_USER']!,
    password: process.env['MARKETS_DB_PASSWORD']!,
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 5000,
    // Add statement timeout as safety net (30 seconds)
    statement_timeout: 30000,
  };

  // Parse AI Model database configuration
  const aiModelConfig: DatabasePoolConfig = {
    host: process.env['AI_MODEL_DB_HOST']!,
    port: parseInt(process.env['AI_MODEL_DB_PORT']!, 10),
    database: process.env['AI_MODEL_DB_NAME']!,
    user: process.env['AI_MODEL_DB_USER']!,
    password: process.env['AI_MODEL_DB_PASSWORD']!,
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 5000,
    // Add statement timeout as safety net (30 seconds)
    statement_timeout: 30000,
  };

  return {
    markets: marketsConfig,
    aiModel: aiModelConfig,
  };
}

/**
 * Get connection string for Markets database (for logging/debugging)
 * Password is masked for security
 */
export function getMarketsConnectionString(): string {
  const config = getDatabaseConfig().markets;
  return `postgresql://${config.user}:***@${config.host}:${config.port}/${config.database}`;
}

/**
 * Get connection string for AI Model database (for logging/debugging)
 * Password is masked for security
 */
export function getAiModelConnectionString(): string {
  const config = getDatabaseConfig().aiModel;
  return `postgresql://${config.user}:***@${config.host}:${config.port}/${config.database}`;
}
