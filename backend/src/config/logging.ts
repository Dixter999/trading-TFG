/**
 * Structured Logging Configuration
 *
 * Provides structured JSON logging for production environments.
 * Outputs logs in a format that can be easily parsed by log aggregators
 * like ELK, Loki, or CloudWatch.
 *
 * Usage:
 *   import { logger } from './config/logging';
 *   logger.info('Server started', { port: 3000 });
 *   logger.error('Database connection failed', { error: err.message });
 */

/**
 * Log entry structure
 */
export interface LogEntry {
  timestamp: string;
  level: 'error' | 'warn' | 'info' | 'debug';
  message: string;
  service: string;
  [key: string]: unknown;
}

/**
 * Log level priority for filtering
 */
const LOG_LEVELS = {
  error: 0,
  warn: 1,
  info: 2,
  debug: 3,
} as const;

/**
 * Current log level from environment (default: info)
 */
const CURRENT_LOG_LEVEL = (process.env['LOG_LEVEL'] as keyof typeof LOG_LEVELS) || 'info';

/**
 * Service name for logging
 */
const SERVICE_NAME = process.env['SERVICE_NAME'] || 'trading-backend';

/**
 * Check if a log level should be output based on current configuration
 */
function shouldLog(level: keyof typeof LOG_LEVELS): boolean {
  return LOG_LEVELS[level] <= LOG_LEVELS[CURRENT_LOG_LEVEL];
}

/**
 * Core logging function
 *
 * @param level - Log level (error, warn, info, debug)
 * @param message - Log message
 * @param meta - Additional metadata to include in log entry
 */
export function log(
  level: LogEntry['level'],
  message: string,
  meta?: Record<string, unknown>
): void {
  // Skip if log level is below current threshold
  if (!shouldLog(level)) {
    return;
  }

  const entry: LogEntry = {
    timestamp: new Date().toISOString(),
    level,
    message,
    service: SERVICE_NAME,
    ...meta,
  };

  // Output to stdout (captured by Docker logging driver)
  console.log(JSON.stringify(entry));
}

/**
 * Logger interface with convenience methods
 */
export const logger = {
  /**
   * Log error message
   *
   * Use for critical failures, exceptions, and errors that require attention.
   *
   * @example
   * logger.error('Database connection failed', {
   *   error: err.message,
   *   host: 'localhost',
   *   port: 5432
   * });
   */
  error: (message: string, meta?: Record<string, unknown>): void => {
    log('error', message, meta);
  },

  /**
   * Log warning message
   *
   * Use for potential issues, deprecated features, or recoverable errors.
   *
   * @example
   * logger.warn('Slow query detected', {
   *   query: 'SELECT * FROM ...',
   *   duration: 2345
   * });
   */
  warn: (message: string, meta?: Record<string, unknown>): void => {
    log('warn', message, meta);
  },

  /**
   * Log info message
   *
   * Use for important application events and milestones.
   *
   * @example
   * logger.info('Server started', {
   *   port: 3000,
   *   environment: 'production'
   * });
   */
  info: (message: string, meta?: Record<string, unknown>): void => {
    log('info', message, meta);
  },

  /**
   * Log debug message
   *
   * Use for detailed debugging information. Should not be enabled in production.
   *
   * @example
   * logger.debug('Processing request', {
   *   requestId: 'req_123',
   *   path: '/api/markets',
   *   query: { symbol: 'EURUSD' }
   * });
   */
  debug: (message: string, meta?: Record<string, unknown>): void => {
    log('debug', message, meta);
  },
};

/**
 * Create a child logger with additional context
 *
 * Useful for adding consistent metadata (e.g., requestId) to all logs within a scope.
 *
 * @param context - Context to add to all logs
 * @returns Logger with context
 *
 * @example
 * const requestLogger = createChildLogger({ requestId: 'req_123' });
 * requestLogger.info('Processing request'); // Includes requestId in output
 */
export function createChildLogger(context: Record<string, unknown>) {
  return {
    error: (message: string, meta?: Record<string, unknown>): void => {
      logger.error(message, { ...context, ...meta });
    },
    warn: (message: string, meta?: Record<string, unknown>): void => {
      logger.warn(message, { ...context, ...meta });
    },
    info: (message: string, meta?: Record<string, unknown>): void => {
      logger.info(message, { ...context, ...meta });
    },
    debug: (message: string, meta?: Record<string, unknown>): void => {
      logger.debug(message, { ...context, ...meta });
    },
  };
}

/**
 * Log HTTP request
 *
 * Helper for logging HTTP requests with standard format.
 *
 * @param method - HTTP method
 * @param path - Request path
 * @param status - HTTP status code
 * @param duration - Request duration in milliseconds
 * @param meta - Additional metadata
 *
 * @example
 * logRequest('GET', '/api/markets/EURUSD/H1', 200, 234, {
 *   requestId: 'req_123',
 *   query: { start: '2024-01-01', end: '2024-01-31' }
 * });
 */
export function logRequest(
  method: string,
  path: string,
  status: number,
  duration: number,
  meta?: Record<string, unknown>
): void {
  const level = status >= 500 ? 'error' : status >= 400 ? 'warn' : 'info';

  log(level, 'HTTP request', {
    method,
    path,
    status,
    duration,
    ...meta,
  });
}

/**
 * Log database query
 *
 * Helper for logging database queries with standard format.
 *
 * @param query - SQL query (or description)
 * @param duration - Query duration in milliseconds
 * @param rows - Number of rows affected/returned
 * @param meta - Additional metadata
 *
 * @example
 * logQuery('SELECT * FROM candles WHERE ...', 45, 100, {
 *   symbol: 'EURUSD',
 *   timeframe: 'H1'
 * });
 */
export function logQuery(
  query: string,
  duration: number,
  rows?: number,
  meta?: Record<string, unknown>
): void {
  // Warn on slow queries (> 1 second)
  const level = duration > 1000 ? 'warn' : 'debug';

  log(level, 'Database query', {
    query: query.substring(0, 200), // Truncate long queries
    duration,
    rows,
    ...meta,
  });
}

/**
 * Log WebSocket event
 *
 * Helper for logging WebSocket events with standard format.
 *
 * @param event - Event type (connected, disconnected, subscribed, etc.)
 * @param clientId - Client identifier
 * @param meta - Additional metadata
 *
 * @example
 * logWebSocket('connected', 'ws_abc123', {
 *   remoteAddress: '192.168.1.100'
 * });
 */
export function logWebSocket(
  event: string,
  clientId: string,
  meta?: Record<string, unknown>
): void {
  log('info', `WebSocket ${event}`, {
    clientId,
    ...meta,
  });
}

/**
 * Log NATS message
 *
 * Helper for logging NATS message events with standard format.
 *
 * @param subject - NATS subject
 * @param action - Action (received, sent, error)
 * @param meta - Additional metadata
 *
 * @example
 * logNATS('markets.eurusd.h1', 'received', {
 *   size: 512,
 *   processingTime: 23
 * });
 */
export function logNATS(subject: string, action: string, meta?: Record<string, unknown>): void {
  log('info', `NATS message ${action}`, {
    subject,
    ...meta,
  });
}

/**
 * Default export
 */
export default logger;
