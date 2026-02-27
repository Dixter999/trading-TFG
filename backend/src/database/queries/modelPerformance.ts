/**
 * Model Performance Query Functions
 *
 * Functions for retrieving model validation performance data:
 * - Reading performance JSON files from results/model_validation/
 * - Filtering by symbol and direction
 * - Aggregating performance metrics
 *
 * Issue #517 - PerformanceMetrics module with TDD
 */

import * as fs from 'fs';
import * as path from 'path';
import { getAiModelPool } from '../connection';
import { executeQuery } from '../utils/query';

// ============================================================================
// Types
// ============================================================================

/**
 * Model validation performance data
 */
export interface ModelPerformance {
  symbol: string;
  direction: string;
  fold: number;
  validation_period: string;
  total_trades: number;
  win_rate: number;
  avg_win: number;
  avg_loss: number;
  risk_reward_ratio: number;
  expectancy: number;
  profit_factor: number;
  total_pnl: number;
  total_pnl_after_costs: number;
  max_drawdown_pips: number;
  max_drawdown_percent: number;
  sharpe_ratio: number;
  is_profitable: boolean;
  status: string;
}

// ============================================================================
// Constants
// ============================================================================

// Try multiple paths for validation results (Docker and local development)
const POSSIBLE_RESULTS_DIRS = [
  '/results/model_validation',
  path.resolve(__dirname, '../../../../results/model_validation'),
  path.resolve(__dirname, '../../../../../results/model_validation'),
];

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Find first existing path from a list of possible paths
 */
function findExistingPath(paths: string[]): string | null {
  for (const p of paths) {
    if (fs.existsSync(p)) {
      return p;
    }
  }
  return null;
}

/**
 * Read and parse a performance JSON file
 */
function readPerformanceFile(filePath: string): ModelPerformance | null {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const data = JSON.parse(content) as ModelPerformance;
    return data;
  } catch (error) {
    console.error(`Error reading performance file ${filePath}:`, error);
    return null;
  }
}

/**
 * Sort performance data by symbol, then direction
 */
function sortPerformanceData(data: ModelPerformance[]): ModelPerformance[] {
  return data.sort((a, b) => {
    // First sort by symbol
    const symbolCompare = a.symbol.localeCompare(b.symbol);
    if (symbolCompare !== 0) {
      return symbolCompare;
    }
    // Then sort by direction (long before short)
    return a.direction.localeCompare(b.direction);
  });
}

// ============================================================================
// Query Functions
// ============================================================================

/**
 * Get model performance from approved_models database table (Phase 5 results)
 */
async function getModelPerformanceFromDatabase(): Promise<ModelPerformance[]> {
  try {
    const pool = getAiModelPool();
    const sql = `
      SELECT
        symbol, signal_name, direction, timeframe,
        phase5_pf as profit_factor,
        phase5_wr as win_rate
      FROM approved_models
      ORDER BY symbol, direction
    `;
    const result = await executeQuery<any>(pool, sql);

    return result.map(row => ({
      symbol: row.symbol.toUpperCase(),
      direction: row.direction.toLowerCase(),
      fold: 29, // Phase 5 uses fold 29
      validation_period: 'Phase 5 Test',
      total_trades: 0, // Not stored in approved_models
      win_rate: parseFloat(row.win_rate) || 0,
      avg_win: 0,
      avg_loss: 0,
      risk_reward_ratio: 1,
      expectancy: 0,
      profit_factor: parseFloat(row.profit_factor) || 1,
      total_pnl: 0,
      total_pnl_after_costs: 0,
      max_drawdown_pips: 0,
      max_drawdown_percent: 0,
      sharpe_ratio: 0,
      is_profitable: parseFloat(row.profit_factor) >= 1.2,
      status: parseFloat(row.profit_factor) >= 1.2 ? 'PROFITABLE' : 'MARGINAL',
    }));
  } catch (error) {
    console.error('Error querying approved_models:', error);
    return [];
  }
}

/**
 * Get all model performance data
 * First tries JSON files, falls back to database
 *
 * @returns Array of performance data for all symbols and directions
 */
export async function getModelPerformance(): Promise<ModelPerformance[]> {
  const resultsDir = findExistingPath(POSSIBLE_RESULTS_DIRS);

  // Try JSON files first
  if (resultsDir) {
    try {
      const files = fs.readdirSync(resultsDir);
      const performanceFiles = files.filter((f) => f.endsWith('_performance.json'));

      if (performanceFiles.length > 0) {
        const performanceData: ModelPerformance[] = [];

        for (const file of performanceFiles) {
          const filePath = path.join(resultsDir, file);
          const data = readPerformanceFile(filePath);

          if (data) {
            performanceData.push(data);
          }
        }

        if (performanceData.length > 0) {
          return sortPerformanceData(performanceData);
        }
      }
    } catch (error) {
      console.error('Error reading model performance files:', error);
    }
  }

  // Fall back to database (approved_models table with Phase 5 data)
  console.log('Using database for model performance (approved_models table)');
  return getModelPerformanceFromDatabase();
}

/**
 * Get model performance data for a specific symbol
 *
 * @param symbol - Symbol to filter by (case-insensitive)
 * @returns Array of performance data for the symbol (all directions)
 */
export async function getModelPerformanceBySymbol(symbol: string): Promise<ModelPerformance[]> {
  const allData = await getModelPerformance();
  const symbolUpper = symbol.toUpperCase();

  return allData.filter((data) => data.symbol.toUpperCase() === symbolUpper);
}

/**
 * Get model performance data for a specific symbol and direction
 *
 * @param symbol - Symbol to filter by (case-insensitive)
 * @param direction - Direction to filter by (case-insensitive)
 * @returns Array of performance data for the symbol and direction
 */
export async function getModelPerformanceBySymbolAndDirection(
  symbol: string,
  direction: string
): Promise<ModelPerformance[]> {
  const allData = await getModelPerformance();
  const symbolUpper = symbol.toUpperCase();
  const directionLower = direction.toLowerCase();

  return allData.filter(
    (data) =>
      data.symbol.toUpperCase() === symbolUpper && data.direction.toLowerCase() === directionLower
  );
}
