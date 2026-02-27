/**
 * Position Sizing Routes (Issue #630)
 *
 * API routes for dynamic balance-based position sizing:
 * - GET /api/position-sizing/allocation - Calculate position allocations for a given balance
 * - GET /api/position-sizing/symbols - Get symbol configuration and margin requirements
 *
 * These endpoints power the balance visualizer frontend and provide
 * real-time position sizing calculations.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { ValidationError } from '../database/errors';
import { getAiModelPool } from '../database/connection';
import { executeQuery } from '../database/utils/query';

const router = Router();

// ============================================================================
// Types
// ============================================================================

interface RequestWithId extends Request {
  id?: string;
}

interface ApiResponse<T> {
  data: T;
  metadata: {
    count?: number;
    timestamp: string;
    requestId?: string;
  };
}

interface SymbolConfig {
  leverage: number;
  pipValue: number;
  pipSize: number;
  baseCurrency: string;
  marginPerLot: number;
}

interface SymbolAllocation {
  symbol: string;
  maxLots: number;
  marginRequired: number;
  leverage: number;
  status: 'excellent' | 'good' | 'warning' | 'insufficient';
}

interface AllocationResponse {
  balance: number;
  availableBalance: number;
  marginReserve: number;
  tradeableSymbols: number;
  totalSymbols: number;
  maxTotalLots: number;
  marginUtilization: number;
  avgLotSize: number;
  diversificationScore: number;
  allocations: SymbolAllocation[];
}

// ============================================================================
// Symbol Configuration
// ============================================================================

/**
 * Symbol configurations — loaded from trading_config_params (Python is source of truth).
 * Falls back to hardcoded defaults if DB table is not yet populated.
 */
const FALLBACK_SYMBOL_CONFIGS: Record<string, SymbolConfig> = {
  EURUSD: { leverage: 30, pipValue: 10.0, pipSize: 0.0001, baseCurrency: 'EUR', marginPerLot: 3666.67 },
  GBPUSD: { leverage: 30, pipValue: 10.0, pipSize: 0.0001, baseCurrency: 'GBP', marginPerLot: 3666.67 },
  USDJPY: { leverage: 30, pipValue: 6.67, pipSize: 0.01, baseCurrency: 'USD', marginPerLot: 3333.33 },
  USDCHF: { leverage: 30, pipValue: 10.0, pipSize: 0.0001, baseCurrency: 'USD', marginPerLot: 3333.33 },
  EURJPY: { leverage: 20, pipValue: 6.67, pipSize: 0.01, baseCurrency: 'EUR', marginPerLot: 5500.0 },
  EURGBP: { leverage: 20, pipValue: 10.0, pipSize: 0.0001, baseCurrency: 'EUR', marginPerLot: 5500.0 },
  EURCAD: { leverage: 20, pipValue: 7.5, pipSize: 0.0001, baseCurrency: 'EUR', marginPerLot: 5500.0 },
  USDCAD: { leverage: 20, pipValue: 7.5, pipSize: 0.0001, baseCurrency: 'USD', marginPerLot: 5000.0 },
};

// Cache: DB-loaded configs with 5-min TTL
let _cachedSymbolConfigs: Record<string, SymbolConfig> | null = null;
let _cacheTimestamp = 0;
const CACHE_TTL_MS = 5 * 60 * 1000;

async function getSymbolConfigs(): Promise<Record<string, SymbolConfig>> {
  const now = Date.now();
  if (_cachedSymbolConfigs && now - _cacheTimestamp < CACHE_TTL_MS) {
    return _cachedSymbolConfigs;
  }

  try {
    const pool = getAiModelPool();
    const sql = `
      SELECT symbol, param_value, metadata
      FROM trading_config_params
      WHERE category = 'margin' AND symbol IS NOT NULL
    `;
    const rows = await executeQuery<{
      symbol: string;
      param_value: string;
      metadata: { leverage: number; pip_value: number; pip_size: number; base_currency: string } | null;
    }>(pool, sql);

    if (rows.length === 0) {
      return FALLBACK_SYMBOL_CONFIGS;
    }

    const configs: Record<string, SymbolConfig> = {};
    for (const row of rows) {
      const meta = row.metadata;
      configs[row.symbol] = {
        leverage: meta?.leverage ?? 20,
        pipValue: meta?.pip_value ?? 10.0,
        pipSize: meta?.pip_size ?? 0.0001,
        baseCurrency: meta?.base_currency ?? '',
        marginPerLot: parseFloat(row.param_value),
      };
    }
    _cachedSymbolConfigs = configs;
    _cacheTimestamp = now;
    return configs;
  } catch {
    return _cachedSymbolConfigs ?? FALLBACK_SYMBOL_CONFIGS;
  }
}


// Diversification rules
const RULES = {
  MAX_SINGLE_SYMBOL_ALLOCATION: 0.25, // Max 25% per symbol
  MAX_SINGLE_DIRECTION_ALLOCATION: 0.6,
  MIN_POSITION_SIZE: 0.01,
  MARGIN_RESERVE_RATIO: 0.1,
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Parse and validate balance parameter
 */
function parseBalance(value: unknown): number {
  if (value === undefined || value === null || value === '') {
    throw new ValidationError('Balance parameter is required', {
      field: 'balance',
    });
  }

  const balance = parseFloat(String(value));
  if (isNaN(balance) || balance < 0) {
    throw new ValidationError('Balance must be a positive number', {
      field: 'balance',
      value,
    });
  }

  return balance;
}

/**
 * Calculate allocation for all symbols based on balance
 */
function calculateAllocation(balance: number, symbolConfigs: Record<string, SymbolConfig>): AllocationResponse {
  const availableBalance = balance * (1 - RULES.MARGIN_RESERVE_RATIO);
  const maxPerSymbol = availableBalance * RULES.MAX_SINGLE_SYMBOL_ALLOCATION;

  const allocations: SymbolAllocation[] = [];
  let totalLots = 0;
  let totalMargin = 0;
  let tradeableCount = 0;

  for (const [symbol, config] of Object.entries(symbolConfigs)) {
    const maxLots = Math.floor((maxPerSymbol / config.marginPerLot) * 100) / 100;
    const marginRequired = maxLots * config.marginPerLot;

    let status: SymbolAllocation['status'];
    if (maxLots >= 0.2) status = 'excellent';
    else if (maxLots >= 0.1) status = 'good';
    else if (maxLots >= RULES.MIN_POSITION_SIZE) status = 'warning';
    else status = 'insufficient';

    if (maxLots >= RULES.MIN_POSITION_SIZE) {
      tradeableCount++;
      totalLots += maxLots;
      totalMargin += marginRequired;
    }

    allocations.push({
      symbol,
      maxLots: Math.max(0, maxLots),
      marginRequired: Math.round(marginRequired * 100) / 100,
      leverage: config.leverage,
      status,
    });
  }

  // Sort by maxLots descending
  allocations.sort((a, b) => b.maxLots - a.maxLots);

  // Calculate diversification score
  const maxSymbols = Object.keys(symbolConfigs).length;
  const symbolScore = (tradeableCount / maxSymbols) * 100;

  const lots = allocations.map((a) => a.maxLots).filter((l) => l >= RULES.MIN_POSITION_SIZE);
  let evenScore = 100;
  if (lots.length > 1) {
    const avg = lots.reduce((a, b) => a + b, 0) / lots.length;
    const variance = lots.reduce((sum, l) => sum + Math.pow((l - avg) / avg, 2), 0) / lots.length;
    evenScore = Math.max(0, 100 - variance * 100);
  }

  const diversificationScore = Math.min(100, Math.round(symbolScore * 0.4 + evenScore * 0.3 + 100 * 0.3));

  return {
    balance,
    availableBalance: Math.round(availableBalance * 100) / 100,
    marginReserve: Math.round(balance * RULES.MARGIN_RESERVE_RATIO * 100) / 100,
    tradeableSymbols: tradeableCount,
    totalSymbols: maxSymbols,
    maxTotalLots: Math.round(totalLots * 100) / 100,
    marginUtilization:
      availableBalance > 0 ? Math.round((totalMargin / availableBalance) * 100) : 0,
    avgLotSize: tradeableCount > 0 ? Math.round((totalLots / tradeableCount) * 100) / 100 : 0,
    diversificationScore,
    allocations,
  };
}

/**
 * Format standard API response
 */
function formatResponse<T>(data: T, req: RequestWithId, count?: number): ApiResponse<T> {
  return {
    data,
    metadata: {
      count,
      timestamp: new Date().toISOString(),
      requestId: req.id,
    },
  };
}

// ============================================================================
// Route Handlers
// ============================================================================

/**
 * GET /api/position-sizing/allocation
 *
 * Calculate position allocations for a given balance.
 *
 * Query params:
 * - balance: Account balance in USD (required)
 *
 * Returns allocation details for all symbols including:
 * - maxLots: Maximum tradeable lots for each symbol
 * - marginRequired: Margin needed for max position
 * - status: excellent/good/warning/insufficient
 * - diversificationScore: Overall portfolio diversification rating
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/position-sizing/allocation',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const balance = parseBalance(req.query['balance']);
      const configs = await getSymbolConfigs();
      const allocation = calculateAllocation(balance, configs);

      res.json(formatResponse(allocation, req, allocation.allocations.length));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/position-sizing/symbols
 *
 * Get all symbol configurations and margin requirements.
 *
 * Returns configuration for each tradeable symbol including:
 * - leverage
 * - pipValue
 * - marginPerLot
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/position-sizing/symbols',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      const configs = await getSymbolConfigs();
      const symbols = Object.entries(configs).map(([symbol, config]) => ({
        symbol,
        ...config,
      }));

      res.json(formatResponse(symbols, req, symbols.length));
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /api/position-sizing/rules
 *
 * Get the diversification rules used for allocation.
 */
// eslint-disable-next-line @typescript-eslint/no-misused-promises
router.get(
  '/api/position-sizing/rules',
  async (req: RequestWithId, res: Response, next: NextFunction) => {
    try {
      res.json(
        formatResponse(
          {
            maxSingleSymbolAllocation: RULES.MAX_SINGLE_SYMBOL_ALLOCATION,
            maxSingleDirectionAllocation: RULES.MAX_SINGLE_DIRECTION_ALLOCATION,
            minPositionSize: RULES.MIN_POSITION_SIZE,
            marginReserveRatio: RULES.MARGIN_RESERVE_RATIO,
          },
          req
        )
      );
    } catch (error) {
      next(error);
    }
  }
);

export default router;
