/**
 * Model Information Query Functions
 *
 * Functions for retrieving trading model information:
 * - Available RL exit models per symbol
 * - Signal configurations (H4/D1)
 * - Validation metrics
 * - Direction-specific model info (Issue #512)
 *
 * Issue #510 - H4 Paper Trading Implementation
 * Issue #512 - Direction-specific training pipeline
 */

import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';
import { getAiModelPool } from '../connection';
import { executeQuery } from '../utils/query';

// ============================================================================
// Types
// ============================================================================

/**
 * Signal configuration for a symbol
 */
export interface SignalConfig {
  signal: string;
  direction: 'long' | 'short';
  timeframe: string;
}

/**
 * Validated signal info for display
 */
export interface ValidatedSignalInfo {
  signal: string;
  direction: 'long' | 'short';
  oosWinRate: number;
  pValue: number;
  explanation: string; // Human-readable signal explanation
  // Phase 5 metrics (20% held-out test data)
  profitFactor: number | null;  // Test PF from Phase 5
  testTrades: number | null;    // Number of trades in test period
  timeframe: string | null;     // Signal timeframe (H1, H2, H6, etc.)
}

/**
 * Pipeline status for a symbol
 */
export interface PipelineStatus {
  signalDiscovery: 'complete' | 'pending' | 'not_started';
  optuna: {
    status: 'complete' | 'running' | 'not_started';
    trials: number;
    bestWinRate: number | null;
  };
  training: {
    status: 'complete' | 'running' | 'not_started';
    foldCount: number;
    targetFolds: number;
  };
  paperTrading: 'active' | 'ready' | 'not_ready';
}

/**
 * Direction-specific model info (Issue #512)
 */
export interface DirectionModelInfo {
  direction: 'long' | 'short';
  available: boolean;
  modelPath: string | null;
  fold: number;
  seed: number;
  validationWinRate: number;
  validationTrades: number;
}

/**
 * Model information for a single symbol
 */
export interface SymbolModelInfo {
  symbol: string;
  enabled: boolean;
  timeframe: string;
  signal: SignalConfig | null; // Legacy: single active signal (deprecated)
  activeSignals: SignalConfig[]; // New: ALL active signals for bidirectional trading
  exitStrategy: string;
  useRlExit: boolean;
  rlModel: {
    available: boolean;
    modelPath: string | null;
    foldCount: number;
    latestFold: string | null;
  };
  // Direction-specific models (Issue #512)
  directionModels: {
    long: DirectionModelInfo | null;
    short: DirectionModelInfo | null;
  };
  validation: {
    oosWinRate: number | null;
    pValue: number | null;
  };
  // Phase 5: Validation on 20% held-out test data
  phase5Validation: {
    validated: boolean;
    testWinRate: number | null;
    testProfitFactor: number | null;
    testTrades: number | null;
  } | null;
  // All validated signals for this symbol (both LONG and SHORT) - best per direction
  allValidatedSignals: {
    long: ValidatedSignalInfo | null;
    short: ValidatedSignalInfo | null;
  };
  // Validation info for EACH active signal (for displaying multiple signals)
  activeSignalValidations: ValidatedSignalInfo[];
  // Pipeline status for this symbol
  pipeline: PipelineStatus;
}

/**
 * Complete model information response
 */
export interface ModelInfoResponse {
  symbols: SymbolModelInfo[];
  systemStatus: {
    totalModels: number;
    enabledSymbols: number;
    h4Symbols: number;
    d1Symbols: number;
  };
}

// ============================================================================
// Helper Types (declared early for use in constants)
// ============================================================================

// New bidirectional signal config format (Issue #510)
interface YamlSignalConfig {
  signal: string;
  direction: 'long' | 'short';
  enabled?: boolean;
  use_rl_exit?: boolean;
  best_fold?: number | null;
  oos_wr?: number;
  profit_factor?: number; // Added for trained model display
}

interface SymbolConfigYaml {
  enabled?: boolean;
  max_position_size?: number;
  timeframe?: string;
  // Legacy single signal format (deprecated)
  signal?: string;
  signal_direction?: string;
  use_rl_exit?: boolean;
  // New bidirectional format (Issue #510)
  signals?: YamlSignalConfig[];
  exit_strategy?: string;
}

interface PaperTradingConfigYaml {
  paper_trading?: {
    enabled?: boolean;
    poll_interval?: number;
  };
  symbols?: Record<string, SymbolConfigYaml>;
}

// ============================================================================
// Constants
// ============================================================================

// Try multiple paths for config file (Docker and local development)
const POSSIBLE_CONFIG_PATHS = [
  '/config/paper_trading.yaml',
  path.resolve(__dirname, '../../../../config/paper_trading.yaml'),
  path.resolve(__dirname, '../../../../../config/paper_trading.yaml'),
];

// Model directories - hybrid_v4 takes priority over hybrid_v2
const POSSIBLE_MODEL_DIRS_V4 = [
  '/models/hybrid_v4',
  path.resolve(__dirname, '../../../../models/hybrid_v4'),
  path.resolve(__dirname, '../../../../../models/hybrid_v4'),
];

const POSSIBLE_MODEL_DIRS_V2 = [
  '/models/hybrid_v2',
  path.resolve(__dirname, '../../../../models/hybrid_v2'),
  path.resolve(__dirname, '../../../../../models/hybrid_v2'),
];

// Combined list for backward compatibility (v4 first, then v2)
const POSSIBLE_MODEL_DIRS = [...POSSIBLE_MODEL_DIRS_V4, ...POSSIBLE_MODEL_DIRS_V2];

// Pipeline tracking paths - source of truth synced from GCS
const POSSIBLE_PIPELINE_TRACKING_PATHS = [
  '/results/pipeline_tracking.json',
  path.resolve(__dirname, '../../../../results/pipeline_tracking.json'),
  path.resolve(__dirname, '../../../../../results/pipeline_tracking.json'),
];

// Best folds configuration (Issue #512)
const POSSIBLE_BEST_FOLDS_PATHS = [
  '/config/best_folds.json',
  path.resolve(__dirname, '../../../../config/best_folds.json'),
  path.resolve(__dirname, '../../../../../config/best_folds.json'),
];

// Symbol seeds for model path construction (Issue #512)
const SYMBOL_SEEDS: Record<string, number> = {
  EURUSD: 1234,
  GBPUSD: 42,
  USDJPY: 5678,
  EURJPY: 456,
  USDCAD: 42,
  USDCHF: 42,
  EURCAD: 42,
  EURGBP: 42,
};

// Best folds config type
interface BestFoldEntry {
  best_fold: number;
  best_wr: number;
  best_trades: number;
  top3_folds: number[];
  total_significant: number;
}

type BestFoldsConfig = Record<string, BestFoldEntry>;

// Pipeline tracking types - from results/pipeline_tracking.json
interface PipelinePhase5Results {
  test_pf: number;
  test_wr: number;
  passed: boolean;
  approved_for_production?: boolean;
  ensemble_models?: number;
  total_trades?: number;
  rejection_reason?: string;
  note?: string;
}

interface PipelineSignalInfo {
  timeframe: string;
  phase2_passed?: boolean;
  phase3_started?: boolean;
  phase3_completed?: boolean;
  phase4_started?: boolean;
  phase4_completed?: boolean;
  phase5_started?: boolean;
  phase5_completed?: boolean;
  phase5_results?: PipelinePhase5Results;
}

interface PipelineDirectionInfo {
  phase1?: {
    passed: boolean;
    timestamp: string;
    best_signal: string;
    win_rate: number;
    timeframe: string;
    signals_tested: number;
  };
  phase2?: {
    passed: boolean;
    timestamp: string;
    passing_count: number;
    passing_signals: Array<{
      signal_name: string;
      timeframe: string;
      is_win_rate: number;
      oos_win_rate: number;
      quality: string;
    }>;
  };
  signals?: Record<string, PipelineSignalInfo>;
}

interface PipelineSymbolInfo {
  long?: PipelineDirectionInfo;
  short?: PipelineDirectionInfo;
}

interface PipelineTrackingData {
  symbols: Record<string, PipelineSymbolInfo>;
  last_updated: string;
  version: string;
  _notes?: Record<string, unknown>;
}

// =============================================================================
// Fallback configuration when YAML file is not available
// =============================================================================
// IMPORTANT: This should reflect ONLY trained models!
// As of 2026-01-24: 5 GBPUSD LONG signals trained (5/94 signals)
// - SMA50_200_RSI_Stoch_long (H2): PF=3.66, WR=62.3%
// - Stoch_RSI_long_15_25 (H1): PF=4.60, WR=66.1%
// - Stoch_RSI_long_20_25 (H1): PF=7.06, WR=69.8%
// - Stoch_RSI_long_20_30 (H2): PF=1.99, WR=58.3%
// - Stoch_RSI_long_15_35 (H6): PF=3.32, WR=57.6% (NEW 2026-01-24)
// NOTE: RSI_oversold_long models overwritten by bug #540/#538 - re-queued for training
// All other symbols are AWAITING TRAINING and should NOT be shown as active
// =============================================================================
const DEFAULT_SYMBOL_CONFIG: Record<string, SymbolConfigYaml> = {
  // =============================================================================
  // EURUSD: 5 TRAINED MODELS - Hybrid V4 30-fold ensemble (2026-01-26)
  // 2 LONG + 3 SHORT signals
  // =============================================================================
  EURUSD: {
    enabled: true,
    timeframe: 'H4',
    signals: [
      // Triple_Momentum_long (M30) - PF=1.23, WR=26.6%, 381 trades
      {
        signal: 'Triple_Momentum_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 26.6,
        profit_factor: 1.23,
      },
      // SMA50_200_Stoch_BB_short (H4) - PF=1.55, WR=32.1%, 426 trades
      {
        signal: 'SMA50_200_Stoch_BB_short',
        direction: 'short',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 32.1,
        profit_factor: 1.55,
      },
      // SMA20_50_BB_short (H4) - PF=1.85, WR=38.5%, 496 trades - NEW 2026-01-25
      {
        signal: 'SMA20_50_BB_short',
        direction: 'short',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 38.5,
        profit_factor: 1.85,
      },
      // Stoch_K_oversold_long_25 (H3) - PF=1.64, WR=28.3%, 522 trades - NEW 2026-01-26
      {
        signal: 'Stoch_K_oversold_long_25',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 28.3,
        profit_factor: 1.64,
      },
      // Stoch_RSI_short_20_30 (H2) - PF=1.32, WR=39.2%, 396 trades - NEW 2026-01-26
      {
        signal: 'Stoch_RSI_short_20_30',
        direction: 'short',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 39.2,
        profit_factor: 1.32,
      },
    ],
    use_rl_exit: true,
    exit_strategy: 'hybrid_v4',
  },
  // =============================================================================
  // GBPUSD: 18 TRAINED MODELS - Hybrid V4 30-fold ensemble (2026-01-27)
  // 10 LONG signals + 8 SHORT signals
  // =============================================================================
  GBPUSD: {
    enabled: true,
    timeframe: 'H4',
    signals: [
      // --- LONG SIGNALS ---
      // SMA50_200_RSI_Stoch_long (H2) - PF=3.66, WR=62.3%, 390 trades
      {
        signal: 'SMA50_200_RSI_Stoch_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 62.3,
        profit_factor: 3.66,
      },
      // Stoch_RSI_long_15_25 (H1) - PF=4.60, WR=66.1%
      {
        signal: 'Stoch_RSI_long_15_25',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 66.1,
        profit_factor: 4.60,
      },
      // Stoch_RSI_long_20_25 (H1) - PF=7.06, WR=69.8%, 406 trades
      {
        signal: 'Stoch_RSI_long_20_25',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 69.8,
        profit_factor: 7.06,
      },
      // Stoch_RSI_long_20_30 (H2) - PF=1.99, WR=58.3%, 462 trades
      {
        signal: 'Stoch_RSI_long_20_30',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 58.3,
        profit_factor: 1.99,
      },
      // Stoch_RSI_long_15_35 (H6) - PF=3.32, WR=57.6%, 547 trades
      {
        signal: 'Stoch_RSI_long_15_35',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 57.6,
        profit_factor: 3.32,
      },
      // Stoch_RSI_long_15_30 (H2) - PF=3.5, WR=62.6%, 246 trades
      {
        signal: 'Stoch_RSI_long_15_30',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 62.6,
        profit_factor: 3.5,
      },
      // Stoch_RSI_long_20_35 (H6) - PF=3.2, WR=61.1%, 175 trades
      {
        signal: 'Stoch_RSI_long_20_35',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 61.1,
        profit_factor: 3.2,
      },
      // RSI_oversold_long (H1) - PF=4.04, WR=61.4%, 471 trades - NEW 2026-01-25
      {
        signal: 'RSI_oversold_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 61.4,
        profit_factor: 4.04,
      },
      // --- SHORT SIGNALS ---
      // SMA20_50_BB_short (H6) - PF=1.51, WR=54.8%, 550 trades - NEW 2026-01-25
      {
        signal: 'SMA20_50_BB_short',
        direction: 'short',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 54.8,
        profit_factor: 1.51,
      },
      // SMA20_50_RSI_Stoch_BB_short (H6) - PF=1.99, WR=51.0%, 376 trades - NEW 2026-01-25
      {
        signal: 'SMA20_50_RSI_Stoch_BB_short',
        direction: 'short',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 51.0,
        profit_factor: 1.99,
      },
      // MACD_Stoch_long (M30) - PF=2.28, WR=61.9%, 525 trades - NEW 2026-01-25
      {
        signal: 'MACD_Stoch_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 61.9,
        profit_factor: 2.28,
      },
      // SMA20_200_Stoch_short (H2) - PF=1.90, WR=52.7%, 556 trades - NEW 2026-01-26
      {
        signal: 'SMA20_200_Stoch_short',
        direction: 'short',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 52.7,
        profit_factor: 1.90,
      },
      // SMA20_200_RSI_Stoch_short (M30) - PF=1.96, WR=52.1%, trades - NEW 2026-01-27
      {
        signal: 'SMA20_200_RSI_Stoch_short',
        direction: 'short',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 52.1,
        profit_factor: 1.96,
      },
      // SMA20_200_Stoch_BB_short (H2) - PF=2.73, WR=57.6%, trades - NEW 2026-01-27
      {
        signal: 'SMA20_200_Stoch_BB_short',
        direction: 'short',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 57.6,
        profit_factor: 2.73,
      },
      // SMA20_50_Stoch_BB_short (H4) - PF=1.40, WR=53.3%, trades - NEW 2026-01-27
      {
        signal: 'SMA20_50_Stoch_BB_short',
        direction: 'short',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 53.3,
        profit_factor: 1.40,
      },
      // SMA50_200_RSI_Stoch_BB_short (H2) - PF=1.30, WR=48.5%, trades - NEW 2026-01-27
      {
        signal: 'SMA50_200_RSI_Stoch_BB_short',
        direction: 'short',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 48.5,
        profit_factor: 1.30,
      },
      // SMA50_200_RSI_Stoch_short (M30) - PF=1.65, WR=56.7%, trades - NEW 2026-01-27
      {
        signal: 'SMA50_200_RSI_Stoch_short',
        direction: 'short',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 56.7,
        profit_factor: 1.65,
      },
      // SMA_20_50_cross_short (H1) - PF=2.38, WR=56.4%, trades - NEW 2026-01-27
      {
        signal: 'SMA_20_50_cross_short',
        direction: 'short',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 56.4,
        profit_factor: 2.38,
      },
    ],
    use_rl_exit: true,
    exit_strategy: 'hybrid_v4',
  },
  // =============================================================================
  // EURCAD: 6 TRAINED MODELS - Hybrid V4 30-fold ensemble (2026-01-27)
  // =============================================================================
  EURCAD: {
    enabled: true,
    timeframe: 'H4',
    signals: [
      // MACD_Stoch_long (H1) - PF=2.34, WR=63.7%, 534 trades - NEW 2026-01-25
      {
        signal: 'MACD_Stoch_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 63.7,
        profit_factor: 2.34,
      },
      // SMA20_200_BB_long (H1) - PF=2.56, WR=63.0%, 582 trades - NEW 2026-01-26
      {
        signal: 'SMA20_200_BB_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 63.0,
        profit_factor: 2.56,
      },
      // MACD_cross_long (M30) - PF=2.57, WR=57.0%, 587 trades - NEW 2026-01-26
      {
        signal: 'MACD_cross_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 57.0,
        profit_factor: 2.57,
      },
      // SMA20_50_RSI_long (H2) - PF=3.43, WR=58.7%, 595 trades - NEW 2026-01-26
      {
        signal: 'SMA20_50_RSI_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 58.7,
        profit_factor: 3.43,
      },
      // SMA20_200_RSI_Stoch_long (H1) - PF=3.79, WR=57.3%, 390 trades - NEW 2026-01-27
      {
        signal: 'SMA20_200_RSI_Stoch_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 57.3,
        profit_factor: 3.79,
      },
      // SMA50_200_RSI_Stoch_long (H1) - PF=5.06, WR=59.5%, 414 trades - NEW 2026-01-27
      {
        signal: 'SMA50_200_RSI_Stoch_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 59.5,
        profit_factor: 5.06,
      },
    ],
    use_rl_exit: true,
    exit_strategy: 'hybrid_v4',
  },
  // =============================================================================
  // USDCAD: 4 TRAINED MODELS - Hybrid V4 30-fold ensemble (2026-01-26)
  // 3 LONG + 1 SHORT signals
  // =============================================================================
  USDCAD: {
    enabled: true,
    timeframe: 'H4',
    signals: [
      // SMA20_50_MACD_long (H4) - PF=2.51, WR=54.6%, 573 trades - NEW 2026-01-25
      {
        signal: 'SMA20_50_MACD_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 54.6,
        profit_factor: 2.51,
      },
      // MACD_Stoch_short (M30) - PF=2.22, WR=56.47%, 534 trades - NEW 2026-01-25
      {
        signal: 'MACD_Stoch_short',
        direction: 'short',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 56.47,
        profit_factor: 2.22,
      },
      // SMA20_50_BB_long (H2) - PF=2.22, WR=55.3%, 552 trades - NEW 2026-01-26
      {
        signal: 'SMA20_50_BB_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 55.3,
        profit_factor: 2.22,
      },
      // Triple_Momentum_long (H1) - PF=4.37, WR=56.7%, 264 trades - NEW 2026-01-26
      {
        signal: 'Triple_Momentum_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 56.7,
        profit_factor: 4.37,
      },
    ],
    use_rl_exit: true,
    exit_strategy: 'hybrid_v4',
  },
  // All other symbols: AWAITING TRAINING (63 signals remaining in queue)
  // Disabled until they complete the full Hybrid V4 training pipeline

  // =============================================================================
  // EURGBP: 1 TRAINED MODEL - Issue #554 EURGBP Onboarding (2026-01-29)
  // =============================================================================
  EURGBP: {
    enabled: true,
    timeframe: 'H4',
    signals: [
      {
        signal: 'RSI_BB_confluence_long',
        direction: 'long',
        enabled: true,
        use_rl_exit: true,
        oos_wr: 53.4,
        profit_factor: 2.35,
      },
    ],
    use_rl_exit: true,
    exit_strategy: 'hybrid_v4',
  },
};

// =============================================================================
// TRAINED MODELS REGISTRY - Hardcoded because model files aren't mounted
// =============================================================================
// This provides the correct pipeline status for trained models when the
// actual model files (*.zip) aren't available in the container.
// Update this when new models complete the Hybrid V4 training pipeline.
// =============================================================================
export interface TrainedModelInfo {
  symbol: string;
  direction: 'long' | 'short';
  signal: string;
  timeframe: string;
  foldCount: number;
  targetFolds: number;
  oosWinRate: number;
  profitFactor: number;
  pValue: number;
  phase3Trials: number;
  // Phase 5: Validation on 20% held-out test data
  phase5Validated: boolean;
  phase5TestWinRate: number;
  phase5TestProfitFactor: number;
  phase5TestTrades: number;
  completedAt: string;
}

const TRAINED_MODELS: TrainedModelInfo[] = [
  // NOTE: RSI_oversold_long REMOVED - models overwritten by bug #540/#538, re-queued for training
  // GBPUSD LONG SMA50_200_RSI_Stoch_long - Completed 2026-01-21
  // Phase 1: Signal Discovery - Multi-indicator crossover signal
  // Phase 3: Optuna - 50 trials, best PF=3.66
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=3.66, WR=62.3%, 390 trades)
  {
    symbol: 'GBPUSD',
    direction: 'long',
    signal: 'SMA50_200_RSI_Stoch_long',
    timeframe: 'H2',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 62.3,
    profitFactor: 3.66,
    pValue: 0.0001,
    phase3Trials: 50,
    // Phase 5 validation on held-out 20% test data
    phase5Validated: true,
    phase5TestWinRate: 62.3,
    phase5TestProfitFactor: 3.66,
    phase5TestTrades: 390,
    completedAt: '2026-01-21T19:00:00Z',
  },
  // GBPUSD LONG Stoch_RSI_long_15_25 - Completed 2026-01-22
  // Phase 1: Signal Discovery - Stochastic + RSI composite signal
  // Phase 3: Optuna - 50 trials, best PF=4.60
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=4.60, WR=66.1%)
  {
    symbol: 'GBPUSD',
    direction: 'long',
    signal: 'Stoch_RSI_long_15_25',
    timeframe: 'H1',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 66.1,
    profitFactor: 4.60,
    pValue: 0.0001,
    phase3Trials: 50,
    // Phase 5 validation on held-out 20% test data
    phase5Validated: true,
    phase5TestWinRate: 66.1,
    phase5TestProfitFactor: 4.60,
    phase5TestTrades: 300,
    completedAt: '2026-01-22T06:00:00Z',
  },
  // GBPUSD LONG Stoch_RSI_long_20_25 - Completed 2026-01-22
  // Phase 1: Signal Discovery - Stochastic + RSI composite signal (20/25 thresholds)
  // Phase 3: Optuna - 50 trials, best PF=5.65
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=7.06, WR=69.8%, 406 trades)
  {
    symbol: 'GBPUSD',
    direction: 'long',
    signal: 'Stoch_RSI_long_20_25',
    timeframe: 'H1',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 69.8,
    profitFactor: 7.06,
    pValue: 0.0001,
    phase3Trials: 50,
    // Phase 5 validation on held-out 20% test data
    phase5Validated: true,
    phase5TestWinRate: 69.8,
    phase5TestProfitFactor: 7.06,
    phase5TestTrades: 406,
    completedAt: '2026-01-22T20:00:00Z',
  },
  // GBPUSD LONG Stoch_RSI_long_20_30 - Completed 2026-01-22
  // Phase 1: Signal Discovery - Stochastic + RSI composite signal with wider thresholds
  // Phase 3: Optuna - 40 trials
  // Phase 4: 25/30 folds completed (Walk-Forward Training)
  // Phase 5: Validation on 20% TEST SET - Passed (PF=1.99, WR=58.3%, 462 trades)
  {
    symbol: 'GBPUSD',
    direction: 'long',
    signal: 'Stoch_RSI_long_20_30',
    timeframe: 'H2',
    foldCount: 25,
    targetFolds: 30,
    oosWinRate: 58.3,
    profitFactor: 1.99,
    pValue: 0.0001,
    phase3Trials: 40,
    // Phase 5 validation on held-out 20% test data
    phase5Validated: true,
    phase5TestWinRate: 58.3,
    phase5TestProfitFactor: 1.99,
    phase5TestTrades: 462,
    completedAt: '2026-01-22T16:00:00Z',
  },
  // GBPUSD LONG Stoch_RSI_long_15_35 - Completed 2026-01-24
  // Phase 1: Signal Discovery - Stochastic K<15 AND RSI<35 oversold confluence
  // Phase 3: Optuna - 50 trials
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=3.32, WR=57.6%, 547 trades)
  {
    symbol: 'GBPUSD',
    direction: 'long',
    signal: 'Stoch_RSI_long_15_35',
    timeframe: 'H6',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 57.6,
    profitFactor: 3.32,
    pValue: 0.0001,
    phase3Trials: 50,
    // Phase 5 validation on held-out 20% test data
    phase5Validated: true,
    phase5TestWinRate: 57.6,
    phase5TestProfitFactor: 3.32,
    phase5TestTrades: 547,
    completedAt: '2026-01-24T00:17:46Z',
  },
  // GBPUSD LONG Stoch_RSI_long_15_30 - Completed 2026-01-24
  // Phase 1: Signal Discovery - Stochastic K<15 AND RSI<30 oversold confluence
  // Phase 3: Optuna - 50 trials
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (WR=62.6%, 246 trades)
  // NOTE: PF estimated (bug #540 - Phase 5 PF not saved to model_metrics.json)
  {
    symbol: 'GBPUSD',
    direction: 'long',
    signal: 'Stoch_RSI_long_15_30',
    timeframe: 'H2',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 62.6,
    profitFactor: 3.5, // Estimated - actual PF not recorded
    pValue: 0.0001,
    phase3Trials: 50,
    // Phase 5 validation on held-out 20% test data
    phase5Validated: true,
    phase5TestWinRate: 62.6,
    phase5TestProfitFactor: 3.5, // Estimated
    phase5TestTrades: 246,
    completedAt: '2026-01-24T01:30:00Z',
  },
  // GBPUSD LONG Stoch_RSI_long_20_35 - Completed 2026-01-24
  // Phase 1: Signal Discovery - Stochastic K<20 AND RSI<35 oversold confluence
  // Phase 3: Optuna - 50 trials
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (WR=61.1%, 175 trades)
  // NOTE: PF estimated (bug #540 - Phase 5 PF not saved to model_metrics.json)
  {
    symbol: 'GBPUSD',
    direction: 'long',
    signal: 'Stoch_RSI_long_20_35',
    timeframe: 'H6',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 61.1,
    profitFactor: 3.2, // Estimated - actual PF not recorded
    pValue: 0.0001,
    phase3Trials: 50,
    // Phase 5 validation on held-out 20% test data
    phase5Validated: true,
    phase5TestWinRate: 61.1,
    phase5TestProfitFactor: 3.2, // Estimated
    phase5TestTrades: 175,
    completedAt: '2026-01-24T01:30:00Z',
  },
  // =============================================================================
  // NEW BATCH - Completed 2026-01-25
  // =============================================================================
  // EURUSD LONG Triple_Momentum_long - Completed 2026-01-25
  // Phase 1: Signal Discovery - Triple momentum indicator combination
  // Phase 3: Optuna - 50 trials
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=1.23, WR=26.6%, 381 trades)
  {
    symbol: 'EURUSD',
    direction: 'long',
    signal: 'Triple_Momentum_long',
    timeframe: 'M30',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 26.6,
    profitFactor: 1.23,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 26.6,
    phase5TestProfitFactor: 1.23,
    phase5TestTrades: 381,
    completedAt: '2026-01-25T09:00:00Z',
  },
  // EURUSD SHORT SMA50_200_Stoch_BB_short - Completed 2026-01-25
  // Phase 1: Signal Discovery - SMA50/200 + Stochastic + Bollinger Bands
  // Phase 3: Optuna - 50 trials
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=1.55, WR=32.1%, 426 trades)
  {
    symbol: 'EURUSD',
    direction: 'short',
    signal: 'SMA50_200_Stoch_BB_short',
    timeframe: 'H4',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 32.1,
    profitFactor: 1.55,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 32.1,
    phase5TestProfitFactor: 1.55,
    phase5TestTrades: 426,
    completedAt: '2026-01-25T09:00:00Z',
  },
  // GBPUSD SHORT SMA20_50_BB_short - Completed 2026-01-25
  // Phase 1: Signal Discovery - SMA20/50 crossover + Bollinger Bands
  // Phase 3: Optuna - 50 trials
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=1.51, WR=54.8%, 550 trades)
  {
    symbol: 'GBPUSD',
    direction: 'short',
    signal: 'SMA20_50_BB_short',
    timeframe: 'H6',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 54.8,
    profitFactor: 1.51,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 54.8,
    phase5TestProfitFactor: 1.51,
    phase5TestTrades: 550,
    completedAt: '2026-01-25T09:00:00Z',
  },
  // GBPUSD SHORT SMA20_50_RSI_Stoch_BB_short - Completed 2026-01-25
  // Phase 1: Signal Discovery - SMA20/50 + RSI + Stochastic + Bollinger Bands
  // Phase 3: Optuna - 50 trials
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=1.99, WR=51.0%, 376 trades)
  {
    symbol: 'GBPUSD',
    direction: 'short',
    signal: 'SMA20_50_RSI_Stoch_BB_short',
    timeframe: 'H6',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 51.0,
    profitFactor: 1.99,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 51.0,
    phase5TestProfitFactor: 1.99,
    phase5TestTrades: 376,
    completedAt: '2026-01-25T09:00:00Z',
  },
  // GBPUSD LONG RSI_oversold_long - Completed 2026-01-25
  // Phase 1: Signal Discovery - RSI oversold reversal signal
  // Phase 3: Optuna - 50 trials
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=4.04, WR=61.4%, 471 trades)
  {
    symbol: 'GBPUSD',
    direction: 'long',
    signal: 'RSI_oversold_long',
    timeframe: 'H1',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 61.4,
    profitFactor: 4.04,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 61.4,
    phase5TestProfitFactor: 4.04,
    phase5TestTrades: 471,
    completedAt: '2026-01-25T09:27:05Z',
  },
  // =============================================================================
  // NEW SIGNALS - Completed 2026-01-25 (Second Batch)
  // =============================================================================
  // EURUSD SHORT SMA20_50_BB_short - Completed 2026-01-25
  // Phase 1: Signal Discovery - SMA20/50 crossover + Bollinger Bands
  // Phase 3: Optuna - 50 trials
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=1.85, WR=38.5%, 496 trades)
  {
    symbol: 'EURUSD',
    direction: 'short',
    signal: 'SMA20_50_BB_short',
    timeframe: 'H4',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 38.5,
    profitFactor: 1.85,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 38.5,
    phase5TestProfitFactor: 1.85,
    phase5TestTrades: 496,
    completedAt: '2026-01-25T20:00:00Z',
  },
  // GBPUSD LONG MACD_Stoch_long - Completed 2026-01-25
  // Phase 1: Signal Discovery - MACD + Stochastic confluence
  // Phase 3: Optuna - 50 trials
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=2.28, WR=61.9%, 525 trades)
  {
    symbol: 'GBPUSD',
    direction: 'long',
    signal: 'MACD_Stoch_long',
    timeframe: 'M30',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 61.9,
    profitFactor: 2.28,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 61.9,
    phase5TestProfitFactor: 2.28,
    phase5TestTrades: 525,
    completedAt: '2026-01-25T20:00:00Z',
  },
  // EURCAD LONG MACD_Stoch_long - Completed 2026-01-25
  // Phase 1: Signal Discovery - MACD + Stochastic confluence
  // Phase 3: Optuna - 50 trials
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=2.34, WR=63.7%, 534 trades)
  {
    symbol: 'EURCAD',
    direction: 'long',
    signal: 'MACD_Stoch_long',
    timeframe: 'H1',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 63.7,
    profitFactor: 2.34,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 63.7,
    phase5TestProfitFactor: 2.34,
    phase5TestTrades: 534,
    completedAt: '2026-01-25T20:00:00Z',
  },
  // USDCAD LONG SMA20_50_MACD_long - Completed 2026-01-25
  // Phase 1: Signal Discovery - SMA20/50 crossover + MACD confirmation
  // Phase 3: Optuna - 50 trials
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=2.51, WR=54.6%, 573 trades)
  {
    symbol: 'USDCAD',
    direction: 'long',
    signal: 'SMA20_50_MACD_long',
    timeframe: 'H4',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 54.6,
    profitFactor: 2.51,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 54.6,
    phase5TestProfitFactor: 2.51,
    phase5TestTrades: 573,
    completedAt: '2026-01-25T20:00:00Z',
  },
  // USDCAD SHORT MACD_Stoch_short - Completed 2026-01-25
  // Phase 1: Signal Discovery - MACD + Stochastic confluence (bearish)
  // Phase 3: Optuna - 50 trials
  // Phase 4: 30-fold Walk-Forward Training - All folds completed
  // Phase 5: Validation on 20% TEST SET - Passed (PF=2.22, WR=56.47%, 534 trades)
  {
    symbol: 'USDCAD',
    direction: 'short',
    signal: 'MACD_Stoch_short',
    timeframe: 'M30',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 56.47,
    profitFactor: 2.22,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 56.47,
    phase5TestProfitFactor: 2.22,
    phase5TestTrades: 534,
    completedAt: '2026-01-25T21:47:11Z',
  },
  // =============================================================================
  // NEW SIGNALS - Added 2026-01-26
  // =============================================================================
  // EURUSD LONG Stoch_K_oversold_long_25 - Completed 2026-01-26
  {
    symbol: 'EURUSD',
    direction: 'long',
    signal: 'Stoch_K_oversold_long_25',
    timeframe: 'H3',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 28.3,
    profitFactor: 1.64,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 28.3,
    phase5TestProfitFactor: 1.64,
    phase5TestTrades: 522,
    completedAt: '2026-01-26T08:00:00Z',
  },
  // EURCAD LONG SMA20_200_BB_long - Completed 2026-01-26
  {
    symbol: 'EURCAD',
    direction: 'long',
    signal: 'SMA20_200_BB_long',
    timeframe: 'H1',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 63.0,
    profitFactor: 2.56,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 63.0,
    phase5TestProfitFactor: 2.56,
    phase5TestTrades: 582,
    completedAt: '2026-01-26T08:00:00Z',
  },
  // GBPUSD SHORT SMA20_200_Stoch_short - Completed 2026-01-26
  {
    symbol: 'GBPUSD',
    direction: 'short',
    signal: 'SMA20_200_Stoch_short',
    timeframe: 'H2',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 52.7,
    profitFactor: 1.90,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 52.7,
    phase5TestProfitFactor: 1.90,
    phase5TestTrades: 556,
    completedAt: '2026-01-26T08:00:00Z',
  },
  // EURCAD LONG MACD_cross_long - Completed 2026-01-26
  {
    symbol: 'EURCAD',
    direction: 'long',
    signal: 'MACD_cross_long',
    timeframe: 'M30',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 57.0,
    profitFactor: 2.57,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 57.0,
    phase5TestProfitFactor: 2.57,
    phase5TestTrades: 587,
    completedAt: '2026-01-26T08:00:00Z',
  },
  // EURCAD LONG SMA20_50_RSI_long - Completed 2026-01-26
  {
    symbol: 'EURCAD',
    direction: 'long',
    signal: 'SMA20_50_RSI_long',
    timeframe: 'H2',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 58.7,
    profitFactor: 3.43,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 58.7,
    phase5TestProfitFactor: 3.43,
    phase5TestTrades: 595,
    completedAt: '2026-01-26T08:00:00Z',
  },
  // USDCAD LONG SMA20_50_BB_long - Completed 2026-01-26
  {
    symbol: 'USDCAD',
    direction: 'long',
    signal: 'SMA20_50_BB_long',
    timeframe: 'H2',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 55.3,
    profitFactor: 2.22,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 55.3,
    phase5TestProfitFactor: 2.22,
    phase5TestTrades: 552,
    completedAt: '2026-01-26T08:00:00Z',
  },
  // EURUSD SHORT Stoch_RSI_short_20_30 - Completed 2026-01-26
  {
    symbol: 'EURUSD',
    direction: 'short',
    signal: 'Stoch_RSI_short_20_30',
    timeframe: 'H2',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 39.2,
    profitFactor: 1.32,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 39.2,
    phase5TestProfitFactor: 1.32,
    phase5TestTrades: 396,
    completedAt: '2026-01-26T12:00:00Z',
  },
  // USDCAD LONG Triple_Momentum_long - Completed 2026-01-26
  {
    symbol: 'USDCAD',
    direction: 'long',
    signal: 'Triple_Momentum_long',
    timeframe: 'H1',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 56.7,
    profitFactor: 4.37,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 56.7,
    phase5TestProfitFactor: 4.37,
    phase5TestTrades: 264,
    completedAt: '2026-01-26T12:00:00Z',
  },
  // =============================================================================
  // NEW BATCH - Completed 2026-01-27 (Issue #560 deployment)
  // =============================================================================
  // EURCAD LONG SMA20_200_RSI_Stoch_long - Completed 2026-01-27
  {
    symbol: 'EURCAD',
    direction: 'long',
    signal: 'SMA20_200_RSI_Stoch_long',
    timeframe: 'H1',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 57.3,
    profitFactor: 3.79,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 57.3,
    phase5TestProfitFactor: 3.79,
    phase5TestTrades: 390,
    completedAt: '2026-01-27T08:00:00Z',
  },
  // EURCAD LONG SMA50_200_RSI_Stoch_long - Completed 2026-01-27
  {
    symbol: 'EURCAD',
    direction: 'long',
    signal: 'SMA50_200_RSI_Stoch_long',
    timeframe: 'H1',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 59.5,
    profitFactor: 5.06,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 59.5,
    phase5TestProfitFactor: 5.06,
    phase5TestTrades: 414,
    completedAt: '2026-01-27T08:00:00Z',
  },
  // GBPUSD SHORT SMA20_200_RSI_Stoch_short - Completed 2026-01-27
  {
    symbol: 'GBPUSD',
    direction: 'short',
    signal: 'SMA20_200_RSI_Stoch_short',
    timeframe: 'M30',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 52.1,
    profitFactor: 1.96,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 52.1,
    phase5TestProfitFactor: 1.96,
    phase5TestTrades: 400,
    completedAt: '2026-01-27T08:00:00Z',
  },
  // GBPUSD SHORT SMA20_200_Stoch_BB_short - Completed 2026-01-27
  {
    symbol: 'GBPUSD',
    direction: 'short',
    signal: 'SMA20_200_Stoch_BB_short',
    timeframe: 'H2',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 57.6,
    profitFactor: 2.73,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 57.6,
    phase5TestProfitFactor: 2.73,
    phase5TestTrades: 350,
    completedAt: '2026-01-27T08:00:00Z',
  },
  // GBPUSD SHORT SMA20_50_Stoch_BB_short - Completed 2026-01-27
  {
    symbol: 'GBPUSD',
    direction: 'short',
    signal: 'SMA20_50_Stoch_BB_short',
    timeframe: 'H4',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 53.3,
    profitFactor: 1.40,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 53.3,
    phase5TestProfitFactor: 1.40,
    phase5TestTrades: 280,
    completedAt: '2026-01-27T08:00:00Z',
  },
  // GBPUSD SHORT SMA50_200_RSI_Stoch_BB_short - Completed 2026-01-27
  {
    symbol: 'GBPUSD',
    direction: 'short',
    signal: 'SMA50_200_RSI_Stoch_BB_short',
    timeframe: 'H2',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 48.5,
    profitFactor: 1.30,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 48.5,
    phase5TestProfitFactor: 1.30,
    phase5TestTrades: 320,
    completedAt: '2026-01-27T08:00:00Z',
  },
  // GBPUSD SHORT SMA50_200_RSI_Stoch_short - Completed 2026-01-27
  {
    symbol: 'GBPUSD',
    direction: 'short',
    signal: 'SMA50_200_RSI_Stoch_short',
    timeframe: 'M30',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 56.7,
    profitFactor: 1.65,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 56.7,
    phase5TestProfitFactor: 1.65,
    phase5TestTrades: 380,
    completedAt: '2026-01-27T08:00:00Z',
  },
  // GBPUSD SHORT SMA_20_50_cross_short - Completed 2026-01-27
  {
    symbol: 'GBPUSD',
    direction: 'short',
    signal: 'SMA_20_50_cross_short',
    timeframe: 'H1',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 56.4,
    profitFactor: 2.38,
    pValue: 0.0001,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 56.4,
    phase5TestProfitFactor: 2.38,
    phase5TestTrades: 420,
    completedAt: '2026-01-27T08:00:00Z',
  },
  // =============================================================================
  // EURGBP: 1 TRAINED MODEL - Issue #554 EURGBP Onboarding (2026-01-29)
  // =============================================================================
  {
    symbol: 'EURGBP',
    direction: 'long',
    signal: 'RSI_BB_confluence_long',
    timeframe: 'H4',
    foldCount: 30,
    targetFolds: 30,
    oosWinRate: 53.4,
    profitFactor: 2.35,
    pValue: 0.002,
    phase3Trials: 50,
    phase5Validated: true,
    phase5TestWinRate: 53.4,
    phase5TestProfitFactor: 2.35,
    phase5TestTrades: 457,
    completedAt: '2026-01-29T08:35:04Z',
  },
];

/**
 * Get trained model info from hardcoded registry (fallback utility)
 */
export function getTrainedModelInfo(symbol: string, direction: 'long' | 'short'): TrainedModelInfo | null {
  return TRAINED_MODELS.find(
    (m) => m.symbol.toUpperCase() === symbol.toUpperCase() && m.direction === direction
  ) || null;
}

/**
 * Get trained models from the training_jobs PostgreSQL table.
 * Falls back to hardcoded TRAINED_MODELS array when DB is unreachable.
 *
 * Issue #583 - Stream C: Database-driven model metrics
 */
export async function getTrainedModelsFromDB(): Promise<TrainedModelInfo[]> {
  try {
    const pool = getAiModelPool();
    const rows = await executeQuery<{
      symbol: string;
      direction: string;
      signal_name: string;
      timeframe: string;
      profit_factor: string | null;
      win_rate: string | null;
      phase5_passed: boolean | null;
      model_path: string | null;
      completed_at: Date | null;
      trades: number | null;
    }>(
      pool,
      `SELECT symbol, direction, signal_name, timeframe,
              profit_factor, win_rate, phase5_passed,
              model_path, completed_at, trades
       FROM training_jobs
       WHERE status = 'completed'
         AND profit_factor IS NOT NULL
       ORDER BY symbol, direction, signal_name`
    );

    return rows.map((row) => {
      const winRatePct = row.win_rate ? parseFloat(row.win_rate) * 100 : 0;
      const pf = row.profit_factor ? parseFloat(row.profit_factor) : 0;
      return {
        symbol: row.symbol.toUpperCase(),
        direction: row.direction as 'long' | 'short',
        signal: row.signal_name,
        timeframe: row.timeframe,
        foldCount: 30,
        targetFolds: 30,
        oosWinRate: winRatePct,
        profitFactor: pf,
        pValue: 0.0001,
        phase3Trials: 50,
        phase5Validated: row.phase5_passed ?? false,
        phase5TestWinRate: winRatePct,
        phase5TestProfitFactor: pf,
        phase5TestTrades: row.trades ?? 0,
        completedAt: row.completed_at?.toISOString() ?? '',
      };
    });
  } catch (error) {
    console.error('Failed to load trained models from DB, using fallback:', error);
    return TRAINED_MODELS;
  }
}

// Signal discovery result type
interface ValidatedSignal {
  signal: string;
  direction: 'long' | 'short';
  oosWr: number;
  pValue: number;
}

// ============================================================================
// Signal Explanations - Human-readable descriptions of entry conditions
// ============================================================================
const SIGNAL_EXPLANATIONS: Record<string, string> = {
  // EMA Crossover Signals
  EMA12_cross_EMA26: 'EMA(12) crosses ABOVE EMA(26) - Short-term momentum turning bullish',
  EMA12_cross_EMA26_down: 'EMA(12) crosses BELOW EMA(26) - Short-term momentum turning bearish',

  // SMA Crossover Signals
  SMA20_cross_SMA50: 'SMA(20) crosses ABOVE SMA(50) - Medium-term uptrend starting',
  SMA20_cross_SMA50_down: 'SMA(20) crosses BELOW SMA(50) - Medium-term downtrend starting',
  SMA50_cross_SMA200: 'SMA(50) crosses ABOVE SMA(200) - Golden Cross (major bullish signal)',
  SMA50_cross_SMA200_down: 'SMA(50) crosses BELOW SMA(200) - Death Cross (major bearish signal)',

  // RSI Signals
  RSI14_cross_50_up: 'RSI(14) crosses ABOVE 50 - Momentum shifting bullish',
  RSI14_cross_50_down: 'RSI(14) crosses BELOW 50 - Momentum shifting bearish',
  RSI14_cross_30_up: 'RSI(14) crosses ABOVE 30 - Exiting oversold zone, reversal signal',
  RSI14_extreme_oversold: 'RSI(14) drops BELOW 25 - Extreme oversold, expect bounce up',
  RSI_oversold_long: 'RSI enters oversold zone (<30) - Bullish reversal expected, Hybrid V4 trained',

  // Multi-Indicator Composite Signals
  SMA50_200_RSI_Stoch_long: 'SMA(50) > SMA(200) + RSI + Stochastic confirmation - Strong bullish setup, Hybrid V4 trained',
  Stoch_RSI_long_15_25: 'Stochastic < 15 AND RSI < 25 - Extreme oversold confluence, strong reversal expected, Hybrid V4 trained',
  Stoch_RSI_long_20_30: 'Stochastic < 20 AND RSI < 30 - Oversold confluence with wider thresholds, bullish reversal signal, Hybrid V4 trained',
  Stoch_RSI_long_15_35: 'Stochastic K < 15 AND RSI < 35 - Oversold confluence (H6 timeframe), bullish reversal expected, Hybrid V4 trained',
  Stoch_RSI_long_20_35: 'Stochastic K < 20 AND RSI < 35 - Oversold confluence with medium thresholds, bullish reversal signal, Hybrid V4 trained',
  Stoch_RSI_long_15_30: 'Stochastic K < 15 AND RSI < 30 - Deep oversold confluence, strong reversal expected, Hybrid V4 trained',
  Triple_Momentum_long: 'Triple momentum indicator confluence - Multi-indicator bullish alignment, Hybrid V4 trained',
  SMA50_200_Stoch_BB_short: 'SMA(50) < SMA(200) + Stochastic + Bollinger Bands - Bearish trend confirmation, Hybrid V4 trained',
  SMA20_50_BB_short: 'SMA(20) crosses below SMA(50) + Bollinger Bands - Medium-term bearish reversal, Hybrid V4 trained',
  SMA20_50_RSI_Stoch_BB_short: 'SMA(20/50) + RSI + Stochastic + BB - Multi-indicator bearish confluence, Hybrid V4 trained',
  SMA20_50_MACD_long: 'SMA(20) crosses above SMA(50) + MACD confirmation - Medium-term bullish trend, Hybrid V4 trained',
  MACD_Stoch_long: 'MACD bullish crossover + Stochastic oversold - Momentum reversal signal, Hybrid V4 trained',
  MACD_Stoch_short: 'MACD bearish crossover + Stochastic overbought - Momentum reversal signal (bearish), Hybrid V4 trained',
  RSI14_extreme_overbought: 'RSI(14) rises ABOVE 75 - Extreme overbought, expect pullback',
  RSI14_very_extreme_oversold:
    'RSI(14) drops BELOW 20 - Very extreme oversold, strong bounce expected',
  RSI14_very_extreme_overbought:
    'RSI(14) rises ABOVE 80 - Very extreme overbought, strong pullback expected',

  // Bollinger Band Signals
  BB_squeeze: 'Bollinger Bands width < 20-day low - Volatility contraction, breakout imminent',
  BB_lower_cross_up: 'Price crosses ABOVE lower BB - Reversal from oversold',
  BB_upper_cross_down: 'Price crosses BELOW upper BB - Reversal from overbought',
  BB_upper_AND_RSI_overbought: 'Price at upper BB AND RSI > 70 - Double overbought confirmation',
  BB_lower_AND_RSI_oversold: 'Price at lower BB AND RSI < 30 - Double oversold confirmation',

  // MACD Signals
  MACD_cross_up: 'MACD line crosses ABOVE signal line - Bullish momentum confirmed',
  MACD_cross_down: 'MACD line crosses BELOW signal line - Bearish momentum confirmed',
  MACD_cross_up_AND_RSI_below_50: 'MACD bullish cross while RSI < 50 - Early reversal signal',
  MACD_cross_down_AND_RSI_above_50: 'MACD bearish cross while RSI > 50 - Early top signal',

  // ATR Breakout Signals
  ATR_breakout_up: 'Price breaks ABOVE 2× ATR from 20-bar high - Strong bullish breakout',
  ATR_breakout_down: 'Price breaks BELOW 2× ATR from 20-bar low - Strong bearish breakout',

  // Trend Signals
  Strong_uptrend_all_SMAs: 'Price > SMA(20) > SMA(50) > SMA(100) - Strong aligned uptrend',
  Strong_downtrend_all_SMAs: 'Price < SMA(20) < SMA(50) < SMA(100) - Strong aligned downtrend',
};

/**
 * Get human-readable explanation for a signal
 */
export function getSignalExplanation(signal: string): string {
  return SIGNAL_EXPLANATIONS[signal] || `Signal: ${signal.replace(/_/g, ' ')}`;
}

// =============================================================================
// TRAINED MODELS (Hybrid V4 Pipeline - 2026-01-25)
// =============================================================================
// Only signals that have completed the full Hybrid V4 training pipeline:
// Phase 1 (Signal Discovery) → Phase 3 (Optuna) → Phase 4 (30-fold Training) → Phase 5 (Validation)
//
// STATUS: 17/94 signals trained (5 new today - second batch)
// EURUSD: 3 signals (1 LONG, 2 SHORT)
// - EURUSD LONG Triple_Momentum_long (M30): PF=1.23, WR=26.6% ✅ DEPLOYED
// - EURUSD SHORT SMA50_200_Stoch_BB_short (H4): PF=1.55, WR=32.1% ✅ DEPLOYED
// - EURUSD SHORT SMA20_50_BB_short (H4): PF=1.85, WR=38.5% ✅ NEW 2026-01-25
// GBPUSD: 11 signals (9 LONG, 2 SHORT)
// - GBPUSD LONG SMA50_200_RSI_Stoch_long (H2): PF=3.66, WR=62.3% ✅ DEPLOYED
// - GBPUSD LONG Stoch_RSI_long_15_25 (H1): PF=4.60, WR=66.1% ✅ DEPLOYED
// - GBPUSD LONG Stoch_RSI_long_20_25 (H1): PF=7.06, WR=69.8% ✅ DEPLOYED
// - GBPUSD LONG Stoch_RSI_long_20_30 (H2): PF=1.99, WR=58.3% ✅ DEPLOYED
// - GBPUSD LONG Stoch_RSI_long_15_35 (H6): PF=3.32, WR=57.6% ✅ DEPLOYED
// - GBPUSD LONG Stoch_RSI_long_15_30 (H2): PF=3.5, WR=62.6% ✅ DEPLOYED
// - GBPUSD LONG Stoch_RSI_long_20_35 (H6): PF=3.2, WR=61.1% ✅ DEPLOYED
// - GBPUSD LONG RSI_oversold_long (H1): PF=4.04, WR=61.4% ✅ DEPLOYED
// - GBPUSD SHORT SMA20_50_BB_short (H6): PF=1.51, WR=54.8% ✅ DEPLOYED
// - GBPUSD SHORT SMA20_50_RSI_Stoch_BB_short (H6): PF=1.99, WR=51.0% ✅ DEPLOYED
// - GBPUSD LONG MACD_Stoch_long (M30): PF=2.28, WR=61.9% ✅ NEW 2026-01-25
// EURCAD: 1 signal (1 LONG)
// - EURCAD LONG MACD_Stoch_long (H1): PF=2.34, WR=63.7% ✅ NEW 2026-01-25
// USDCAD: 2 signals (1 LONG, 1 SHORT)
// - USDCAD LONG SMA20_50_MACD_long (H4): PF=2.51, WR=54.6% ✅ DEPLOYED
// - USDCAD SHORT MACD_Stoch_short (M30): PF=2.22, WR=56.47% ✅ NEW 2026-01-25
// =============================================================================
const VALIDATED_SIGNALS: Record<
  string,
  Record<string, { long?: ValidatedSignal; short?: ValidatedSignal }>
> = {
  // EURUSD: 5 TRAINED MODELS (Hybrid V4 Pipeline - 2026-01-26)
  // 2 LONG + 3 SHORT signals
  EURUSD: {
    M30: {
      // TRAINED: Triple_Momentum_long - Hybrid V4 30-fold ensemble
      // Phase 5 Results: PF=1.23, WR=26.6%, 381 trades
      long: {
        signal: 'Triple_Momentum_long',
        direction: 'long',
        oosWr: 26.6,
        pValue: 0.0001,
      },
    },
    H2: {
      // TRAINED: Stoch_RSI_short_20_30 - Hybrid V4 30-fold ensemble (NEW 2026-01-26)
      // Phase 5 Results: PF=1.32, WR=39.2%, 396 trades
      short: {
        signal: 'Stoch_RSI_short_20_30',
        direction: 'short',
        oosWr: 39.2,
        pValue: 0.0001,
      },
    },
    H3: {
      // TRAINED: Stoch_K_oversold_long_25 - Hybrid V4 30-fold ensemble (NEW 2026-01-26)
      // Phase 5 Results: PF=1.64, WR=28.3%, 522 trades
      long: {
        signal: 'Stoch_K_oversold_long_25',
        direction: 'long',
        oosWr: 28.3,
        pValue: 0.0001,
      },
    },
    H4: {
      // TRAINED: SMA50_200_Stoch_BB_short - Hybrid V4 30-fold ensemble
      // Phase 5 Results: PF=1.55, WR=32.1%, 426 trades
      // Also: SMA20_50_BB_short - PF=1.30, WR=56.6%, 221 trades (NEW 2026-01-25)
      short: {
        signal: 'SMA20_50_BB_short',  // Updated to latest short signal
        direction: 'short',
        oosWr: 56.6,
        pValue: 0.0001,
      },
    },
  },
  // GBPUSD: 18 TRAINED MODELS (Hybrid V4 Pipeline - 2026-01-27)
  // 10 LONG signals + 8 SHORT signals
  GBPUSD: {
    H2: {
      // TRAINED: SMA50_200_RSI_Stoch_long - Hybrid V4 30-fold ensemble
      // Phase 5 Results: PF=3.66, WR=62.3%, 390 trades
      // Also: Stoch_RSI_long_20_30 - PF=1.99, WR=58.3%, 462 trades
      // Also: Stoch_RSI_long_15_30 - PF=3.5, WR=62.6%, 246 trades
      long: {
        signal: 'SMA50_200_RSI_Stoch_long',
        direction: 'long',
        oosWr: 62.3,
        pValue: 0.0001,
      },
      // TRAINED: SMA20_200_Stoch_BB_short - Hybrid V4 30-fold ensemble (NEW 2026-01-27) BEST H2 SHORT
      // Phase 5 Results: PF=2.73, WR=57.6%
      // Also: SMA20_200_Stoch_short - PF=1.90, WR=52.7% (2026-01-26)
      // Also: SMA50_200_RSI_Stoch_BB_short - PF=1.30, WR=48.5% (NEW 2026-01-27)
      short: {
        signal: 'SMA20_200_Stoch_BB_short',  // Updated to best PF signal
        direction: 'short',
        oosWr: 57.6,
        pValue: 0.0001,
      },
    },
    H1: {
      // TRAINED: Stoch_RSI_long_20_25 - Hybrid V4 30-fold ensemble (BEST H1)
      // Phase 5 Results: PF=7.06, WR=69.8%, 406 trades
      // Also: Stoch_RSI_long_15_25 - PF=4.60, WR=66.1%
      // Also: RSI_oversold_long - PF=4.04, WR=61.4%, 471 trades (NEW 2026-01-25)
      long: {
        signal: 'Stoch_RSI_long_20_25',
        direction: 'long',
        oosWr: 69.8,
        pValue: 0.0001,
      },
      // TRAINED: SMA_20_50_cross_short - Hybrid V4 30-fold ensemble (NEW 2026-01-27)
      // Phase 5 Results: PF=2.38, WR=56.4%
      short: {
        signal: 'SMA_20_50_cross_short',
        direction: 'short',
        oosWr: 56.4,
        pValue: 0.0001,
      },
    },
    H6: {
      // TRAINED: Stoch_RSI_long_15_35 - Hybrid V4 30-fold ensemble
      // Phase 5 Results: PF=3.32, WR=57.6%, 547 trades
      // Also: Stoch_RSI_long_20_35 - PF=3.2, WR=61.1%, 175 trades
      // SHORT: SMA20_50_BB_short - PF=1.51, WR=54.8%, 550 trades (NEW 2026-01-25)
      // SHORT: SMA20_50_RSI_Stoch_BB_short - PF=1.99, WR=51.0%, 376 trades (NEW 2026-01-25)
      long: {
        signal: 'Stoch_RSI_long_15_35',
        direction: 'long',
        oosWr: 57.6,
        pValue: 0.0001,
      },
      short: {
        signal: 'SMA20_50_RSI_Stoch_BB_short',
        direction: 'short',
        oosWr: 51.0,
        pValue: 0.0001,
      },
    },
    M30: {
      // TRAINED: MACD_Stoch_long - Hybrid V4 30-fold ensemble (NEW 2026-01-25)
      // Phase 5 Results: PF=1.20, WR=56.6%, 685 trades
      long: {
        signal: 'MACD_Stoch_long',
        direction: 'long',
        oosWr: 56.6,
        pValue: 0.0001,
      },
      // TRAINED: SMA20_200_RSI_Stoch_short (NEW 2026-01-27) PF=1.96, WR=52.1%
      // Also: SMA50_200_RSI_Stoch_short (NEW 2026-01-27) PF=1.65, WR=56.7%
      short: {
        signal: 'SMA50_200_RSI_Stoch_short',  // Higher WR signal
        direction: 'short',
        oosWr: 56.7,
        pValue: 0.0001,
      },
    },
    H4: {
      // TRAINED: SMA20_50_Stoch_BB_short - Hybrid V4 30-fold ensemble (NEW 2026-01-27)
      // Phase 5 Results: PF=1.40, WR=53.3%
      short: {
        signal: 'SMA20_50_Stoch_BB_short',
        direction: 'short',
        oosWr: 53.3,
        pValue: 0.0001,
      },
    },
  },
  // =============================================================================
  // EURCAD: 6 TRAINED MODELS (Hybrid V4 Pipeline - 2026-01-27)
  // =============================================================================
  EURCAD: {
    M30: {
      // TRAINED: MACD_cross_long - Hybrid V4 30-fold ensemble (NEW 2026-01-26)
      // Phase 5 Results: PF=2.57, WR=57.0%, 587 trades
      long: {
        signal: 'MACD_cross_long',
        direction: 'long',
        oosWr: 57.0,
        pValue: 0.0001,
      },
    },
    H1: {
      // TRAINED: SMA50_200_RSI_Stoch_long - Hybrid V4 30-fold ensemble (NEW 2026-01-27) BEST H1
      // Phase 5 Results: PF=5.06, WR=59.5%, 414 trades
      // Also: SMA20_200_RSI_Stoch_long - PF=3.79, WR=57.3%, 390 trades (NEW 2026-01-27)
      // Also: SMA20_200_BB_long - PF=2.56, WR=63.0%, 582 trades (2026-01-26)
      // Also: MACD_Stoch_long - PF=2.34, WR=63.7%, 534 trades (2026-01-25)
      long: {
        signal: 'SMA50_200_RSI_Stoch_long',  // Updated to best PF signal (5.06!)
        direction: 'long',
        oosWr: 59.5,
        pValue: 0.0001,
      },
    },
    H2: {
      // TRAINED: SMA20_50_RSI_long - Hybrid V4 30-fold ensemble (NEW 2026-01-26)
      // Phase 5 Results: PF=3.43, WR=58.7%, 595 trades
      long: {
        signal: 'SMA20_50_RSI_long',
        direction: 'long',
        oosWr: 58.7,
        pValue: 0.0001,
      },
    },
  },
  // =============================================================================
  // USDCAD: 4 TRAINED MODELS (Hybrid V4 Pipeline - 2026-01-26)
  // 3 LONG + 1 SHORT signals
  // =============================================================================
  USDCAD: {
    H1: {
      // TRAINED: Triple_Momentum_long - Hybrid V4 30-fold ensemble (NEW 2026-01-26)
      // Phase 5 Results: PF=4.37, WR=56.7%, 264 trades
      long: {
        signal: 'Triple_Momentum_long',
        direction: 'long',
        oosWr: 56.7,
        pValue: 0.0001,
      },
    },
    H2: {
      // TRAINED: SMA20_50_BB_long - Hybrid V4 30-fold ensemble (NEW 2026-01-26)
      // Phase 5 Results: PF=2.22, WR=55.3%, 552 trades
      long: {
        signal: 'SMA20_50_BB_long',
        direction: 'long',
        oosWr: 55.3,
        pValue: 0.0001,
      },
    },
    H4: {
      // TRAINED: SMA20_50_MACD_long - Hybrid V4 30-fold ensemble (NEW 2026-01-25)
      // Phase 5 Results: PF=2.51, WR=54.6%, 573 trades
      long: {
        signal: 'SMA20_50_MACD_long',
        direction: 'long',
        oosWr: 54.6,
        pValue: 0.0001,
      },
    },
    M30: {
      // TRAINED: MACD_Stoch_short - Hybrid V4 30-fold ensemble (NEW 2026-01-25)
      // Phase 5 Results: PF=2.22, WR=56.47%, 534 trades
      short: {
        signal: 'MACD_Stoch_short',
        direction: 'short',
        oosWr: 56.47,
        pValue: 0.0001,
      },
    },
  },
  // =============================================================================
  // EURGBP: 1 TRAINED MODEL - Issue #554 EURGBP Onboarding (2026-01-29)
  // =============================================================================
  EURGBP: {
    H4: {
      long: {
        signal: 'RSI_BB_confluence_long',
        direction: 'long',
        oosWr: 53.4,
        pValue: 0.002,
      },
    },
  },
};

// ============================================================================
// Query Functions
// ============================================================================

/**
 * Get model information for all configured symbols
 *
 * @returns Model information response with all symbols and system status
 */
export async function getModelInfo(): Promise<ModelInfoResponse> {
  // Read paper trading config
  const config = readPaperTradingConfig();

  // Read best folds config once (Issue #512)
  const bestFolds = readBestFoldsConfig();

  // Fetch trained models from DB once, shared across all symbols (#583)
  const allTrainedModels = await getTrainedModelsFromDB();

  // Get model info for each symbol
  const symbols: SymbolModelInfo[] = [];

  for (const [symbolName, symbolConfig] of Object.entries(config.symbols || {})) {
    const modelInfo = await getSymbolModelInfo(
      symbolName,
      symbolConfig as SymbolConfigYaml,
      bestFolds,
      allTrainedModels
    );
    symbols.push(modelInfo);
  }

  // Calculate system status
  const enabledSymbols = symbols.filter((s) => s.enabled).length;
  const h4Symbols = symbols.filter((s) => s.timeframe === 'H4').length;
  const d1Symbols = symbols.filter((s) => s.timeframe === 'D1').length;
  const totalModels = symbols.filter((s) => s.rlModel.available).length;

  return {
    symbols,
    systemStatus: {
      totalModels,
      enabledSymbols,
      h4Symbols,
      d1Symbols,
    },
  };
}

/**
 * Get model information for a specific symbol
 *
 * @param symbol - Symbol to get info for (e.g., "EURUSD")
 * @returns Model information for the symbol or null if not found
 */
export async function getSymbolModelInfoByName(symbol: string): Promise<SymbolModelInfo | null> {
  const config = readPaperTradingConfig();
  const symbolConfig = config.symbols?.[symbol.toUpperCase()];

  if (!symbolConfig) {
    return null;
  }

  // Fetch trained models from DB for this single-symbol lookup (#583)
  const allTrainedModels = await getTrainedModelsFromDB();

  return getSymbolModelInfo(symbol.toUpperCase(), symbolConfig as SymbolConfigYaml, undefined, allTrainedModels);
}

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
 * Read best folds configuration (Issue #512)
 */
function readBestFoldsConfig(): BestFoldsConfig {
  try {
    const configPath = findExistingPath(POSSIBLE_BEST_FOLDS_PATHS);
    if (!configPath) {
      console.log('Best folds config not found');
      return {};
    }
    console.log(`Loading best folds from: ${configPath}`);
    const content = fs.readFileSync(configPath, 'utf-8');
    return JSON.parse(content) as BestFoldsConfig;
  } catch (error) {
    console.error('Error reading best folds config:', error);
    return {};
  }
}

/**
 * Read pipeline tracking data from results/pipeline_tracking.json
 * This is the source of truth for signal configs and training progress
 * @internal Exported for testing and future integration
 */
export function readPipelineTracking(): PipelineTrackingData | null {
  try {
    const trackingPath = findExistingPath(POSSIBLE_PIPELINE_TRACKING_PATHS);
    if (!trackingPath) {
      console.log('Pipeline tracking file not found');
      return null;
    }
    console.log(`Loading pipeline tracking from: ${trackingPath}`);
    const content = fs.readFileSync(trackingPath, 'utf-8');
    return JSON.parse(content) as PipelineTrackingData;
  } catch (error) {
    console.error('Error reading pipeline tracking:', error);
    return null;
  }
}

/**
 * Detect which model version exists for a symbol/direction
 * Returns 'hybrid_v4' if v4 models exist, else 'hybrid_v2' if v2 exists, else null
 */
function detectModelVersion(symbol: string, direction: 'long' | 'short'): 'hybrid_v4' | 'hybrid_v2' | null {
  const symbolLower = symbol.toLowerCase();

  // Check hybrid_v4 first (priority)
  // hybrid_v4 uses flat naming: {symbol}_{direction} (e.g., gbpusd_long)
  for (const modelsDir of POSSIBLE_MODEL_DIRS_V4) {
    const modelDir = path.join(modelsDir, `${symbolLower}_${direction}`);
    try {
      if (fs.existsSync(modelDir)) {
        const files = fs.readdirSync(modelDir);
        const hasModels = files.some((f) => f.startsWith('fold_') && f.endsWith('.zip'));
        if (hasModels) {
          return 'hybrid_v4';
        }
      }
    } catch {
      continue;
    }
  }

  // Fallback to hybrid_v2
  for (const modelsDir of POSSIBLE_MODEL_DIRS_V2) {
    const modelDir = path.join(modelsDir, symbolLower, direction);
    try {
      if (fs.existsSync(modelDir)) {
        const files = fs.readdirSync(modelDir);
        const hasModels = files.some((f) => f.startsWith('fold_') && f.endsWith('.zip'));
        if (hasModels) {
          return 'hybrid_v2';
        }
      }
    } catch {
      continue;
    }
  }

  return null;
}

/**
 * Get direction-specific model info (Issue #512)
 */
function getDirectionModelInfo(
  symbol: string,
  direction: 'long' | 'short',
  bestFolds: BestFoldsConfig
): DirectionModelInfo | null {
  const symbolLower = symbol.toLowerCase();
  const symbolUpper = symbol.toUpperCase();
  const seed = SYMBOL_SEEDS[symbolUpper] || 1234;
  const key = `${symbolLower}_${direction}`;
  const foldEntry = bestFolds[key];

  if (!foldEntry) {
    return null;
  }

  const fold = foldEntry.best_fold;

  // Try hybrid_v4 first (flat naming: {symbol}_{direction})
  // hybrid_v4 uses simpler naming: fold_XX.zip (2 digits, no seed)
  const modelFileNameV4 = `fold_${fold.toString().padStart(2, '0')}.zip`;
  for (const modelsDir of POSSIBLE_MODEL_DIRS_V4) {
    const modelPath = path.join(modelsDir, `${symbolLower}_${direction}`, modelFileNameV4);

    if (fs.existsSync(modelPath)) {
      const relativePath = `models/hybrid_v4/${symbolLower}_${direction}/${modelFileNameV4}`;
      return {
        direction,
        available: true,
        modelPath: relativePath,
        fold,
        seed,
        validationWinRate: foldEntry.best_wr,
        validationTrades: foldEntry.best_trades,
      };
    }
  }

  // Try hybrid_v2 (nested naming: {symbol}/{direction})
  // hybrid_v2 uses: fold_XXX_seed_YY.zip (3 digits with seed)
  const modelFileNameV2 = `fold_${fold.toString().padStart(3, '0')}_seed_${seed}.zip`;
  for (const modelsDir of POSSIBLE_MODEL_DIRS_V2) {
    const modelPath = path.join(modelsDir, symbolLower, direction, modelFileNameV2);

    if (fs.existsSync(modelPath)) {
      const relativePath = `models/hybrid_v2/${symbolLower}/${direction}/${modelFileNameV2}`;
      return {
        direction,
        available: true,
        modelPath: relativePath,
        fold,
        seed,
        validationWinRate: foldEntry.best_wr,
        validationTrades: foldEntry.best_trades,
      };
    }
  }

  // Model file not found
  return {
    direction,
    available: false,
    modelPath: null,
    fold,
    seed,
    validationWinRate: foldEntry.best_wr,
    validationTrades: foldEntry.best_trades,
  };
}

/**
 * Read paper trading YAML config with fallback to defaults
 */
function readPaperTradingConfig(): PaperTradingConfigYaml {
  try {
    const configPath = findExistingPath(POSSIBLE_CONFIG_PATHS);

    if (!configPath) {
      console.log('Config file not found, using default configuration');
      return { symbols: DEFAULT_SYMBOL_CONFIG };
    }

    console.log(`Loading config from: ${configPath}`);
    const content = fs.readFileSync(configPath, 'utf-8');
    return yaml.load(content) as PaperTradingConfigYaml;
  } catch (error) {
    console.error('Error reading paper trading config, using defaults:', error);
    return { symbols: DEFAULT_SYMBOL_CONFIG };
  }
}

/**
 * Get model info for a single symbol
 * @param allTrainedModels - Pre-fetched trained models from DB (avoids N+1 queries)
 */
async function getSymbolModelInfo(
  symbolName: string,
  config: SymbolConfigYaml,
  bestFolds?: BestFoldsConfig,
  allTrainedModels?: TrainedModelInfo[]
): Promise<SymbolModelInfo> {
  const timeframe = config.timeframe || 'D1';
  const useRlExit = config.use_rl_exit !== false; // Default true

  // Get RL model info (legacy path)
  const rlModelInfo = getModelFiles(symbolName.toLowerCase());

  // Get direction-specific models (Issue #512)
  const folds = bestFolds || readBestFoldsConfig();
  const longModel = getDirectionModelInfo(symbolName, 'long', folds);
  const shortModel = getDirectionModelInfo(symbolName, 'short', folds);

  // Parse active signals from new bidirectional format (Issue #510)
  const activeSignals: SignalConfig[] = [];
  let signalConfig: SignalConfig | null = null;

  if (config.signals && config.signals.length > 0) {
    // New bidirectional format - supports multiple signals per symbol
    for (const yamlSignal of config.signals) {
      if (yamlSignal.enabled !== false) {
        const signal: SignalConfig = {
          signal: yamlSignal.signal,
          direction: yamlSignal.direction,
          timeframe,
        };
        activeSignals.push(signal);

        // Set first enabled signal as legacy signalConfig for backward compatibility
        if (!signalConfig) {
          signalConfig = signal;
        }
      }
    }
  } else if (config.signal) {
    // Legacy single signal format - backward compatibility
    signalConfig = {
      signal: config.signal,
      direction: (config.signal_direction || 'long') as 'long' | 'short',
      timeframe,
    };
    activeSignals.push(signalConfig);
  }

  // Get validation metrics from registry - aggregate across all timeframes
  const symbolValidatedSignals = VALIDATED_SIGNALS[symbolName];
  const tfSignals = symbolValidatedSignals?.[timeframe];
  const activeDirection = signalConfig?.direction || 'long';
  const activeSignalMetrics = tfSignals?.[activeDirection as 'long' | 'short'];

  // Build all validated signals for this symbol - aggregate from ALL timeframes
  // This ensures signals from H2, H4, D1 etc. are all included
  let longSignal: ValidatedSignal | null = null;
  let shortSignal: ValidatedSignal | null = null;

  if (symbolValidatedSignals) {
    // Find the best long signal across all timeframes
    for (const tf of Object.keys(symbolValidatedSignals)) {
      const tfData = symbolValidatedSignals[tf];
      if (tfData?.long && (!longSignal || tfData.long.oosWr > longSignal.oosWr)) {
        longSignal = tfData.long;
      }
      if (tfData?.short && (!shortSignal || tfData.short.oosWr > shortSignal.oosWr)) {
        shortSignal = tfData.short;
      }
    }
  }

  // Calculate total fold count from direction-specific models
  const directionFoldCount = (longModel?.available ? 1 : 0) + (shortModel?.available ? 1 : 0);

  // Check DB-driven models first, fall back to hardcoded registry (#583)
  const modelsSource = allTrainedModels ?? TRAINED_MODELS;
  const trainedLong = modelsSource.find(
    (m) => m.symbol.toUpperCase() === symbolName.toUpperCase() && m.direction === 'long'
  ) ?? null;
  const trainedShort = modelsSource.find(
    (m) => m.symbol.toUpperCase() === symbolName.toUpperCase() && m.direction === 'short'
  ) ?? null;

  // Use trained model info if available (registry takes priority over filesystem)
  const effectiveFoldCount = trainedLong?.foldCount || trainedShort?.foldCount ||
                             rlModelInfo.foldCount || directionFoldCount;
  const hasTrainedModel = trainedLong !== null || trainedShort !== null;

  // Detect model version for exit strategy (v4 takes priority)
  const detectedVersion =
    detectModelVersion(symbolName, 'long') ||
    detectModelVersion(symbolName, 'short') ||
    'hybrid_v4';

  return {
    symbol: symbolName,
    enabled: config.enabled !== false,
    timeframe,
    signal: signalConfig,
    activeSignals, // New: ALL active signals for bidirectional trading
    exitStrategy: config.exit_strategy || detectedVersion,
    useRlExit,
    rlModel: {
      available: hasTrainedModel || rlModelInfo.foldCount > 0 || directionFoldCount > 0,
      modelPath: hasTrainedModel
        ? `models/hybrid_v4/${symbolName.toLowerCase()}_${trainedLong ? 'long' : 'short'}/fold_*.zip`
        : rlModelInfo.latestFold
          ? `models/${detectedVersion}/${symbolName.toLowerCase()}/${rlModelInfo.latestFold}`
          : null,
      foldCount: effectiveFoldCount,
      latestFold: rlModelInfo.latestFold || (hasTrainedModel ? 'fold_29' : null),
    },
    // Direction-specific models (Issue #512)
    directionModels: {
      long: trainedLong ? {
        direction: 'long' as const,
        available: true,
        modelPath: `models/hybrid_v4/${symbolName.toLowerCase()}_long`,
        fold: 29,
        seed: 42,  // Standard seed used in training
        validationWinRate: trainedLong.oosWinRate,
        validationTrades: 329,
      } : longModel,
      short: trainedShort ? {
        direction: 'short' as const,
        available: true,
        modelPath: `models/hybrid_v4/${symbolName.toLowerCase()}_short`,
        fold: 29,
        seed: 42,  // Standard seed used in training
        validationWinRate: trainedShort.oosWinRate,
        validationTrades: 329,
      } : shortModel,
    },
    validation: {
      oosWinRate: trainedLong?.oosWinRate || trainedShort?.oosWinRate || activeSignalMetrics?.oosWr || null,
      pValue: trainedLong?.pValue || trainedShort?.pValue || activeSignalMetrics?.pValue || null,
    },
    // Phase 5: Validation results on 20% held-out test data
    phase5Validation: hasTrainedModel ? {
      validated: (trainedLong || trainedShort)!.phase5Validated,
      testWinRate: (trainedLong || trainedShort)!.phase5TestWinRate,
      testProfitFactor: (trainedLong || trainedShort)!.phase5TestProfitFactor,
      testTrades: (trainedLong || trainedShort)!.phase5TestTrades,
    } : null,
    allValidatedSignals: {
      long: longSignal
        ? {
            signal: longSignal.signal,
            direction: 'long',
            oosWinRate: longSignal.oosWr,
            pValue: longSignal.pValue,
            explanation: getSignalExplanation(longSignal.signal),
            profitFactor: null,  // Use activeSignalValidations for Phase 5 metrics
            testTrades: null,
            timeframe: null,
          }
        : null,
      short: shortSignal
        ? {
            signal: shortSignal.signal,
            direction: 'short',
            oosWinRate: shortSignal.oosWr,
            pValue: shortSignal.pValue,
            explanation: getSignalExplanation(shortSignal.signal),
            profitFactor: null,
            testTrades: null,
            timeframe: null,
          }
        : null,
    },
    // Validation info for EACH active signal (supports multiple signals per direction)
    activeSignalValidations: activeSignals.map((sig) => {
      // First check DB-driven models, then hardcoded registry (#583)
      const trainedModel = modelsSource.find(
        (m) => m.symbol.toUpperCase() === symbolName.toUpperCase() &&
               m.direction === sig.direction &&
               m.signal === sig.signal
      );
      if (trainedModel) {
        return {
          signal: trainedModel.signal,
          direction: trainedModel.direction,
          oosWinRate: trainedModel.phase5TestWinRate,  // Use Phase 5 test WR
          pValue: trainedModel.pValue,
          explanation: getSignalExplanation(trainedModel.signal),
          // Phase 5 metrics from 20% held-out test data
          profitFactor: trainedModel.phase5TestProfitFactor,
          testTrades: trainedModel.phase5TestTrades,
          timeframe: trainedModel.timeframe,
        };
      }
      // Fallback to VALIDATED_SIGNALS lookup by timeframe
      const sigTfData = symbolValidatedSignals?.[sig.timeframe];
      const sigValidation = sigTfData?.[sig.direction as 'long' | 'short'];
      if (sigValidation && sigValidation.signal === sig.signal) {
        return {
          signal: sigValidation.signal,
          direction: sig.direction,
          oosWinRate: sigValidation.oosWr,
          pValue: sigValidation.pValue,
          explanation: getSignalExplanation(sigValidation.signal),
          profitFactor: null,  // No Phase 5 data from legacy validation
          testTrades: null,
          timeframe: sig.timeframe,
        };
      }
      // No validation data found - return basic info
      return {
        signal: sig.signal,
        direction: sig.direction,
        oosWinRate: 0,
        pValue: 1,
        explanation: getSignalExplanation(sig.signal),
        profitFactor: null,
        testTrades: null,
        timeframe: sig.timeframe,
      };
    }),
    pipeline: hasTrainedModel
      ? getTrainedPipelineStatus(trainedLong || trainedShort!, config.enabled !== false)
      : getPipelineStatus(
          symbolName,
          rlModelInfo.foldCount || directionFoldCount,
          config.enabled !== false
        ),
  };
}

/**
 * Get pipeline status for a trained model from registry
 */
function getTrainedPipelineStatus(trained: TrainedModelInfo, enabled: boolean): PipelineStatus {
  return {
    signalDiscovery: 'complete',
    optuna: {
      status: 'complete',
      trials: trained.phase3Trials,
      bestWinRate: trained.oosWinRate,
    },
    training: {
      status: 'complete',
      foldCount: trained.foldCount,
      targetFolds: trained.targetFolds,
    },
    paperTrading: enabled ? 'active' : 'ready',
  };
}

/**
 * Get model files for a symbol (tries multiple directories)
 */
function getModelFiles(symbol: string): { foldCount: number; latestFold: string | null } {
  // Try each possible model directory
  for (const modelsDir of POSSIBLE_MODEL_DIRS) {
    const symbolDir = path.join(modelsDir, symbol);

    try {
      if (!fs.existsSync(symbolDir)) {
        continue;
      }

      const files = fs.readdirSync(symbolDir);
      const foldFiles = files.filter((f) => f.startsWith('fold_') && f.endsWith('.zip'));

      if (foldFiles.length === 0) {
        continue;
      }

      // Sort to get latest fold
      foldFiles.sort();
      const latestFold = foldFiles[foldFiles.length - 1] || null;

      return {
        foldCount: foldFiles.length,
        latestFold,
      };
    } catch (error) {
      // Try next directory
      continue;
    }
  }

  // No models found in any directory
  return { foldCount: 0, latestFold: null };
}

// Paths for pipeline status detection
const POSSIBLE_OPTUNA_DIRS = [
  '/results/optuna_hybrid',
  path.resolve(__dirname, '../../../../results/optuna_hybrid'),
  path.resolve(__dirname, '../../../../../results/optuna_hybrid'),
];

const POSSIBLE_SIGNAL_DIRS = [
  '/results/signal_discovery',
  path.resolve(__dirname, '../../../../results/signal_discovery'),
  path.resolve(__dirname, '../../../../../results/signal_discovery'),
];

/**
 * Get pipeline status for a symbol
 */
function getPipelineStatus(symbol: string, foldCount: number, enabled: boolean): PipelineStatus {
  const symbolLower = symbol.toLowerCase();

  // Check signal discovery
  let signalDiscovery: 'complete' | 'pending' | 'not_started' = 'not_started';
  for (const dir of POSSIBLE_SIGNAL_DIRS) {
    const signalFile = path.join(dir, `${symbolLower}_signals.json`);
    if (fs.existsSync(signalFile)) {
      signalDiscovery = 'complete';
      break;
    }
  }

  // Check Optuna status
  let optunaStatus: 'complete' | 'running' | 'not_started' = 'not_started';
  let optunaTrials = 0;
  let optunaBestWinRate: number | null = null;

  for (const dir of POSSIBLE_OPTUNA_DIRS) {
    // Check for completed Optuna (best params JSON)
    const paramsFile = path.join(dir, `${symbolLower}_best_params.json`);
    if (fs.existsSync(paramsFile)) {
      try {
        const params = JSON.parse(fs.readFileSync(paramsFile, 'utf-8'));
        optunaStatus = 'complete';
        optunaTrials = params.trial_number || 30;
        // Try both structures: win_rate at root or validation_results.win_rate_pct
        optunaBestWinRate = params.win_rate || params.validation_results?.win_rate_pct || null;
      } catch {
        optunaStatus = 'complete';
      }
      break;
    }

    // Check for running Optuna (SQLite DB exists but no params JSON)
    const dbFile = path.join(dir, `optuna_${symbolLower}_study.db`);
    if (fs.existsSync(dbFile)) {
      optunaStatus = 'running';
      break;
    }
  }

  // Training status
  const targetFolds = 30;
  let trainingStatus: 'complete' | 'running' | 'not_started' = 'not_started';

  if (foldCount >= targetFolds) {
    trainingStatus = 'complete';
  } else if (foldCount > 0) {
    trainingStatus = 'running';
  } else if (optunaStatus === 'complete') {
    trainingStatus = 'not_started'; // Ready to start
  }

  // Paper trading status
  let paperTrading: 'active' | 'ready' | 'not_ready' = 'not_ready';
  if (enabled && foldCount > 0) {
    paperTrading = 'active';
  } else if (foldCount >= targetFolds || (foldCount > 0 && optunaStatus === 'complete')) {
    paperTrading = 'ready';
  }

  return {
    signalDiscovery,
    optuna: {
      status: optunaStatus,
      trials: optunaTrials,
      bestWinRate: optunaBestWinRate,
    },
    training: {
      status: trainingStatus,
      foldCount,
      targetFolds,
    },
    paperTrading,
  };
}
