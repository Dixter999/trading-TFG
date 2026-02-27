"""
RL Model Evaluator for Backtesting Trained Models.

This module implements RLEvaluator that:
- Loads trained PPO/DQN models from Stable-Baselines3
- Creates gym environment wrapper for backtesting
- Runs through RealisticBacktester with trading costs
- Generates comparison reports against baseline strategies

TDD Phase: GREEN - Minimal implementation to pass tests.

Issue: #246 (Stream B - Model Evaluation Framework)
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.base_class import BaseAlgorithm

from trading_patterns.backtest.realistic_backtester import RealisticBacktester

logger = logging.getLogger(__name__)


class RLEvaluator:
    """
    RL Model Evaluator for backtesting trained RL models.

    This class loads trained reinforcement learning models (PPO/DQN)
    and evaluates their performance using RealisticBacktester with
    trading costs (commission, spread, slippage).

    Attributes:
        initial_capital: Starting capital for backtesting
        commission_pct: Commission percentage (default 0.05%)
        spread_pips: Spread in pips (default 1.0)
        slippage_pips: Slippage in pips (default 0.5)

    Example:
        >>> evaluator = RLEvaluator(initial_capital=100000)
        >>> model = evaluator.load_model("models/ppo_h4.zip", "ppo")
        >>> results = evaluator.run_backtest(model, data, "H4")
        >>> comparison = evaluator.compare_with_baseline(results['metrics'], baseline_metrics)
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_pct: float = 0.05,
        spread_pips: float = 1.0,
        slippage_pips: float = 0.5,
        risk_percentage: float = 2.0,
        atr_multiplier: float = 2.0
    ):
        """
        Initialize RL Evaluator.

        Args:
            initial_capital: Starting capital for backtesting (default 100,000)
            commission_pct: Commission percentage (default 0.05%)
            spread_pips: Spread in pips (default 1.0)
            slippage_pips: Slippage in pips (default 0.5)
            risk_percentage: Risk percentage per trade (default 2.0%)
            atr_multiplier: ATR multiplier for stop loss (default 2.0)
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.risk_percentage = risk_percentage
        self.atr_multiplier = atr_multiplier

        logger.info(
            f"RLEvaluator initialized: capital=${initial_capital}, "
            f"commission={commission_pct}%, spread={spread_pips} pips, "
            f"slippage={slippage_pips} pips"
        )

    def load_model(self, model_path: str, model_type: str) -> BaseAlgorithm:
        """
        Load a trained RL model from disk.

        Args:
            model_path: Path to the saved model (.zip file)
            model_type: Type of model ('ppo' or 'dqn')

        Returns:
            Loaded Stable-Baselines3 model

        Raises:
            ValueError: If model_type is not 'ppo' or 'dqn'
            FileNotFoundError: If model_path does not exist

        Example:
            >>> evaluator = RLEvaluator()
            >>> model = evaluator.load_model("models/ppo_h4.zip", "ppo")
        """
        # Validate model type
        model_type = model_type.lower()
        if model_type not in ['ppo', 'dqn']:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported types: 'ppo', 'dqn'"
            )

        # Check if file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model based on type
        if model_type == 'ppo':
            model = PPO.load(model_path)
            logger.info(f"Loaded PPO model from {model_path}")
        elif model_type == 'dqn':
            model = DQN.load(model_path)
            logger.info(f"Loaded DQN model from {model_path}")

        return model

    def _create_evaluation_env(self, data: pd.DataFrame, timeframe: str):
        """
        Create a gym environment for evaluation.

        This is a placeholder for creating a gym environment.
        In practice, this would create a PatternAwareEnv or MTFTradingEnv.

        Args:
            data: DataFrame with OHLCV and indicator data
            timeframe: Timeframe string ('H1', 'H4', 'D1')

        Returns:
            Gym environment instance
        """
        # Placeholder: In real implementation, this would create PatternAwareEnv
        # For now, just return a mock object that has the required attributes
        class MockEnv:
            def __init__(self):
                self.observation_space = None
                self.action_space = None

            def reset(self):
                return np.zeros(26), {}

            def step(self, action):
                return np.zeros(26), 0.0, False, False, {}

        return MockEnv()

    def run_backtest(
        self,
        model: BaseAlgorithm,
        data: pd.DataFrame,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Run backtest with trained RL model.

        This method:
        1. Creates a gym environment from the data
        2. Runs the model through the environment
        3. Tracks trades and applies trading costs via RealisticBacktester
        4. Returns performance metrics and trade list

        Args:
            model: Trained Stable-Baselines3 model
            data: DataFrame with OHLCV, indicators, and pattern detection
            timeframe: Timeframe string ('H1', 'H4', 'D1')

        Returns:
            Dictionary with:
                - 'metrics': Performance metrics (return, Sharpe, drawdown, etc.)
                - 'trades': List of executed trades

        Example:
            >>> results = evaluator.run_backtest(model, data, "H4")
            >>> print(results['metrics']['total_return_pct'])
        """
        logger.info(f"Starting backtest on {timeframe} with {len(data)} candles")

        # Create backtester with trading costs
        backtester = RealisticBacktester(
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            spread_pips=self.spread_pips,
            slippage_pips=self.slippage_pips,
            risk_percentage=self.risk_percentage,
            atr_multiplier=self.atr_multiplier
        )

        # Create environment for model inference
        env = self._create_evaluation_env(data, timeframe)

        # Run through all data
        for idx in range(len(data)):
            row = data.iloc[idx]

            # Get observation (simplified - in real version would use PatternAwareObservationSpace)
            observation = np.zeros(26)  # Placeholder

            # Get model prediction
            action, _ = model.predict(observation, deterministic=True)

            # Convert action to trading decision
            # Actions: 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
            current_price = row['close']
            timestamp = str(row['timestamp']) if 'timestamp' in row else f"idx_{idx}"

            # Get current data for position sizing
            current_data = {
                'atr_14': row.get('atr_14', 0.0002)
            }

            # Execute action
            if action == 1 and backtester.position == 0:
                # Enter LONG
                backtester.enter_long(
                    current_price,
                    timestamp,
                    "RL Model Signal",
                    current_data=current_data
                )
            elif action == 2 and backtester.position == 0:
                # Enter SHORT
                backtester.enter_short(
                    current_price,
                    timestamp,
                    "RL Model Signal",
                    current_data=current_data
                )
            elif action == 3 and backtester.position != 0:
                # Close position
                backtester.exit_position(
                    current_price,
                    timestamp,
                    "RL Model Exit"
                )

        # Close any open position at the end
        if backtester.position != 0:
            final_row = data.iloc[-1]
            backtester.exit_position(
                final_row['close'],
                str(final_row.get('timestamp', 'end')),
                "Backtest End"
            )

        # Get metrics from backtester
        metrics = backtester.get_metrics()
        trades = backtester.trades

        logger.info(
            f"Backtest complete: {metrics['total_trades']} trades, "
            f"{metrics['total_return']:.2f}% return"
        )

        return {
            'metrics': metrics,
            'trades': trades
        }

    def compare_with_baseline(
        self,
        rl_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Compare RL model performance with baseline strategy.

        Args:
            rl_metrics: Performance metrics from RL model backtest
            baseline_metrics: Performance metrics from baseline strategy

        Returns:
            Dictionary with:
                - 'rl_metrics': RL model metrics
                - 'baseline_metrics': Baseline strategy metrics
                - 'improvements': Improvement calculations
                - 'is_better_than_baseline': Boolean indicating if RL beats baseline

        Example:
            >>> comparison = evaluator.compare_with_baseline(rl_metrics, baseline_metrics)
            >>> print(comparison['improvements']['return_improvement_pct'])
        """
        # Calculate improvements
        return_improvement = (
            rl_metrics['total_return'] - baseline_metrics['total_return']
        )
        sharpe_improvement = (
            rl_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']
        )
        drawdown_improvement = (
            baseline_metrics['max_drawdown'] - rl_metrics['max_drawdown']
        )  # Positive means RL has lower drawdown (better)

        improvements = {
            'return_improvement': return_improvement,
            'sharpe_improvement': sharpe_improvement,
            'drawdown_improvement': drawdown_improvement
        }

        # Determine if RL is better (higher return is primary metric)
        is_better = rl_metrics['total_return'] > baseline_metrics['total_return']

        comparison = {
            'rl_metrics': rl_metrics,
            'baseline_metrics': baseline_metrics,
            'improvements': improvements,
            'is_better_than_baseline': is_better
        }

        logger.info(
            f"Comparison complete: RL return={rl_metrics['total_return']:.2f}%, "
            f"Baseline return={baseline_metrics['total_return']:.2f}%, "
            f"Improvement={return_improvement:+.2f}%"
        )

        return comparison

    def generate_report(
        self,
        comparison: Dict[str, Any],
        model_path: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Generate evaluation report.

        Args:
            comparison: Comparison dictionary from compare_with_baseline()
            model_path: Path to the evaluated model
            timeframe: Timeframe used for evaluation

        Returns:
            Report dictionary with all evaluation details

        Example:
            >>> report = evaluator.generate_report(comparison, "models/ppo.zip", "H4")
        """
        report = {
            'model_path': model_path,
            'timeframe': timeframe,
            'evaluation_date': datetime.now().isoformat(),
            'comparison': comparison
        }

        logger.info(f"Report generated for {model_path} on {timeframe}")

        return report

    def export_to_csv(self, results: Dict[str, Any], csv_path: str):
        """
        Export backtest results to CSV file.

        Args:
            results: Results dictionary from run_backtest()
            csv_path: Path to save CSV file

        Example:
            >>> evaluator.export_to_csv(results, "results/rl_backtest.csv")
        """
        # Extract metrics
        metrics = results['metrics']

        # Create DataFrame with metrics
        df = pd.DataFrame([metrics])

        # Save to CSV
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

        logger.info(f"Results exported to {csv_path}")

    def export_comparison_to_json(self, comparison: Dict[str, Any], json_path: str):
        """
        Export comparison report to JSON file.

        Args:
            comparison: Comparison dictionary from compare_with_baseline()
            json_path: Path to save JSON file

        Example:
            >>> evaluator.export_comparison_to_json(comparison, "results/comparison.json")
        """
        # Create directory if it doesn't exist
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON
        with open(json_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Comparison exported to {json_path}")
