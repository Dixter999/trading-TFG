"""
Walk-Forward Validation Module.

Provides TRUE out-of-sample validation using the validation data split (20%)
that was NEVER seen during Signal Discovery. This eliminates data leakage
and provides genuine validation of discovered signals.

Issue: #527 - True Walk-Forward Validation
Epic: hybrid-v4

Usage:
    from validation.walk_forward import validate_signal, filter_validated_signals

    # Validate a single signal
    result = validate_signal(
        signal_func=my_signal,
        signal_name="rsi_oversold",
        full_df=df,
        symbol="EURUSD",
        is_metrics=discovery_metrics,
    )

    # Filter multiple signals, keeping only those that pass
    passed_signals = filter_validated_signals(signals, df, "EURUSD")
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from data.splitter import get_validation_data
from indicator_discovery.signals import SignalDefinition
from indicator_discovery.tester import SignalTester, TradingCosts


# Pass criteria constants
MIN_PROFIT_FACTOR = 1.2  # Minimum PF for signal to pass validation
MAX_WR_DEGRADATION = 0.10  # Maximum win rate degradation (10%)
MIN_TRADES = 20  # Minimum trades for statistical validity


@dataclass
class ValidationResult:
    """
    Result of Walk-Forward Validation on out-of-sample data.

    Contains both out-of-sample (OOS) metrics from validation data
    and in-sample (IS) metrics from discovery for comparison.

    Attributes:
        signal_name: Name of the validated signal
        symbol: Trading symbol (e.g., 'EURUSD')
        profit_factor: OOS profit factor
        win_rate: OOS win rate
        trade_count: Number of trades on validation data
        net_pnl_pips: Net PnL after costs
        is_profit_factor: In-sample profit factor from discovery
        is_win_rate: In-sample win rate from discovery
        pf_degradation: IS_PF - OOS_PF
        wr_degradation: IS_WR - OOS_WR
        passes: Whether signal passed all validation criteria
        failure_reasons: List of reasons for failure (empty if passes)
    """

    signal_name: str
    symbol: str
    profit_factor: float
    win_rate: float
    trade_count: int
    net_pnl_pips: float
    is_profit_factor: float
    is_win_rate: float
    pf_degradation: float
    wr_degradation: float
    passes: bool
    failure_reasons: list[str]

    def to_dict(self) -> dict:
        """Convert ValidationResult to dictionary for serialization."""
        return {
            "signal_name": self.signal_name,
            "symbol": self.symbol,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "trade_count": self.trade_count,
            "net_pnl_pips": self.net_pnl_pips,
            "is_profit_factor": self.is_profit_factor,
            "is_win_rate": self.is_win_rate,
            "pf_degradation": self.pf_degradation,
            "wr_degradation": self.wr_degradation,
            "passes": self.passes,
            "failure_reasons": self.failure_reasons,
        }


def validate_signal(
    signal_func: Callable,
    signal_name: str,
    full_df: pd.DataFrame,
    symbol: str,
    is_metrics: dict,
    costs: TradingCosts = None,
) -> ValidationResult:
    """
    Validate a discovered signal on TRUE unseen validation data.

    Uses ONLY the validation data split (20%) that was never seen during
    signal discovery. Applies full trading costs and checks pass criteria.

    Args:
        signal_func: The signal function to validate (or SignalDefinition)
        signal_name: Name for logging/identification
        full_df: Full dataset (will be split to get validation portion)
        symbol: Symbol name (e.g., 'EURUSD')
        is_metrics: In-sample metrics from Discovery phase, must contain:
            - 'profit_factor': float
            - 'win_rate': float
            - 'trades': int (optional)
        costs: Trading costs to apply (defaults to TradingCosts())

    Returns:
        ValidationResult with pass/fail status and detailed metrics

    Pass Criteria:
        - Profit Factor >= 1.2
        - Win Rate Degradation <= 10%
        - Trade Count >= 20
    """
    if costs is None:
        costs = TradingCosts()

    # Get ONLY validation data - never seen during discovery
    val_df = get_validation_data(full_df, symbol)

    # Create signal definition if needed
    if isinstance(signal_func, SignalDefinition):
        signal_def = signal_func
    else:
        # Wrap callable in SignalDefinition
        # Default to 'long' direction if not specified
        signal_def = SignalDefinition(
            name=signal_name,
            indicator="custom",
            direction="long",
            timeframe="H4",
            condition=signal_func,
        )

    # Create tester and evaluate on validation data WITH COSTS
    tester = SignalTester()
    oos_result = tester.test_with_costs(
        df=val_df,
        signal=signal_def,
        costs=costs,
        symbol=symbol,
    )

    # Extract OOS metrics
    oos_pf = oos_result.profit_factor
    oos_wr = oos_result.win_rate
    oos_trades = oos_result.total_trades
    oos_net_pnl = oos_result.net_pnl_pips

    # Extract IS metrics
    is_pf = is_metrics.get("profit_factor", 0.0)
    is_wr = is_metrics.get("win_rate", 0.0)

    # Calculate degradation
    pf_degradation = is_pf - oos_pf
    wr_degradation = is_wr - oos_wr

    # Check pass criteria
    failure_reasons = []

    if oos_pf < MIN_PROFIT_FACTOR:
        failure_reasons.append(
            f"PF {oos_pf:.2f} < {MIN_PROFIT_FACTOR} minimum"
        )

    if wr_degradation > MAX_WR_DEGRADATION:
        failure_reasons.append(
            f"WR degradation {wr_degradation:.1%} > {MAX_WR_DEGRADATION:.0%} maximum"
        )

    if oos_trades < MIN_TRADES:
        failure_reasons.append(
            f"Trade count {oos_trades} < {MIN_TRADES} minimum"
        )

    passes = len(failure_reasons) == 0

    return ValidationResult(
        signal_name=signal_name,
        symbol=symbol,
        profit_factor=oos_pf,
        win_rate=oos_wr,
        trade_count=oos_trades,
        net_pnl_pips=oos_net_pnl,
        is_profit_factor=is_pf,
        is_win_rate=is_wr,
        pf_degradation=pf_degradation,
        wr_degradation=wr_degradation,
        passes=passes,
        failure_reasons=failure_reasons,
    )


def filter_validated_signals(
    signals: list[tuple[str, Callable, dict]],
    full_df: pd.DataFrame,
    symbol: str,
    costs: TradingCosts = None,
) -> list[ValidationResult]:
    """
    Validate all discovered signals and return only those that pass.

    This is the gate function that filters signals before they advance
    to Optuna tuning. Only signals passing walk-forward validation
    should proceed to the next phase.

    Args:
        signals: List of tuples (name, signal_func, is_metrics)
            - name: Signal name for identification
            - signal_func: Signal function or SignalDefinition
            - is_metrics: Dict with 'profit_factor', 'win_rate', 'trades'
        full_df: Full dataset (will be split for validation)
        symbol: Symbol name
        costs: Trading costs to apply (defaults to TradingCosts())

    Returns:
        List of ValidationResult for signals that PASS validation only.
        Results for failed signals are logged but not returned.

    Example:
        signals = [
            ("rsi_oversold", rsi_signal, {"profit_factor": 1.8, "win_rate": 0.60}),
            ("macd_cross", macd_signal, {"profit_factor": 1.5, "win_rate": 0.55}),
        ]
        passed = filter_validated_signals(signals, df, "EURUSD")
    """
    if not signals:
        return []

    if costs is None:
        costs = TradingCosts()

    results = []
    passed = []
    failed = []

    for name, func, is_metrics in signals:
        result = validate_signal(
            signal_func=func,
            signal_name=name,
            full_df=full_df,
            symbol=symbol,
            is_metrics=is_metrics,
            costs=costs,
        )
        results.append(result)

        if result.passes:
            passed.append(result)
            print(f"  PASS {name}: PF={result.profit_factor:.2f}, WR={result.win_rate:.1%}")
        else:
            failed.append(result)
            print(f"  FAIL {name}: {', '.join(result.failure_reasons)}")

    print(f"\nValidation Summary: {len(passed)}/{len(results)} signals passed")

    return passed
