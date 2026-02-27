"""
Suspicious Pattern Detector.

Detects unrealistic performance patterns that indicate potential lookahead bias
or data leakage. This is a critical final check for validation.

The detector checks for:
1. Perfect or near-perfect win rates (>95%)
2. Extreme Sharpe ratios (>10)
3. Zero or near-zero volatility
4. No drawdown or minimal drawdown (<1%)
5. Impossible consecutive wins (>50)
6. High correlation between actions and future prices (>0.9)

Strict mode: False positives are better than false negatives.
"""

from typing import Dict, Any, List
import numpy as np


class SuspiciousPatternDetector:
    """
    Detects suspicious performance patterns that indicate potential data issues.

    This detector is strict - it will flag potential issues even if uncertain.
    False positives are acceptable, false negatives are NOT.

    Usage:
        detector = SuspiciousPatternDetector(
            max_sharpe=10.0,
            max_win_rate=0.95,
            min_volatility=0.001
        )
        result = detector.detect({'returns': daily_returns})
        if result['is_suspicious']:
            raise ValueError(f"Suspicious patterns: {result['suspicious_patterns']}")
    """

    def __init__(
        self,
        max_sharpe: float = 10.0,
        max_win_rate: float = 0.95,
        min_volatility: float = 0.001,
        min_drawdown: float = 0.01,
        max_consecutive_wins: int = 50,
        max_action_price_correlation: float = 0.9,
    ):
        """
        Initialize detector with thresholds.

        Args:
            max_sharpe: Maximum acceptable Sharpe ratio (default: 10)
            max_win_rate: Maximum acceptable win rate (default: 0.95 = 95%)
            min_volatility: Minimum acceptable volatility (default: 0.001)
            min_drawdown: Minimum acceptable max drawdown (default: 0.01 = 1%)
            max_consecutive_wins: Maximum consecutive wins (default: 50)
            max_action_price_correlation: Maximum action-price correlation (default: 0.9)
        """
        self.max_sharpe = max_sharpe
        self.max_win_rate = max_win_rate
        self.min_volatility = min_volatility
        self.min_drawdown = min_drawdown
        self.max_consecutive_wins = max_consecutive_wins
        self.max_action_price_correlation = max_action_price_correlation

    def detect(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect suspicious patterns in returns and actions.

        Args:
            data: Dictionary containing:
                - returns: List[float] - Daily/trade returns
                - actions: Optional[List[float]] - Trading actions
                - future_prices: Optional[List[float]] - Future prices (for correlation)

        Returns:
            {
                'is_suspicious': bool,
                'suspicious_patterns': List[str],
                'sharpe_ratio': float,
                'win_rate': float,
                'return_std': float,
                'max_drawdown': float,
                'max_consecutive_wins': int,
                'action_price_correlation': float or None,
                'message': str,
            }
        """
        result = {
            "is_suspicious": False,
            "suspicious_patterns": [],
            "sharpe_ratio": None,
            "win_rate": None,
            "return_std": None,
            "max_drawdown": None,
            "max_consecutive_wins": None,
            "action_price_correlation": None,
            "message": "No suspicious patterns detected",
        }

        # Get returns (optional - correlation can work without it)
        has_returns = "returns" in data and data["returns"]

        if not has_returns:
            # Check action-price correlation even without returns
            if "actions" in data and "future_prices" in data:
                correlation = self._calculate_action_price_correlation(
                    data["actions"], data["future_prices"]
                )
                result["action_price_correlation"] = float(correlation)

                if correlation > self.max_action_price_correlation:
                    result["is_suspicious"] = True
                    result["suspicious_patterns"].append("lookahead_correlation")
                    result["message"] = f"Suspicious patterns detected: lookahead_correlation"

            return result

        # Process returns

        returns = np.array(data["returns"])

        # Calculate Sharpe ratio
        sharpe = self._calculate_sharpe(returns)
        result["sharpe_ratio"] = float(sharpe)

        if sharpe > self.max_sharpe:
            result["is_suspicious"] = True
            result["suspicious_patterns"].append("sharpe_ratio")

        # Calculate win rate
        win_rate = self._calculate_win_rate(returns)
        result["win_rate"] = float(win_rate)

        if win_rate >= self.max_win_rate:
            result["is_suspicious"] = True
            result["suspicious_patterns"].append("win_rate")

        # Calculate volatility
        return_std = float(np.std(returns))
        result["return_std"] = return_std

        if return_std < self.min_volatility:
            result["is_suspicious"] = True
            result["suspicious_patterns"].append("zero_volatility")

        # Calculate max drawdown
        max_dd = self._calculate_max_drawdown(returns)
        result["max_drawdown"] = float(max_dd)

        if max_dd < self.min_drawdown:
            result["is_suspicious"] = True
            result["suspicious_patterns"].append("no_drawdown")

        # Calculate consecutive wins
        max_consec = self._calculate_max_consecutive_wins(returns)
        result["max_consecutive_wins"] = int(max_consec)

        if max_consec >= self.max_consecutive_wins:
            result["is_suspicious"] = True
            result["suspicious_patterns"].append("consecutive_wins")

        # Calculate action-price correlation if available
        if "actions" in data and "future_prices" in data:
            correlation = self._calculate_action_price_correlation(
                data["actions"], data["future_prices"]
            )
            result["action_price_correlation"] = float(correlation)

            if correlation > self.max_action_price_correlation:
                result["is_suspicious"] = True
                result["suspicious_patterns"].append("lookahead_correlation")

        # Generate message
        if result["is_suspicious"]:
            patterns = ", ".join(result["suspicious_patterns"])
            result["message"] = f"Suspicious patterns detected: {patterns}"

        return result

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """
        Calculate Sharpe ratio (assuming risk-free rate = 0).

        Args:
            returns: Array of returns

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            # If no volatility, return extreme value
            return 999.0 if mean_return > 0 else 0.0

        # Annualized Sharpe (assuming daily returns)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return float(sharpe)

    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """
        Calculate win rate (percentage of positive returns).

        Args:
            returns: Array of returns

        Returns:
            Win rate (0.0 to 1.0)
        """
        if len(returns) == 0:
            return 0.0

        winning_trades = np.sum(returns > 0)
        total_trades = len(returns)

        return winning_trades / total_trades

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Args:
            returns: Array of returns

        Returns:
            Maximum drawdown (0.0 to 1.0)
        """
        if len(returns) == 0:
            return 0.0

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)

        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max

        # Return absolute maximum drawdown
        max_dd = abs(np.min(drawdown))
        return float(max_dd)

    def _calculate_max_consecutive_wins(self, returns: np.ndarray) -> int:
        """
        Calculate maximum consecutive winning trades.

        Args:
            returns: Array of returns

        Returns:
            Maximum consecutive wins
        """
        if len(returns) == 0:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for ret in returns:
            if ret > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_action_price_correlation(
        self, actions: List[float], future_prices: List[float]
    ) -> float:
        """
        Calculate correlation between actions and future price movements.

        High correlation indicates potential lookahead bias.

        Args:
            actions: List of trading actions
            future_prices: List of future prices

        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(actions) == 0 or len(future_prices) == 0:
            return 0.0

        # Ensure same length
        min_len = min(len(actions), len(future_prices))
        actions = np.array(actions[:min_len])
        future_prices = np.array(future_prices[:min_len])

        # Calculate correlation
        if len(actions) < 2:
            return 0.0

        correlation = np.corrcoef(actions, future_prices)[0, 1]

        # Handle NaN (happens when std = 0)
        if np.isnan(correlation):
            return 0.0

        return abs(correlation)  # Use absolute value
