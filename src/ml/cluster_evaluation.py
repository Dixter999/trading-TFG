"""
Cluster evaluation system for regime-aware pattern recognition.

This module provides performance metrics calculation and statistical significance
testing for discovered clusters. Helps identify profitable trading patterns.

Following TDD methodology:
- Stream A: Performance Metrics (Sharpe, win rate, profit factor, etc.)
- Stream B: Statistical Significance Testing (t-test, binomial, Cohen's d, bootstrap CI)
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from scipy import stats
from scipy.stats import ttest_1samp


# ============================================================================
# Stream A: Performance Metrics
# ============================================================================


@dataclass
class ClusterMetrics:
    """
    Performance metrics for a cluster.

    Attributes:
        sharpe_ratio: Risk-adjusted return (mean - rf) / std
        win_rate: Percentage of positive returns (0.0 to 1.0)
        mean_return: Average return in percentage points
        std_return: Standard deviation of returns
        profit_factor: Total profits / total losses
        sortino_ratio: Return / downside deviation (downside risk only)
        max_drawdown: Maximum peak-to-trough decline (negative value)
        skewness: Return distribution skewness
        kurtosis: Return distribution kurtosis (excess)
        sample_count: Number of samples in cluster
    """

    sharpe_ratio: float
    win_rate: float
    mean_return: float
    std_return: float
    profit_factor: float
    sortino_ratio: float
    max_drawdown: float
    skewness: float
    kurtosis: float
    sample_count: int


def calculate_cluster_metrics(
    cluster_returns: np.ndarray,
    risk_free_rate: float = 0.02,
) -> ClusterMetrics:
    """
    Calculate comprehensive performance metrics for a cluster.

    Computes risk-adjusted returns, win rates, profit factors, and distribution
    statistics to evaluate cluster profitability and quality.

    Args:
        cluster_returns: Forward returns for cluster samples (percentage points).
                        Example: [0.5, -0.2, 0.3] represents 0.5%, -0.2%, 0.3% returns.
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculation (default: 0.02 = 2%).
                       Automatically converted to daily rate by dividing by 252.

    Returns:
        ClusterMetrics containing:
        - sharpe_ratio: Risk-adjusted return
        - win_rate: Fraction of positive returns
        - mean_return: Average return
        - std_return: Standard deviation (sample std with ddof=1)
        - profit_factor: Gross profits / gross losses
        - sortino_ratio: Downside-only risk-adjusted return
        - max_drawdown: Maximum peak-to-trough decline
        - skewness: Return distribution skewness
        - kurtosis: Return distribution excess kurtosis
        - sample_count: Number of samples

    Note:
        - Uses sample standard deviation (ddof=1) for unbiased estimates
        - Adds small epsilon (1e-8) to denominators to prevent division by zero
        - Maximum drawdown is calculated from cumulative returns
    """
    # Basic statistics
    mean_ret = np.mean(cluster_returns)
    std_ret = np.std(cluster_returns, ddof=1)  # Sample std
    sample_count = len(cluster_returns)

    # Sharpe ratio: (mean_return - risk_free) / std_dev
    rf_daily = risk_free_rate / 252  # Convert annual to daily
    sharpe = (mean_ret - rf_daily) / (std_ret + 1e-8)

    # Win rate: % positive returns
    win_count = (cluster_returns > 0).sum()
    win_rate = win_count / sample_count if sample_count > 0 else 0.0

    # Profit factor: sum(wins) / sum(losses)
    wins = cluster_returns[cluster_returns > 0].sum()
    losses = np.abs(cluster_returns[cluster_returns < 0].sum())
    profit_factor = wins / (losses + 1e-8)

    # Sortino ratio: only penalize downside volatility
    downside_returns = cluster_returns[cluster_returns < 0]
    downside_std = (
        np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 1e-8
    )
    sortino = (mean_ret - rf_daily) / (downside_std + 1e-8)

    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + cluster_returns / 100)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Return distribution
    skewness = stats.skew(cluster_returns)
    kurtosis_val = stats.kurtosis(cluster_returns)

    return ClusterMetrics(
        sharpe_ratio=sharpe,
        win_rate=win_rate,
        mean_return=mean_ret,
        std_return=std_ret,
        profit_factor=profit_factor,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        skewness=skewness,
        kurtosis=kurtosis_val,
        sample_count=sample_count,
    )


# ============================================================================
# Stream B: Statistical Significance Testing
# ============================================================================


@dataclass
class SignificanceTest:
    """
    Statistical significance test results.

    Attributes:
        t_statistic: T-test statistic (positive = mean > 0)
        t_pvalue: One-tailed p-value for t-test
        binom_pvalue: One-tailed p-value for binomial test
        cohens_d: Effect size (standardized mean difference)
        ci_lower: Lower bound of bootstrap confidence interval
        ci_upper: Upper bound of bootstrap confidence interval
        is_significant: True if BOTH tests are significant (p < 0.05)

    Example:
        >>> returns = np.array([0.5, 0.6, 0.4, 0.7, 0.5])
        >>> sig = SignificanceTest(
        ...     t_statistic=8.66,
        ...     t_pvalue=0.001,
        ...     binom_pvalue=0.031,
        ...     cohens_d=5.48,
        ...     ci_lower=0.42,
        ...     ci_upper=0.58,
        ...     is_significant=True
        ... )
    """

    t_statistic: float
    t_pvalue: float
    binom_pvalue: float
    cohens_d: float
    ci_lower: float
    ci_upper: float
    is_significant: bool


def _calculate_one_tailed_t_pvalue(t_stat: float, t_pvalue: float) -> float:
    """
    Convert two-tailed t-test p-value to one-tailed.

    We only care if mean > 0 (right tail), so convert accordingly.

    Args:
        t_stat: T-statistic from t-test
        t_pvalue: Two-tailed p-value

    Returns:
        One-tailed p-value
    """
    return t_pvalue / 2 if t_stat > 0 else 1.0 - t_pvalue / 2


def _calculate_binomial_pvalue(cluster_returns: np.ndarray) -> float:
    """
    Calculate binomial test p-value for win rate > 50%.

    Args:
        cluster_returns: Array of forward returns

    Returns:
        One-tailed binomial test p-value
    """
    win_count = (cluster_returns > 0).sum()
    n_samples = len(cluster_returns)

    # Try modern scipy.stats.binomtest, fallback to deprecated binom_test
    try:
        from scipy.stats import binomtest

        binom_result = binomtest(win_count, n_samples, 0.5, alternative="greater")
        return binom_result.pvalue
    except ImportError:
        # Fallback for older scipy versions
        from scipy.stats import binom_test

        return binom_test(win_count, n_samples, 0.5, alternative="greater")


def _calculate_cohens_d(cluster_returns: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Cohen's d = mean / std (comparing to zero baseline)
    Interpretation:
    - |d| < 0.2: negligible
    - |d| < 0.5: small
    - |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        cluster_returns: Array of forward returns

    Returns:
        Cohen's d effect size
    """
    mean_ret = np.mean(cluster_returns)
    std_ret = np.std(cluster_returns, ddof=0)  # Population std for effect size
    return mean_ret / (std_ret + 1e-8)


def _bootstrap_confidence_interval(
    cluster_returns: np.ndarray,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    """
    Calculate bootstrap confidence interval for mean return.

    Uses resampling with replacement to estimate distribution of mean.

    Args:
        cluster_returns: Array of forward returns
        confidence_level: Confidence level (default 0.95 = 95%)
        n_bootstrap: Number of bootstrap samples (default 1000)

    Returns:
        Tuple of (ci_lower, ci_upper)
    """
    n_samples = len(cluster_returns)
    bootstrap_means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(cluster_returns, size=n_samples, replace=True)
        bootstrap_means[i] = np.mean(sample)

    # Calculate confidence interval percentiles
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return ci_lower, ci_upper


def test_statistical_significance(
    cluster_returns: np.ndarray,
    confidence_level: float = 0.95,
    significance_threshold: float = 0.05,
) -> SignificanceTest:
    """
    Test if cluster returns are statistically significant.

    Performs multiple statistical tests to verify that profitability
    is not due to random chance:
    - T-test: Is mean return significantly > 0?
    - Binomial test: Is win rate significantly > 50%?
    - Cohen's d: What is the effect size?
    - Bootstrap CI: What is the confidence interval?

    Overall significance requires BOTH t-test AND binomial test to pass.

    Args:
        cluster_returns: Forward returns for cluster samples
        confidence_level: Confidence level for CI (default 0.95 = 95%)
        significance_threshold: P-value threshold (default 0.05)

    Returns:
        SignificanceTest with p-values, effect sizes, and significance flag

    Raises:
        ValueError: If cluster_returns is empty

    Example:
        >>> returns = np.array([0.5, 0.6, 0.4, 0.7, 0.5, 0.3, 0.8])
        >>> sig = test_statistical_significance(returns)
        >>> print(f"Significant: {sig.is_significant}")
        >>> print(f"Effect size: {sig.cohens_d:.2f}")
        >>> print(f"95% CI: [{sig.ci_lower:.2f}, {sig.ci_upper:.2f}]")
    """
    if len(cluster_returns) == 0:
        raise ValueError("cluster_returns cannot be empty")

    # T-test: mean return significantly > 0?
    t_stat, t_pvalue_two_tailed = ttest_1samp(cluster_returns, 0)
    t_pvalue = _calculate_one_tailed_t_pvalue(t_stat, t_pvalue_two_tailed)

    # Binomial test: win rate significantly > 50%?
    binom_pvalue = _calculate_binomial_pvalue(cluster_returns)

    # Effect size (Cohen's d)
    cohens_d = _calculate_cohens_d(cluster_returns)

    # Bootstrap confidence interval
    ci_lower, ci_upper = _bootstrap_confidence_interval(
        cluster_returns, confidence_level
    )

    # Determine overall significance
    # Require BOTH t-test AND binomial test to be significant
    is_significant = (
        t_pvalue < significance_threshold and binom_pvalue < significance_threshold
    )

    return SignificanceTest(
        t_statistic=t_stat,
        t_pvalue=t_pvalue,
        binom_pvalue=binom_pvalue,
        cohens_d=cohens_d,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_significant=is_significant,
    )


# ============================================================================
# Stream C: Regime Affinity Analysis
# ============================================================================


@dataclass
class RegimeAffinity:
    """
    Regime affinity analysis results for a cluster.

    Analyzes how cluster performance varies across different market regimes
    (volatility, trend, momentum). Identifies which regimes favor which patterns.

    Attributes:
        regime_performance: Performance metrics by regime. Dict mapping regime name
                           to performance dict containing:
                           - sample_count: Number of samples in regime
                           - sharpe: Sharpe ratio in regime
                           - mean_return: Average return in regime
                           - win_rate: Win rate in regime
        best_regime: Regime with highest Sharpe ratio
        worst_regime: Regime with lowest Sharpe ratio
        regime_correlation: Correlation between returns and regime indicator.
                          Dict mapping regime name to correlation coefficient.

    Example:
        >>> affinity = RegimeAffinity(
        ...     regime_performance={
        ...         "low_vol": {"sample_count": 50, "sharpe": 1.2,
        ...                     "mean_return": 0.5, "win_rate": 0.58},
        ...         "high_vol": {"sample_count": 30, "sharpe": 0.8,
        ...                      "mean_return": 0.3, "win_rate": 0.52}
        ...     },
        ...     best_regime="low_vol",
        ...     worst_regime="high_vol",
        ...     regime_correlation={"low_vol": 0.65, "high_vol": 0.35}
        ... )
    """

    regime_performance: Dict[str, Dict[str, float]]
    best_regime: str
    worst_regime: str
    regime_correlation: Dict[str, float]


def analyze_regime_affinity(
    cluster_returns: np.ndarray,
    timestamps: "pd.DatetimeIndex",
    regime_labels: np.ndarray,
) -> RegimeAffinity:
    """
    Analyze cluster performance across different market regimes.

    Calculates performance metrics for each regime, identifies best/worst
    regimes, and computes correlation between returns and regime indicators.

    Args:
        cluster_returns: Forward returns for cluster samples (percentage points).
                        Example: [0.5, -0.2, 0.3] represents 0.5%, -0.2%, 0.3% returns.
        timestamps: Date/time index for each sample (used for alignment).
        regime_labels: Regime classification for each timestamp.
                      Example: ["low", "high", "low", "medium"]

    Returns:
        RegimeAffinity containing:
        - regime_performance: Metrics per regime
        - best_regime: Regime with highest Sharpe
        - worst_regime: Regime with lowest Sharpe
        - regime_correlation: Correlation per regime

    Note:
        - Requires minimum 2 samples per regime (allows std dev calculation with ddof=1)
        - Uses calculate_cluster_metrics() to compute regime performance
        - Correlation is point-biserial (binary regime indicator vs returns)
        - In production, 10+ samples per regime recommended for reliable statistics

    Example:
        >>> import pandas as pd
        >>> returns = np.array([0.5, 0.6, 0.3, 0.4, 0.7])
        >>> timestamps = pd.date_range("2024-01-01", periods=5, freq="D")
        >>> regimes = np.array(["low", "low", "high", "high", "low"])
        >>> affinity = analyze_regime_affinity(returns, timestamps, regimes)
        >>> print(f"Best regime: {affinity.best_regime}")
        >>> print(f"Low vol Sharpe: {affinity.regime_performance['low']['sharpe']:.2f}")
    """
    import pandas as pd

    regime_performance = {}
    unique_regimes = np.unique(regime_labels)

    # Minimum samples required for statistical validity
    # 2 samples is absolute minimum (allows std dev calculation with ddof=1)
    # In practice, 10+ samples recommended for reliable statistics
    MIN_SAMPLES_PER_REGIME = 2

    # Calculate performance for each regime
    for regime in unique_regimes:
        # Create mask for this regime
        mask = regime_labels == regime
        regime_returns = cluster_returns[mask]

        # Only include regimes with enough samples
        if len(regime_returns) >= MIN_SAMPLES_PER_REGIME:
            # Calculate metrics using Stream A function
            metrics = calculate_cluster_metrics(regime_returns)

            regime_performance[regime] = {
                "sample_count": int(mask.sum()),
                "sharpe": float(metrics.sharpe_ratio),
                "mean_return": float(metrics.mean_return),
                "win_rate": float(metrics.win_rate),
            }

    # Find best and worst regimes by Sharpe ratio
    if regime_performance:
        # Get regime with highest Sharpe
        best_regime = max(
            regime_performance.items(), key=lambda x: x[1]["sharpe"]
        )[0]

        # Get regime with lowest Sharpe
        worst_regime = min(
            regime_performance.items(), key=lambda x: x[1]["sharpe"]
        )[0]
    else:
        # No regimes met minimum sample requirement
        best_regime = ""
        worst_regime = ""

    # Calculate correlation between returns and each regime
    regime_correlation = {}
    for regime in regime_performance.keys():
        # Create binary indicator: 1 if in regime, 0 otherwise
        regime_indicator = (regime_labels == regime).astype(float)

        # Calculate point-biserial correlation
        correlation = np.corrcoef(cluster_returns, regime_indicator)[0, 1]

        # Handle NaN from correlation (constant values)
        if np.isnan(correlation):
            correlation = 0.0

        regime_correlation[regime] = float(correlation)

    return RegimeAffinity(
        regime_performance=regime_performance,
        best_regime=best_regime,
        worst_regime=worst_regime,
        regime_correlation=regime_correlation,
    )


# ============================================================================
# Stream D: Filtering, Ranking & Persistence
# ============================================================================


@dataclass
class ProfitableClusterRanking:
    """
    Ranking of profitable clusters after filtering.

    Attributes:
        profitable_clusters: List of cluster metadata dicts with:
            - cluster_id: Cluster identifier
            - sharpe: Sharpe ratio
            - win_rate: Win rate (0.0 to 1.0)
            - mean_return: Average return
            - profit_factor: Gross profit / gross loss
            - p_value: Statistical significance p-value
        total_clusters: Total number of clusters evaluated
        profitable_count: Number of clusters passing all filters
        profitability_rate: Fraction of profitable clusters (0.0 to 1.0)
    """

    profitable_clusters: List[Dict[str, float]]
    total_clusters: int
    profitable_count: int
    profitability_rate: float


@dataclass
class PersistenceAnalysis:
    """
    Out-of-sample persistence validation results.

    Attributes:
        train_sharpe: Sharpe ratio on training set
        test_sharpe: Sharpe ratio on test set
        sharpe_degradation: Fractional degradation (train - test) / train
        persistence_score: Persistence quality score (1.0 - abs(degradation))
        is_persistent: True if persistence_score > 0.7
    """

    train_sharpe: float
    test_sharpe: float
    sharpe_degradation: float
    persistence_score: float
    is_persistent: bool


def filter_and_rank_clusters(
    all_cluster_metrics: Dict[int, ClusterMetrics],
    all_significance: Dict[int, SignificanceTest],
    sharpe_threshold: float = 0.5,
    win_rate_threshold: float = 0.45,
    sample_count_threshold: int = 50,
    significance_threshold: float = 0.05,
) -> ProfitableClusterRanking:
    """
    Filter clusters by performance thresholds and rank by profitability.

    Applies multiple filters to identify genuinely profitable clusters:
    1. Sharpe ratio > sharpe_threshold (default 0.5)
    2. Win rate > win_rate_threshold (default 0.45 = 45%)
    3. Sample count >= sample_count_threshold (default 50)
    4. Statistical significance (is_significant = True)

    Clusters passing all filters are ranked by Sharpe ratio (descending).

    Args:
        all_cluster_metrics: ClusterMetrics for each cluster ID
        all_significance: SignificanceTest for each cluster ID
        sharpe_threshold: Minimum Sharpe ratio (default 0.5)
        win_rate_threshold: Minimum win rate (default 0.45)
        sample_count_threshold: Minimum sample count (default 50)
        significance_threshold: Maximum p-value (default 0.05, unused - uses is_significant)

    Returns:
        ProfitableClusterRanking with filtered and ranked clusters

    Example:
        >>> metrics = {0: ClusterMetrics(...), 1: ClusterMetrics(...)}
        >>> sig_tests = {0: SignificanceTest(...), 1: SignificanceTest(...)}
        >>> ranking = filter_and_rank_clusters(metrics, sig_tests)
        >>> print(f"Profitable: {ranking.profitable_count}/{ranking.total_clusters}")
        >>> print(f"Top cluster: {ranking.profitable_clusters[0]['cluster_id']}")
    """
    profitable_clusters = []

    for cluster_id, metrics in all_cluster_metrics.items():
        sig_test = all_significance[cluster_id]

        # Apply all filters
        if (
            metrics.sharpe_ratio > sharpe_threshold
            and metrics.win_rate > win_rate_threshold
            and metrics.sample_count >= sample_count_threshold
            and sig_test.is_significant
        ):
            profitable_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "sharpe": metrics.sharpe_ratio,
                    "win_rate": metrics.win_rate,
                    "mean_return": metrics.mean_return,
                    "profit_factor": metrics.profit_factor,
                    "p_value": sig_test.t_pvalue,
                }
            )

    # Sort by Sharpe ratio (descending)
    profitable_clusters.sort(key=lambda x: x["sharpe"], reverse=True)

    # Calculate profitability rate
    total_clusters = len(all_cluster_metrics)
    profitable_count = len(profitable_clusters)
    profitability_rate = profitable_count / total_clusters if total_clusters > 0 else 0.0

    return ProfitableClusterRanking(
        profitable_clusters=profitable_clusters,
        total_clusters=total_clusters,
        profitable_count=profitable_count,
        profitability_rate=profitability_rate,
    )


def validate_cluster_persistence(
    cluster_returns_train: np.ndarray,
    cluster_returns_test: np.ndarray,
) -> PersistenceAnalysis:
    """
    Verify cluster profitability persists out-of-sample.

    Compares training set performance to test set performance to ensure
    profitability isn't due to overfitting. A persistent cluster shows
    similar Sharpe ratios on both sets.

    Args:
        cluster_returns_train: Returns on training set
        cluster_returns_test: Returns on test set

    Returns:
        PersistenceAnalysis with degradation and persistence scores

    Note:
        - Sharpe degradation = (train_sharpe - test_sharpe) / train_sharpe
        - Persistence score = 1.0 - abs(degradation)
        - is_persistent = True if persistence_score > 0.7

    Example:
        >>> train_returns = np.array([0.5, 0.6, 0.4, 0.7])
        >>> test_returns = np.array([0.45, 0.55, 0.35, 0.65])
        >>> persistence = validate_cluster_persistence(train_returns, test_returns)
        >>> print(f"Persistent: {persistence.is_persistent}")
        >>> print(f"Degradation: {persistence.sharpe_degradation:.1%}")
    """
    # Calculate metrics for both sets
    metrics_train = calculate_cluster_metrics(cluster_returns_train)
    metrics_test = calculate_cluster_metrics(cluster_returns_test)

    # Calculate Sharpe degradation
    train_sharpe = metrics_train.sharpe_ratio
    test_sharpe = metrics_test.sharpe_ratio

    sharpe_degradation = (train_sharpe - test_sharpe) / (train_sharpe + 1e-8)

    # Calculate persistence score (1.0 = perfect persistence)
    persistence_score = 1.0 - abs(sharpe_degradation)

    # Threshold: persistence_score > 0.7 is considered persistent
    is_persistent = bool(persistence_score > 0.7)

    return PersistenceAnalysis(
        train_sharpe=train_sharpe,
        test_sharpe=test_sharpe,
        sharpe_degradation=sharpe_degradation,
        persistence_score=persistence_score,
        is_persistent=is_persistent,
    )
