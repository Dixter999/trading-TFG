"""
Continuous learning pipeline for pattern evolution.

This module implements automated continuous learning to evolve pattern templates
over time through:
- Weekly re-clustering with new market data (Stream A)
- Pattern pruning of underperformers (Stream B)
- Regime shift detection (Stream B)
- Performance monitoring and dashboards (Stream C)

Following TDD methodology:
- Stream A: Re-Clustering Pipeline & Pattern Discovery
- Stream B: Pattern Pruning & Regime Detection
- Stream C: Performance Monitoring & Dashboard

Issue: #291
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)

# Stream A imports
from database.connection_manager import DatabaseManager
from ml.data_preparation import (
    load_historical_data,
    extract_rolling_windows,
    normalize_window,
    generate_forward_labels,
)
from ml.clustering import cluster_kmeans
from ml.cluster_evaluation import (
    calculate_cluster_metrics,
    test_statistical_significance,
    analyze_regime_affinity,
)
from ml.template_extraction import (
    FuzzyPattern,
    extract_cluster_centers,
    calculate_fuzzy_boundaries,
    create_template_from_cluster,
    calculate_fuzzy_similarity,
)


# ============================================================================
# Stream C: Data Structures for Performance Monitoring & Dashboard
# ============================================================================


@dataclass
class PatternMonitoring:
    """
    Performance monitoring metrics for a single pattern.

    Attributes:
        name: Pattern identifier
        sharpe_rolling: 4-week rolling Sharpe ratio
        win_rate_rolling: 4-week rolling win rate
        matches_this_week: Number of matches in current week
        drawdown_current: Current drawdown from peak
        trend: Performance trend ("improving", "stable", "degrading")
    """

    name: str
    sharpe_rolling: float
    win_rate_rolling: float
    matches_this_week: int
    drawdown_current: float
    trend: str  # "improving", "stable", "degrading"


@dataclass
class PerformanceMonitoring:
    """
    Complete performance monitoring snapshot for all patterns.

    Attributes:
        timestamp: Monitoring timestamp
        patterns: Dict mapping pattern name to PatternMonitoring
        alerts: List of active alerts (sharpe drops, win rate drops, no matches)
    """

    timestamp: datetime
    patterns: Dict[str, PatternMonitoring]
    alerts: List[Dict[str, Any]]


@dataclass
class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard for pattern library evolution.

    Attributes:
        total_patterns: Total patterns in library
        patterns_by_performance: Pattern names sorted by Sharpe (descending)
        patterns_by_family: Dict mapping family to pattern list
        patterns_by_regime: Dict mapping regime to pattern list
        active_alerts: List of active alerts
        regime_shifts: List of detected regime shifts
        performance_trends: Dict mapping pattern name to trend classification
    """

    total_patterns: int
    patterns_by_performance: List[str]
    patterns_by_family: Dict[str, List[str]]
    patterns_by_regime: Dict[str, List[str]]
    active_alerts: List[Dict[str, Any]]
    regime_shifts: List["RegimeShift"]
    performance_trends: Dict[str, str]


# ============================================================================
# Stream B: Data Structures for Regime Shift Detection
# ============================================================================


@dataclass
class RegimeShift:
    """
    Detected regime shift with metadata.

    Attributes:
        type: Shift type ("volatility", "trend", "correlation_breakdown")
        magnitude: Change magnitude (e.g., 1.5 = 150% increase)
        from_regime: Previous regime classification
        to_regime: New regime classification
        timestamp: Detection timestamp (auto-generated)
    """

    type: str  # "volatility", "trend", "correlation_breakdown"
    magnitude: float
    from_regime: str
    to_regime: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RegimeShiftAnalysis:
    """
    Analysis result from regime shift detection.

    Attributes:
        shifts_detected: List of detected regime shifts
        is_significant: Whether any shifts exceed threshold
        recommendation: Suggested action based on shift severity
    """

    shifts_detected: List[RegimeShift]
    is_significant: bool
    recommendation: str


# ============================================================================
# Stream B: Configuration
# ============================================================================

PRUNING_CONFIG = {
    "performance_threshold": 0.0,  # Sharpe < 0.0 = remove
    "evaluation_window": 4,  # Weeks of data
    "min_trades_to_evaluate": 20,  # Need 20+ matches to prune
    "grace_period": 2,  # New patterns: 2 weeks before evaluation
}

REGIME_SHIFT_CONFIG = {
    "detection_method": "statistical",
    "comparison_windows": [4, 8, 16],  # Compare 4/8/16-week periods
    "shift_threshold": 0.5,  # 50% change in key metrics
    "metrics_to_monitor": [
        "volatility",
        "trend_strength",
        "pattern_performance",
        "correlation",
    ],
}

# ============================================================================
# Stream C: Configuration
# ============================================================================

MONITORING_CONFIG = {
    "metrics_to_track": {
        "sharpe_rolling": {"window": 4},  # 4-week rolling Sharpe
        "win_rate_rolling": {"window": 4},  # 4-week rolling win rate
        "matches_per_week": {},
        "drawdown_current": {},
        "correlation_with_market": {},
    },
    "alert_thresholds": {
        "sharpe_drop": 0.5,  # Alert if Sharpe drops below 0.5
        "win_rate_drop": 0.10,  # Alert if win rate drops 10%+
        "no_matches": 7,  # Alert if no matches for 7 days
    },
    "trend_classification": {
        "improving_threshold": 1.0,  # Sharpe > 1.0 = improving
        "degrading_threshold": 0.5,  # Sharpe < 0.5 = degrading
        "stable_range": (0.5, 1.0),  # Between = stable
    },
}


# ============================================================================
# Stream B: Pattern Pruning Functions
# ============================================================================


def prune_underperforming_patterns(
    library: Dict[str, Any],
    evaluation_window_weeks: int,
    performance_threshold: float,
    min_trades_to_evaluate: int,
    grace_period_weeks: int,
) -> Dict[str, Any]:
    """
    Remove underperforming patterns from library.

    Prunes patterns that meet ALL criteria:
    1. Sharpe ratio < performance_threshold
    2. At least min_trades_to_evaluate matches
    3. Age > grace_period_weeks (protection for new patterns)

    Args:
        library: FuzzyPatternLibrary with performance tracking
        evaluation_window_weeks: Weeks of data to evaluate (default: 4)
        performance_threshold: Minimum Sharpe to keep (default: 0.0)
        min_trades_to_evaluate: Minimum matches required (default: 20)
        grace_period_weeks: Weeks before new patterns can be pruned (default: 2)

    Returns:
        Dict with:
            - pruned_count: Number of patterns removed
            - pruned_patterns: List of pruned pattern names
            - reason: Dict mapping pattern name to pruning reason

    Example:
        >>> library = load_fuzzy_pattern_library()
        >>> result = prune_underperforming_patterns(
        ...     library=library,
        ...     evaluation_window_weeks=4,
        ...     performance_threshold=0.0,
        ...     min_trades_to_evaluate=20,
        ...     grace_period_weeks=2,
        ... )
        >>> print(f"Pruned {result['pruned_count']} patterns")
        Pruned 3 patterns
    """
    pruned_patterns = []
    prune_reasons = {}
    now = datetime.now(timezone.utc)

    patterns = library.get("patterns", {})
    logger.info(
        f"Starting pattern pruning evaluation: {len(patterns)} patterns, "
        f"grace_period={grace_period_weeks}w, min_trades={min_trades_to_evaluate}, "
        f"threshold={performance_threshold}"
    )

    for pattern_name, pattern_data in patterns.items():
        # Extract pattern and performance data
        pattern = pattern_data.get("pattern")
        if pattern is None:
            continue

        performance_history = pattern_data.get("performance_history", {})

        # 1. Check grace period (protect new patterns)
        discovery_date_str = pattern.discovery_date
        discovery_date = datetime.fromisoformat(discovery_date_str)
        age_days = (now - discovery_date).days
        age_weeks = age_days / 7.0

        if age_weeks <= grace_period_weeks:
            # Pattern is too new, skip evaluation
            logger.debug(
                f"Pattern {pattern_name} protected by grace period "
                f"({age_weeks:.1f}w old < {grace_period_weeks}w threshold)"
            )
            continue

        # 2. Check minimum trades requirement
        matches = performance_history.get("matches", 0)
        if matches < min_trades_to_evaluate:
            # Not enough data to evaluate
            logger.debug(
                f"Pattern {pattern_name} has insufficient matches "
                f"({matches} < {min_trades_to_evaluate})"
            )
            continue

        # 3. Check performance threshold
        sharpe_rolling = performance_history.get("sharpe_rolling", 0.0)
        if sharpe_rolling < performance_threshold:
            # Underperforming - prune this pattern
            logger.warning(
                f"Pruning pattern {pattern_name}: Sharpe {sharpe_rolling:.2f} < {performance_threshold:.2f} "
                f"({matches} matches over {age_weeks:.1f} weeks)"
            )
            pruned_patterns.append(pattern_name)
            prune_reasons[pattern_name] = (
                f"Sharpe {sharpe_rolling:.2f} < {performance_threshold:.2f} "
                f"({matches} matches over {age_weeks:.1f} weeks)"
            )

    logger.info(
        f"Pruning complete: {len(pruned_patterns)} patterns pruned, "
        f"{len(patterns) - len(pruned_patterns)} patterns retained"
    )

    return {
        "pruned_count": len(pruned_patterns),
        "pruned_patterns": pruned_patterns,
        "reason": prune_reasons,
    }


# ============================================================================
# Stream B: Regime Shift Detection Functions
# ============================================================================


def detect_regime_shifts(
    monitoring_data: Dict[str, Any], shift_threshold: float
) -> RegimeShiftAnalysis:
    """
    Detect significant regime shifts in market conditions.

    Compares recent period (4 weeks) vs historical period (8 weeks) to detect:
    - Volatility regime changes (low → medium → high)
    - Trend reversals (bullish ↔ bearish)
    - Correlation breakdown (pattern relationships weakening)

    A shift is significant if the change magnitude exceeds shift_threshold.

    Args:
        monitoring_data: Dict with "recent_period" and "historical_period" metrics
        shift_threshold: Minimum change to trigger alert (0.5 = 50%)

    Returns:
        RegimeShiftAnalysis with detected shifts and recommendations

    Example:
        >>> data = {
        ...     "recent_period": {"volatility": 0.020, "trend_strength": 45.0, "correlation": 0.85},
        ...     "historical_period": {"volatility": 0.008, "trend_strength": 42.0, "correlation": 0.82},
        ... }
        >>> analysis = detect_regime_shifts(data, shift_threshold=0.5)
        >>> if analysis.is_significant:
        ...     print(f"Detected {len(analysis.shifts_detected)} regime shifts")
        Detected 1 regime shifts
    """
    shifts_detected = []
    recent = monitoring_data.get("recent_period", {})
    historical = monitoring_data.get("historical_period", {})

    logger.info("Starting regime shift detection analysis")

    # 1. Detect volatility regime shifts
    recent_vol = recent.get("volatility", 0.0)
    historical_vol = historical.get("volatility", 0.0)

    if historical_vol > 0:
        vol_change = (recent_vol - historical_vol) / historical_vol
        if abs(vol_change) > shift_threshold:
            # Classify regimes
            from_regime = classify_volatility(historical_vol)
            to_regime = classify_volatility(recent_vol)

            if from_regime != to_regime:
                logger.warning(
                    f"Volatility regime shift detected: {from_regime} → {to_regime} "
                    f"(magnitude: {abs(vol_change):.1%})"
                )
                shifts_detected.append(
                    RegimeShift(
                        type="volatility",
                        magnitude=abs(vol_change),
                        from_regime=from_regime,
                        to_regime=to_regime,
                    )
                )

    # 2. Detect trend reversals
    recent_trend = recent.get("trend_strength", 0.0)
    historical_trend = historical.get("trend_strength", 0.0)

    # Check for sign reversal (bullish ↔ bearish)
    if (recent_trend * historical_trend) < 0:  # Opposite signs
        # Trend reversal detected
        from_regime_trend = classify_trend(historical_trend)
        to_regime_trend = classify_trend(recent_trend)

        # Calculate magnitude as absolute change
        trend_magnitude = abs(recent_trend - historical_trend) / max(
            abs(historical_trend), 1.0
        )

        logger.warning(
            f"Trend reversal detected: {from_regime_trend} → {to_regime_trend} "
            f"(magnitude: {trend_magnitude:.1%})"
        )
        shifts_detected.append(
            RegimeShift(
                type="trend",
                magnitude=trend_magnitude,
                from_regime=from_regime_trend,
                to_regime=to_regime_trend,
            )
        )

    # 3. Detect correlation breakdown
    recent_corr = recent.get("correlation", 0.0)
    historical_corr = historical.get("correlation", 0.0)

    if historical_corr > 0:
        corr_change = (recent_corr - historical_corr) / historical_corr
        if corr_change < -shift_threshold:  # Negative change (breakdown)
            logger.warning(
                f"Correlation breakdown detected: {historical_corr:.2f} → {recent_corr:.2f} "
                f"(drop: {abs(corr_change):.1%})"
            )
            shifts_detected.append(
                RegimeShift(
                    type="correlation_breakdown",
                    magnitude=abs(corr_change),
                    from_regime=f"corr_{historical_corr:.2f}",
                    to_regime=f"corr_{recent_corr:.2f}",
                )
            )

    # Determine if shifts are significant
    is_significant = len(shifts_detected) > 0

    if is_significant:
        logger.info(f"Regime shift analysis complete: {len(shifts_detected)} shifts detected")
    else:
        logger.info("Regime shift analysis complete: No significant shifts detected")

    # Generate recommendation based on shift severity
    if not is_significant:
        recommendation = "No significant regime shifts detected. Continue monitoring."
    else:
        # Count shift types
        shift_types = [s.type for s in shifts_detected]
        severity = len(shifts_detected)

        if severity >= 2:
            recommendation = (
                f"CRITICAL: Multiple regime shifts detected ({severity} shifts). "
                "Consider re-clustering patterns immediately to adapt to new market conditions."
            )
        else:
            shift_type = shifts_detected[0].type
            if shift_type == "volatility":
                recommendation = (
                    f"Volatility regime changed from {shifts_detected[0].from_regime} to "
                    f"{shifts_detected[0].to_regime}. Monitor pattern performance closely."
                )
            elif shift_type == "trend":
                recommendation = (
                    f"Trend reversal detected: {shifts_detected[0].from_regime} → "
                    f"{shifts_detected[0].to_regime}. Review pattern regime affinities."
                )
            else:  # correlation_breakdown
                recommendation = (
                    "Correlation breakdown detected. Pattern relationships weakening. "
                    "Consider re-evaluating pattern library."
                )

    return RegimeShiftAnalysis(
        shifts_detected=shifts_detected,
        is_significant=is_significant,
        recommendation=recommendation,
    )


# ============================================================================
# Stream B: Helper Classifier Functions
# ============================================================================


def classify_volatility(volatility: float) -> str:
    """
    Classify volatility into regime categories.

    Args:
        volatility: ATR-based volatility metric

    Returns:
        "low", "medium", or "high"

    Example:
        >>> classify_volatility(0.008)
        'low'
        >>> classify_volatility(0.015)
        'medium'
        >>> classify_volatility(0.025)
        'high'
    """
    # Thresholds based on typical forex ATR values
    # Low: < 0.010 (100 pips)
    # Medium: 0.010 - 0.020
    # High: > 0.020
    if volatility < 0.010:
        return "low"
    elif volatility < 0.020:
        return "medium"
    else:
        return "high"


def classify_trend(trend_strength: float) -> str:
    """
    Classify trend strength into regime categories.

    Args:
        trend_strength: ADX-based trend metric (positive=bullish, negative=bearish)

    Returns:
        "strong_bullish", "bullish", "ranging", "bearish", or "strong_bearish"

    Example:
        >>> classify_trend(55.0)
        'strong_bullish'
        >>> classify_trend(35.0)
        'bullish'
        >>> classify_trend(0.0)
        'ranging'
        >>> classify_trend(-40.0)
        'bearish'
        >>> classify_trend(-55.0)
        'strong_bearish'
    """
    # Thresholds based on ADX strength
    # Ranging: -20 to +20
    # Bullish/Bearish: 20 to 45
    # Strong Bullish/Bearish: > 45
    if trend_strength > 45.0:
        return "strong_bullish"
    elif trend_strength > 20.0:
        return "bullish"
    elif trend_strength > -20.0:
        return "ranging"
    elif trend_strength > -45.0:
        return "bearish"
    else:
        return "strong_bearish"


# ============================================================================
# Stream A: Data Structures for Re-Clustering Pipeline
# ============================================================================


@dataclass
class WeeklyResults:
    """
    Results from weekly re-clustering pipeline execution.

    Attributes:
        new_patterns_added: Number of new patterns added to library
        patterns_pruned: Number of patterns removed (from Stream B)
        total_patterns: Total patterns in library after update
        new_patterns: List of FuzzyPattern instances added
        pruned_patterns: List of pattern names removed
        monitoring: Performance monitoring data (from Stream C)
    """

    new_patterns_added: int
    patterns_pruned: int
    total_patterns: int
    new_patterns: List[FuzzyPattern]
    pruned_patterns: List[str]
    monitoring: Optional[Any] = None  # Will be defined by Stream C


# ============================================================================
# Stream A: Re-Clustering Pipeline Functions
# ============================================================================


def weekly_reclustering_pipeline(
    symbol: str,
    timeframe: str,
    data_window_days: int = 90,
    clustering_k: int = 50,
    pattern_library: Any = None,
    sharpe_threshold: float = 0.5,
    pvalue_threshold: float = 0.05,
    similarity_threshold: float = 0.75,
) -> WeeklyResults:
    """
    Execute weekly re-clustering pipeline to discover new patterns.

    This function orchestrates the complete weekly pattern discovery workflow:
    1. Load last N days of market data from database
    2. Extract and normalize rolling windows
    3. Run K-Means clustering (k=50 default)
    4. Evaluate cluster profitability and statistical significance
    5. Compare to existing patterns in library
    6. Add high-confidence new patterns (Sharpe > 0.5, p < 0.05)
    7. Return summary of changes

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: Timeframe code (e.g., 'H1')
        data_window_days: Days of historical data to analyze (default: 90)
        clustering_k: Number of clusters for K-Means (default: 50)
        pattern_library: FuzzyPatternLibrary instance
        sharpe_threshold: Minimum Sharpe ratio to add pattern (default: 0.5)
        pvalue_threshold: Maximum p-value for significance (default: 0.05)
        similarity_threshold: Threshold for pattern similarity matching (default: 0.75)

    Returns:
        WeeklyResults with summary of patterns added/pruned

    Example:
        >>> from pattern_system.tracker import FuzzyPatternLibrary
        >>> library = FuzzyPatternLibrary()
        >>> result = weekly_reclustering_pipeline(
        ...     symbol='EURUSD',
        ...     timeframe='H1',
        ...     data_window_days=90,
        ...     clustering_k=50,
        ...     pattern_library=library,
        ... )
        >>> print(f"Added {result.new_patterns_added} new patterns")
    """
    # Initialize tracking
    new_patterns = []

    # Step 1: Load historical data
    with DatabaseManager() as db:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=data_window_days)

        df = load_historical_data(
            db_manager=db,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            min_candles=100,  # Minimum for clustering
        )

    # Step 2: Extract rolling windows
    windows_list = extract_rolling_windows(
        prices=df,
        window_sizes=[20],  # Fixed window size for pattern matching
        overlap=0.75,
    )

    # Step 3: Normalize windows
    normalized_windows = []
    for window in windows_list:
        # Convert window array to DataFrame for normalize_window
        window_df = pd.DataFrame(window, columns=['open', 'high', 'low', 'close'])
        normalized = normalize_window(window_df)
        normalized_windows.append(normalized)

    # Stack into 3D array: (n_samples, window_size, n_features)
    normalized_array = np.array(normalized_windows)

    # Step 4: Generate forward labels for performance evaluation
    forward_labels = generate_forward_labels(
        prices=df,
        horizons=[5],  # 5-candle forward return
        threshold=0.5,
    )

    # Step 5: Run K-Means clustering
    kmeans_result = cluster_kmeans(
        windows=normalized_array,
        n_clusters=clustering_k,
        random_state=42,
    )

    # Step 6: Evaluate each cluster for profitability
    cluster_centers_dict = extract_cluster_centers(
        windows=normalized_array,
        cluster_labels=kmeans_result.labels,
        profitable_cluster_ids=list(range(clustering_k)),
    )

    # Step 7: Find profitable clusters and compare to library
    for cluster_id in range(clustering_k):
        # Get cluster samples
        cluster_mask = kmeans_result.labels == cluster_id
        cluster_returns = forward_labels[cluster_mask]['return_5'].values

        # Skip empty clusters
        if len(cluster_returns) == 0:
            continue

        # Calculate performance metrics
        metrics = calculate_cluster_metrics(cluster_returns)

        # Test statistical significance
        sig_test = test_statistical_significance(cluster_returns)

        # Filter by performance thresholds
        if (
            metrics.sharpe_ratio < sharpe_threshold
            or not sig_test.is_significant
            or sig_test.t_pvalue > pvalue_threshold
        ):
            continue  # Skip low-performance clusters

        # Check if similar pattern already exists in library
        cluster_center = cluster_centers_dict[cluster_id]
        similar_pattern = find_similar_existing_pattern(
            cluster_center=cluster_center,
            pattern_library=pattern_library,
            similarity_threshold=similarity_threshold,
        )

        if similar_pattern is not None:
            continue  # Skip if similar pattern exists

        # Create new pattern template
        fuzzy_boundaries = calculate_fuzzy_boundaries(
            windows=normalized_array,
            cluster_labels=kmeans_result.labels,
            cluster_id=cluster_id,
            tolerance_level="medium",
        )

        # Analyze regime affinity (dummy regimes for now)
        cluster_timestamps = df[cluster_mask]['rate_time']
        regime_labels = np.array(['low'] * len(cluster_returns))  # Placeholder

        regime_affinity = analyze_regime_affinity(
            cluster_returns=cluster_returns,
            timestamps=cluster_timestamps,
            regime_labels=regime_labels,
        )

        # Create FuzzyPattern
        new_pattern = create_template_from_cluster(
            cluster_id=cluster_id,
            cluster_center=cluster_center,
            fuzzy_boundaries=fuzzy_boundaries,
            cluster_metrics=metrics,
            regime_affinity=regime_affinity,
            cluster_family="Re-clustered",  # Family assignment
        )

        # Add to library
        pattern_library.add_pattern(new_pattern)
        new_patterns.append(new_pattern)

    # Return results
    return WeeklyResults(
        new_patterns_added=len(new_patterns),
        patterns_pruned=0,  # Stream B will populate this
        total_patterns=pattern_library.pattern_count(),
        new_patterns=new_patterns,
        pruned_patterns=[],  # Stream B will populate this
        monitoring=None,  # Stream C will populate this
    )


def find_similar_existing_pattern(
    cluster_center: np.ndarray,
    pattern_library: Any,
    similarity_threshold: float = 0.75,
) -> Optional[FuzzyPattern]:
    """
    Find existing pattern similar to new cluster center.

    Uses fuzzy similarity matching to determine if a new cluster is
    sufficiently different from existing library patterns. If a similar
    pattern exists (similarity >= threshold), returns the existing pattern.
    Otherwise returns None, indicating the cluster is novel.

    Args:
        cluster_center: Center pattern from clustering (window_size, n_features)
        pattern_library: FuzzyPatternLibrary with get_all_patterns() method
        similarity_threshold: Minimum similarity to consider patterns equivalent (default: 0.75)

    Returns:
        Most similar FuzzyPattern if above threshold, otherwise None

    Example:
        >>> cluster_center = np.random.randn(20, 4)
        >>> similar = find_similar_existing_pattern(
        ...     cluster_center=cluster_center,
        ...     pattern_library=library,
        ...     similarity_threshold=0.75,
        ... )
        >>> if similar is None:
        ...     print("Novel pattern detected!")
    """
    # Get all existing patterns from library
    existing_patterns = pattern_library.get_all_patterns()

    # Return None if library is empty
    if not existing_patterns:
        return None

    # Calculate similarity to each existing pattern
    max_similarity = 0.0
    most_similar_pattern = None

    for pattern in existing_patterns:
        # Calculate fuzzy similarity
        similarity = calculate_fuzzy_similarity(
            window=cluster_center,
            pattern=pattern,
        )

        # Track most similar pattern
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_pattern = pattern

    # Return pattern if above threshold, otherwise None
    if max_similarity >= similarity_threshold:
        return most_similar_pattern
    else:
        return None


# ============================================================================
# Stream C: Performance Monitoring Functions
# ============================================================================


def update_performance_monitoring(
    library: Dict[str, Any],
    monitoring_window_weeks: int = 4,
    alert_sharpe_threshold: float = 0.5,
    alert_win_rate_drop: float = 0.10,
    alert_no_matches_days: int = 7,
    baseline_win_rate: float = 0.55,
) -> PerformanceMonitoring:
    """
    Update performance monitoring for all patterns in library.

    Tracks 4-week rolling metrics for each pattern:
    - Sharpe ratio
    - Win rate
    - Matches this week
    - Current drawdown
    - Performance trend (improving/stable/degrading)

    Generates alerts for:
    - Sharpe drop below threshold
    - Win rate drop exceeding threshold
    - No matches for N days

    Args:
        library: FuzzyPatternLibrary with performance tracking
        monitoring_window_weeks: Window for rolling metrics (default: 4)
        alert_sharpe_threshold: Alert if Sharpe drops below this (default: 0.5)
        alert_win_rate_drop: Alert if win rate drops this much (default: 0.10)
        alert_no_matches_days: Alert if no matches for N days (default: 7)
        baseline_win_rate: Expected baseline win rate (default: 0.55)

    Returns:
        PerformanceMonitoring with all pattern metrics and alerts

    Example:
        >>> library = load_fuzzy_pattern_library()
        >>> monitoring = update_performance_monitoring(library=library)
        >>> print(f"Tracked {len(monitoring.patterns)} patterns")
        >>> print(f"Active alerts: {len(monitoring.alerts)}")
    """
    patterns_monitoring = {}
    alerts = []
    now = datetime.now(timezone.utc)

    patterns = library.get("patterns", {})
    logger.info(
        f"Updating performance monitoring for {len(patterns)} patterns "
        f"(window={monitoring_window_weeks}w)"
    )

    for pattern_name, pattern_data in patterns.items():
        # Extract performance history
        performance_history = pattern_data.get("performance_history", {})

        # Get rolling metrics
        sharpe_rolling = performance_history.get("sharpe_rolling", 0.0)
        win_rate_rolling = performance_history.get("win_rate_rolling", 0.0)
        matches = performance_history.get("matches", 0)
        drawdown_current = performance_history.get("drawdown_current", 0.0)
        last_match_date_str = performance_history.get("last_match_date")

        # Calculate performance trend
        trend = _classify_performance_trend(sharpe_rolling)

        # Create PatternMonitoring
        pattern_mon = PatternMonitoring(
            name=pattern_name,
            sharpe_rolling=sharpe_rolling,
            win_rate_rolling=win_rate_rolling,
            matches_this_week=matches,
            drawdown_current=drawdown_current,
            trend=trend,
        )

        patterns_monitoring[pattern_name] = pattern_mon

        # Generate alerts

        # 1. Sharpe drop alert
        if sharpe_rolling < alert_sharpe_threshold:
            alerts.append(
                {
                    "type": "sharpe_drop",
                    "pattern": pattern_name,
                    "message": f"Sharpe ratio dropped to {sharpe_rolling:.2f} (threshold: {alert_sharpe_threshold:.2f})",
                    "severity": "high" if sharpe_rolling < 0.0 else "medium",
                }
            )

        # 2. Win rate drop alert
        if win_rate_rolling < (baseline_win_rate - alert_win_rate_drop):
            alerts.append(
                {
                    "type": "win_rate_drop",
                    "pattern": pattern_name,
                    "message": f"Win rate dropped to {win_rate_rolling:.2%} (baseline: {baseline_win_rate:.2%})",
                    "severity": "medium",
                }
            )

        # 3. No matches alert
        if last_match_date_str:
            last_match_date = datetime.fromisoformat(last_match_date_str)
            days_since_match = (now - last_match_date).days

            if days_since_match >= alert_no_matches_days:
                alerts.append(
                    {
                        "type": "no_matches",
                        "pattern": pattern_name,
                        "message": f"No matches for {days_since_match} days (threshold: {alert_no_matches_days} days)",
                        "severity": "low",
                    }
                )

    logger.info(
        f"Performance monitoring complete: {len(patterns_monitoring)} patterns tracked, "
        f"{len(alerts)} alerts generated"
    )

    return PerformanceMonitoring(
        timestamp=now,
        patterns=patterns_monitoring,
        alerts=alerts,
    )


def _classify_performance_trend(sharpe: float) -> str:
    """
    Classify pattern performance trend based on Sharpe ratio.

    Args:
        sharpe: Current Sharpe ratio

    Returns:
        "improving", "stable", or "degrading"
    """
    config = MONITORING_CONFIG["trend_classification"]

    if sharpe >= config["improving_threshold"]:
        return "improving"
    elif sharpe < config["degrading_threshold"]:
        return "degrading"
    else:
        return "stable"


# ============================================================================
# Stream C: Dashboard Generation Functions
# ============================================================================


def generate_monitoring_dashboard(
    performance_monitoring: PerformanceMonitoring,
    regime_shifts: List[RegimeShift],
    library: Dict[str, Any],
) -> MonitoringDashboard:
    """
    Generate comprehensive monitoring dashboard for pattern library.

    Creates dashboard with:
    - Total pattern count
    - Patterns ranked by performance (Sharpe ratio)
    - Patterns grouped by family
    - Patterns grouped by best regime
    - Active alerts
    - Detected regime shifts
    - Performance trends

    Args:
        performance_monitoring: Current performance monitoring snapshot
        regime_shifts: List of detected regime shifts
        library: FuzzyPatternLibrary with pattern metadata

    Returns:
        MonitoringDashboard with all visualization components

    Example:
        >>> monitoring = update_performance_monitoring(library)
        >>> shifts = detect_regime_shifts(monitoring_data, shift_threshold=0.5)
        >>> dashboard = generate_monitoring_dashboard(
        ...     performance_monitoring=monitoring,
        ...     regime_shifts=shifts.shifts_detected,
        ...     library=library,
        ... )
        >>> print(f"Total patterns: {dashboard.total_patterns}")
    """
    patterns = library.get("patterns", {})

    logger.info(f"Generating monitoring dashboard for {len(patterns)} patterns")

    # 1. Total patterns
    total_patterns = len(patterns)

    # 2. Patterns ranked by performance (Sharpe descending)
    patterns_by_performance = _sort_patterns_by_sharpe(performance_monitoring.patterns)

    # 3. Patterns grouped by family
    patterns_by_family = _group_patterns_by_family(library)

    # 4. Patterns grouped by regime
    patterns_by_regime = _group_patterns_by_regime(library)

    # 5. Active alerts
    active_alerts = performance_monitoring.alerts

    # 6. Performance trends
    performance_trends = {
        name: mon.trend for name, mon in performance_monitoring.patterns.items()
    }

    logger.info(
        f"Dashboard generated: {total_patterns} patterns, "
        f"{len(active_alerts)} alerts, {len(regime_shifts)} regime shifts"
    )

    return MonitoringDashboard(
        total_patterns=total_patterns,
        patterns_by_performance=patterns_by_performance,
        patterns_by_family=patterns_by_family,
        patterns_by_regime=patterns_by_regime,
        active_alerts=active_alerts,
        regime_shifts=regime_shifts,
        performance_trends=performance_trends,
    )


def _sort_patterns_by_sharpe(
    patterns: Dict[str, PatternMonitoring]
) -> List[str]:
    """
    Sort patterns by Sharpe ratio (descending).

    Args:
        patterns: Dict mapping pattern name to PatternMonitoring

    Returns:
        List of pattern names sorted by Sharpe (highest first)
    """
    sorted_patterns = sorted(
        patterns.items(),
        key=lambda item: item[1].sharpe_rolling,
        reverse=True,
    )
    return [name for name, _ in sorted_patterns]


def _group_patterns_by_family(library: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Group patterns by family.

    Args:
        library: FuzzyPatternLibrary with pattern metadata

    Returns:
        Dict mapping family name to list of pattern names
    """
    families = {}
    patterns = library.get("patterns", {})

    for pattern_name, pattern_data in patterns.items():
        pattern = pattern_data.get("pattern")
        if pattern is None:
            continue

        family = pattern.pattern_family
        if family not in families:
            families[family] = []

        families[family].append(pattern_name)

    return families


def _group_patterns_by_regime(library: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Group patterns by best regime.

    Args:
        library: FuzzyPatternLibrary with pattern metadata

    Returns:
        Dict mapping regime to list of pattern names
    """
    regimes = {}
    patterns = library.get("patterns", {})

    for pattern_name, pattern_data in patterns.items():
        pattern = pattern_data.get("pattern")
        if pattern is None:
            continue

        best_regime = pattern.best_regime
        if best_regime not in regimes:
            regimes[best_regime] = []

        regimes[best_regime].append(pattern_name)

    return regimes
