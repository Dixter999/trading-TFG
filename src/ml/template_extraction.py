"""
Template extraction from profitable clusters.

This module converts profitable clusters from ML pattern discovery into fuzzy
pattern templates. Extracts cluster centers, defines fuzzy matching parameters,
and generates performance metadata for integration into FuzzyPatternLibrary.

Following TDD methodology:
- Stream A: Center Extraction & Fuzzy Boundaries
- Stream B: Template Creation & Metadata
- Stream C: Library Integration & Validation
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from ml.cluster_evaluation import ClusterMetrics, RegimeAffinity

# ============================================================================
# Configuration
# ============================================================================

FUZZY_TOLERANCE = {
    "tight": 0.25,  # 25% tolerance (strict matching)
    "medium": 0.50,  # 50% tolerance (balanced)
    "loose": 0.75,  # 75% tolerance (permissive)
}


# ============================================================================
# Stream A: Center Extraction & Fuzzy Boundaries
# ============================================================================


@dataclass
class FuzzyBoundaries:
    """
    Fuzzy matching boundaries for pattern templates.

    Attributes:
        nominal: Ideal pattern (cluster center)
        lower: Lower bound (nominal - tolerance * std)
        upper: Upper bound (nominal + tolerance * std)
        tolerance_level: "tight" (0.25), "medium" (0.50), or "loose" (0.75)
        std_per_feature: Standard deviation for each feature
    """

    nominal: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    tolerance_level: str
    std_per_feature: np.ndarray


def extract_cluster_centers(
    windows: np.ndarray,
    cluster_labels: np.ndarray,
    profitable_cluster_ids: list[int],
) -> dict[int, np.ndarray]:
    """
    Extract center pattern for each profitable cluster.

    Calculates the centroid (mean) for each cluster across all samples.
    The center represents the "ideal" pattern that defines the cluster.

    Args:
        windows: Feature windows (n_samples, window_size, n_features)
        cluster_labels: Cluster assignment per sample
        profitable_cluster_ids: Clusters to extract

    Returns:
        Dict mapping cluster_id to center pattern (centroid)

    Example:
        >>> windows = np.array([[[1.0, 2.0]], [[1.2, 2.2]]])
        >>> labels = np.array([0, 0])
        >>> centers = extract_cluster_centers(windows, labels, [0])
        >>> centers[0]
        array([[1.1, 2.1]])
    """
    cluster_centers = {}

    for cluster_id in profitable_cluster_ids:
        # Get all windows belonging to this cluster
        mask = cluster_labels == cluster_id
        cluster_windows = windows[mask]

        # Calculate centroid (mean across samples)
        center = cluster_windows.mean(axis=0)

        cluster_centers[cluster_id] = center

    return cluster_centers


def calculate_fuzzy_boundaries(
    windows: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_id: int,
    tolerance_level: str = "medium",
) -> FuzzyBoundaries:
    """
    Calculate fuzzy matching boundaries for cluster.

    Computes statistical boundaries for fuzzy pattern matching:
    - Nominal pattern: Cluster center (mean)
    - Lower bound: nominal - (tolerance * std_dev)
    - Upper bound: nominal + (tolerance * std_dev)

    The tolerance level controls how strictly patterns must match:
    - "tight": +/-25% of std dev (strict matching)
    - "medium": +/-50% of std dev (balanced)
    - "loose": +/-75% of std dev (permissive)

    Args:
        windows: Feature windows (n_samples, window_size, n_features)
        cluster_labels: Cluster assignments
        cluster_id: Cluster to analyze
        tolerance_level: "tight", "medium", or "loose"

    Returns:
        FuzzyBoundaries with nominal/lower/upper for each feature

    Example:
        >>> windows = np.array([[[0.0]], [[1.0]], [[2.0]]])
        >>> labels = np.array([0, 0, 0])
        >>> boundaries = calculate_fuzzy_boundaries(windows, labels, 0, "medium")
        >>> boundaries.nominal  # Mean = 1.0
        array([[1.0]])
        >>> boundaries.lower  # 1.0 - 0.5 * 1.0 = 0.5
        array([[0.5]])
        >>> boundaries.upper  # 1.0 + 0.5 * 1.0 = 1.5
        array([[1.5]])
    """
    # Get all windows belonging to this cluster
    mask = cluster_labels == cluster_id
    cluster_windows = windows[mask]

    # Calculate statistics per feature
    nominal_pattern = cluster_windows.mean(axis=0)
    std_per_feature = cluster_windows.std(axis=0)

    # Get tolerance percentage
    tolerance_pct = FUZZY_TOLERANCE[tolerance_level]

    # Create boundaries: nominal +/- (tolerance * std)
    tolerance_amount = std_per_feature * tolerance_pct

    lower_bound = nominal_pattern - tolerance_amount
    upper_bound = nominal_pattern + tolerance_amount

    return FuzzyBoundaries(
        nominal=nominal_pattern,
        lower=lower_bound,
        upper=upper_bound,
        tolerance_level=tolerance_level,
        std_per_feature=std_per_feature,
    )


# ============================================================================
# Stream B: Template Creation & Metadata
# ============================================================================


@dataclass
class PerformanceMetadata:
    """
    Performance metrics for pattern templates.

    Attributes:
        sharpe_ratio: Risk-adjusted return (mean - rf) / std
        win_rate: Percentage of positive returns (0.0 to 1.0)
        mean_return: Average return in percentage points
        profit_factor: Total profits / total losses
        source_samples: Number of samples in cluster (training data size)
    """

    sharpe_ratio: float
    win_rate: float
    mean_return: float
    profit_factor: float
    source_samples: int


@dataclass
class FuzzyPattern:
    """
    Pattern template with fuzzy matching capability.

    Represents a discovered pattern from ML clustering with all metadata
    needed for fuzzy matching and performance evaluation.

    Attributes:
        name: Pattern identifier (e.g., "ML_Pattern_00")
        pattern_type: Pattern source type ("ml_discovered")
        center: Center pattern (window_size, n_features) from cluster centroid
        fuzzy_boundaries: Fuzzy matching boundaries from Stream A

        performance: Performance metrics (Sharpe, win rate, etc.)
        regime_affinity: Regime preference data from Task 020
        best_regime: Regime with highest Sharpe ratio

        confidence: Pattern reliability score (0.0 to 1.0)
        discovery_date: ISO 8601 UTC timestamp of pattern discovery
        pattern_family: Hierarchical family from Task 019 clustering
        source_cluster_id: Original cluster ID from clustering
    """

    name: str
    pattern_type: str
    center: np.ndarray
    fuzzy_boundaries: FuzzyBoundaries

    # Performance metadata
    performance: PerformanceMetadata
    regime_affinity: RegimeAffinity
    best_regime: str

    # Confidence and discovery metadata
    confidence: float
    discovery_date: str
    pattern_family: str
    source_cluster_id: int


def create_template_from_cluster(
    cluster_id: int,
    cluster_center: np.ndarray,
    fuzzy_boundaries: FuzzyBoundaries,
    cluster_metrics: ClusterMetrics,
    regime_affinity: RegimeAffinity,
    cluster_family: str,
) -> FuzzyPattern:
    """
    Create FuzzyPattern template from cluster data.

    Combines cluster center, fuzzy boundaries, performance metrics, and
    regime affinity into a complete pattern template ready for use in
    the ConfluenceEngine.

    Args:
        cluster_id: Source cluster ID (used for pattern naming)
        cluster_center: Center pattern from extract_cluster_centers()
        fuzzy_boundaries: Matching boundaries from calculate_fuzzy_boundaries()
        cluster_metrics: Performance metrics from Task 020
        regime_affinity: Regime preferences from Task 020
        cluster_family: Pattern family (from Task 019)

    Returns:
        FuzzyPattern ready for ConfluenceEngine integration

    Example:
        >>> cluster_center = np.array([[1.0, 2.0]] * 20)  # window_size=20
        >>> fuzzy_boundaries = calculate_fuzzy_boundaries(...)
        >>> cluster_metrics = ClusterMetrics(sharpe_ratio=1.5, win_rate=0.6, ...)
        >>> regime_affinity = RegimeAffinity(best_regime="low", ...)
        >>> template = create_template_from_cluster(
        ...     cluster_id=5,
        ...     cluster_center=cluster_center,
        ...     fuzzy_boundaries=fuzzy_boundaries,
        ...     cluster_metrics=cluster_metrics,
        ...     regime_affinity=regime_affinity,
        ...     cluster_family="Up Trend"
        ... )
        >>> template.name
        'ML_Pattern_05'
        >>> template.confidence
        0.675  # (1.5/2.0)*0.5 + 0.6*0.5 = 0.375 + 0.3 = 0.675
    """
    # Calculate confidence from metrics
    # High Sharpe + High win rate = High confidence
    # Formula: min(1.0, (sharpe/2.0)*0.5 + win_rate*0.5)
    # - Sharpe contribution: Assumes Sharpe of 2.0 = max (0.5 contribution)
    # - Win rate contribution: Win rate directly contributes (0.5 max)
    # - Total capped at 1.0
    confidence = min(
        1.0,
        (cluster_metrics.sharpe_ratio / 2.0) * 0.5  # Sharpe contribution
        + (cluster_metrics.win_rate * 0.5),  # Win rate contribution
    )

    # Create performance metadata
    performance = PerformanceMetadata(
        sharpe_ratio=cluster_metrics.sharpe_ratio,
        win_rate=cluster_metrics.win_rate,
        mean_return=cluster_metrics.mean_return,
        profit_factor=cluster_metrics.profit_factor,
        source_samples=cluster_metrics.sample_count,
    )

    # Create FuzzyPattern template
    template = FuzzyPattern(
        name=f"ML_Pattern_{cluster_id:02d}",  # Zero-padded 2-digit ID
        pattern_type="ml_discovered",
        center=cluster_center,
        fuzzy_boundaries=fuzzy_boundaries,
        # Performance metadata
        performance=performance,
        regime_affinity=regime_affinity,
        best_regime=regime_affinity.best_regime,
        # Confidence and metadata
        confidence=confidence,
        discovery_date=datetime.now(timezone.utc).isoformat(),
        pattern_family=cluster_family,
        source_cluster_id=cluster_id,
    )

    return template


# ============================================================================
# Stream C: Library Integration & Validation
# ============================================================================


@dataclass
class IntegrationResult:
    """
    Result of integrating templates into pattern library.

    Attributes:
        patterns_added: Number of new patterns added to library
        patterns_updated: Number of existing patterns updated with better performance
        patterns_skipped: Number of patterns skipped (lower confidence than existing)
        total_library_size: Total patterns in library after integration
    """

    patterns_added: int
    patterns_updated: int
    patterns_skipped: int
    total_library_size: int


@dataclass
class TemplateValidation:
    """
    Validation results for pattern templates on historical data.

    Attributes:
        validation_results: Dict mapping pattern_name to metrics dict
            Each metrics dict contains:
            - matches_found: Number of matches found
            - match_percentage: Percentage of validation windows matched
            - mean_return: Average return for matched windows
            - win_rate: Percentage of positive returns
            - sharpe: Sharpe ratio of matched returns
        avg_match_rate: Average match percentage across all patterns
        avg_sharpe: Average Sharpe ratio across all patterns
    """

    validation_results: dict[str, dict[str, float]]
    avg_match_rate: float
    avg_sharpe: float


@dataclass
class ExtractionReport:
    """
    Complete extraction pipeline report.

    Attributes:
        templates_created: Number of templates created from clusters
        templates_integrated: Number of templates added/updated in library
        total_library_size: Final library size after integration
        validation_results: Validation metrics for all templates
    """

    templates_created: int
    templates_integrated: int
    total_library_size: int
    validation_results: TemplateValidation


def calculate_fuzzy_similarity(
    window: np.ndarray,
    pattern: FuzzyPattern,
) -> float:
    """
    Calculate similarity between window and fuzzy pattern.

    Computes how well a window matches the pattern's fuzzy boundaries.
    Returns 1.0 for perfect match to nominal pattern, and decreases
    based on how far features deviate from fuzzy boundaries.

    Similarity calculation:
    - For each feature timestep:
      - If value within fuzzy boundaries: high score (based on distance from nominal)
      - If value outside fuzzy boundaries: low score (based on distance from boundary)
    - Overall similarity is mean across all features

    Args:
        window: Feature window to match (window_size, n_features)
        pattern: FuzzyPattern with fuzzy boundaries

    Returns:
        Similarity score in [0.0, 1.0]
        - 1.0 = perfect match to nominal pattern
        - 0.8-1.0 = within fuzzy boundaries, close to nominal
        - 0.5-0.8 = within fuzzy boundaries, near edges
        - 0.0-0.5 = outside fuzzy boundaries

    Example:
        >>> nominal = np.array([[1.0, 2.0]])
        >>> pattern = FuzzyPattern(
        ...     center=nominal,
        ...     fuzzy_boundaries=FuzzyBoundaries(
        ...         nominal=nominal,
        ...         lower=np.array([[0.5, 1.5]]),
        ...         upper=np.array([[1.5, 2.5]]),
        ...         tolerance_level="medium",
        ...         std_per_feature=np.ones_like(nominal),
        ...     ),
        ...     ...
        ... )
        >>> window = np.array([[1.0, 2.0]])  # Perfect match
        >>> calculate_fuzzy_similarity(window, pattern)
        1.0
    """
    fuzzy_bounds = pattern.fuzzy_boundaries
    nominal = fuzzy_bounds.nominal
    lower = fuzzy_bounds.lower
    upper = fuzzy_bounds.upper

    # Calculate normalized distance from nominal for each feature
    # Distance is normalized by the fuzzy boundary range
    boundary_range = upper - lower

    # Avoid division by zero for constant features
    boundary_range = np.where(boundary_range == 0, 1.0, boundary_range)

    # Calculate distance from nominal
    distance_from_nominal = np.abs(window - nominal)

    # Normalize distance by boundary range
    normalized_distance = distance_from_nominal / boundary_range

    # Calculate feature-wise similarity scores
    # If within boundaries (distance <= range/2): high similarity
    # If outside boundaries: low similarity based on how far outside
    feature_similarities = np.zeros_like(window)

    # Check which features are within boundaries
    within_lower = window >= lower
    within_upper = window <= upper
    within_boundaries = np.logical_and(within_lower, within_upper)

    # For features within boundaries: similarity = 1.0 - normalized_distance
    # This gives 1.0 for perfect match to nominal, decreasing toward boundaries
    feature_similarities[within_boundaries] = 1.0 - normalized_distance[within_boundaries]

    # For features outside boundaries: similarity decreases rapidly
    # Use exponential decay based on how far outside
    outside_boundaries = np.logical_not(within_boundaries)
    if np.any(outside_boundaries):
        # Distance beyond boundaries
        distance_below = np.maximum(0, lower - window)
        distance_above = np.maximum(0, window - upper)
        distance_beyond = distance_below + distance_above

        # Normalize by boundary range
        normalized_beyond = distance_beyond / boundary_range

        # Exponential decay: e^(-2*distance)
        # This gives ~0.14 at 1x boundary range beyond, ~0.02 at 2x, etc.
        feature_similarities[outside_boundaries] = np.exp(
            -2.0 * normalized_beyond[outside_boundaries]
        )

    # Overall similarity is mean across all features
    overall_similarity = float(np.mean(feature_similarities))

    # Clamp to [0.0, 1.0] for safety
    overall_similarity = max(0.0, min(1.0, overall_similarity))

    return overall_similarity


def integrate_templates_into_library(
    fuzzy_patterns: list[FuzzyPattern],
    library: object,  # FuzzyPatternLibrary interface
) -> IntegrationResult:
    """
    Add ML-discovered templates to pattern library.

    Integrates fuzzy pattern templates into the existing pattern library.
    Handles three cases:
    1. New patterns: Added to library
    2. Existing patterns with lower confidence: Updated
    3. Existing patterns with higher confidence: Skipped

    Args:
        fuzzy_patterns: Templates to integrate
        library: Pattern library with interface:
            - get_pattern(name) -> pattern or None
            - add_pattern(pattern) -> None
            - update_pattern(name, pattern) -> None
            - pattern_count() -> int

    Returns:
        IntegrationResult with summary of additions/updates/skips

    Example:
        >>> patterns = [FuzzyPattern(...), FuzzyPattern(...)]
        >>> library = FuzzyPatternLibrary()
        >>> result = integrate_templates_into_library(patterns, library)
        >>> result.patterns_added
        2
        >>> result.total_library_size
        12
    """
    integration_summary = {
        "added": [],
        "updated": [],
        "skipped": [],
    }

    for pattern in fuzzy_patterns:
        # Check if pattern with same name exists
        existing = library.get_pattern(pattern.name)

        if existing is None:
            # Add new pattern
            library.add_pattern(pattern)
            integration_summary["added"].append(pattern.name)

        else:
            # Update only if new confidence is higher
            if pattern.confidence > existing.confidence:
                library.update_pattern(pattern.name, pattern)
                integration_summary["updated"].append(pattern.name)
            else:
                integration_summary["skipped"].append(pattern.name)

    return IntegrationResult(
        patterns_added=len(integration_summary["added"]),
        patterns_updated=len(integration_summary["updated"]),
        patterns_skipped=len(integration_summary["skipped"]),
        total_library_size=library.pattern_count(),
    )


def validate_templates(
    fuzzy_patterns: list[FuzzyPattern],
    validation_windows: np.ndarray,
    validation_returns: np.ndarray,
    similarity_threshold: float = 0.75,
) -> TemplateValidation:
    """
    Validate templates produce valid pattern matches on historical data.

    For each template, finds matches in validation data and calculates
    performance metrics (match rate, Sharpe ratio, win rate).

    Validation process:
    1. For each pattern, scan validation windows
    2. Calculate fuzzy similarity for each window
    3. Windows above similarity_threshold are considered matches
    4. Calculate performance metrics for matched windows
    5. Aggregate statistics across all patterns

    Args:
        fuzzy_patterns: Templates to validate
        validation_windows: Test data windows (n_samples, window_size, n_features)
        validation_returns: Forward returns for windows (n_samples,)
        similarity_threshold: Minimum similarity for match (default: 0.75)

    Returns:
        TemplateValidation with match statistics per pattern and averages

    Example:
        >>> patterns = [FuzzyPattern(...)]
        >>> windows = np.random.randn(1000, 20, 4)
        >>> returns = np.random.randn(1000) * 0.01
        >>> validation = validate_templates(patterns, windows, returns)
        >>> validation.avg_match_rate
        3.2  # 3.2% of windows matched
        >>> validation.avg_sharpe
        1.05  # Sharpe ratio of matched returns
    """
    validation_results = {}

    for pattern in fuzzy_patterns:
        matches = []

        # Find matches in validation data
        for i, window in enumerate(validation_windows):
            similarity = calculate_fuzzy_similarity(window, pattern)

            if similarity >= similarity_threshold:
                matches.append(
                    {
                        "window_idx": i,
                        "similarity": similarity,
                        "return": validation_returns[i],
                    }
                )

        if len(matches) > 0:
            match_returns = np.array([m["return"] for m in matches])

            validation_results[pattern.name] = {
                "matches_found": len(matches),
                "match_percentage": 100.0 * len(matches) / len(validation_windows),
                "mean_return": float(match_returns.mean()),
                "win_rate": float((match_returns > 0).mean()),
                "sharpe": _calculate_sharpe(match_returns),
            }

    # Calculate averages
    if validation_results:
        avg_match_rate = np.mean(
            [v["match_percentage"] for v in validation_results.values()]
        )
        avg_sharpe = np.mean([v["sharpe"] for v in validation_results.values()])
    else:
        avg_match_rate = 0.0
        avg_sharpe = 0.0

    return TemplateValidation(
        validation_results=validation_results,
        avg_match_rate=avg_match_rate,
        avg_sharpe=avg_sharpe,
    )


def _calculate_sharpe(returns: np.ndarray) -> float:
    """
    Calculate annualized Sharpe ratio from returns.

    Args:
        returns: Array of returns

    Returns:
        Annualized Sharpe ratio (assumes 252 trading days)
        Returns 0.0 if returns are empty or have zero std
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))  # Annualized
