"""
ML clustering module for pattern discovery.

This module provides unsupervised clustering algorithms to identify
natural pattern groups in normalized price action.

TDD Phase: REFACTOR - Optimized implementation with logging and improved performance.
"""

from dataclasses import dataclass
from typing import Optional
import logging
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class KMeansResult:
    """Result from K-Means clustering.

    Attributes:
        labels: Cluster label per sample (n_samples,)
        centers: Cluster center positions (n_clusters, n_features)
        inertia: Sum of squared distances to closest cluster center
        n_iter: Number of iterations until convergence
    """

    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int


@dataclass
class DBSCANResult:
    """Result from DBSCAN outlier detection.

    Attributes:
        labels: Cluster labels for each sample (-1 = outlier)
        n_outliers: Number of outliers detected
        outlier_percentage: Percentage of samples classified as outliers
        n_core_clusters: Number of core clusters (excluding outliers)
    """

    labels: np.ndarray
    n_outliers: int
    outlier_percentage: float
    n_core_clusters: int


@dataclass
class HierarchicalResult:
    """Result from hierarchical clustering of K-Means centers.

    Attributes:
        family_labels: Family assignment for each K-Means cluster (n_kmeans_clusters, 1) or (n_kmeans_clusters,)
        linkage_matrix: Hierarchical linkage matrix from scipy (n_kmeans_clusters-1, 4)
        cluster_families: Dict mapping family_id -> list of cluster_ids belonging to that family
    """

    family_labels: np.ndarray
    linkage_matrix: np.ndarray
    cluster_families: dict[int, list[int]]


@dataclass
class ClusterQuality:
    """Quality metrics for clustering evaluation.

    Attributes:
        silhouette_score: Silhouette coefficient in range [-1, 1] (higher is better, target >0.4)
        davies_bouldin_score: Davies-Bouldin index (lower is better, target <1.5)
        calinski_harabasz_score: Calinski-Harabasz index (higher is better, target >100)
        quality_rating: Human-readable quality rating ("Excellent", "Good", "Fair", "Poor")
    """

    silhouette_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float
    quality_rating: str


@dataclass
class StabilityMetrics:
    """Clustering stability metrics across multiple runs.

    Attributes:
        mean_stability: Mean Adjusted Rand Index (ARI) across all pairwise comparisons
        min_stability: Minimum ARI observed
        max_stability: Maximum ARI observed
        all_scores: List of all pairwise ARI scores
    """

    mean_stability: float
    min_stability: float
    max_stability: float
    all_scores: list[float]


@dataclass
class ClusterStatistics:
    """Per-cluster statistics including size, returns, and performance.

    Attributes:
        stats: Dict mapping cluster_id -> statistics dict containing:
               - size: Number of samples in cluster
               - percentage: Percentage of total samples
               - mean_return_5h, mean_return_10h, mean_return_20h: Average returns
               - std_return_5h: Standard deviation of 5h returns
               - win_rate_5h, win_rate_10h: Percentage of positive returns
               - median_return_5h: Median 5h return
               - pattern_variance: Average variance of patterns in cluster
    """

    stats: dict[int, dict[str, float]]


def cluster_kmeans(
    windows: np.ndarray,
    n_clusters: int = 50,
    random_state: int = 42,
) -> KMeansResult:
    """
    Cluster price patterns using K-Means algorithm.

    Uses k-means++ initialization for better convergence and reproducibility.
    Windows are flattened from 3D (n_samples, window_size, n_features) to 2D
    (n_samples, window_size * n_features) before clustering.

    Performance:
    - 10,000 samples: ~5 minutes on standard CPU (per spec)
    - Convergence typically <100 iterations with k-means++
    - Memory: O(n_samples * window_size * n_features)

    Args:
        windows: Feature array (n_samples, window_size, n_features)
                Typically normalized price windows from data_preparation.normalize_window()
        n_clusters: Number of clusters (default: 50 per spec)
                   Rationale: 10,000 samples / 50 = 200 samples per cluster (good granularity)
        random_state: Random seed for reproducibility (default: 42 per spec)

    Returns:
        KMeansResult with cluster labels, centers, inertia, and iteration count

    Raises:
        ValueError: If windows array is empty or has invalid dimensions

    Example:
        >>> from src.ml.data_preparation import load_historical_data, extract_rolling_windows
        >>> windows = extract_rolling_windows(df, window_size=20)
        >>> result = cluster_kmeans(windows, n_clusters=50)
        >>> print(f"Converged in {result.n_iter} iterations")
        >>> print(f"Found {len(np.unique(result.labels))} clusters")
    """
    # Validate input
    if windows.size == 0:
        raise ValueError("Input windows array is empty")
    if windows.ndim not in (2, 3):
        raise ValueError(
            f"Input windows must be 2D or 3D array, got {windows.ndim}D"
        )

    logger.info(
        f"Starting K-Means clustering: n_samples={windows.shape[0]}, "
        f"n_clusters={n_clusters}, random_state={random_state}"
    )

    # Flatten windows for clustering: (n_samples, window_size, n_features) -> (n_samples, window_size * n_features)
    n_samples = windows.shape[0]
    if windows.ndim == 3:
        flattened_features = windows.shape[1] * windows.shape[2]
        windows_flat = windows.reshape(n_samples, flattened_features)
        logger.debug(
            f"Flattened windows: {windows.shape} -> {windows_flat.shape}"
        )
    else:
        # Already 2D
        windows_flat = windows
        logger.debug(f"Using pre-flattened windows: {windows_flat.shape}")

    # Initialize K-Means with k-means++ for better convergence
    logger.debug("Initializing K-Means with k-means++ initialization")
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",  # Smart initialization (spec requirement)
        n_init=20,  # Multiple random starts for robustness
        max_iter=500,  # Convergence threshold (spec requirement)
        random_state=random_state,
        algorithm="lloyd",  # Standard K-Means algorithm
        verbose=0,  # Suppress sklearn output
    )

    # Fit and predict cluster labels
    logger.info("Fitting K-Means model...")
    cluster_labels = kmeans.fit_predict(windows_flat)
    cluster_centers = kmeans.cluster_centers_

    # Log convergence statistics
    logger.info(
        f"K-Means converged in {kmeans.n_iter_} iterations "
        f"(inertia={kmeans.inertia_:.2f})"
    )

    # Cluster size distribution
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    logger.info(
        f"Cluster size distribution: min={counts.min()}, "
        f"max={counts.max()}, median={np.median(counts):.0f}, "
        f"mean={counts.mean():.1f}"
    )

    # Check for convergence issues
    if kmeans.n_iter_ >= 500:
        logger.warning(
            f"K-Means reached max_iter={500} without full convergence. "
            f"Consider increasing max_iter or adjusting n_clusters."
        )

    return KMeansResult(
        labels=cluster_labels,
        centers=cluster_centers,
        inertia=kmeans.inertia_,
        n_iter=kmeans.n_iter_,
    )


def detect_outliers_dbscan(
    windows: np.ndarray, eps: float = 0.8, min_samples: int = 20
) -> DBSCANResult:
    """
    Identify outlier patterns using DBSCAN clustering.

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    identifies clusters based on density. Points that don't belong to any
    cluster are classified as outliers (-1 label).

    This function is designed to detect rare/novel price patterns that don't
    fit into normal pattern groups. Outliers may indicate:
    - Market anomalies
    - Regime shifts
    - Rare but significant events

    Args:
        windows: Feature array with shape (n_samples, window_size, n_features)
                 Example: (10000, 20, 4) for 10k windows of 20 candles with OHLC
        eps: Maximum distance between two samples for one to be considered
             in the neighborhood of the other (default: 0.8)
             - Smaller eps = stricter neighborhood = more outliers
             - Larger eps = looser neighborhood = fewer outliers
        min_samples: Minimum samples in a neighborhood for a point to be
                     considered a core point (default: 20)
                     - Higher min_samples = stricter cluster formation = more outliers
                     - Lower min_samples = easier cluster formation = fewer outliers

    Returns:
        DBSCANResult containing:
        - labels: Array of cluster assignments (-1 for outliers)
        - n_outliers: Count of outlier samples
        - outlier_percentage: Percentage of samples classified as outliers
        - n_core_clusters: Number of distinct clusters (excluding outliers)

    Example:
        >>> windows = np.random.randn(1000, 20, 4)  # 1000 normalized price windows
        >>> result = detect_outliers_dbscan(windows, eps=0.8, min_samples=20)
        >>> print(f"Found {result.n_outliers} outliers ({result.outlier_percentage:.1f}%)")
        >>> print(f"Core clusters: {result.n_core_clusters}")

    Expected Behavior:
        - Typical datasets: 1-5% outliers (per spec)
        - Empty input: 0 outliers, 0 clusters
        - Sparse data (all outliers): high outlier percentage
        - Dense data (no outliers): 0% outliers

    Performance:
        - Time complexity: O(n log n) with efficient indexing
        - Space complexity: O(n) for labels storage
        - Typical runtime: ~2 minutes for 10,000 samples on CPU
    """
    # Validate input
    if windows.size == 0:
        logger.warning("Empty input to DBSCAN - returning empty result")
        return DBSCANResult(
            labels=np.array([], dtype=np.int32),
            n_outliers=0,
            outlier_percentage=0.0,
            n_core_clusters=0,
        )

    if windows.ndim not in (2, 3):
        raise ValueError(
            f"Input windows must be 2D or 3D array, got {windows.ndim}D"
        )

    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    if min_samples < 1:
        raise ValueError(f"min_samples must be at least 1, got {min_samples}")

    logger.info(
        f"Starting DBSCAN outlier detection: "
        f"n_samples={windows.shape[0]}, eps={eps}, min_samples={min_samples}"
    )

    # Flatten windows from (n_samples, window_size, n_features) to (n_samples, features)
    # Example: (10000, 20, 4) -> (10000, 80)
    n_samples = windows.shape[0]
    if windows.ndim == 3:
        flattened_features = windows.shape[1] * windows.shape[2]
        windows_flat = windows.reshape(n_samples, flattened_features)
        logger.debug(
            f"Flattened windows: {windows.shape} -> {windows_flat.shape}"
        )
    else:
        # Already 2D
        windows_flat = windows
        logger.debug(f"Using pre-flattened windows: {windows_flat.shape}")

    # Run DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    outlier_labels = dbscan.fit_predict(windows_flat)

    # Calculate outlier statistics
    # DBSCAN labels outliers as -1
    n_outliers = (outlier_labels == -1).sum()
    outlier_pct = 100.0 * n_outliers / len(outlier_labels)

    # Count core clusters (excluding outliers)
    # Labels are: -1 (outlier), 0, 1, 2, ... (cluster IDs)
    unique_labels = np.unique(outlier_labels)
    core_labels = unique_labels[unique_labels >= 0]  # Exclude -1 (outliers)
    n_core_clusters = len(core_labels)

    # Log cluster size distribution
    if n_core_clusters > 0:
        core_cluster_sizes = [
            (outlier_labels == label).sum() for label in core_labels
        ]
        logger.info(
            f"Core cluster size distribution: "
            f"min={min(core_cluster_sizes)}, "
            f"max={max(core_cluster_sizes)}, "
            f"median={np.median(core_cluster_sizes):.0f}, "
            f"mean={np.mean(core_cluster_sizes):.1f}"
        )

    logger.info(
        f"DBSCAN complete: {n_outliers} outliers ({outlier_pct:.2f}%), "
        f"{n_core_clusters} core clusters, "
        f"{(outlier_labels >= 0).sum()} samples in core clusters"
    )

    # Warn if outlier percentage is unusually high or low
    if outlier_pct > 20.0:
        logger.warning(
            f"High outlier percentage ({outlier_pct:.1f}%) - "
            f"consider increasing eps or decreasing min_samples"
        )
    elif outlier_pct == 0.0 and n_samples > 100:
        logger.warning(
            "No outliers found - "
            "consider decreasing eps or increasing min_samples for stricter detection"
        )

    return DBSCANResult(
        labels=outlier_labels,
        n_outliers=int(n_outliers),
        outlier_percentage=float(outlier_pct),
        n_core_clusters=int(n_core_clusters),
    )


def cluster_hierarchical(
    kmeans_result: KMeansResult,
    n_clusters: int = 10,
) -> HierarchicalResult:
    """
    Create hierarchical clustering of K-Means cluster centers to identify pattern families.

    This function groups K-Means clusters (typically 50) into a smaller number of
    pattern families (typically 10) using hierarchical clustering with Ward linkage.
    Ward linkage minimizes within-cluster variance, making it ideal for grouping
    similar patterns.

    Use Case:
    - Primary clustering: K-Means creates 50 fine-grained clusters (200 samples each)
    - Secondary clustering: Hierarchical groups these 50 into 10 broader families (~5 clusters each)
    - Benefit: Enables both detailed pattern analysis (50 clusters) and high-level categorization (10 families)

    Args:
        kmeans_result: Result from cluster_kmeans() containing cluster centers
                      Expected: 50 cluster centers from K-Means
        n_clusters: Number of pattern families to create (default: 10 per spec)
                   Rationale: 50 clusters / 10 families = ~5 clusters per family (good balance)

    Returns:
        HierarchicalResult containing:
        - family_labels: Array (n_kmeans_clusters,) or (n_kmeans_clusters, 1) with family ID per cluster
        - linkage_matrix: Scipy linkage matrix (n_kmeans_clusters-1, 4) describing merge hierarchy
        - cluster_families: Dict {family_id: [cluster_ids]} for easy lookup

    Raises:
        ValueError: If n_clusters <= 0 or > number of K-Means clusters

    Example:
        >>> from src.ml.clustering import cluster_kmeans, cluster_hierarchical
        >>> # First: K-Means clustering
        >>> kmeans_result = cluster_kmeans(windows, n_clusters=50)
        >>> # Second: Hierarchical grouping
        >>> hier_result = cluster_hierarchical(kmeans_result, n_clusters=10)
        >>> print(f"Created {len(hier_result.cluster_families)} pattern families")
        >>> print(f"Family 0 contains clusters: {hier_result.cluster_families[0]}")

    Performance:
        - Time complexity: O(n^2 log n) for Ward linkage on n centers
        - For 50 centers: <1 second on CPU
        - Space complexity: O(n^2) for distance matrix
        - Deterministic: same input = same output (no random seed needed)

    Technical Notes:
        - Uses Ward linkage (minimizes variance within families)
        - Operates on cluster centers, NOT raw samples (much faster)
        - scipy.cluster.hierarchy.cut_tree splits dendrogram at n_clusters level
        - Linkage matrix format: [idx1, idx2, distance, sample_count] per merge
    """
    from scipy.cluster.hierarchy import linkage, cut_tree

    # Validate input
    n_kmeans_clusters = len(kmeans_result.centers)

    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}")

    if n_clusters > n_kmeans_clusters:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed number of K-Means clusters ({n_kmeans_clusters})"
        )

    logger.info(
        f"Starting hierarchical clustering: grouping {n_kmeans_clusters} K-Means clusters "
        f"into {n_clusters} pattern families using Ward linkage"
    )

    # Extract cluster centers (shape: n_kmeans_clusters, n_features)
    # These are already in flattened form from K-Means
    cluster_centers = kmeans_result.centers

    logger.debug(
        f"K-Means centers shape: {cluster_centers.shape} "
        f"({n_kmeans_clusters} clusters, {cluster_centers.shape[1]} features)"
    )

    # Compute hierarchical clustering with Ward linkage
    # Ward linkage minimizes within-cluster variance (best for pattern grouping)
    logger.debug("Computing Ward linkage matrix...")
    linkage_matrix = linkage(
        cluster_centers,
        method="ward",  # Minimize variance (spec requirement)
        metric="euclidean",  # Ward requires Euclidean distance
    )

    # Cut dendrogram at n_clusters level
    # Returns (n_kmeans_clusters, 1) array of family labels
    logger.debug(f"Cutting dendrogram at {n_clusters} families...")
    family_labels = cut_tree(linkage_matrix, n_clusters=n_clusters)

    # Flatten family_labels from (n, 1) to (n,) for consistency
    family_labels_flat = family_labels.flatten()

    # Build cluster_families dict: {family_id: [cluster_ids]}
    cluster_families: dict[int, list[int]] = {}
    for family_id in range(n_clusters):
        # Find all K-Means cluster IDs belonging to this family
        cluster_ids = np.where(family_labels_flat == family_id)[0].tolist()
        cluster_families[family_id] = cluster_ids

    # Log family size distribution
    family_sizes = [len(clusters) for clusters in cluster_families.values()]
    logger.info(
        f"Hierarchical clustering complete: {n_clusters} families created"
    )
    logger.info(
        f"Family size distribution: "
        f"min={min(family_sizes)}, "
        f"max={max(family_sizes)}, "
        f"median={np.median(family_sizes):.0f}, "
        f"mean={np.mean(family_sizes):.1f}"
    )

    # Log sample family composition
    logger.debug("Sample family composition:")
    for family_id in range(min(3, n_clusters)):  # Show first 3 families
        clusters = cluster_families[family_id]
        logger.debug(
            f"  Family {family_id}: {len(clusters)} clusters {clusters[:5]}..."
            if len(clusters) > 5
            else f"  Family {family_id}: {len(clusters)} clusters {clusters}"
        )

    return HierarchicalResult(
        family_labels=family_labels_flat,  # Return flattened (n,) array
        linkage_matrix=linkage_matrix,
        cluster_families=cluster_families,
    )


def calculate_cluster_quality(
    windows: np.ndarray,
    labels: np.ndarray,
) -> ClusterQuality:
    """
    Calculate quality metrics for clustering results.

    Evaluates clustering quality using three complementary metrics:
    - Silhouette score: Measures separation between clusters (higher is better)
    - Davies-Bouldin score: Measures cluster compactness and separation (lower is better)
    - Calinski-Harabasz score: Ratio of between-cluster to within-cluster variance (higher is better)

    Args:
        windows: Feature array (n_samples, window_size, n_features)
                Example: (10000, 20, 4) for 10k price windows
        labels: Cluster assignments (n_samples,)
               Values should be integers in range [0, n_clusters-1]

    Returns:
        ClusterQuality with scores and overall rating

    Raises:
        ValueError: If windows and labels have different lengths
        ValueError: If labels contain only one unique value (no clustering)

    Example:
        >>> result = cluster_kmeans(windows, n_clusters=50)
        >>> quality = calculate_cluster_quality(windows, result.labels)
        >>> print(f"Silhouette: {quality.silhouette_score:.3f}")
        >>> print(f"Quality: {quality.quality_rating}")

    Quality Thresholds (per spec):
        - Excellent: silhouette >0.7, davies_bouldin <1.0
        - Good: silhouette >0.4, davies_bouldin <1.5
        - Fair: silhouette >0.2, davies_bouldin <2.0
        - Poor: otherwise
    """
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score,
    )

    # Validate input
    if len(windows) != len(labels):
        raise ValueError(
            f"windows ({len(windows)}) and labels ({len(labels)}) must have same length"
        )

    if len(np.unique(labels)) == 1:
        raise ValueError("labels must contain at least 2 unique values for quality metrics")

    logger.info(
        f"Calculating cluster quality metrics: "
        f"n_samples={len(windows)}, n_clusters={len(np.unique(labels))}"
    )

    # Flatten windows from (n_samples, window_size, n_features) to (n_samples, features)
    n_samples = windows.shape[0]
    if windows.ndim == 3:
        flattened_features = windows.shape[1] * windows.shape[2]
        windows_flat = windows.reshape(n_samples, flattened_features)
        logger.debug(f"Flattened windows: {windows.shape} -> {windows_flat.shape}")
    else:
        windows_flat = windows
        logger.debug(f"Using pre-flattened windows: {windows_flat.shape}")

    # Calculate quality metrics
    logger.debug("Computing silhouette score...")
    silhouette = silhouette_score(windows_flat, labels)

    logger.debug("Computing Davies-Bouldin score...")
    davies_bouldin = davies_bouldin_score(windows_flat, labels)

    logger.debug("Computing Calinski-Harabasz score...")
    calinski_harabasz = calinski_harabasz_score(windows_flat, labels)

    # Determine quality rating based on thresholds
    if silhouette > 0.7 and davies_bouldin < 1.0:
        rating = "Excellent"
    elif silhouette > 0.4 and davies_bouldin < 1.5:
        rating = "Good"
    elif silhouette > 0.2 and davies_bouldin < 2.0:
        rating = "Fair"
    else:
        rating = "Poor"

    logger.info(
        f"Quality metrics: silhouette={silhouette:.3f}, "
        f"davies_bouldin={davies_bouldin:.3f}, "
        f"calinski_harabasz={calinski_harabasz:.1f}, "
        f"rating={rating}"
    )

    return ClusterQuality(
        silhouette_score=float(silhouette),
        davies_bouldin_score=float(davies_bouldin),
        calinski_harabasz_score=float(calinski_harabasz),
        quality_rating=rating,
    )


def validate_cluster_stability(
    windows: np.ndarray,
    n_runs: int = 5,
    n_clusters: int = 50,
) -> StabilityMetrics:
    """
    Validate clustering stability across multiple runs with different random seeds.

    Measures how consistent clustering results are by running K-Means multiple times
    and comparing cluster assignments using Adjusted Rand Index (ARI). High stability
    indicates robust, reproducible clustering.

    Args:
        windows: Feature array (n_samples, window_size, n_features)
        n_runs: Number of independent clustering runs (default: 5 per spec)
               More runs = more reliable stability estimate, but slower
        n_clusters: Number of clusters per run (default: 50 per spec)

    Returns:
        StabilityMetrics with mean/min/max ARI and all pairwise scores

    Raises:
        ValueError: If n_runs < 2 (need at least 2 runs to compare)

    Example:
        >>> stability = validate_cluster_stability(windows, n_runs=5, n_clusters=50)
        >>> print(f"Mean stability: {stability.mean_stability:.3f}")
        >>> print(f"Range: [{stability.min_stability:.3f}, {stability.max_stability:.3f}]")

    Stability Interpretation:
        - >0.9: Excellent (very stable clustering)
        - 0.8-0.9: Good (stable, suitable for production)
        - 0.6-0.8: Fair (acceptable but monitor)
        - <0.6: Poor (unstable, consider different k or algorithm)

    Performance:
        - For 10k samples, 5 runs, 50 clusters: ~5 minutes on CPU
        - Time complexity: O(n_runs * K-Means time)
        - Number of comparisons: n_runs*(n_runs-1)/2
    """
    from sklearn.metrics import adjusted_rand_score

    # Validate input
    if n_runs < 2:
        raise ValueError(f"n_runs must be at least 2, got {n_runs}")

    logger.info(
        f"Validating cluster stability: "
        f"n_samples={len(windows)}, n_runs={n_runs}, n_clusters={n_clusters}"
    )

    # Run K-Means multiple times with different random seeds
    results = []
    for i in range(n_runs):
        logger.debug(f"Stability run {i+1}/{n_runs}...")
        # Use different random seed for each run
        result = cluster_kmeans(windows, n_clusters=n_clusters, random_state=42 + i)
        results.append(result.labels)

    # Compare all pairs of runs using Adjusted Rand Index
    stability_scores = []
    n_comparisons = n_runs * (n_runs - 1) // 2
    logger.debug(f"Computing {n_comparisons} pairwise ARI scores...")

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            ari = adjusted_rand_score(results[i], results[j])
            stability_scores.append(ari)
            logger.debug(f"  ARI(run_{i}, run_{j}) = {ari:.4f}")

    # Calculate summary statistics
    mean_stability = np.mean(stability_scores)
    min_stability = np.min(stability_scores)
    max_stability = np.max(stability_scores)

    logger.info(
        f"Stability validation complete: "
        f"mean={mean_stability:.3f}, min={min_stability:.3f}, max={max_stability:.3f}"
    )

    # Warn if stability is low
    if mean_stability < 0.6:
        logger.warning(
            f"Low stability detected ({mean_stability:.3f}). "
            f"Consider adjusting n_clusters or using different algorithm."
        )

    return StabilityMetrics(
        mean_stability=float(mean_stability),
        min_stability=float(min_stability),
        max_stability=float(max_stability),
        all_scores=[float(s) for s in stability_scores],
    )


def calculate_cluster_statistics(
    windows: np.ndarray,
    labels: np.ndarray,
    forward_returns: "pd.DataFrame",
) -> ClusterStatistics:
    """
    Calculate comprehensive statistics for each cluster.

    Computes per-cluster metrics including size, percentage, return distributions,
    win rates, and pattern variance. Useful for identifying profitable patterns
    and understanding cluster characteristics.

    Args:
        windows: Feature array (n_samples, window_size, n_features)
        labels: Cluster assignments (n_samples,)
        forward_returns: DataFrame with columns:
                        - return_5: 5-hour forward returns
                        - return_10: 10-hour forward returns
                        - return_20: 20-hour forward returns

    Returns:
        ClusterStatistics with per-cluster metrics dict

    Raises:
        ValueError: If windows, labels, and forward_returns have different lengths
        ValueError: If forward_returns missing required columns

    Example:
        >>> result = cluster_kmeans(windows, n_clusters=50)
        >>> stats = calculate_cluster_statistics(windows, result.labels, forward_returns)
        >>> cluster_0 = stats.stats[0]
        >>> print(f"Cluster 0: {cluster_0['size']} samples, "
        ...       f"mean return {cluster_0['mean_return_5h']:.2f}%, "
        ...       f"win rate {cluster_0['win_rate_5h']:.1%}")

    Statistics Computed:
        - size: Sample count
        - percentage: % of total
        - mean_return_*: Average returns (5h/10h/20h)
        - std_return_5h: Volatility
        - win_rate_*: % positive returns
        - median_return_5h: Median performance
        - pattern_variance: Pattern diversity
    """
    import pandas as pd

    # Validate input lengths
    if not (len(windows) == len(labels) == len(forward_returns)):
        raise ValueError(
            f"Length mismatch: windows={len(windows)}, "
            f"labels={len(labels)}, forward_returns={len(forward_returns)}"
        )

    # Validate forward_returns columns
    required_columns = {"return_5", "return_10", "return_20"}
    if not required_columns.issubset(forward_returns.columns):
        missing = required_columns - set(forward_returns.columns)
        raise ValueError(f"forward_returns missing columns: {missing}")

    logger.info(
        f"Calculating cluster statistics: "
        f"n_samples={len(windows)}, n_clusters={len(np.unique(labels))}"
    )

    # Calculate statistics for each cluster
    stats = {}
    unique_clusters = np.unique(labels)
    total_samples = len(labels)

    for cluster_id in unique_clusters:
        # Get samples belonging to this cluster
        mask = labels == cluster_id
        cluster_samples = windows[mask]
        cluster_returns = forward_returns[mask]

        # Size statistics
        size = mask.sum()
        percentage = 100.0 * size / total_samples

        # Return statistics
        mean_return_5h = cluster_returns["return_5"].mean()
        mean_return_10h = cluster_returns["return_10"].mean()
        mean_return_20h = cluster_returns["return_20"].mean()
        std_return_5h = cluster_returns["return_5"].std()
        median_return_5h = cluster_returns["return_5"].median()

        # Win rates (percentage of positive returns)
        win_rate_5h = (cluster_returns["return_5"] > 0).mean()
        win_rate_10h = (cluster_returns["return_10"] > 0).mean()

        # Pattern variance (average variance across all features)
        pattern_variance = cluster_samples.var(axis=0).mean()

        stats[int(cluster_id)] = {
            "size": int(size),
            "percentage": float(percentage),
            "mean_return_5h": float(mean_return_5h),
            "mean_return_10h": float(mean_return_10h),
            "mean_return_20h": float(mean_return_20h),
            "std_return_5h": float(std_return_5h),
            "win_rate_5h": float(win_rate_5h),
            "win_rate_10h": float(win_rate_10h),
            "median_return_5h": float(median_return_5h),
            "pattern_variance": float(pattern_variance),
        }

        logger.debug(
            f"Cluster {cluster_id}: size={size} ({percentage:.1f}%), "
            f"mean_return_5h={mean_return_5h:.3f}, win_rate={win_rate_5h:.2%}"
        )

    logger.info(f"Statistics calculated for {len(stats)} clusters")

    return ClusterStatistics(stats=stats)
