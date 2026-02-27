"""
Feature Engineering Utilities for Pattern-Aware RL Training.

This module provides utility functions for feature extraction, normalization,
and analysis of the 26-feature observation space.

TDD Phase: GREEN - Implementing utilities

Functions:
- calculate_feature_correlation: Compute correlation matrix from observation history
- normalize_feature_vector: Batch normalization for multiple observations
- validate_observation: Check observation vector for issues (NaN, Inf, range)
"""


import numpy as np
import pandas as pd


def calculate_feature_correlation(
    observations: np.ndarray, feature_names: list[str] | None = None
) -> pd.DataFrame:
    """
    Calculate correlation matrix for features from observation history.

    This helps identify highly correlated features that might be redundant
    for RL training.

    Args:
        observations: Array of shape (n_samples, 26) with observation history
        feature_names: Optional list of 26 feature names for column labels

    Returns:
        DataFrame with correlation matrix (26x26)

    Example:
        >>> observations = np.random.randn(1000, 26)
        >>> corr_matrix = calculate_feature_correlation(observations)
        >>> print(corr_matrix.shape)
        (26, 26)
        >>> print(corr_matrix.iloc[0, 0])  # Should be 1.0 (self-correlation)
        1.0
    """
    if observations.ndim != 2:
        raise ValueError(
            f"observations must be 2D array, got shape {observations.shape}"
        )

    if observations.shape[1] != 26:
        raise ValueError(
            f"observations must have 26 features, got {observations.shape[1]}"
        )

    # Calculate correlation matrix using numpy
    corr_matrix = np.corrcoef(observations.T)

    # Convert to DataFrame with feature names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(26)]

    corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)

    return corr_df


def validate_observation(observation: np.ndarray) -> tuple[bool, str]:
    """
    Validate observation vector for common issues.

    Checks for:
    - Correct shape (26,)
    - Correct dtype (float32)
    - No NaN values
    - No Inf values
    - Pattern flags are binary (indices 16-21)

    Args:
        observation: Observation vector to validate

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if observation is valid, False otherwise
        - error_message: Empty string if valid, error description otherwise

    Example:
        >>> obs = np.random.randn(26).astype(np.float32)
        >>> is_valid, error = validate_observation(obs)
        >>> print(is_valid)
        True
        >>> print(error)
        ""
    """
    # Check shape
    if observation.shape != (26,):
        return False, f"Invalid shape: expected (26,), got {observation.shape}"

    # Check dtype
    if observation.dtype != np.float32:
        return False, f"Invalid dtype: expected float32, got {observation.dtype}"

    # Check for NaN
    if np.any(np.isnan(observation)):
        nan_indices = np.where(np.isnan(observation))[0]
        return False, f"Contains NaN at indices: {nan_indices.tolist()}"

    # Check for Inf
    if np.any(np.isinf(observation)):
        inf_indices = np.where(np.isinf(observation))[0]
        return False, f"Contains Inf at indices: {inf_indices.tolist()}"

    # Check pattern flags are binary (indices 16-21)
    pattern_flags = observation[16:22]
    if not np.all((pattern_flags == 0.0) | (pattern_flags == 1.0)):
        invalid_indices = (
            np.where((pattern_flags != 0.0) & (pattern_flags != 1.0))[0] + 16
        )
        return False, f"Pattern flags not binary at indices: {invalid_indices.tolist()}"

    return True, ""


def normalize_feature_vector(
    features: np.ndarray, method: str = "standardize"
) -> np.ndarray:
    """
    Normalize feature vector using specified method.

    This is useful for batch normalization of observation history.

    Args:
        features: Array of shape (n_samples, 26) with feature vectors
        method: Normalization method:
            - "standardize": Z-score normalization (mean=0, std=1)
            - "minmax": Min-max normalization to [0, 1]
            - "robust": Robust scaling using median and IQR

    Returns:
        Normalized feature array of same shape

    Example:
        >>> features = np.random.randn(100, 26)
        >>> normalized = normalize_feature_vector(features, method="standardize")
        >>> print(normalized.shape)
        (100, 26)
        >>> print(np.mean(normalized, axis=0))  # Should be close to 0
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D array, got shape {features.shape}")

    if features.shape[1] != 26:
        raise ValueError(f"features must have 26 columns, got {features.shape[1]}")

    if method == "standardize":
        # Z-score normalization
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
        normalized = (features - mean) / std

    elif method == "minmax":
        # Min-max normalization to [0, 1]
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1.0, range_val)
        normalized = (features - min_val) / range_val

    elif method == "robust":
        # Robust scaling using median and IQR
        median = np.median(features, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        q25 = np.percentile(features, 25, axis=0)
        iqr = q75 - q25
        # Avoid division by zero
        iqr = np.where(iqr == 0, 1.0, iqr)
        normalized = (features - median) / iqr

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def get_feature_statistics(observations: np.ndarray) -> pd.DataFrame:
    """
    Calculate summary statistics for each feature in observation history.

    Args:
        observations: Array of shape (n_samples, 26) with observation history

    Returns:
        DataFrame with statistics for each feature:
        - mean, std, min, max, median, q25, q75

    Example:
        >>> observations = np.random.randn(1000, 26)
        >>> stats = get_feature_statistics(observations)
        >>> print(stats.shape)
        (26, 7)
        >>> print(stats.columns)
        ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
    """
    if observations.ndim != 2:
        raise ValueError(
            f"observations must be 2D array, got shape {observations.shape}"
        )

    if observations.shape[1] != 26:
        raise ValueError(
            f"observations must have 26 features, got {observations.shape[1]}"
        )

    stats = {
        "mean": np.mean(observations, axis=0),
        "std": np.std(observations, axis=0),
        "min": np.min(observations, axis=0),
        "max": np.max(observations, axis=0),
        "median": np.median(observations, axis=0),
        "q25": np.percentile(observations, 25, axis=0),
        "q75": np.percentile(observations, 75, axis=0),
    }

    stats_df = pd.DataFrame(stats)
    stats_df.index = [f"feature_{i}" for i in range(26)]

    return stats_df
