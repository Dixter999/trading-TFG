"""
MLflow configuration for experiment tracking.

Provides centralized configuration for MLflow tracking server,
artifact storage, and experiment management.
"""

import os


class MLflowConfig:
    """
    Configuration for MLflow tracking server.

    Provides tracking URI, artifact location, and default experiment settings
    for logging training runs, metrics, and model artifacts.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
        default_experiment: str | None = None,
    ):
        """
        Initialize MLflow configuration.

        Args:
            tracking_uri: MLflow tracking server URI (defaults to https://mlflow.local.pro4.es/)
            artifact_location: Path for artifact storage (defaults to ./mlruns)
            default_experiment: Default experiment name (defaults to gym-trading-env)
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "https://mlflow.local.pro4.es/"
        )

        self.artifact_location = artifact_location or os.getenv(
            "MLFLOW_ARTIFACT_LOCATION", "./mlruns"
        )

        self.default_experiment = default_experiment or os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "gym-trading-env"
        )

    def __repr__(self) -> str:
        return (
            f"MLflowConfig("
            f"tracking_uri={self.tracking_uri}, "
            f"artifact_location={self.artifact_location}, "
            f"default_experiment={self.default_experiment})"
        )
