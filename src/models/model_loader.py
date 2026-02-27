"""
Model Loader Utility for Stable-Baselines3 Models.

This module provides ModelLoader class that:
- Loads trained PPO/DQN models from .zip files
- Supports model versioning and checkpoints
- Provides model metadata extraction
- Lists available models in directories

TDD Phase: GREEN - Minimal implementation to pass tests.

Issue: #249 (Stream A - Evaluation Environment & Model Loading)
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""

    PPO = "ppo"
    DQN = "dqn"


class ModelLoader:
    """
    Utility class for loading Stable-Baselines3 models.

    This class provides methods to:
    - Load trained models from .zip files
    - Extract model metadata
    - List available models

    Example:
        >>> loader = ModelLoader()
        >>> model = loader.load_model("models/ppo_h4.zip", ModelType.PPO)
        >>> info = loader.get_model_info("models/ppo_h4.zip")
    """

    def __init__(self):
        """Initialize ModelLoader."""
        logger.info("ModelLoader initialized")

    def load_model(
        self, model_path: str, model_type: ModelType | str
    ) -> BaseAlgorithm:
        """
        Load a trained RL model from disk.

        Args:
            model_path: Path to the saved model (.zip file)
            model_type: Type of model (ModelType enum or string 'ppo'/'dqn')

        Returns:
            Loaded Stable-Baselines3 model

        Raises:
            ValueError: If model_type is not supported
            FileNotFoundError: If model_path does not exist

        Example:
            >>> loader = ModelLoader()
            >>> model = loader.load_model("models/ppo_h4.zip", "ppo")
            >>> model = loader.load_model("models/dqn.zip", ModelType.DQN)
        """
        # Convert string to ModelType if necessary
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type.lower())
            except ValueError as e:
                raise ValueError(
                    f"Unsupported model type: {model_type}. "
                    f"Supported types: ppo, dqn"
                ) from e

        # Check if file exists
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading {model_type.value.upper()} model from {model_path}")

        # Load model based on type
        if model_type == ModelType.PPO:
            model = PPO.load(model_path)
            logger.info(f"Successfully loaded PPO model from {model_path}")
        elif model_type == ModelType.DQN:
            model = DQN.load(model_path)
            logger.info(f"Successfully loaded DQN model from {model_path}")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model

    def get_model_info(self, model_path: str) -> dict[str, Any]:
        """
        Extract metadata from model file.

        Args:
            model_path: Path to model file

        Returns:
            Dictionary with model metadata:
                - path: Full path to model
                - filename: Model filename
                - size_bytes: File size in bytes

        Example:
            >>> loader = ModelLoader()
            >>> info = loader.get_model_info("models/ppo_h4.zip")
            >>> print(f"Model size: {info['size_bytes']} bytes")
        """
        model_path_obj = Path(model_path)

        info = {
            "path": str(model_path_obj.absolute()),
            "filename": model_path_obj.name,
            "size_bytes": (
                model_path_obj.stat().st_size if model_path_obj.exists() else 0
            ),
        }

        return info

    def list_available_models(self, models_dir: str) -> list[str]:
        """
        List all available model files (.zip) in directory.

        Args:
            models_dir: Directory containing model files

        Returns:
            List of model file paths (relative to models_dir)

        Example:
            >>> loader = ModelLoader()
            >>> models = loader.list_available_models("models/")
            >>> for model in models:
            ...     print(f"Found model: {model}")
        """
        models_dir_obj = Path(models_dir)

        if not models_dir_obj.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return []

        # Find all .zip files
        model_files = list(models_dir_obj.glob("*.zip"))

        # Return relative paths as strings
        model_paths = [str(f.relative_to(models_dir_obj)) for f in model_files]

        logger.info(f"Found {len(model_paths)} models in {models_dir}")

        return model_paths
