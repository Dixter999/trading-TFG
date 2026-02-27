"""
Callbacks module for gym_trading_env.

Provides integration with stable-baselines3 callbacks for:
- TensorBoard logging
- MLflow tracking
- Episode tracking and CSV export
- Custom metrics logging
"""

from gym_trading_env.callbacks.episode_tracker import EpisodeTrackerCallback
from gym_trading_env.callbacks.tensorboard_config import (
    TensorBoardConfig,
    create_tensorboard_callback,
)

# Import MLflowCallback if available (Stream A)
try:
    from gym_trading_env.callbacks.mlflow_callback import MLflowCallback

    __all__ = [
        "EpisodeTrackerCallback",
        "TensorBoardConfig",
        "create_tensorboard_callback",
        "MLflowCallback",
    ]
except ImportError:
    __all__ = [
        "EpisodeTrackerCallback",
        "TensorBoardConfig",
        "create_tensorboard_callback",
    ]
