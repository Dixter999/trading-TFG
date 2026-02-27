"""
TensorBoard configuration for stable-baselines3 integration.

TDD Phase: GREEN - Minimal implementation to pass tests.

This module provides TensorBoard logging configuration for RL training:
- Event file creation
- Training curve logging (loss, episode reward, episode length)
- Integration with stable-baselines3 callbacks
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Default metric names following stable-baselines3 conventions
DEFAULT_METRIC_NAMES = {
    "episode_reward": "rollout/ep_rew_mean",
    "episode_length": "rollout/ep_len_mean",
    "loss": "train/loss",
}


@dataclass
class TensorBoardConfig:
    """
    Configuration for TensorBoard logging.

    Attributes:
        log_dir: Base directory for TensorBoard logs
        experiment_name: Name of the experiment (subdirectory)
        run_name: Name of the specific run (subdirectory)
        metric_names: Custom metric names (defaults to stable-baselines3 conventions)
        full_log_path: Complete path where logs will be written (computed)
    """

    log_dir: str
    experiment_name: str
    run_name: str
    metric_names: dict[str, str] = field(
        default_factory=lambda: DEFAULT_METRIC_NAMES.copy()
    )

    def __post_init__(self):
        """
        Initialize configuration and create directory structure.

        Creates: log_dir/experiment_name/run_name/
        """
        # Compute full log path
        self.full_log_path = os.path.join(
            self.log_dir, self.experiment_name, self.run_name
        )

        # Create directory structure
        Path(self.full_log_path).mkdir(parents=True, exist_ok=True)

        logger.info(f"TensorBoard logs will be written to: {self.full_log_path}")


def create_tensorboard_callback(config: TensorBoardConfig):
    """
    Create TensorBoard callback for stable-baselines3.

    Args:
        config: TensorBoard configuration

    Returns:
        TensorBoard callback instance compatible with stable-baselines3

    Raises:
        ImportError: If stable-baselines3 is not installed
    """
    try:
        from stable_baselines3.common.callbacks import BaseCallback
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 and tensorboard are required for TensorBoard logging. "
            "Install with: pip install stable-baselines3 tensorboard"
        ) from e

    class TensorBoardCallback(BaseCallback):
        """
        Custom TensorBoard callback for detailed training logging.

        Logs:
        - Episode rewards (mean)
        - Episode lengths (mean)
        - Training loss
        - Custom metrics
        """

        def __init__(self, config: TensorBoardConfig, verbose: int = 0):
            super().__init__(verbose)
            self.config = config
            self.writer: SummaryWriter | None = None
            self.last_logged_episode_count = 0

        def _on_training_start(self) -> None:
            """
            Initialize TensorBoard writer when training starts.
            """
            self.writer = SummaryWriter(log_dir=self.config.full_log_path)
            logger.info(f"TensorBoard writer initialized: {self.config.full_log_path}")

        def _on_step(self) -> bool:
            """
            Log metrics at each training step.

            Returns:
                True to continue training
            """
            if self.writer is None:
                return True

            # Get current step
            current_step = self.num_timesteps

            # Log episode metrics if available
            if "episode_rewards" in self.locals:
                episode_rewards = self.locals.get("episode_rewards", [])
                if len(episode_rewards) > 0:
                    # Log mean episode reward
                    mean_reward = sum(episode_rewards) / len(episode_rewards)
                    self.writer.add_scalar(
                        self.config.metric_names["episode_reward"],
                        mean_reward,
                        current_step,
                    )

            # Log episode lengths if available
            if "episode_lengths" in self.locals:
                episode_lengths = self.locals.get("episode_lengths", [])
                if len(episode_lengths) > 0:
                    mean_length = sum(episode_lengths) / len(episode_lengths)
                    self.writer.add_scalar(
                        self.config.metric_names["episode_length"],
                        mean_length,
                        current_step,
                    )

            # Log training loss if available
            if "loss" in self.locals:
                loss = self.locals.get("loss")
                if loss is not None:
                    self.writer.add_scalar(
                        self.config.metric_names["loss"], loss, current_step
                    )

            # Flush writer periodically
            if current_step % 100 == 0:
                self.writer.flush()

            return True

        def _on_training_end(self) -> None:
            """
            Close TensorBoard writer when training ends.
            """
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
                logger.info("TensorBoard writer closed")

    return TensorBoardCallback(config)
