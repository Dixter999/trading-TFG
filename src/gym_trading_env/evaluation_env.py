"""
Evaluation Environment Wrapper for Gym Trading Environments.

This module provides EvaluationEnvWrapper that:
- Wraps existing gym environments for evaluation mode
- Forces deterministic action selection (no exploration)
- Disables training-specific features (noise, randomness)
- Maintains full compatibility with Stable-Baselines3

TDD Phase: GREEN - Minimal implementation to pass tests.

Issue: #249 (Stream A - Evaluation Environment & Model Loading)
"""

import logging
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class EvaluationEnvWrapper(gym.Wrapper):
    """
    Environment wrapper for evaluation mode.

    This wrapper ensures that the environment operates in deterministic mode
    for model evaluation, disabling all sources of randomness that are used
    during training (exploration noise, action randomization, etc.).

    Attributes:
        deterministic: Always True - forces deterministic behavior
        training: Always False - disables training mode

    Example:
        >>> from src.gym_trading_env.mtf_trading_env import MTFTradingEnv
        >>> train_env = MTFTradingEnv(data)
        >>> eval_env = EvaluationEnvWrapper(train_env)
        >>> # Now eval_env will operate deterministically
    """

    def __init__(self, env: gym.Env):
        """
        Initialize evaluation wrapper.

        Args:
            env: The gym environment to wrap
        """
        super().__init__(env)

        # Set deterministic mode
        self.deterministic = True

        # Disable training mode
        self.training = False

        # Disable exploration noise if environment has it
        if hasattr(env, "exploration_noise"):
            self.exploration_noise = 0.0

        logger.info(
            f"EvaluationEnvWrapper initialized for {type(env).__name__} "
            f"(deterministic=True, training=False)"
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed (optional)
            options: Additional reset options (optional)

        Returns:
            Tuple of (observation, info dict)
            Info dict includes 'deterministic': True flag
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Add deterministic flag to info
        info["deterministic"] = True

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        return self.env.step(action)

    def seed(self, seed: int):
        """
        Set random seed for deterministic behavior.

        Args:
            seed: Random seed value
        """
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        logger.debug(f"Set evaluation seed to {seed}")

    def render(self):
        """
        Render the environment.

        Returns:
            Render output from wrapped environment
        """
        return self.env.render()

    def close(self):
        """Close the environment and release resources."""
        self.env.close()

    @property
    def metadata(self):
        """Get environment metadata."""
        return self.env.metadata
