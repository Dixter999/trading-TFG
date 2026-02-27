"""
Episode tracking callback for stable-baselines3 training.

TDD Phase: GREEN - Minimal implementation to pass tests.

Tracks episode metrics during training and provides summary statistics.
"""

import csv
import logging
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class EpisodeTrackerCallback(BaseCallback):
    """
    Custom callback for tracking episode metrics during training.

    This callback extends stable-baselines3's BaseCallback to track:
    - Episode reward (cumulative reward for entire episode)
    - Episode length (number of steps in episode)
    - Timestep when episode completed
    - Custom metrics from environment (e.g., final PnL)

    The tracked episodes can be:
    - Exported to CSV for analysis
    - Summarized with mean/std statistics
    - Accessed directly via the episodes attribute

    Args:
        verbose: Verbosity level (0: no output, 1: info, 2: debug)

    Attributes:
        episodes: List of episode data dictionaries
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episodes: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        """
        Called on each step during training.

        Records episode data when episodes complete.

        Returns:
            True to continue training, False to stop
        """
        # Get dones and infos from locals (provided by stable-baselines3)
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        # Track metrics for each completed episode
        for done, info in zip(dones, infos, strict=False):
            if done and "episode" in info:
                episode_data = self._extract_episode_data(info)
                self.episodes.append(episode_data)

                if self.verbose >= 1:
                    logger.info(
                        f"Episode {len(self.episodes)}: "
                        f"reward={episode_data.get('episode_reward', 0):.2f}, "
                        f"length={episode_data.get('episode_length', 0)}"
                    )

        # Continue training
        return True

    def _extract_episode_data(self, info: dict[str, Any]) -> dict[str, Any]:
        """
        Extract episode data from info dict.

        Args:
            info: Info dict containing episode information

        Returns:
            Dict of episode data with standard and custom metrics
        """
        episode_data = {
            "timestep": self.num_timesteps,
        }

        # Extract standard episode metrics from SB3
        episode_info = info.get("episode", {})
        if "r" in episode_info:
            episode_data["episode_reward"] = float(episode_info["r"])
        if "l" in episode_info:
            episode_data["episode_length"] = int(episode_info["l"])

        # Extract custom trading metrics
        if "final_pnl" in info:
            episode_data["final_pnl"] = float(info["final_pnl"])

        if "sharpe_ratio" in info:
            episode_data["sharpe_ratio"] = float(info["sharpe_ratio"])

        if "max_drawdown" in info:
            episode_data["max_drawdown"] = float(info["max_drawdown"])

        return episode_data

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics for tracked episodes.

        Returns:
            Dict with summary statistics:
            - total_episodes: Number of episodes tracked
            - mean_reward: Mean episode reward
            - std_reward: Standard deviation of episode reward
            - mean_length: Mean episode length
            - std_length: Standard deviation of episode length
            - min_reward: Minimum episode reward
            - max_reward: Maximum episode reward
        """
        if not self.episodes:
            return {
                "total_episodes": 0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_length": 0.0,
                "std_length": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
            }

        rewards = [ep.get("episode_reward", 0) for ep in self.episodes]
        lengths = [ep.get("episode_length", 0) for ep in self.episodes]

        # Calculate statistics
        import statistics

        summary = {
            "total_episodes": len(self.episodes),
            "mean_reward": statistics.mean(rewards) if rewards else 0.0,
            "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            "mean_length": statistics.mean(lengths) if lengths else 0.0,
            "std_length": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
        }

        return summary

    def export_to_csv(self, filepath: str) -> None:
        """
        Export episode data to CSV file.

        Args:
            filepath: Path to CSV file to write

        Raises:
            ValueError: If no episodes have been tracked
        """
        if not self.episodes:
            raise ValueError("No episodes to export")

        # Get all unique keys from all episodes
        all_keys = set()
        for episode in self.episodes:
            all_keys.update(episode.keys())

        # Sort keys for consistent column order
        fieldnames = sorted(all_keys)

        # Write CSV
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.episodes)

        logger.info(f"Exported {len(self.episodes)} episodes to {filepath}")
