"""
MLflow callback for stable-baselines3 training.

Logs training parameters, episode metrics, and model checkpoints to MLflow.
"""

from datetime import datetime
from typing import Any

import mlflow
from stable_baselines3.common.callbacks import BaseCallback


class MLflowCallback(BaseCallback):
    """
    Custom callback for logging training metrics to MLflow.

    This callback extends stable-baselines3's BaseCallback to log:
    - Dataset split boundaries (train/validation dates) on training start
    - Normalization parameters on training start
    - Episode metrics (reward, length, PnL) on each completed episode
    - Model checkpoints at specified intervals

    Args:
        run_name: Optional name for the MLflow run
        experiment_name: Optional name for the MLflow experiment
        save_model_every_n_steps: Optional interval for saving model checkpoints
        verbose: Verbosity level (0: no output, 1: info, 2: debug)
    """

    def __init__(
        self,
        run_name: str | None = None,
        experiment_name: str | None = None,
        save_model_every_n_steps: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.save_model_every_n_steps = save_model_every_n_steps

    def _on_training_start(self) -> None:
        """
        Called when training starts.

        Logs dataset split boundaries and normalization parameters to MLflow.
        """
        env = self.training_env
        params = self._extract_training_params(env)

        if params:
            mlflow.log_params(params)

    def _extract_training_params(self, env) -> dict[str, Any]:
        """
        Extract training parameters from environment.

        Args:
            env: Training environment

        Returns:
            Dict of parameters to log
        """
        params = {}

        # Define datetime parameters that need formatting
        datetime_params = [
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
        ]
        for param_name in datetime_params:
            if hasattr(env, param_name):
                param_value = getattr(env, param_name)
                if param_value is not None:
                    params[param_name] = self._format_datetime(param_value)

        # Define numeric parameters that can be logged directly
        numeric_params = ["normalization_mean", "normalization_std"]
        for param_name in numeric_params:
            if hasattr(env, param_name):
                param_value = getattr(env, param_name)
                if param_value is not None:
                    params[param_name] = param_value

        return params

    def _on_step(self) -> bool:
        """
        Called on each step during training.

        Logs episode metrics when episodes complete and saves model checkpoints
        at specified intervals.

        Returns:
            True to continue training, False to stop
        """
        # Get dones and infos from locals (provided by stable-baselines3)
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        # Log metrics for each completed episode
        for done, info in zip(dones, infos, strict=False):
            if done and "episode" in info:
                metrics = self._extract_episode_metrics(info)
                if metrics:
                    mlflow.log_metrics(metrics, step=self.num_timesteps)

        # Save model checkpoint if interval is set and reached
        if self.save_model_every_n_steps is not None:
            if self.num_timesteps % self.save_model_every_n_steps == 0:
                self._save_model_checkpoint()

        # Continue training
        return True

    def _extract_episode_metrics(self, info: dict[str, Any]) -> dict[str, float]:
        """
        Extract episode metrics from info dict.

        Args:
            info: Info dict containing episode information

        Returns:
            Dict of metrics to log
        """
        metrics = {}

        # Extract standard episode metrics
        episode_info = info.get("episode", {})
        if "r" in episode_info:
            metrics["episode_reward"] = float(episode_info["r"])
        if "l" in episode_info:
            metrics["episode_length"] = float(episode_info["l"])

        # Extract custom trading metrics
        if "final_pnl" in info:
            metrics["episode_pnl"] = float(info["final_pnl"])

        return metrics

    def _save_model_checkpoint(self) -> None:
        """
        Save model checkpoint to local filesystem.

        The model is saved using stable-baselines3's .save() method,
        which saves the model weights and configuration to a zip file.

        Note: This currently saves to local filesystem. Future enhancement
        could log the model artifact directly to MLflow using mlflow.log_artifact()
        or mlflow.sklearn.log_model() for full MLflow model registry integration.
        """
        import tempfile

        # Create a temporary file for the model checkpoint
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_step_{self.num_timesteps}.zip"
        ) as tmp_file:
            checkpoint_path = tmp_file.name

        # Save the model using stable-baselines3's save method
        # The .save() method is provided by all SB3 models (PPO, DQN, A2C, etc.)
        self.model.save(checkpoint_path)

        # Log the model artifact to MLflow (optional - can be enabled later)
        # mlflow.log_artifact(checkpoint_path, artifact_path="models")

        # Clean up temporary file (optional - keep for debugging)
        # Path(checkpoint_path).unlink()

    def _format_datetime(self, dt: datetime) -> str:
        """
        Format datetime as ISO 8601 string.

        Args:
            dt: Datetime to format

        Returns:
            ISO 8601 formatted string
        """
        return dt.isoformat()
