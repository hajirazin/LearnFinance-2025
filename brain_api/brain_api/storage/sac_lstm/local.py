"""Local filesystem storage for SAC + LSTM model artifacts.

SAC stores more artifacts than PPO due to twin critics and target networks.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.portfolio_rl.sac_networks import GaussianActor, TwinCritic
from brain_api.core.sac_lstm.config import SACLSTMConfig
from brain_api.storage.base import DEFAULT_DATA_PATH


@dataclass
class SACLSTMArtifacts:
    """Loaded SAC + LSTM model artifacts for inference.

    Contains everything needed to run inference:
    - config: SAC hyperparameters
    - scaler: fitted PortfolioScaler for state normalization
    - actor: SAC actor network
    - critic: SAC twin critic networks (for evaluation)
    - critic_target: SAC target critic networks
    - log_alpha: entropy coefficient
    - symbol_order: ordered list of symbols
    - version: the version string these artifacts came from
    """

    config: SACLSTMConfig
    scaler: PortfolioScaler
    actor: GaussianActor
    critic: TwinCritic
    critic_target: TwinCritic
    log_alpha: torch.Tensor
    symbol_order: list[str]
    version: str


class SACLSTMLocalStorage:
    """Local filesystem storage for SAC + LSTM model artifacts.

    Artifacts are stored under:
        {base_path}/models/sac_lstm/{version}/
            - actor.pt              (Actor network weights)
            - critic.pt             (Twin critic weights)
            - critic_target.pt      (Target critic weights)
            - log_alpha.pt          (Entropy coefficient)
            - scaler.pkl            (PortfolioScaler)
            - config.json           (SAC hyperparameters)
            - symbol_order.json     (ordered list of symbols)
            - metadata.json         (training info, metrics, data window)

    The current version pointer is stored at:
        {base_path}/models/sac_lstm/current
    """

    def __init__(self, base_path: Path | str | None = None):
        """Initialize storage.

        Args:
            base_path: Base path for data storage. Defaults to 'data/'.
        """
        if base_path is None:
            base_path = DEFAULT_DATA_PATH
        self.base_path = Path(base_path)
        self._model_path = self.base_path / "models" / "sac_lstm"

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return "sac_lstm"

    def _version_path(self, version: str) -> Path:
        """Get the path for a specific version."""
        return self._model_path / version

    def version_exists(self, version: str) -> bool:
        """Check if a version already exists."""
        return self._version_path(version).exists()

    def write_artifacts(
        self,
        version: str,
        actor: GaussianActor,
        critic: TwinCritic,
        critic_target: TwinCritic,
        log_alpha: torch.Tensor,
        scaler: PortfolioScaler,
        config: SACLSTMConfig,
        symbol_order: list[str],
        metadata: dict[str, Any],
    ) -> Path:
        """Write model artifacts for a version.

        Args:
            version: Version string (e.g., 'v2025-01-08-abc123')
            actor: Trained SAC actor network
            critic: Trained twin critic networks
            critic_target: Target critic networks
            log_alpha: Entropy coefficient tensor
            scaler: Fitted PortfolioScaler for state normalization
            config: SAC configuration
            symbol_order: Ordered list of symbols
            metadata: Training metadata (includes window, metrics, etc.)

        Returns:
            Path to the version directory
        """
        version_dir = self._version_path(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save actor weights
        actor_path = version_dir / "actor.pt"
        torch.save(actor.state_dict(), actor_path)

        # Save critic weights
        critic_path = version_dir / "critic.pt"
        torch.save(critic.state_dict(), critic_path)

        # Save target critic weights
        critic_target_path = version_dir / "critic_target.pt"
        torch.save(critic_target.state_dict(), critic_target_path)

        # Save log_alpha
        log_alpha_path = version_dir / "log_alpha.pt"
        torch.save(log_alpha, log_alpha_path)

        # Save scaler
        scaler_path = version_dir / "scaler.pkl"
        scaler.save(scaler_path)

        # Save config
        config_path = version_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # Save symbol order
        symbol_order_path = version_dir / "symbol_order.json"
        with open(symbol_order_path, "w") as f:
            json.dump(symbol_order, f, indent=2)

        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return version_dir

    def read_current_version(self) -> str | None:
        """Read the current version pointer.

        Returns:
            Version string if current exists, None otherwise.
        """
        current_file = self._model_path / "current"
        if not current_file.exists():
            return None
        return current_file.read_text().strip()

    def read_metadata(self, version: str) -> dict[str, Any] | None:
        """Read metadata for a version.

        Args:
            version: Version string

        Returns:
            Metadata dict if exists, None otherwise.
        """
        metadata_path = self._version_path(version) / "metadata.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path) as f:
            return json.load(f)

    def promote_version(self, version: str) -> None:
        """Atomically promote a version to current.

        Args:
            version: Version string to promote
        """
        self._model_path.mkdir(parents=True, exist_ok=True)

        current_file = self._model_path / "current"

        # Atomic write
        fd, temp_path = tempfile.mkstemp(
            dir=self._model_path,
            prefix=".current_",
            suffix=".tmp",
        )
        try:
            os.write(fd, version.encode())
            os.close(fd)
            os.rename(temp_path, current_file)
        except Exception:
            os.close(fd) if fd else None
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def load_config(self, version: str) -> SACLSTMConfig:
        """Load model configuration for a version.

        Args:
            version: Version string

        Returns:
            SACLSTMConfig reconstructed from config.json
        """
        config_path = self._version_path(version) / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        return SACLSTMConfig.from_dict(config_dict)

    def load_scaler(self, version: str) -> PortfolioScaler:
        """Load fitted scaler for a version.

        Args:
            version: Version string

        Returns:
            PortfolioScaler loaded from pickle.
        """
        scaler_path = self._version_path(version) / "scaler.pkl"
        return PortfolioScaler.load(scaler_path)

    def load_symbol_order(self, version: str) -> list[str]:
        """Load symbol order for a version.

        Args:
            version: Version string

        Returns:
            Ordered list of symbols.
        """
        symbol_order_path = self._version_path(version) / "symbol_order.json"
        with open(symbol_order_path) as f:
            return json.load(f)

    def load_artifacts(self, version: str) -> SACLSTMArtifacts:
        """Load all artifacts for a version.

        Args:
            version: Version string

        Returns:
            SACLSTMArtifacts containing all loaded components.
        """
        config = self.load_config(version)
        scaler = self.load_scaler(version)
        symbol_order = self.load_symbol_order(version)

        # Compute state dimension from config
        n_stocks = len(symbol_order)
        from brain_api.core.portfolio_rl.state import StateSchema
        schema = StateSchema(n_stocks=n_stocks)
        state_dim = schema.state_dim
        action_dim = n_stocks + 1  # stocks + CASH

        # Initialize actor
        actor = GaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )

        # Load actor weights
        actor_path = self._version_path(version) / "actor.pt"
        actor.load_state_dict(torch.load(actor_path, weights_only=True))
        actor.eval()

        # Initialize and load critics
        critic = TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )
        critic_path = self._version_path(version) / "critic.pt"
        critic.load_state_dict(torch.load(critic_path, weights_only=True))
        critic.eval()

        critic_target = TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )
        critic_target_path = self._version_path(version) / "critic_target.pt"
        critic_target.load_state_dict(torch.load(critic_target_path, weights_only=True))
        critic_target.eval()

        # Load log_alpha
        log_alpha_path = self._version_path(version) / "log_alpha.pt"
        log_alpha = torch.load(log_alpha_path, weights_only=True)

        return SACLSTMArtifacts(
            config=config,
            scaler=scaler,
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            log_alpha=log_alpha,
            symbol_order=symbol_order,
            version=version,
        )

    def load_current_artifacts(self) -> SACLSTMArtifacts:
        """Load artifacts for the current version.

        Returns:
            SACLSTMArtifacts for current version.

        Raises:
            ValueError: If no current version is set.
        """
        version = self.read_current_version()
        if version is None:
            raise ValueError("No current SAC LSTM model version available")
        return self.load_artifacts(version)


def create_sac_lstm_metadata(
    version: str,
    data_window_start: str,
    data_window_end: str,
    symbols: list[str],
    config: SACLSTMConfig,
    promoted: bool,
    prior_version: str | None,
    actor_loss: float,
    critic_loss: float,
    avg_episode_return: float,
    avg_episode_sharpe: float,
    eval_sharpe: float,
    eval_cagr: float,
    eval_max_drawdown: float,
) -> dict[str, Any]:
    """Create metadata dictionary for SAC + LSTM model.

    Args:
        version: Model version string
        data_window_start: Training data start date (ISO format)
        data_window_end: Training data end date (ISO format)
        symbols: List of symbols used
        config: SAC configuration
        promoted: Whether this version was promoted
        prior_version: Previous version (if any)
        actor_loss: Final actor loss
        critic_loss: Final critic loss
        avg_episode_return: Average episode return
        avg_episode_sharpe: Average episode Sharpe
        eval_sharpe: Evaluation Sharpe ratio
        eval_cagr: Evaluation CAGR
        eval_max_drawdown: Evaluation max drawdown

    Returns:
        Metadata dictionary.
    """
    return {
        "model_type": "sac_lstm",
        "version": version,
        "training_timestamp": datetime.utcnow().isoformat(),
        "data_window": {
            "start": data_window_start,
            "end": data_window_end,
        },
        "symbols": symbols,
        "config": config.to_dict(),
        "promoted": promoted,
        "prior_version": prior_version,
        "metrics": {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "avg_episode_return": avg_episode_return,
            "avg_episode_sharpe": avg_episode_sharpe,
            "eval_sharpe": eval_sharpe,
            "eval_cagr": eval_cagr,
            "eval_max_drawdown": eval_max_drawdown,
        },
    }

