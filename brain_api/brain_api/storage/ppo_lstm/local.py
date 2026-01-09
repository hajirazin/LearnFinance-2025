"""Local filesystem storage for PPO + LSTM model artifacts.

PPO uses PortfolioScaler instead of StandardScaler, so we can't directly
use the base class. This storage handles PPO-specific artifact structure.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.ppo_lstm.config import PPOLSTMConfig
from brain_api.core.ppo_lstm.model import PPOActorCritic
from brain_api.storage.base import DEFAULT_DATA_PATH


@dataclass
class PPOLSTMArtifacts:
    """Loaded PPO + LSTM model artifacts for inference.

    Contains everything needed to run inference:
    - config: PPO hyperparameters
    - scaler: fitted PortfolioScaler for state normalization
    - model: PPO actor-critic model with loaded weights
    - symbol_order: ordered list of symbols
    - version: the version string these artifacts came from
    """

    config: PPOLSTMConfig
    scaler: PortfolioScaler
    model: PPOActorCritic
    symbol_order: list[str]
    version: str


class PPOLSTMLocalStorage:
    """Local filesystem storage for PPO + LSTM model artifacts.

    Artifacts are stored under:
        {base_path}/models/ppo_lstm/{version}/
            - weights.pt            (PyTorch model weights)
            - scaler.pkl            (PortfolioScaler)
            - config.json           (PPO hyperparameters)
            - symbol_order.json     (ordered list of symbols)
            - metadata.json         (training info, metrics, data window)

    The current version pointer is stored at:
        {base_path}/models/ppo_lstm/current
    """

    def __init__(self, base_path: Path | str | None = None):
        """Initialize storage.

        Args:
            base_path: Base path for data storage. Defaults to 'data/'.
        """
        if base_path is None:
            base_path = DEFAULT_DATA_PATH
        self.base_path = Path(base_path)
        self._model_path = self.base_path / "models" / "ppo_lstm"

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return "ppo_lstm"

    def _version_path(self, version: str) -> Path:
        """Get the path for a specific version."""
        return self._model_path / version

    def version_exists(self, version: str) -> bool:
        """Check if a version already exists."""
        return self._version_path(version).exists()

    def write_artifacts(
        self,
        version: str,
        model: PPOActorCritic,
        scaler: PortfolioScaler,
        config: PPOLSTMConfig,
        symbol_order: list[str],
        metadata: dict[str, Any],
    ) -> Path:
        """Write model artifacts for a version.

        Args:
            version: Version string (e.g., 'v2025-01-08-abc123')
            model: Trained PPO actor-critic model
            scaler: Fitted PortfolioScaler for state normalization
            config: PPO configuration
            symbol_order: Ordered list of symbols
            metadata: Training metadata (includes window, metrics, etc.)

        Returns:
            Path to the version directory
        """
        version_dir = self._version_path(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        weights_path = version_dir / "weights.pt"
        torch.save(model.state_dict(), weights_path)

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

    def load_config(self, version: str) -> PPOLSTMConfig:
        """Load model configuration for a version.

        Args:
            version: Version string

        Returns:
            PPOLSTMConfig reconstructed from config.json
        """
        config_path = self._version_path(version) / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        return PPOLSTMConfig.from_dict(config_dict)

    def load_scaler(self, version: str) -> PortfolioScaler:
        """Load fitted scaler for a version.

        Args:
            version: Version string

        Returns:
            PortfolioScaler loaded from scaler.pkl
        """
        scaler_path = self._version_path(version) / "scaler.pkl"
        return PortfolioScaler.load(scaler_path)

    def load_symbol_order(self, version: str) -> list[str]:
        """Load symbol order for a version.

        Args:
            version: Version string

        Returns:
            Ordered list of symbols
        """
        symbol_order_path = self._version_path(version) / "symbol_order.json"
        with open(symbol_order_path) as f:
            return json.load(f)

    def load_model(
        self,
        version: str,
        config: PPOLSTMConfig | None = None,
        symbol_order: list[str] | None = None,
    ) -> PPOActorCritic:
        """Load trained model for a version.

        Args:
            version: Version string
            config: Optional config; if not provided, will load from config.json
            symbol_order: Optional symbol order; if not provided, will load from file

        Returns:
            PPOActorCritic with weights loaded
        """
        if config is None:
            config = self.load_config(version)
        if symbol_order is None:
            symbol_order = self.load_symbol_order(version)

        weights_path = self._version_path(version) / "weights.pt"

        # Compute state/action dimensions
        from brain_api.core.portfolio_rl.state import StateSchema

        schema = StateSchema(n_stocks=len(symbol_order))

        model = PPOActorCritic(
            state_dim=schema.state_dim,
            action_dim=len(symbol_order) + 1,  # +1 for CASH
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        return model

    def load_current_artifacts(self) -> PPOLSTMArtifacts:
        """Load all artifacts for the current promoted version.

        Returns:
            PPOLSTMArtifacts containing everything needed for inference

        Raises:
            ValueError: if no current version is set
        """
        version = self.read_current_version()
        if version is None:
            raise ValueError(
                "No current PPO_LSTM version set. "
                "Train a model first with POST /train/ppo_lstm"
            )

        config = self.load_config(version)
        scaler = self.load_scaler(version)
        symbol_order = self.load_symbol_order(version)
        model = self.load_model(version, config, symbol_order)

        return PPOLSTMArtifacts(
            config=config,
            scaler=scaler,
            model=model,
            symbol_order=symbol_order,
            version=version,
        )
