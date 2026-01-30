"""HuggingFace Hub storage for PPO model artifacts (unified with dual forecasts).

PPO requires storing the actor-critic model plus scaler, config, and symbol order.
"""

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

from brain_api.core.config import get_hf_ppo_model_repo, get_hf_token
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.ppo.config import PPOConfig
from brain_api.core.ppo.model import PPOActorCritic
from brain_api.storage.ppo.local import (
    PPOArtifacts,
    PPOLocalStorage,
)

logger = logging.getLogger(__name__)

__all__ = ["HFModelInfo", "PPOHuggingFaceModelStorage"]


@dataclass
class HFModelInfo:
    """Information about a model stored on HuggingFace Hub."""

    repo_id: str
    version: str
    revision: str  # Git commit/tag on HF


class PPOHuggingFaceModelStorage:
    """HuggingFace Hub storage for PPO model artifacts.

    PPO stores:
        - weights.pt            (PPO actor-critic network weights)
        - scaler.pkl            (PortfolioScaler for state normalization)
        - config.json           (PPO hyperparameters)
        - symbol_order.json     (Ordered list of symbols)
        - metadata.json         (Training info, metrics, data window)

    Versions are managed as git tags/branches on the HF repo.
    The 'main' branch typically points to the current promoted version.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        token: str | None = None,
        local_cache: PPOLocalStorage | None = None,
    ):
        """Initialize PPO HuggingFace model storage.

        Args:
            repo_id: HuggingFace repo ID. Defaults to HF_PPO_MODEL_REPO env var.
            token: HuggingFace API token.
            local_cache: Optional local storage for caching downloaded models.
        """
        self.repo_id = repo_id or get_hf_ppo_model_repo()
        self.token = token or get_hf_token()
        self.local_cache = local_cache or PPOLocalStorage()
        self.api = HfApi(token=self.token)

        if not self.repo_id:
            raise ValueError(
                "HuggingFace PPO model repo not configured. "
                "Set HF_PPO_MODEL_REPO environment variable or pass repo_id."
            )

    @property
    def model_type(self) -> str:
        return "ppo"

    def _ensure_repo_exists(self) -> None:
        """Create the HF repo if it doesn't exist."""
        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type="model")
        except RepositoryNotFoundError:
            logger.info(f"Creating HuggingFace model repo: {self.repo_id}")
            self.api.create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                exist_ok=True,
            )

    def upload_model(
        self,
        version: str,
        model: PPOActorCritic,
        scaler: PortfolioScaler,
        config: PPOConfig,
        symbol_order: list[str],
        metadata: dict[str, Any],
        make_current: bool = False,
    ) -> HFModelInfo:
        """Upload PPO model artifacts to HuggingFace Hub.

        Args:
            version: Version string (e.g., 'v2025-01-08-abc123')
            model: Trained PPO actor-critic model
            scaler: Fitted PortfolioScaler for state normalization
            config: PPO configuration
            symbol_order: Ordered list of symbols
            metadata: Training metadata
            make_current: If True, also update 'main' branch to point to this version

        Returns:
            HFModelInfo with upload details
        """
        self._ensure_repo_exists()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Save model weights
            weights_path = tmppath / "weights.pt"
            torch.save(model.state_dict(), weights_path)

            # Save scaler
            scaler_path = tmppath / "scaler.pkl"
            scaler.save(scaler_path)

            # Save config
            config_path = tmppath / "config.json"
            with open(config_path, "w") as f:
                json.dump(config.to_dict(), f, indent=2)

            # Save symbol order
            symbol_order_path = tmppath / "symbol_order.json"
            with open(symbol_order_path, "w") as f:
                json.dump(symbol_order, f, indent=2)

            # Save metadata
            metadata_path = tmppath / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # Create README
            readme_path = tmppath / "README.md"
            readme_content = self._generate_readme(version, metadata, symbol_order)
            with open(readme_path, "w") as f:
                f.write(readme_content)

            # Upload all files to the version branch/tag
            logger.info(f"Uploading PPO model {version} to {self.repo_id}")

            # Create the version branch first
            try:
                self.api.create_branch(
                    repo_id=self.repo_id,
                    repo_type="model",
                    branch=version,
                )
                logger.info(f"Created branch {version} on {self.repo_id}")
            except Exception as e:
                # Branch may already exist, which is fine
                if (
                    "already exists" not in str(e).lower()
                    and "reference already exists" not in str(e).lower()
                ):
                    logger.warning(f"Could not create branch {version}: {e}")

            self.api.upload_folder(
                folder_path=tmpdir,
                repo_id=self.repo_id,
                repo_type="model",
                revision=version,
                commit_message=f"Add PPO model version {version}",
            )

            if make_current:
                logger.info(f"Setting {version} as current (main branch)")
                self.api.upload_folder(
                    folder_path=tmpdir,
                    repo_id=self.repo_id,
                    repo_type="model",
                    revision="main",
                    commit_message=f"Promote PPO version {version} to current",
                )

        return HFModelInfo(
            repo_id=self.repo_id,
            version=version,
            revision=version,
        )

    def download_model(
        self,
        version: str | None = None,
        use_cache: bool = True,
    ) -> PPOArtifacts:
        """Download PPO model artifacts from HuggingFace Hub.

        Args:
            version: Version string to download. If None, downloads 'main' (current).
            use_cache: If True, check local cache first and update it after download.

        Returns:
            PPOArtifacts ready for inference
        """
        revision = version or "main"

        # Check local cache first
        if use_cache and version and self.local_cache.version_exists(version):
            logger.info(f"Loading PPO model {version} from local cache")
            return self.local_cache.load_current_artifacts()

        logger.info(f"Downloading PPO model from {self.repo_id} (revision: {revision})")

        # Download all files to a temp directory
        local_dir = snapshot_download(
            repo_id=self.repo_id,
            revision=revision,
            repo_type="model",
            token=self.token,
        )
        local_path = Path(local_dir)

        # Load config
        config_path = local_path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        config = PPOConfig.from_dict(config_dict)

        # Load symbol order
        symbol_order_path = local_path / "symbol_order.json"
        with open(symbol_order_path) as f:
            symbol_order = json.load(f)

        # Load scaler
        scaler_path = local_path / "scaler.pkl"
        scaler = PortfolioScaler.load(scaler_path)

        # Compute dimensions from symbol order
        n_stocks = len(symbol_order)
        from brain_api.core.portfolio_rl.state import StateSchema

        schema = StateSchema(n_stocks=n_stocks)
        state_dim = schema.state_dim
        action_dim = n_stocks + 1  # stocks + CASH

        # Initialize and load model
        model = PPOActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )
        weights_path = local_path / "weights.pt"
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()

        # Determine actual version from metadata if we downloaded 'main'
        actual_version = version
        if not version:
            metadata_path = local_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    actual_version = metadata.get("version", "main")

        # Cache locally if we have a version
        if use_cache and actual_version and actual_version != "main":
            logger.info(f"Caching PPO model {actual_version} locally")
            metadata_path = local_path / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

            self.local_cache.write_artifacts(
                version=actual_version,
                model=model,
                scaler=scaler,
                config=config,
                symbol_order=symbol_order,
                metadata=metadata,
            )
            self.local_cache.promote_version(actual_version)

        return PPOArtifacts(
            config=config,
            scaler=scaler,
            model=model,
            symbol_order=symbol_order,
            version=actual_version or "main",
        )

    def get_current_version(self) -> str | None:
        """Get the current (main branch) version from HF Hub."""
        try:
            metadata_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="metadata.json",
                repo_type="model",
                token=self.token,
            )
            with open(metadata_path) as f:
                metadata = json.load(f)
            return metadata.get("version")
        except Exception:
            return None

    def get_current_metadata(self) -> dict[str, Any] | None:
        """Get metadata for current (main branch) version without downloading model.

        This is useful for checking prior version info when local storage is empty
        (e.g., in GCP where local storage is ephemeral).

        Returns:
            Full metadata dict if available, None otherwise.
        """
        try:
            metadata_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="metadata.json",
                repo_type="model",
                token=self.token,
            )
            with open(metadata_path) as f:
                return json.load(f)
        except Exception:
            return None

    def list_versions(self) -> list[str]:
        """List all available model versions (branches) on HF Hub."""
        try:
            refs = self.api.list_repo_refs(repo_id=self.repo_id, repo_type="model")
            versions = [
                branch.name for branch in refs.branches if branch.name.startswith("v")
            ]
            return sorted(versions, reverse=True)
        except Exception:
            return []

    def _generate_readme(
        self, version: str, metadata: dict[str, Any], symbol_order: list[str]
    ) -> str:
        """Generate README content for the model card."""
        metrics = metadata.get("metrics", {})
        return f"""---
tags:
- ppo
- reinforcement-learning
- portfolio-optimization
- finance
- dual-forecasts
- learnfinance
---

# LearnFinance PPO Model - {version}

Proximal Policy Optimization (PPO) portfolio allocation agent using dual forecasts (LSTM + PatchTST) as features.

## Model Details

- **Version**: {version}
- **Model Type**: PPO (Proximal Policy Optimization) with dual forecasts
- **Training Window**: {metadata.get("data_window", {}).get("start", "N/A")} to {metadata.get("data_window", {}).get("end", "N/A")}
- **Symbols**: {len(symbol_order)} stocks

## Components

- `weights.pt` - PPO actor-critic network weights
- `scaler.pkl` - PortfolioScaler for state normalization
- `symbol_order.json` - Ordered list of portfolio symbols

## Metrics

- Policy Loss: {metrics.get("policy_loss", "N/A")}
- Value Loss: {metrics.get("value_loss", "N/A")}
- Avg Episode Return: {metrics.get("avg_episode_return", "N/A")}
- Avg Episode Sharpe: {metrics.get("avg_episode_sharpe", "N/A")}
- Eval Sharpe: {metrics.get("eval_sharpe", "N/A")}
- Eval CAGR: {metrics.get("eval_cagr", "N/A")}
- Eval Max Drawdown: {metrics.get("eval_max_drawdown", "N/A")}

## Usage

```python
from brain_api.storage.ppo import PPOHuggingFaceModelStorage

storage = PPOHuggingFaceModelStorage(repo_id="{self.repo_id}")
artifacts = storage.download_model(version="{version}")
```
"""
