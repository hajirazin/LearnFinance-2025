"""HuggingFace Hub storage for SAC model artifacts (unified with dual forecasts).

SAC requires storing multiple components (actor, critic, target critic, log_alpha)
plus the scaler, config, and symbol order. This is more complex than single-model
storage like LSTM or PatchTST.
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

from brain_api.core.config import get_hf_sac_model_repo, get_hf_token
from brain_api.core.portfolio_rl.sac_networks import GaussianActor, TwinCritic
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.sac.config import SACConfig
from brain_api.storage.sac.local import (
    SACArtifacts,
    SACLocalStorage,
)

logger = logging.getLogger(__name__)

__all__ = ["HFModelInfo", "SACHuggingFaceModelStorage"]


@dataclass
class HFModelInfo:
    """Information about a model stored on HuggingFace Hub."""

    repo_id: str
    version: str
    revision: str  # Git commit/tag on HF


class SACHuggingFaceModelStorage:
    """HuggingFace Hub storage for SAC model artifacts.

    SAC stores multiple components unlike single-model storage:
        - actor.pt              (Gaussian actor network weights)
        - critic.pt             (Twin critic weights)
        - critic_target.pt      (Target critic weights)
        - log_alpha.pt          (Entropy coefficient)
        - scaler.pkl            (PortfolioScaler for state normalization)
        - config.json           (SAC hyperparameters)
        - symbol_order.json     (Ordered list of symbols)
        - metadata.json         (Training info, metrics, data window)

    Versions are managed as git tags/branches on the HF repo.
    The 'main' branch typically points to the current promoted version.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        token: str | None = None,
        local_cache: SACLocalStorage | None = None,
    ):
        """Initialize SAC HuggingFace model storage.

        Args:
            repo_id: HuggingFace repo ID. Defaults to HF_SAC_MODEL_REPO env var.
            token: HuggingFace API token.
            local_cache: Optional local storage for caching downloaded models.
        """
        self.repo_id = repo_id or get_hf_sac_model_repo()
        self.token = token or get_hf_token()
        self.local_cache = local_cache or SACLocalStorage()
        self.api = HfApi(token=self.token)

        if not self.repo_id:
            raise ValueError(
                "HuggingFace SAC model repo not configured. "
                "Set HF_SAC_MODEL_REPO environment variable or pass repo_id."
            )

    @property
    def model_type(self) -> str:
        return "sac"

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
        actor: GaussianActor,
        critic: TwinCritic,
        critic_target: TwinCritic,
        log_alpha: torch.Tensor,
        scaler: PortfolioScaler,
        config: SACConfig,
        symbol_order: list[str],
        metadata: dict[str, Any],
        make_current: bool = False,
    ) -> HFModelInfo:
        """Upload SAC model artifacts to HuggingFace Hub.

        Args:
            version: Version string (e.g., 'v2025-01-08-abc123')
            actor: Trained SAC actor network
            critic: Trained twin critic networks
            critic_target: Target critic networks
            log_alpha: Entropy coefficient tensor
            scaler: Fitted PortfolioScaler for state normalization
            config: SAC configuration
            symbol_order: Ordered list of symbols
            metadata: Training metadata
            make_current: If True, also update 'main' branch to point to this version

        Returns:
            HFModelInfo with upload details
        """
        self._ensure_repo_exists()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Save actor weights
            actor_path = tmppath / "actor.pt"
            torch.save(actor.state_dict(), actor_path)

            # Save critic weights
            critic_path = tmppath / "critic.pt"
            torch.save(critic.state_dict(), critic_path)

            # Save target critic weights
            critic_target_path = tmppath / "critic_target.pt"
            torch.save(critic_target.state_dict(), critic_target_path)

            # Save log_alpha
            log_alpha_path = tmppath / "log_alpha.pt"
            torch.save(log_alpha, log_alpha_path)

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
            logger.info(f"Uploading SAC model {version} to {self.repo_id}")

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
                commit_message=f"Add SAC model version {version}",
            )

            if make_current:
                logger.info(f"Setting {version} as current (main branch)")
                self.api.upload_folder(
                    folder_path=tmpdir,
                    repo_id=self.repo_id,
                    repo_type="model",
                    revision="main",
                    commit_message=f"Promote SAC version {version} to current",
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
    ) -> SACArtifacts:
        """Download SAC model artifacts from HuggingFace Hub.

        Args:
            version: Version string to download. If None, downloads 'main' (current).
            use_cache: If True, check local cache first and update it after download.

        Returns:
            SACArtifacts ready for inference
        """
        revision = version or "main"

        # Check local cache first
        if use_cache and version and self.local_cache.version_exists(version):
            logger.info(f"Loading SAC model {version} from local cache")
            return self.local_cache.load_artifacts(version)

        logger.info(f"Downloading SAC model from {self.repo_id} (revision: {revision})")

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
        config = SACConfig.from_dict(config_dict)

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

        # Initialize and load actor
        actor = GaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )
        actor_path = local_path / "actor.pt"
        actor.load_state_dict(torch.load(actor_path, weights_only=True))
        actor.eval()

        # Initialize and load critic
        critic = TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )
        critic_path = local_path / "critic.pt"
        critic.load_state_dict(torch.load(critic_path, weights_only=True))
        critic.eval()

        # Initialize and load target critic
        critic_target = TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )
        critic_target_path = local_path / "critic_target.pt"
        critic_target.load_state_dict(torch.load(critic_target_path, weights_only=True))
        critic_target.eval()

        # Load log_alpha
        log_alpha_path = local_path / "log_alpha.pt"
        log_alpha = torch.load(log_alpha_path, weights_only=True)

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
            logger.info(f"Caching SAC model {actual_version} locally")
            metadata_path = local_path / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

            self.local_cache.write_artifacts(
                version=actual_version,
                actor=actor,
                critic=critic,
                critic_target=critic_target,
                log_alpha=log_alpha,
                scaler=scaler,
                config=config,
                symbol_order=symbol_order,
                metadata=metadata,
            )
            self.local_cache.promote_version(actual_version)

        return SACArtifacts(
            config=config,
            scaler=scaler,
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            log_alpha=log_alpha,
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
- sac
- reinforcement-learning
- portfolio-optimization
- finance
- dual-forecasts
- learnfinance
---

# LearnFinance SAC Model - {version}

Soft Actor-Critic (SAC) portfolio allocation agent using dual forecasts (LSTM + PatchTST) as features.

## Model Details

- **Version**: {version}
- **Model Type**: SAC (Soft Actor-Critic) with dual forecasts
- **Training Window**: {metadata.get("data_window", {}).get("start", "N/A")} to {metadata.get("data_window", {}).get("end", "N/A")}
- **Symbols**: {len(symbol_order)} stocks

## Components

- `actor.pt` - Gaussian policy network
- `critic.pt` - Twin Q-value networks
- `critic_target.pt` - Target Q-value networks
- `log_alpha.pt` - Entropy temperature coefficient
- `scaler.pkl` - PortfolioScaler for state normalization
- `symbol_order.json` - Ordered list of portfolio symbols

## Metrics

- Actor Loss: {metrics.get("actor_loss", "N/A")}
- Critic Loss: {metrics.get("critic_loss", "N/A")}
- Avg Episode Return: {metrics.get("avg_episode_return", "N/A")}
- Avg Episode Sharpe: {metrics.get("avg_episode_sharpe", "N/A")}
- Eval Sharpe: {metrics.get("eval_sharpe", "N/A")}
- Eval CAGR: {metrics.get("eval_cagr", "N/A")}
- Eval Max Drawdown: {metrics.get("eval_max_drawdown", "N/A")}

## Usage

```python
from brain_api.storage.sac import SACHuggingFaceModelStorage

storage = SACHuggingFaceModelStorage(repo_id="{self.repo_id}")
artifacts = storage.download_model(version="{version}")
```
"""
