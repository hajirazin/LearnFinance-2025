"""PPO + PatchTST portfolio allocator.

This variant uses PatchTST predicted weekly returns as the forecast feature
in the PPO state vector. The environment, constraints, and reward logic
come from the shared portfolio_rl module.

This is structurally identical to ppo_lstm, but uses PatchTST as the
forecast provider instead of LSTM.
"""

from brain_api.core.ppo_lstm.finetune import PPOFinetuneConfig

# Reuse model from ppo_lstm (identical architecture)
from brain_api.core.ppo_lstm.model import (
    PPOActorCritic,
    PPOPolicy,
    PPOValueNetwork,
)
from brain_api.core.ppo_patchtst.config import (
    DEFAULT_PPO_PATCHTST_CONFIG,
    PPOPatchTSTConfig,
)
from brain_api.core.ppo_patchtst.inference import (
    run_ppo_patchtst_inference,
)
from brain_api.core.ppo_patchtst.training import (
    PPOPatchTSTTrainingResult,
    finetune_ppo_patchtst,
    train_ppo_patchtst,
)
from brain_api.core.ppo_patchtst.version import compute_version

__all__ = [
    "DEFAULT_PPO_PATCHTST_CONFIG",
    "PPOActorCritic",
    # Training (fine-tune)
    "PPOFinetuneConfig",
    # Config
    "PPOPatchTSTConfig",
    # Training (full)
    "PPOPatchTSTTrainingResult",
    # Model (shared with ppo_lstm)
    "PPOPolicy",
    "PPOValueNetwork",
    # Version
    "compute_version",
    "finetune_ppo_patchtst",
    # Inference
    "run_ppo_patchtst_inference",
    "train_ppo_patchtst",
]

