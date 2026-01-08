"""PPO + PatchTST portfolio allocator.

This variant uses PatchTST predicted weekly returns as the forecast feature
in the PPO state vector. The environment, constraints, and reward logic
come from the shared portfolio_rl module.

This is structurally identical to ppo_lstm, but uses PatchTST as the
forecast provider instead of LSTM.
"""

from brain_api.core.ppo_patchtst.config import (
    PPOPatchTSTConfig,
    DEFAULT_PPO_PATCHTST_CONFIG,
)
from brain_api.core.ppo_patchtst.training import (
    PPOPatchTSTTrainingResult,
    train_ppo_patchtst,
    finetune_ppo_patchtst,
)
from brain_api.core.ppo_lstm.finetune import PPOFinetuneConfig
from brain_api.core.ppo_patchtst.inference import (
    run_ppo_patchtst_inference,
)
from brain_api.core.ppo_patchtst.version import compute_version

# Reuse model from ppo_lstm (identical architecture)
from brain_api.core.ppo_lstm.model import (
    PPOPolicy,
    PPOValueNetwork,
    PPOActorCritic,
)

__all__ = [
    # Config
    "PPOPatchTSTConfig",
    "DEFAULT_PPO_PATCHTST_CONFIG",
    # Training (full)
    "PPOPatchTSTTrainingResult",
    "train_ppo_patchtst",
    # Training (fine-tune)
    "PPOFinetuneConfig",
    "finetune_ppo_patchtst",
    # Inference
    "run_ppo_patchtst_inference",
    # Model (shared with ppo_lstm)
    "PPOPolicy",
    "PPOValueNetwork",
    "PPOActorCritic",
    # Version
    "compute_version",
]

