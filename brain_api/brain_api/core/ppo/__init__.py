"""PPO portfolio allocator with dual forecasts (LSTM + PatchTST).

Uses both LSTM and PatchTST predicted weekly returns as forecast features
in the PPO state vector. The environment, constraints, and reward logic
come from the shared portfolio_rl module.
"""

from brain_api.core.ppo.config import (
    DEFAULT_PPO_CONFIG,
    PPOConfig,
)
from brain_api.core.ppo.data import (
    TrainingData,
    build_training_data,
)
from brain_api.core.ppo.finetune import (
    PPOFinetuneConfig,
    finetune_ppo,
)
from brain_api.core.ppo.inference import (
    PPOInferenceResult,
    run_ppo_inference,
)
from brain_api.core.ppo.model import (
    PPOActorCritic,
    PPOPolicy,
    PPOValueNetwork,
)
from brain_api.core.ppo.train import (
    PPOTrainingResult,
    evaluate_policy,
    train_ppo,
)
from brain_api.core.ppo.version import compute_version

__all__ = [
    "DEFAULT_PPO_CONFIG",
    "PPOActorCritic",
    # Config
    "PPOConfig",
    # Training (fine-tune)
    "PPOFinetuneConfig",
    # Inference
    "PPOInferenceResult",
    # Model
    "PPOPolicy",
    # Training (full)
    "PPOTrainingResult",
    "PPOValueNetwork",
    # Data
    "TrainingData",
    "build_training_data",
    # Version
    "compute_version",
    "evaluate_policy",
    "finetune_ppo",
    "run_ppo_inference",
    "train_ppo",
]
