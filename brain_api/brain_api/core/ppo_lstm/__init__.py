"""PPO + LSTM portfolio allocator.

This variant uses LSTM predicted weekly returns as the forecast feature
in the PPO state vector. The environment, constraints, and reward logic
come from the shared portfolio_rl module.
"""

from brain_api.core.ppo_lstm.config import (
    DEFAULT_PPO_LSTM_CONFIG,
    PPOLSTMConfig,
)
from brain_api.core.ppo_lstm.data import (
    TrainingData,
    build_training_data,
)
from brain_api.core.ppo_lstm.finetune import (
    PPOFinetuneConfig,
    finetune_ppo_lstm,
)
from brain_api.core.ppo_lstm.inference import (
    PPOInferenceResult,
    run_ppo_inference,
)
from brain_api.core.ppo_lstm.model import (
    PPOActorCritic,
    PPOPolicy,
    PPOValueNetwork,
)
from brain_api.core.ppo_lstm.train import (
    PPOTrainingResult,
    evaluate_policy,
    train_ppo_lstm,
)
from brain_api.core.ppo_lstm.version import compute_version

__all__ = [
    "DEFAULT_PPO_LSTM_CONFIG",
    "PPOActorCritic",
    # Training (fine-tune)
    "PPOFinetuneConfig",
    # Inference
    "PPOInferenceResult",
    # Config
    "PPOLSTMConfig",
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
    "finetune_ppo_lstm",
    "run_ppo_inference",
    "train_ppo_lstm",
]
