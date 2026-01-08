"""PPO + LSTM portfolio allocator.

This variant uses LSTM predicted weekly returns as the forecast feature
in the PPO state vector. The environment, constraints, and reward logic
come from the shared portfolio_rl module.
"""

from brain_api.core.ppo_lstm.config import (
    PPOLSTMConfig,
    DEFAULT_PPO_LSTM_CONFIG,
)
from brain_api.core.ppo_lstm.data import (
    TrainingData,
    build_training_data,
)
from brain_api.core.ppo_lstm.train import (
    PPOTrainingResult,
    train_ppo_lstm,
    evaluate_policy,
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
    PPOPolicy,
    PPOValueNetwork,
    PPOActorCritic,
)
from brain_api.core.ppo_lstm.version import compute_version

__all__ = [
    # Config
    "PPOLSTMConfig",
    "DEFAULT_PPO_LSTM_CONFIG",
    # Data
    "TrainingData",
    "build_training_data",
    # Training (full)
    "PPOTrainingResult",
    "train_ppo_lstm",
    "evaluate_policy",
    # Training (fine-tune)
    "PPOFinetuneConfig",
    "finetune_ppo_lstm",
    # Inference
    "PPOInferenceResult",
    "run_ppo_inference",
    # Model
    "PPOPolicy",
    "PPOValueNetwork",
    "PPOActorCritic",
    # Version
    "compute_version",
]
