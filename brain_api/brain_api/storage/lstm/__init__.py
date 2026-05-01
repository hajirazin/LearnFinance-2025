"""LSTM model storage module."""

from brain_api.storage.lstm.huggingface import HFModelInfo, HuggingFaceModelStorage
from brain_api.storage.lstm.local import (
    LocalModelStorage,  # Backward compatibility alias
    LSTMArtifacts,
    LSTMHalalNewModelStorage,
    LSTMLocalStorage,  # Backward compatibility alias
)
from brain_api.storage.metadata import create_training_metadata

__all__ = [
    "HFModelInfo",
    "HuggingFaceModelStorage",
    "LSTMArtifacts",
    "LSTMHalalNewModelStorage",
    "LSTMLocalStorage",  # Backward compatibility alias
    "LocalModelStorage",  # Backward compatibility alias
    "create_training_metadata",
]
