"""HuggingFace Hub storage for model artifacts and datasets.

This module re-exports from model-specific submodules for backward compatibility.
"""

# LSTM HuggingFace storage
# Dataset storage
from brain_api.storage.datasets.huggingface import HuggingFaceDatasetStorage
from brain_api.storage.lstm.huggingface import (
    HFModelInfo,
    HuggingFaceModelStorage,
)

# PatchTST HuggingFace storage
from brain_api.storage.patchtst.huggingface import (
    PatchTSTHuggingFaceModelStorage,
)

__all__ = [
    "HFModelInfo",
    # Datasets
    "HuggingFaceDatasetStorage",
    # LSTM
    "HuggingFaceModelStorage",
    # PatchTST
    "PatchTSTHuggingFaceModelStorage",
]
