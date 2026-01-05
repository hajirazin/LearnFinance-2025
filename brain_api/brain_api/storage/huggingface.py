"""HuggingFace Hub storage for model artifacts and datasets.

This module re-exports from model-specific submodules for backward compatibility.
"""

# LSTM HuggingFace storage
from brain_api.storage.lstm.huggingface import (
    HFModelInfo,
    HuggingFaceModelStorage,
)

# PatchTST HuggingFace storage
from brain_api.storage.patchtst.huggingface import (
    PatchTSTHuggingFaceModelStorage,
)

# Dataset storage
from brain_api.storage.datasets.huggingface import HuggingFaceDatasetStorage

__all__ = [
    # LSTM
    "HuggingFaceModelStorage",
    "HFModelInfo",
    # PatchTST
    "PatchTSTHuggingFaceModelStorage",
    # Datasets
    "HuggingFaceDatasetStorage",
]
