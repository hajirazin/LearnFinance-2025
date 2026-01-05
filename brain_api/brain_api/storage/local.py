"""Local filesystem storage for model artifacts.

This module re-exports from model-specific submodules for backward compatibility.
"""

# LSTM storage
from brain_api.storage.lstm.local import (
    LSTMArtifacts,
    LocalModelStorage,
    create_metadata,
)

# PatchTST storage
from brain_api.storage.patchtst.local import (
    PatchTSTArtifacts,
    PatchTSTModelStorage,
    create_metadata as create_patchtst_metadata,
)

# Shared utilities
from brain_api.storage.base import DEFAULT_DATA_PATH

__all__ = [
    # LSTM
    "LSTMArtifacts",
    "LocalModelStorage",
    "create_metadata",
    # PatchTST
    "PatchTSTArtifacts",
    "PatchTSTModelStorage",
    "create_patchtst_metadata",
    # Shared
    "DEFAULT_DATA_PATH",
]
