"""Local filesystem storage for model artifacts.

This module re-exports from model-specific submodules for backward compatibility.
"""

# LSTM storage
# Shared utilities

from brain_api.storage.base import DEFAULT_DATA_PATH
from brain_api.storage.lstm.local import (
    LocalModelStorage,  # Backward compatibility alias
    LSTMArtifacts,
    LSTMHalalNewModelStorage,
    LSTMLocalStorage,  # Backward compatibility alias
)
from brain_api.storage.metadata import create_training_metadata

# PatchTST storage
from brain_api.storage.patchtst.local import (
    PatchTSTArtifacts,
    PatchTSTHalalNewModelStorage,
    PatchTSTIndiaModelStorage,  # Backward compatibility alias
    PatchTSTModelStorage,  # Backward compatibility alias
    PatchTSTNiftyShariah500ModelStorage,
)

# SAC storage (PatchTST forecast features)
from brain_api.storage.sac.local import (
    SACArtifacts,
    SACHalalFilteredModelStorage,
    SACHalalModelStorage,
    SACLocalStorage,  # Backward compatibility alias
    create_sac_metadata,
)

__all__ = [
    # Shared
    "DEFAULT_DATA_PATH",
    # LSTM
    "LSTMArtifacts",
    "LSTMHalalNewModelStorage",
    "LSTMLocalStorage",  # Backward compatibility alias
    "LocalModelStorage",  # Backward compatibility alias
    # PatchTST
    "PatchTSTArtifacts",
    "PatchTSTHalalNewModelStorage",
    "PatchTSTIndiaModelStorage",  # Backward compatibility alias
    "PatchTSTModelStorage",  # Backward compatibility alias
    "PatchTSTNiftyShariah500ModelStorage",
    # SAC (PatchTST forecast features)
    "SACArtifacts",
    "SACHalalFilteredModelStorage",
    "SACHalalModelStorage",
    "SACLocalStorage",  # Backward compatibility alias
    "create_sac_metadata",
    "create_training_metadata",
]
