"""HuggingFace Hub storage for model artifacts and datasets.

This module re-exports from model-specific submodules for backward compatibility.
"""

# Dataset storage
from brain_api.storage.datasets.huggingface import HuggingFaceDatasetStorage

# LSTM HuggingFace storage
from brain_api.storage.lstm.huggingface import (
    HFModelInfo,
    HuggingFaceModelStorage,
)

# PatchTST HuggingFace storage
from brain_api.storage.patchtst.huggingface import (
    PatchTSTHalalNewHuggingFaceModelStorage,
    PatchTSTHuggingFaceModelStorage,  # Backward compatibility alias
    PatchTSTIndiaHuggingFaceModelStorage,  # Backward compatibility alias
    PatchTSTNiftyShariah500HuggingFaceModelStorage,
)

# SAC HuggingFace storage (PatchTST forecast features)
from brain_api.storage.sac.huggingface import (
    SACHuggingFaceModelStorage,
)

__all__ = [
    "HFModelInfo",
    # Datasets
    "HuggingFaceDatasetStorage",
    # LSTM
    "HuggingFaceModelStorage",
    # PatchTST
    "PatchTSTHalalNewHuggingFaceModelStorage",
    "PatchTSTHuggingFaceModelStorage",  # Backward compatibility alias
    "PatchTSTIndiaHuggingFaceModelStorage",  # Backward compatibility alias
    "PatchTSTNiftyShariah500HuggingFaceModelStorage",
    # SAC (PatchTST forecast features)
    "SACHuggingFaceModelStorage",
]
