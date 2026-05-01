"""PatchTST model storage module."""

from brain_api.storage.metadata import create_training_metadata
from brain_api.storage.patchtst.huggingface import (
    PatchTSTHalalNewHuggingFaceModelStorage,
    PatchTSTHuggingFaceModelStorage,  # Backward compatibility alias
    PatchTSTIndiaHuggingFaceModelStorage,  # Backward compatibility alias
    PatchTSTNiftyShariah500HuggingFaceModelStorage,
)
from brain_api.storage.patchtst.local import (
    PatchTSTArtifacts,
    PatchTSTHalalNewModelStorage,
    PatchTSTIndiaModelStorage,  # Backward compatibility alias
    PatchTSTModelStorage,  # Backward compatibility alias
    PatchTSTNiftyShariah500ModelStorage,
)

__all__ = [
    "PatchTSTArtifacts",
    "PatchTSTHalalNewHuggingFaceModelStorage",
    "PatchTSTHalalNewModelStorage",
    "PatchTSTHuggingFaceModelStorage",  # Backward compatibility alias
    "PatchTSTIndiaHuggingFaceModelStorage",  # Backward compatibility alias
    "PatchTSTIndiaModelStorage",  # Backward compatibility alias
    "PatchTSTModelStorage",  # Backward compatibility alias
    "PatchTSTNiftyShariah500HuggingFaceModelStorage",
    "PatchTSTNiftyShariah500ModelStorage",
    "create_training_metadata",
]
