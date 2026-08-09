"""SAC model storage for the PatchTST-featured allocator."""

from brain_api.storage.sac.artifacts import (
    SACArtifactCompatibilityError,
    SACV3AuxiliaryArtifacts,
)
from brain_api.storage.sac.huggingface import (
    HFModelInfo,
    SACHuggingFaceModelStorage,
)
from brain_api.storage.sac.local import (
    SACArtifacts,
    SACHalalFilteredModelStorage,
    SACHalalModelStorage,
    SACLocalStorage,  # Backward compatibility alias
    create_sac_metadata,
)

__all__ = [
    "HFModelInfo",
    "SACArtifactCompatibilityError",
    "SACArtifacts",
    "SACHalalFilteredModelStorage",
    "SACHalalModelStorage",
    "SACHuggingFaceModelStorage",
    "SACLocalStorage",  # Backward compatibility alias
    "SACV3AuxiliaryArtifacts",
    "create_sac_metadata",
]
