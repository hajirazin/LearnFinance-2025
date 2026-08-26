"""Storage package for ppo_discovery artifacts."""

from brain_api.storage.ppo_discovery.huggingface import (
    PPODiscoveryHuggingFaceModelStorage,
)
from brain_api.storage.ppo_discovery.local import (
    PPODiscoveryArtifacts,
    PPODiscoveryHalalNewModelStorage,
)

__all__ = [
    "PPODiscoveryArtifacts",
    "PPODiscoveryHalalNewModelStorage",
    "PPODiscoveryHuggingFaceModelStorage",
]
