"""Regression tests for the PatchTST domain-to-Hugging-Face adapter."""

import pytest

from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.core.patchtst.training import _create_patchtst_model


@pytest.mark.parametrize("adapter", ["inference", "training"])
def test_patchtst_adapter_uses_configured_patch_stride(adapter: str) -> None:
    """The domain stride must control Hugging Face patch extraction."""
    config = PatchTSTConfig()

    if adapter == "inference":
        hf_config = config.to_hf_config()
    else:
        hf_config = _create_patchtst_model(config).config

    assert config.patch_length == 10
    assert config.stride == 5
    assert hf_config.patch_stride == config.stride
    patch_count = (
        config.context_length - config.patch_length
    ) // hf_config.patch_stride + 1
    assert patch_count == 11
