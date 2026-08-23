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


def test_patchtst_training_and_inference_adapters_have_identical_effective_config() -> (
    None
):
    """Training cannot silently drift from the serialized inference architecture."""
    config = PatchTSTConfig()

    inference_config = config.to_hf_config()
    training_config = _create_patchtst_model(config).config

    assert training_config.to_dict() == inference_config.to_dict()
    assert not hasattr(training_config, "dropout")
    assert training_config.attention_dropout == config.dropout
    assert training_config.positional_dropout == config.dropout
    assert training_config.ff_dropout == 0.0
    assert training_config.path_dropout == 0.0
    assert training_config.head_dropout == 0.0
