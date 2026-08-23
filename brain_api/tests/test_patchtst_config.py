"""Regression tests for the PatchTST domain-to-Hugging-Face adapter."""

import pytest

from brain_api.core.patchtst.config import DEFAULT_CONFIG, PatchTSTConfig
from brain_api.core.patchtst.training import _create_patchtst_model


def test_default_config_is_locked_close_only_mean_10_5() -> None:
    """Production defaults must match the locked research contract."""
    assert DEFAULT_CONFIG.num_input_channels == 1
    assert DEFAULT_CONFIG.feature_names == ["close_ret"]
    assert DEFAULT_CONFIG.context_length == 60
    assert DEFAULT_CONFIG.prediction_length == 5
    assert DEFAULT_CONFIG.patch_length == 10
    assert DEFAULT_CONFIG.stride == 5
    assert DEFAULT_CONFIG.epochs == 60
    assert DEFAULT_CONFIG.early_stopping_patience == 8
    assert DEFAULT_CONFIG.weight_decay == 0.0
    assert DEFAULT_CONFIG.batch_size == 256
    assert DEFAULT_CONFIG.learning_rate == 0.0003
    assert DEFAULT_CONFIG.max_grad_norm == 1.0
    assert DEFAULT_CONFIG.dropout == 0.2

    hf_config = DEFAULT_CONFIG.to_hf_config()
    assert hf_config.pooling_type == "mean"
    assert hf_config.channel_attention is False
    assert hf_config.patch_length == 10
    assert hf_config.patch_stride == 5
    assert hf_config.num_input_channels == 1
    assert hf_config.do_mask_input is False
    assert hf_config.scaling == "std"
    assert hf_config.share_embedding is True
    assert hf_config.pre_norm is True
    assert hf_config.norm_type == "batchnorm"
    assert hf_config.attention_dropout == 0.2
    assert hf_config.positional_dropout == 0.2
    assert hf_config.path_dropout == 0.0
    assert hf_config.ff_dropout == 0.0
    assert hf_config.head_dropout == 0.0
    patch_count = (
        DEFAULT_CONFIG.context_length - DEFAULT_CONFIG.patch_length
    ) // hf_config.patch_stride + 1
    assert patch_count == 11


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
