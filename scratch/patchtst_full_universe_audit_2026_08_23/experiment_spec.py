"""Frozen configuration and audit helpers for the full-universe experiment."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import PatchTSTConfig as HFPatchTSTConfig
from transformers import PatchTSTForPrediction

CONTEXT_LENGTH = 60
PREDICTION_LENGTH = 5
FEATURE_NAMES = ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
CLOSE_INDEX = FEATURE_NAMES.index("close_ret")
SEEDS = [20260823, 20260824, 20260825]
TOP_K = 15


@dataclass(frozen=True)
class ExperimentArm:
    """One predeclared PatchTST configuration arm."""

    name: str
    patch_length: int
    patch_stride: int
    pooling_type: str | None
    attention_dropout: float
    positional_dropout: float
    path_dropout: float
    ff_dropout: float
    head_dropout: float
    objective: str


ARMS = {
    "corrected_control_16_8": ExperimentArm(
        name="corrected_control_16_8",
        patch_length=16,
        patch_stride=8,
        pooling_type="mean",
        attention_dropout=0.2,
        positional_dropout=0.2,
        path_dropout=0.0,
        ff_dropout=0.0,
        head_dropout=0.0,
        objective="close_daily_mse",
    ),
    "coherent_candidate_10_5": ExperimentArm(
        name="coherent_candidate_10_5",
        patch_length=10,
        patch_stride=5,
        pooling_type=None,
        attention_dropout=0.0,
        positional_dropout=0.05,
        path_dropout=0.05,
        ff_dropout=0.05,
        head_dropout=0.0,
        objective="scaled_ohlcv_daily_mse",
    ),
}


def patch_count(context_length: int, patch_length: int, patch_stride: int) -> int:
    """Return Hugging Face's unpadded patch count."""
    return (context_length - patch_length) // patch_stride + 1


def hf_config_for_arm(arm: ExperimentArm) -> HFPatchTSTConfig:
    """Construct a fully explicit Transformers 4.57.3 PatchTST config."""
    return HFPatchTSTConfig(
        num_input_channels=5,
        context_length=CONTEXT_LENGTH,
        distribution_output="student_t",
        loss="mse",
        patch_length=arm.patch_length,
        patch_stride=arm.patch_stride,
        num_hidden_layers=2,
        d_model=64,
        num_attention_heads=4,
        share_embedding=True,
        channel_attention=False,
        ffn_dim=128,
        norm_type="batchnorm",
        norm_eps=1e-5,
        attention_dropout=arm.attention_dropout,
        positional_dropout=arm.positional_dropout,
        path_dropout=arm.path_dropout,
        ff_dropout=arm.ff_dropout,
        bias=True,
        activation_function="gelu",
        pre_norm=True,
        positional_encoding_type="sincos",
        use_cls_token=False,
        init_std=0.02,
        share_projection=True,
        scaling="std",
        do_mask_input=False,
        pooling_type=arm.pooling_type,
        head_dropout=arm.head_dropout,
        prediction_length=PREDICTION_LENGTH,
    )


def build_model(arm: ExperimentArm) -> PatchTSTForPrediction:
    return PatchTSTForPrediction(hf_config_for_arm(arm))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def json_dump(path: Path, value: Any) -> None:
    """Atomically write strict JSON, converting nonfinite numbers to null."""

    def safe(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): safe(inner) for key, inner in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(inner) for inner in item]
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            return None
        if isinstance(item, np.integer):
            return int(item)
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(safe(value), indent=2, sort_keys=True, default=str) + "\n"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def runtime_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "device_capability": "mps" if torch.backends.mps.is_available() else "cpu",
    }


def arm_manifest() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, arm in ARMS.items():
        config = hf_config_for_arm(arm)
        output[name] = {
            "declared": asdict(arm),
            "effective_hf_config": config.to_dict(),
            "patch_count": patch_count(
                CONTEXT_LENGTH, config.patch_length, config.patch_stride
            ),
        }
    return output
