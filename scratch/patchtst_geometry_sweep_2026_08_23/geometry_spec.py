"""Frozen domain configuration for the PatchTST patch-geometry sweep."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import PatchTSTConfig, PatchTSTForPrediction

CONTEXT_LENGTH = 60
PREDICTION_LENGTH = 5
SEEDS = (20260823, 20260824, 20260825)
TOP_K = 15
BOOTSTRAP_REPETITIONS = 2_000
BOOTSTRAP_BLOCK_WEEKS = 4
DATA_START = date(2015, 1, 1)
DATA_END_EXCLUSIVE = date(2026, 8, 22)


@dataclass(frozen=True)
class DateWindow:
    """Inclusive Monday decision-date window."""

    start: date
    end: date


@dataclass(frozen=True)
class EvaluationFold:
    """One chronological expanding-window research fold."""

    name: str
    evidence_kind: str
    train: DateWindow
    validation: DateWindow
    evaluation: DateWindow


@dataclass(frozen=True)
class PatchGeometry:
    """The only model property varied by this experiment."""

    name: str
    patch_length: int
    patch_stride: int


PATCH_GEOMETRIES = {
    "patch_8_stride_4": PatchGeometry("patch_8_stride_4", 8, 4),
    "patch_10_stride_5": PatchGeometry("patch_10_stride_5", 10, 5),
    "patch_16_stride_8": PatchGeometry("patch_16_stride_8", 16, 8),
}

EVALUATION_FOLDS = {
    "development_2024": EvaluationFold(
        name="development_2024",
        evidence_kind="development",
        train=DateWindow(date(2015, 5, 4), date(2022, 12, 19)),
        validation=DateWindow(date(2023, 1, 9), date(2023, 12, 18)),
        evaluation=DateWindow(date(2024, 1, 8), date(2024, 12, 23)),
    ),
    "development_2025": EvaluationFold(
        name="development_2025",
        evidence_kind="development",
        train=DateWindow(date(2015, 5, 4), date(2023, 12, 18)),
        validation=DateWindow(date(2024, 1, 8), date(2024, 12, 16)),
        evaluation=DateWindow(date(2025, 1, 6), date(2025, 12, 22)),
    ),
    "confirmatory_2026": EvaluationFold(
        name="confirmatory_2026",
        evidence_kind="confirmatory",
        train=DateWindow(date(2015, 5, 4), date(2024, 12, 16)),
        validation=DateWindow(date(2025, 1, 6), date(2025, 12, 22)),
        evaluation=DateWindow(date(2026, 1, 12), date(2026, 8, 17)),
    ),
}


def patch_count(context_length: int, patch_length: int, patch_stride: int) -> int:
    """Return the unpadded Hugging Face PatchTST token count."""
    if not 0 < patch_length <= context_length:
        raise ValueError("patch_length must be in [1, context_length]")
    if patch_stride <= 0:
        raise ValueError("patch_stride must be positive")
    return (context_length - patch_length) // patch_stride + 1


def hf_config_for_geometry(geometry: PatchGeometry) -> PatchTSTConfig:
    """Construct the fully explicit frozen one-channel model configuration."""
    return PatchTSTConfig(
        num_input_channels=1,
        context_length=CONTEXT_LENGTH,
        distribution_output="student_t",
        loss="mse",
        patch_length=geometry.patch_length,
        patch_stride=geometry.patch_stride,
        num_hidden_layers=2,
        d_model=64,
        num_attention_heads=4,
        share_embedding=True,
        channel_attention=False,
        ffn_dim=128,
        norm_type="batchnorm",
        norm_eps=1e-5,
        attention_dropout=0.2,
        positional_dropout=0.2,
        path_dropout=0.0,
        ff_dropout=0.0,
        bias=True,
        activation_function="gelu",
        pre_norm=True,
        positional_encoding_type="sincos",
        use_cls_token=False,
        init_std=0.02,
        share_projection=True,
        scaling="std",
        do_mask_input=False,
        pooling_type="mean",
        head_dropout=0.0,
        prediction_length=PREDICTION_LENGTH,
    )


def build_patchtst_model(geometry: PatchGeometry) -> PatchTSTForPrediction:
    """Build one uninitialized research model."""
    return PatchTSTForPrediction(hf_config_for_geometry(geometry))


def set_deterministic_seed(seed: int) -> None:
    """Seed every RNG used by the research runner and enforce determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


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


def strict_json_value(value: Any) -> Any:
    """Recursively replace nonfinite scalars with JSON null values."""
    if isinstance(value, dict):
        return {str(key): strict_json_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(inner) for inner in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def json_dump(path: Path, value: Any) -> None:
    """Atomically write indented strict JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(strict_json_value(value), indent=2, sort_keys=True, default=str)
        + "\n"
    )
    temporary.replace(path)


def runtime_manifest() -> dict[str, Any]:
    """Return a sanitized runtime snapshot without host identifiers."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "hardware": {
            "machine": "MacBook Pro",
            "chip": "Apple M5 Pro",
            "cpu_cores": 18,
            "gpu_cores": 20,
            "unified_memory_gb": 48,
        },
    }


def frozen_configuration_manifest() -> dict[str, Any]:
    """Serialize folds, arms, and all common settings for audit output."""
    return {
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "input": "one-channel adjusted-close log returns",
        "seeds": list(SEEDS),
        "geometries": {
            name: {
                **asdict(geometry),
                "patch_count": patch_count(
                    CONTEXT_LENGTH,
                    geometry.patch_length,
                    geometry.patch_stride,
                ),
                "effective_hf_config": hf_config_for_geometry(geometry).to_dict(),
            }
            for name, geometry in PATCH_GEOMETRIES.items()
        },
        "folds": {name: asdict(fold) for name, fold in EVALUATION_FOLDS.items()},
        "training": {
            "objective": "daily_close_mse",
            "optimizer": "Adam",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "batch_size": 256,
            "max_epochs": 60,
            "early_stopping_patience": 8,
            "gradient_clip": 1.0,
            "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=5)",
            "checkpoint_selection": (
                "maximum validation weekly rank IC; validation MSE tie-break"
            ),
        },
    }
