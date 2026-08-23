#!/usr/bin/env python3
"""Data, architecture, and training primitives for the approved scratch suite."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
import transformers
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSTConfig as HFPatchTSTConfig
from transformers import PatchTSTForPrediction


SYMBOLS = [
    "AAPL",
    "AMD",
    "AVGO",
    "GOOGL",
    "LLY",
    "META",
    "MSFT",
    "MU",
    "NVDA",
    "TSLA",
    "TSM",
    "XOM",
]
SEEDS = [20260823, 20260824, 20260825]
FEATURES = ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
CONTEXT = 60
HORIZON = 5
SPLITS: dict[str, tuple[date, date]] = {
    "train": (date(2015, 5, 4), date(2022, 12, 19)),
    "validation": (date(2023, 1, 9), date(2023, 12, 18)),
    "test": (date(2024, 1, 8), date(2025, 12, 22)),
}


@dataclass(frozen=True)
class Architecture:
    name: str
    num_channels: int
    patch_stride: int
    pooling_type: str | None
    channel_attention: bool
    legacy_stride_keyword: bool = False


ARCHITECTURES = {
    "legacy_effective": Architecture("legacy_effective", 5, 1, "mean", False, True),
    "stride_only_fixed": Architecture("stride_only_fixed", 5, 8, "mean", False),
    "canonical_close_only": Architecture("canonical_close_only", 1, 8, None, False),
    "canonical_independent_5ch": Architecture(
        "canonical_independent_5ch", 5, 8, None, False
    ),
    "canonical_mixing_5ch": Architecture("canonical_mixing_5ch", 5, 8, None, True),
}


def json_dump(path: Path, value: Any) -> None:
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
        json.dumps(safe(value), indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def training_fingerprint(
    architecture: Architecture,
    objective: str,
    arrays: tuple[np.ndarray, ...],
) -> str:
    digest = hashlib.sha256()
    identity = {
        "architecture": asdict(architecture),
        "objective": objective,
        "training": {
            "context": CONTEXT,
            "horizon": HORIZON,
            "patch": 16,
            "d_model": 64,
            "heads": 4,
            "layers": 2,
            "ffn": 128,
            "lr": 3e-4,
            "weight_decay": 0.0,
            "max_epochs": 60,
            "patience": 8,
        },
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    digest.update(json.dumps(identity, sort_keys=True).encode())
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def validate_split_contract(splits: dict[str, tuple[date, date]]) -> None:
    required = ["train", "validation", "test"]
    if list(splits) != required:
        raise ValueError(f"split order must be {required}")
    for name, (start, end) in splits.items():
        if start > end:
            raise ValueError(f"{name} starts after it ends")
    if (
        not splits["train"][1]
        < splits["validation"][0]
        < splits["validation"][1]
        < splits["test"][0]
    ):
        raise ValueError("splits overlap or are not chronological")
    if (splits["validation"][0] - splits["train"][1]).days < 21:
        raise ValueError("train/validation embargo is shorter than two weeks")
    if (splits["test"][0] - splits["validation"][1]).days < 21:
        raise ValueError("validation/test embargo is shorter than two weeks")


def patch_count(context_length: int, patch_length: int, stride: int) -> int:
    return (context_length - patch_length) // stride + 1


def build_model(architecture: Architecture) -> PatchTSTForPrediction:
    common: dict[str, Any] = {
        "num_input_channels": architecture.num_channels,
        "context_length": CONTEXT,
        "patch_length": 16,
        "d_model": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "ffn_dim": 128,
        "prediction_length": HORIZON,
        "attention_dropout": 0.2,
        "positional_dropout": 0.2,
        "use_cls_token": False,
        "pooling_type": architecture.pooling_type,
        "channel_attention": architecture.channel_attention,
        "scaling": "std",
    }
    if architecture.legacy_stride_keyword:
        common["stride"] = 8  # intentionally ignored by Transformers 4.57
    else:
        common["patch_stride"] = architecture.patch_stride
    return PatchTSTForPrediction(HFPatchTSTConfig(**common))


def architecture_manifest() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, arch in ARCHITECTURES.items():
        model = build_model(arch)
        result[name] = {
            **asdict(arch),
            "effective_patch_stride": model.config.patch_stride,
            "effective_patch_count": patch_count(
                CONTEXT, 16, model.config.patch_stride
            ),
        }
    return result


def _returns(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    clean = frame[required].astype(float).sort_index()
    if not clean.index.is_unique:
        raise ValueError("OHLCV session index contains duplicates")
    # Match the production transform: a non-positive bar invalidates that
    # return/window; it is never replaced with a sentinel or zero.
    logged = clean.where(clean > 0).map(np.log)
    values = logged.diff().dropna()
    values.columns = FEATURES
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def _split_for(day: date, splits: dict[str, tuple[date, date]]) -> str | None:
    for name, (start, end) in splits.items():
        if start <= day <= end:
            return name
    return None


def build_weekly_panel(
    prices: dict[str, pd.DataFrame],
    splits: dict[str, tuple[date, date]] = SPLITS,
    *,
    include_test_labels: bool,
) -> pd.DataFrame:
    """Build Monday-decision rows; test targets are not read unless unlocked."""
    validate_split_contract(splits)
    if len(prices) < 10:
        raise RuntimeError(f"fewer than 10 symbols remain: {len(prices)}")
    returns = {symbol: _returns(frame) for symbol, frame in prices.items()}
    start = min(v[0] for v in splits.values())
    end = max(v[1] for v in splits.values())
    mondays = pd.date_range(start, end, freq="W-MON")
    rows: list[dict[str, Any]] = []
    exclusions: dict[str, int] = {}
    split_names = list(splits)
    next_split_start = {
        name: (
            splits[split_names[index + 1]][0] if index + 1 < len(split_names) else None
        )
        for index, name in enumerate(split_names)
    }
    for anchor in mondays:
        split = _split_for(anchor.date(), splits)
        if split is None:
            continue
        future_by_symbol = {
            symbol: tuple(
                returns[symbol].index[returns[symbol].index >= anchor][:HORIZON]
            )
            for symbol in sorted(prices)
        }
        complete_futures = [
            value for value in future_by_symbol.values() if len(value) == HORIZON
        ]
        if len(complete_futures) != len(prices) or len(set(complete_futures)) != 1:
            exclusions["misaligned_or_short_target_calendar"] = exclusions.get(
                "misaligned_or_short_target_calendar", 0
            ) + len(prices)
            continue
        common_future_index = pd.DatetimeIndex(complete_futures[0])
        following_start = next_split_start[split]
        if (
            following_start is not None
            and common_future_index[-1].date() >= following_start
        ):
            exclusions["target_crosses_next_split"] = exclusions.get(
                "target_crosses_next_split", 0
            ) + len(prices)
            continue
        week_rows: list[dict[str, Any]] = []
        for symbol in sorted(prices):
            series = returns[symbol]
            past = series.loc[series.index < anchor]
            future_index = common_future_index
            reason = None
            if len(past) < CONTEXT:
                reason = "short_context"
            elif len(future_index) < HORIZON:
                reason = "short_target"
            if reason:
                exclusions[reason] = exclusions.get(reason, 0) + 1
                continue
            x = past.iloc[-CONTEXT:][FEATURES].to_numpy(dtype=np.float32)
            target_locked = split == "test" and not include_test_labels
            y: np.ndarray | None = None
            actual = math.nan
            if not target_locked:
                y = series.loc[future_index, FEATURES].to_numpy(dtype=np.float32)
                actual = float(y[:, 3].sum())
            if not np.isfinite(x).all() or (y is not None and not np.isfinite(y).all()):
                exclusions["non_finite"] = exclusions.get("non_finite", 0) + 1
                continue
            close = x[:, 3]
            week_rows.append(
                {
                    "decision_date": anchor.date(),
                    "split": split,
                    "symbol": symbol,
                    "context_end": past.index[-1].date(),
                    "target_end": future_index[-1].date(),
                    "x": x,
                    "y": y,
                    "actual_weekly_return": actual,
                    "past_week_return": float(close[-5:].sum()),
                    "momentum_4w": float(close[-20:].sum()),
                    "ridge_features": np.array(
                        [
                            close[-5:].sum(),
                            close[-20:].sum(),
                            close.sum(),
                            close[-20:].std(ddof=1),
                            x[-20:, 4].std(ddof=1),
                        ],
                        dtype=np.float32,
                    ),
                }
            )
        if len(week_rows) == len(prices):
            rows.extend(week_rows)
        else:
            exclusions["incomplete_week_missing_rows"] = exclusions.get(
                "incomplete_week_missing_rows", 0
            ) + (len(prices) - len(week_rows))
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RuntimeError(f"no common panel rows; exclusions={exclusions}")
    panel.attrs["exclusions"] = exclusions
    panel.attrs["symbols"] = sorted(prices)
    return panel.sort_values(["decision_date", "symbol"]).reset_index(drop=True)


def panel_arrays(
    panel: pd.DataFrame, split: str, architecture: Architecture
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = panel[panel["split"] == split]
    if rows["y"].isna().any():
        raise RuntimeError(f"{split} labels are locked")
    channels = [3] if architecture.num_channels == 1 else list(range(5))
    x = np.stack([v[:, channels] for v in rows["x"]]).astype(np.float32)
    y = np.stack([v[:, channels] for v in rows["y"]]).astype(np.float32)
    dates = rows["decision_date"].astype(str).to_numpy()
    return x, y, dates


def _objective(
    model: PatchTSTForPrediction,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    group_size: int,
) -> torch.Tensor:
    pred = model(past_values=x).prediction_outputs
    close_idx = 0 if pred.shape[-1] == 1 else 3
    if mode == "daily_mse":
        return F.mse_loss(pred[:, :, close_idx], y[:, :, close_idx])
    if mode != "listnet":
        raise ValueError(mode)
    if len(x) % group_size:
        raise ValueError("ListNet batch must contain whole weeks")
    predicted = pred[:, :, close_idx].sum(dim=1).reshape(-1, group_size)
    actual = y[:, :, close_idx].sum(dim=1).reshape(-1, group_size)
    actual_z = (actual - actual.mean(dim=1, keepdim=True)) / actual.std(
        dim=1, keepdim=True
    ).clamp_min(1e-6)
    rank_loss = (
        -(torch.softmax(actual_z, dim=1) * torch.log_softmax(predicted, dim=1))
        .sum(dim=1)
        .mean()
    )
    return rank_loss + 0.1 * F.huber_loss(predicted, actual)


def _validation_loss(
    model: PatchTSTForPrediction,
    x: np.ndarray,
    y: np.ndarray,
    mode: str,
    group_size: int,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        if mode == "listnet":
            return float(
                _objective(
                    model,
                    torch.from_numpy(x).to(device),
                    torch.from_numpy(y).to(device),
                    mode,
                    group_size,
                )
            )
        weighted_loss = 0.0
        count = 0
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=512
        )
        for bx, by in loader:
            weighted_loss += float(
                _objective(model, bx.to(device), by.to(device), mode, group_size)
            ) * len(bx)
            count += len(bx)
    return weighted_loss / count


def train_model(
    architecture: Architecture,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    objective: str,
    group_size: int,
    model_dir: Path,
    fingerprint: str,
) -> tuple[PatchTSTForPrediction, dict[str, Any]]:
    set_seed(seed)
    device = torch.device("cpu")
    model = build_model(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience = 0
    history: list[dict[str, float | int]] = []
    weeks = np.unique(np.asarray([str(v) for v in range(len(train_x) // group_size)]))
    for epoch in range(1, 61):
        model.train()
        losses: list[float] = []
        if objective == "daily_mse":
            generator = torch.Generator().manual_seed(seed + epoch)
            loader = DataLoader(
                TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                batch_size=256,
                shuffle=True,
                generator=generator,
            )
            batches = [(bx, by) for bx, by in loader]
        else:
            rng = np.random.default_rng(seed + epoch)
            order = rng.permutation(len(weeks))
            batches = []
            for start in range(0, len(order), 16):
                idx = np.concatenate(
                    [
                        np.arange(i * group_size, (i + 1) * group_size)
                        for i in order[start : start + 16]
                    ]
                )
                batches.append(
                    (torch.from_numpy(train_x[idx]), torch.from_numpy(train_y[idx]))
                )
        for bx, by in batches:
            optimizer.zero_grad()
            loss = _objective(
                model, bx.to(device), by.to(device), objective, group_size
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite {objective} loss for {architecture.name}/{seed}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_loss = _validation_loss(model, val_x, val_y, objective, group_size)
        if not math.isfinite(val_loss):
            raise FloatingPointError(
                f"non-finite validation loss for {architecture.name}/{seed}"
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": val_loss,
            }
        )
        if val_loss < best_loss - 1e-10:
            best_loss = val_loss
            best_epoch = epoch
            patience = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience += 1
            if patience >= 8:
                break
    if best_state is None:
        raise RuntimeError(f"no finite checkpoint for {architecture.name}/{seed}")
    restored = build_model(architecture)
    restored.load_state_dict(best_state)
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / "weights.pt"
    torch.save(best_state, weights_path)
    meta = {
        "architecture": asdict(architecture),
        "objective": objective,
        "seed": seed,
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "best_validation_loss": best_loss,
        "n_train": len(train_x),
        "n_validation": len(val_x),
        "training_fingerprint": fingerprint,
        "weights_sha256": sha256_file(weights_path),
        "history": history,
    }
    json_dump(model_dir / "meta.json", meta)
    return restored, meta


def load_model_artifact(
    architecture: Architecture, model_dir: Path
) -> tuple[PatchTSTForPrediction, dict[str, Any]]:
    weights = model_dir / "weights.pt"
    metadata = model_dir / "meta.json"
    if not weights.exists() or not metadata.exists():
        raise FileNotFoundError(model_dir)
    meta = json.loads(metadata.read_text())
    expected_hash = meta.get("weights_sha256")
    if not expected_hash or sha256_file(weights) != expected_hash:
        raise RuntimeError(f"model weight hash mismatch: {model_dir}")
    model = build_model(architecture)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model, meta


def predict(model: PatchTSTForPrediction, x: np.ndarray) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            value = model(
                past_values=torch.from_numpy(x[start : start + 512])
            ).prediction_outputs
            close_idx = 0 if value.shape[-1] == 1 else 3
            outputs.append(value[:, :, close_idx].sum(dim=1).cpu().numpy())
    result = np.concatenate(outputs).astype(float)
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite predictions")
    return result


def model_sensitivity(
    model: PatchTSTForPrediction, x: np.ndarray, close_idx: int
) -> dict[str, float]:
    model.eval()
    base = torch.from_numpy(x.copy()).requires_grad_(True)
    perturbed = x.copy()
    non_close = [i for i in range(x.shape[-1]) if i != close_idx]
    rng = np.random.default_rng(991)
    perturbed[:, :, non_close] = rng.normal(0, 0.05, perturbed[:, :, non_close].shape)
    baseline_out = model(past_values=base).prediction_outputs[:, :, close_idx]
    changed_out = model(past_values=torch.from_numpy(perturbed)).prediction_outputs[
        :, :, close_idx
    ]
    baseline_out.sum().backward()
    grad = base.grad.detach().numpy()
    return {
        "forecast_max_abs_delta": float(
            torch.max(torch.abs(changed_out - baseline_out)).detach()
        ),
        "non_close_grad_l1": float(np.abs(grad[:, :, non_close]).sum()),
    }


def control_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    ordered = panel.sort_values(["decision_date", "symbol"]).copy()
    momentum_4w_source = ordered["momentum_4w"].to_numpy(copy=True)
    for name in [
        "zero_return",
        "historical_mean",
        "majority_sign",
        "persistence_1w",
        "reversal_1w",
    ]:
        ordered[name] = math.nan
    ordered["momentum_4w"] = momentum_4w_source
    known: dict[str, list[float]] = {
        symbol: [] for symbol in ordered["symbol"].unique()
    }
    known_universe: list[float] = []
    pending: list[tuple[date, str, float]] = []
    for decision, indices in ordered.groupby("decision_date", sort=True).groups.items():
        newly_available = [item for item in pending if item[0] < decision]
        pending = [item for item in pending if item[0] >= decision]
        for _target_end, symbol, actual in newly_available:
            known[symbol].append(actual)
            if actual != 0:
                known_universe.append(actual)
        universe_magnitude = (
            float(np.median(np.abs(known_universe))) if known_universe else 0.0
        )
        universe_sign = (
            1.0
            if not known_universe or np.mean(np.asarray(known_universe) > 0) >= 0.5
            else -1.0
        )
        for idx in indices:
            symbol = ordered.at[idx, "symbol"]
            history = known[symbol]
            mean = float(np.mean(history)) if history else 0.0
            ordered.at[idx, "zero_return"] = 0.0
            ordered.at[idx, "historical_mean"] = mean
            # Classification-only baseline: one causal universe-wide class and
            # magnitude per week, deliberately carrying no cross-sectional rank.
            ordered.at[idx, "majority_sign"] = universe_sign * universe_magnitude
            ordered.at[idx, "persistence_1w"] = ordered.at[idx, "past_week_return"]
            ordered.at[idx, "reversal_1w"] = -ordered.at[idx, "past_week_return"]
        for idx in indices:
            actual = ordered.at[idx, "actual_weekly_return"]
            if np.isfinite(actual):
                pending.append(
                    (
                        ordered.at[idx, "target_end"],
                        ordered.at[idx, "symbol"],
                        float(actual),
                    )
                )
    return ordered


def ridge_predictions(panel: pd.DataFrame) -> np.ndarray:
    fit = panel[panel["split"].isin(["train", "validation"])]
    test = panel[panel["split"] == "test"]
    model = Ridge(alpha=1.0)
    model.fit(np.stack(fit["ridge_features"]), fit["actual_weekly_return"].to_numpy())
    return model.predict(np.stack(test["ridge_features"])).astype(float)
