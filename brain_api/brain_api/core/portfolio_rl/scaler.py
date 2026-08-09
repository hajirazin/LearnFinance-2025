"""SAC v3 scaler: standardize only the raw PatchTST median global."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain_api.core.portfolio_rl.state import (
    LEARNED_STATE_DIM,
    MAX_ASSETS,
    STATE_DIM,
)

MEDIAN_GLOBAL_INDEX = MAX_ASSETS * 7


@dataclass
class PortfolioScaler:
    """Training-fold statistics for the sole standardized v3 input."""

    median_mean: float = 0.0
    median_scale: float = 1.0
    is_fitted: bool = False

    @classmethod
    def create(cls, n_stocks: int = MAX_ASSETS) -> PortfolioScaler:
        if not 1 <= n_stocks <= MAX_ASSETS:
            raise ValueError(f"n_stocks must be in [1, {MAX_ASSETS}]")
        return cls()

    @property
    def n_features_to_scale(self) -> int:
        return 1

    @property
    def n_portfolio_weights(self) -> int:
        return MAX_ASSETS + 1

    def fit(self, states: np.ndarray) -> PortfolioScaler:
        batch = np.asarray(states, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != STATE_DIM:
            raise ValueError(f"states must have shape (n, {STATE_DIM})")
        return self.fit_patchtst_medians(batch[:, MEDIAN_GLOBAL_INDEX])

    def fit_patchtst_medians(self, medians: np.ndarray) -> PortfolioScaler:
        """Fit once on every raw training-fold weekly median."""
        medians = np.asarray(medians, dtype=np.float64)
        if medians.ndim != 1 or len(medians) < 1:
            raise ValueError("raw PatchTST medians must be a non-empty vector")
        if not np.all(np.isfinite(medians)):
            raise ValueError("raw PatchTST median values must be finite")
        self.median_mean = float(np.mean(medians))
        scale = float(np.std(medians, ddof=0))
        self.median_scale = scale if scale > 0 else 1.0
        self.is_fitted = True
        return self

    def transform(self, states: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        values = np.asarray(states, dtype=np.float64)
        single = values.ndim == 1
        batch = values.reshape(1, -1) if single else values
        if batch.ndim != 2 or batch.shape[1] != STATE_DIM:
            raise ValueError(f"states must have trailing dimension {STATE_DIM}")
        result = batch.copy()
        result[:, MEDIAN_GLOBAL_INDEX] = (
            result[:, MEDIAN_GLOBAL_INDEX] - self.median_mean
        ) / self.median_scale
        # Ranks, probabilities, weights and masks are intentionally untouched.
        return result[0] if single else result

    def fit_transform(self, states: np.ndarray) -> np.ndarray:
        return self.fit(states).transform(states)

    def inverse_transform(self, states: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
        values = np.asarray(states, dtype=np.float64)
        single = values.ndim == 1
        batch = values.reshape(1, -1) if single else values
        if batch.ndim != 2 or batch.shape[1] != STATE_DIM:
            raise ValueError(f"states must have trailing dimension {STATE_DIM}")
        result = batch.copy()
        result[:, MEDIAN_GLOBAL_INDEX] = (
            result[:, MEDIAN_GLOBAL_INDEX] * self.median_scale + self.median_mean
        )
        return result[0] if single else result

    def to_dict(self) -> dict[str, float | bool | int]:
        return {
            "schema_version": 3,
            "median_mean": self.median_mean,
            "median_scale": self.median_scale,
            "is_fitted": self.is_fitted,
            "learned_state_dim": LEARNED_STATE_DIM,
        }

    def save(self, path: Path | str) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self.to_dict(), handle)

    @classmethod
    def load(cls, path: Path | str) -> PortfolioScaler:
        with Path(path).open("rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, dict) or data.get("schema_version") != 3:
            raise ValueError("legacy SAC scaler is incompatible with SAC schema v3")
        return cls(
            median_mean=float(data["median_mean"]),
            median_scale=float(data["median_scale"]),
            is_fitted=bool(data["is_fitted"]),
        )
