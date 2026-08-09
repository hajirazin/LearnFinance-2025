"""Fixed-shape SAC v3 state packing and cross-sectional feature math."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_ASSETS = 30
MIN_ELIGIBLE_ASSETS = 10
ASSET_FEATURES = 7
GLOBAL_FEATURES = 5
ACTION_DIM = MAX_ASSETS + 1
LEARNED_STATE_DIM = MAX_ASSETS * ASSET_FEATURES + GLOBAL_FEATURES
STATE_DIM = LEARNED_STATE_DIM + MAX_ASSETS

SAC_RAW_SIGNAL_NAMES = (
    "momentum_1w",
    "momentum_4w",
    "momentum_12_1",
    "news_sentiment",
    "realized_vol_20d",
)
SAC_ASSET_FEATURE_NAMES = (
    "patchtst_pred_weekly_cs_rank",
    "momentum_1w_cs_rank",
    "momentum_4w_cs_rank",
    "momentum_12_1_cs_rank",
    "news_sentiment_cs_rank",
    "realized_vol_20d_cs_rank",
    "current_weight",
)
SAC_GLOBAL_FEATURE_NAMES = (
    "median_patchtst_raw",
    "fraction_patchtst_positive",
    "p_calm",
    "p_stress",
    "cash_weight",
)


@dataclass(frozen=True)
class UnpackedState:
    """Structured views unpacked from a serialized SAC state carrier."""

    asset_features: np.ndarray
    globals: np.ndarray
    asset_mask: np.ndarray


@dataclass(frozen=True)
class PortfolioState:
    """Current long-only portfolio weights."""

    current_weights: dict[str, float]
    cash_value: float
    portfolio_value: float
    last_turnover: float = 0.0

    def to_weight_array(self, symbol_order: list[str]) -> np.ndarray:
        weights = np.zeros(ACTION_DIM, dtype=np.float64)
        for index, symbol in enumerate(symbol_order):
            weights[index] = self.current_weights.get(symbol, 0.0)
        weights[-1] = self.current_weights.get("CASH", 0.0)
        return weights


@dataclass(frozen=True)
class StateSchema:
    """SAC v3 fixed state schema.

    ``n_stocks`` is accepted only to make stale call sites fail with a useful
    error when they exceed the fixed slot envelope. It never changes shapes.
    """

    n_stocks: int = MAX_ASSETS

    def __post_init__(self) -> None:
        if not 1 <= self.n_stocks <= MAX_ASSETS:
            raise ValueError(f"n_stocks must be in [1, {MAX_ASSETS}]")

    n_signals_per_stock = len(SAC_RAW_SIGNAL_NAMES)
    n_forecasts_per_stock = 1
    n_portfolio_weights = ACTION_DIM
    state_dim = STATE_DIM
    action_dim = ACTION_DIM

    @property
    def signal_names(self) -> list[str]:
        return list(SAC_RAW_SIGNAL_NAMES)

    def get_asset_feature_indices(self) -> tuple[int, int]:
        return 0, MAX_ASSETS * ASSET_FEATURES

    def get_global_indices(self) -> tuple[int, int]:
        return MAX_ASSETS * ASSET_FEATURES, LEARNED_STATE_DIM

    def get_mask_indices(self) -> tuple[int, int]:
        return LEARNED_STATE_DIM, STATE_DIM


def cross_sectional_rank(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks over valid assets, mapped exactly to ``[-1, 1]``."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("rank values must be a non-empty finite vector")
    if values.size == 1:
        return np.zeros(1, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    return 2.0 * (ranks - 1.0) / (values.size - 1.0) - 1.0


def pack_state(
    asset_features: np.ndarray,
    globals_: np.ndarray,
    asset_mask: np.ndarray,
) -> np.ndarray:
    """Serialize token features, globals, and auxiliary mask into 245 values."""
    assets = np.asarray(asset_features, dtype=np.float64)
    globals_array = np.asarray(globals_, dtype=np.float64)
    mask = np.asarray(asset_mask, dtype=bool)
    if assets.shape != (MAX_ASSETS, ASSET_FEATURES):
        raise ValueError(
            f"asset_features must have shape ({MAX_ASSETS}, {ASSET_FEATURES})"
        )
    if globals_array.shape != (GLOBAL_FEATURES,):
        raise ValueError(f"globals must have shape ({GLOBAL_FEATURES},)")
    if mask.shape != (MAX_ASSETS,) or not np.any(mask):
        raise ValueError(
            f"asset_mask must have shape ({MAX_ASSETS},) with >=1 valid asset"
        )
    if not np.all(np.isfinite(assets[mask])) or not np.all(np.isfinite(globals_array)):
        raise ValueError("valid asset features and globals must be finite")
    canonical_assets = assets.copy()
    canonical_assets[~mask] = 0.0
    return np.concatenate(
        (canonical_assets.reshape(-1), globals_array, mask.astype(np.float64))
    )


def unpack_state(state: np.ndarray) -> UnpackedState:
    """Unpack one serialized state and validate its mask/canonical padding."""
    vector = np.asarray(state, dtype=np.float64)
    if vector.shape != (STATE_DIM,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"state must be a finite vector of shape ({STATE_DIM},)")
    assets_end = MAX_ASSETS * ASSET_FEATURES
    assets = vector[:assets_end].reshape(MAX_ASSETS, ASSET_FEATURES)
    globals_ = vector[assets_end:LEARNED_STATE_DIM]
    raw_mask = vector[LEARNED_STATE_DIM:]
    if not np.all((raw_mask == 0.0) | (raw_mask == 1.0)) or not np.any(raw_mask):
        raise ValueError("state mask must be binary with at least one valid asset")
    mask = raw_mask.astype(bool)
    if np.any(assets[~mask] != 0.0):
        raise ValueError("padded asset features must be canonical zero")
    return UnpackedState(assets, globals_, mask)


def unpack_state_batch(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized unpack for network/scaler input."""
    batch = np.asarray(states)
    if batch.ndim == 1:
        batch = batch.reshape(1, -1)
    if batch.ndim != 2 or batch.shape[1] != STATE_DIM:
        raise ValueError(f"states must have trailing dimension {STATE_DIM}")
    assets_end = MAX_ASSETS * ASSET_FEATURES
    assets = batch[:, :assets_end].reshape(-1, MAX_ASSETS, ASSET_FEATURES)
    globals_ = batch[:, assets_end:LEARNED_STATE_DIM]
    raw_mask = batch[:, LEARNED_STATE_DIM:]
    if not np.all((raw_mask == 0) | (raw_mask == 1)) or np.any(raw_mask.sum(1) < 1):
        raise ValueError(
            "every state mask must be binary with at least one valid asset"
        )
    return assets, globals_, raw_mask.astype(bool)


def build_state_vector(
    signals: dict[str, dict[str, float]],
    patchtst_forecasts: dict[str, float],
    portfolio_weights: np.ndarray,
    symbol_order: list[str],
    schema: StateSchema | None = None,
    *,
    asset_mask: np.ndarray | None = None,
    regime_probabilities: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Build the ranked, padded SAC v3 state from raw per-asset observations."""
    del schema
    n_symbols = len(symbol_order)
    if not 1 <= n_symbols <= MAX_ASSETS or len(set(symbol_order)) != n_symbols:
        raise ValueError(f"symbol_order must contain 1..{MAX_ASSETS} unique symbols")
    weights = np.asarray(portfolio_weights, dtype=np.float64)
    if weights.shape == (n_symbols + 1,):
        padded_weights = np.zeros(ACTION_DIM, dtype=np.float64)
        padded_weights[:n_symbols] = weights[:-1]
        padded_weights[-1] = weights[-1]
        weights = padded_weights
    if weights.shape != (ACTION_DIM,) or not np.all(np.isfinite(weights)):
        raise ValueError(f"portfolio_weights must have shape ({ACTION_DIM},)")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError("portfolio_weights must be a nonnegative simplex")
    mask = np.zeros(MAX_ASSETS, dtype=bool)
    mask[:n_symbols] = True
    if asset_mask is not None:
        supplied = np.asarray(asset_mask, dtype=bool)
        if supplied.shape == (n_symbols,):
            mask[:n_symbols] = supplied
        elif supplied.shape == (MAX_ASSETS,):
            mask = supplied.copy()
            mask[n_symbols:] = False
        else:
            raise ValueError("asset_mask has invalid shape")
    if not np.any(mask):
        raise ValueError("at least one asset must be eligible")

    valid_indices = np.flatnonzero(mask)
    raw = np.empty((len(valid_indices), 6), dtype=np.float64)
    for row, index in enumerate(valid_indices):
        symbol = symbol_order[index]
        if symbol not in signals or symbol not in patchtst_forecasts:
            raise ValueError(f"missing SAC v3 observations for eligible asset {symbol}")
        raw[row, 0] = float(patchtst_forecasts[symbol])
        for column, name in enumerate(SAC_RAW_SIGNAL_NAMES, start=1):
            try:
                raw[row, column] = float(signals[symbol][name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"missing SAC v3 signal {name!r} for {symbol}"
                ) from exc
    if not np.all(np.isfinite(raw)):
        raise ValueError("eligible SAC v3 observations must be finite")

    assets = np.zeros((MAX_ASSETS, ASSET_FEATURES), dtype=np.float64)
    for column in range(6):
        assets[valid_indices, column] = cross_sectional_rank(raw[:, column])
    assets[valid_indices, 6] = weights[valid_indices]
    p_calm, p_stress = map(float, regime_probabilities)
    if min(p_calm, p_stress) < 0 or p_calm + p_stress > 1.0 + 1e-8:
        raise ValueError("regime probabilities must be nonnegative and sum to <= 1")
    globals_ = np.asarray(
        [np.median(raw[:, 0]), np.mean(raw[:, 0] > 0), p_calm, p_stress, weights[-1]],
        dtype=np.float64,
    )
    return pack_state(assets, globals_, mask)


def extract_portfolio_weights_from_state(
    state: np.ndarray, schema: StateSchema | None = None
) -> np.ndarray:
    """Return padded stock weights plus CASH from a v3 state."""
    del schema
    unpacked = unpack_state(state)
    weights = np.zeros(ACTION_DIM, dtype=np.float64)
    weights[:MAX_ASSETS] = unpacked.asset_features[:, 6]
    weights[-1] = unpacked.globals[-1]
    return weights


def state_to_dict(
    state: np.ndarray, symbol_order: list[str], schema: StateSchema | None = None
) -> dict[str, Any]:
    """Return an audit-friendly structured representation."""
    del schema
    unpacked = unpack_state(state)
    return {
        "asset_features": {
            symbol: dict(
                zip(
                    SAC_ASSET_FEATURE_NAMES,
                    map(float, unpacked.asset_features[index]),
                    strict=True,
                )
            )
            for index, symbol in enumerate(symbol_order)
        },
        "globals": dict(
            zip(SAC_GLOBAL_FEATURE_NAMES, map(float, unpacked.globals), strict=True)
        ),
        "asset_mask": unpacked.asset_mask.astype(int).tolist(),
    }
