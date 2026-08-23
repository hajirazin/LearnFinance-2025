"""PatchTST trainer requires a cross-section and checkpoints on rank IC."""

from datetime import date, timedelta

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.core.patchtst.training import train_model_pytorch


def _tiny_config() -> PatchTSTConfig:
    return PatchTSTConfig(
        context_length=32,
        patch_length=8,
        stride=8,
        d_model=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        ffn_dim=32,
        epochs=2,
        early_stopping_patience=10,
        batch_size=4,
    )


def _cross_section_panel(
    config: PatchTSTConfig,
    *,
    n_train_weeks: int = 4,
    symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enough later-week symbols that validation rank IC is defined."""
    rng = np.random.default_rng(0)
    n_channels = config.num_input_channels
    ctx = config.context_length
    horizon = config.prediction_length
    train_dates = [
        date(2024, 1, 5) + timedelta(days=7 * i) for i in range(n_train_weeks)
    ]
    val_date = date(2024, 3, 1)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    dates: list[date] = []
    labels: list[str] = []
    for train_date in train_dates:
        for symbol in symbols:
            xs.append(rng.normal(size=(ctx, n_channels)).astype(np.float32))
            ys.append(rng.normal(size=(horizon, n_channels)).astype(np.float32))
            dates.append(train_date)
            labels.append(symbol)
    for symbol in symbols:
        xs.append(rng.normal(size=(ctx, n_channels)).astype(np.float32))
        ys.append(rng.normal(size=(horizon, n_channels)).astype(np.float32))
        dates.append(val_date)
        labels.append(symbol)

    return (
        np.stack(xs),
        np.stack(ys),
        np.array(dates, dtype=object),
        np.array(labels, dtype=object),
    )


def test_nonempty_train_requires_anchor_dates_and_sample_symbols() -> None:
    config = _tiny_config()
    X, y, dates, symbols = _cross_section_panel(config)
    scaler = StandardScaler()
    with pytest.raises(ValueError, match="sample_symbols"):
        train_model_pytorch(X, y, scaler, config, anchor_dates=dates)
    with pytest.raises(ValueError, match="anchor_dates"):
        train_model_pytorch(X, y, scaler, config, sample_symbols=symbols)


def test_train_model_pytorch_returns_finite_val_rank_ic() -> None:
    config = _tiny_config()
    X, y, dates, symbols = _cross_section_panel(config)
    result = train_model_pytorch(
        X,
        y,
        StandardScaler(),
        config,
        anchor_dates=dates,
        sample_symbols=symbols,
    )
    assert np.isfinite(result.val_rank_ic)
    assert result.stopped_epoch == 2
    assert 1 <= result.best_epoch <= result.stopped_epoch


def test_empty_data_allows_nonfinite_val_rank_ic() -> None:
    config = _tiny_config()
    result = train_model_pytorch(
        np.zeros(
            (0, config.context_length, config.num_input_channels), dtype=np.float32
        ),
        np.zeros(
            (0, config.prediction_length, config.num_input_channels), dtype=np.float32
        ),
        StandardScaler(),
        config,
    )
    assert result.best_epoch == 0
    assert result.stopped_epoch == 0
    assert not np.isfinite(result.val_rank_ic)
