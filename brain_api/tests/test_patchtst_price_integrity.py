"""Regressions for missing sessions, close-only inputs and artifact geometry."""

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from transformers import PatchTSTForPrediction

from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.core.patchtst.data_loaders import align_multivariate_data
from brain_api.core.patchtst.dataset import build_dataset
from brain_api.core.patchtst.inference import (
    build_inference_features,
    run_batch_inference,
)
from brain_api.core.patchtst.price_history import completed_context_dates, session_dates
from brain_api.core.patchtst.score_validation import validate_and_collect_finite_scores
from brain_api.core.prices import load_close_prices_yfinance, load_prices_yfinance
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
from brain_api.storage.patchtst.huggingface import PatchTSTHuggingFaceModelStorage
from brain_api.storage.patchtst.local import PatchTSTModelStorage


def close_history(index):
    return pd.DataFrame(
        {"close": 100 * np.exp(np.arange(len(index)) * 0.001)}, index=index
    )


def test_close_loader_retains_valid_close_despite_missing_other_fields():
    index = pd.to_datetime(["2026-08-27", "2026-08-28"])
    frame = pd.DataFrame(
        {
            "Open": [10, 11],
            "High": [11, np.nan],
            "Low": [9, 10],
            "Close": [10, 11],
            "Volume": [100, np.nan],
        },
        index=index,
    )
    with patch("brain_api.core.prices.yf.download", return_value=frame) as download:
        close = load_close_prices_yfinance(
            ["ADR"], date(2026, 8, 27), date(2026, 8, 28)
        )
        full = load_prices_yfinance(["ADR"], date(2026, 8, 27), date(2026, 8, 28))
    assert list(close["ADR"].index) == list(index)
    assert list(full["ADR"].index) == [index[0]]
    assert download.call_args.kwargs["end"] == "2026-08-29"
    features = build_inference_features(
        "ADR", close["ADR"], PatchTSTConfig(context_length=1), date(2026, 8, 31)
    )
    np.testing.assert_allclose(features.features, [[np.log(11 / 10)]])


@pytest.mark.parametrize("individual", [True, False])
def test_close_loader_accepts_close_only_provider_frame(individual):
    frame = pd.DataFrame(
        {"Close": [10.0, 11.0]}, index=pd.to_datetime(["2026-08-27", "2026-08-28"])
    )
    with (
        patch(
            "brain_api.core.prices.yf.download",
            return_value=pd.DataFrame() if individual else frame,
        ),
        patch("brain_api.core.prices.yf.Ticker") as ticker,
    ):
        ticker.return_value.history.return_value = frame
        loaded = load_close_prices_yfinance(
            ["ADR"], date(2026, 8, 27), date(2026, 8, 28)
        )
    assert list(loaded["ADR"].close) == [10.0, 11.0]


@pytest.mark.parametrize(
    "corruption", ["stale", "gap", "zero", "nan", "duplicate", "reverse"]
)
def test_inference_rejects_corrupt_session_context(corruption):
    config = PatchTSTConfig(context_length=5)
    index = completed_context_dates(date(2026, 9, 28), 10)
    frame = close_history(index)
    if corruption == "stale":
        frame = frame.iloc[:-1]
    elif corruption == "gap":
        frame = frame.drop(index[-3])
    elif corruption in {"zero", "nan"}:
        frame.loc[index[-3], "close"] = 0 if corruption == "zero" else np.nan
    elif corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[-1:]])
    else:
        frame = frame.iloc[::-1]
    result = build_inference_features("AAA", frame, config, date(2026, 9, 28))
    assert result.features is None
    assert not result.has_enough_history


def test_labor_day_holiday_accepts_friday_close_without_fabricating_monday():
    config = PatchTSTConfig(context_length=5)
    index = completed_context_dates(date(2026, 9, 8), 5)
    assert index[-1].date() == date(2026, 9, 4)
    prices = close_history(index).tz_localize("America/New_York")
    result = build_inference_features("AAA", prices, config, date(2026, 9, 8))
    assert result.data_end_date == date(2026, 9, 4)
    np.testing.assert_allclose(result.features, np.full((5, 1), 0.001), atol=1e-8)


def test_stale_batch_cannot_reach_top15_selector():
    config = PatchTSTConfig(context_length=5)
    frame = close_history(completed_context_dates(date(2026, 9, 28), 10)).iloc[:-1]
    model = Mock()
    artifacts = SimpleNamespace(
        config=config, model=model, feature_scaler=None, version="vtest"
    )
    symbols = [f"S{i}" for i in range(20)]
    with patch(
        "brain_api.core.prices.load_close_prices_yfinance",
        return_value=dict.fromkeys(symbols, frame),
    ):
        result = run_batch_inference(symbols, date(2026, 9, 25), artifacts=artifacts)
    model.assert_not_called()
    with pytest.raises(RuntimeError, match="below min_predictions=15"):
        validate_and_collect_finite_scores(result.predictions, 20, 15)


def test_training_keeps_missing_session_invalid_in_both_context_and_target():
    config = PatchTSTConfig(context_length=5)
    index = session_dates(date(2026, 7, 20), date(2026, 9, 30))
    original = close_history(index)
    dirty = original.drop(pd.Timestamp("2026-08-12"))
    aligned = align_multivariate_data({"AAA": dirty}, config)
    assert np.isnan(aligned["AAA"].loc["2026-08-12", "close_ret"])
    result = build_dataset(aligned, {}, config)
    assert date(2026, 8, 7) not in result.anchor_dates  # target includes missing bar
    assert date(2026, 8, 14) not in result.anchor_dates  # context includes missing bar
    assert date(2026, 8, 21) in result.anchor_dates
    assert np.isfinite(result.X).all() and np.isfinite(result.y).all()


def test_good_friday_uses_thursday_anchor_and_missing_friday_does_not():
    config = PatchTSTConfig(context_length=5)
    index = session_dates(date(2024, 3, 1), date(2024, 4, 12))
    frame = pd.DataFrame({"close_ret": np.full(len(index), 0.001)}, index=index)
    clean = build_dataset({"AAA": frame}, {}, config)
    assert date(2024, 3, 28) in clean.anchor_dates
    dirty = frame.drop(pd.Timestamp("2024-03-22"))
    missing = build_dataset({"AAA": dirty}, {}, config)
    assert date(2024, 3, 21) not in missing.anchor_dates
    assert date(2024, 3, 22) not in missing.anchor_dates


@pytest.mark.parametrize("case", ["recorded5_stride1", "current5", "current1"])
def test_checkpoint_reload_preserves_actual_geometry_and_predictions(tmp_path, case):
    recorded_stride_one = case == "recorded5_stride1"
    five_channels = case != "current1"
    config = PatchTSTConfig(
        num_input_channels=5 if five_channels else 1,
        patch_length=16 if five_channels else 10,
        stride=8 if five_channels else 5,
        d_model=8,
        num_attention_heads=2,
        ffn_dim=16,
        feature_names=["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]
        if five_channels
        else ["close_ret"],
    )
    storage = PatchTSTModelStorage(tmp_path)
    hf = config.to_hf_config()
    if recorded_stride_one:
        hf.patch_stride = 1  # Saved effective geometry must take precedence.
    model = PatchTSTForPrediction(hf).eval()
    storage.write_artifacts(
        "vold", model, StandardScaler(), config, {"version": "vold"}
    )
    config_path = storage._version_path("vold") / "config.json"
    saved = json.loads(config_path.read_text())
    expected_stride = 1 if recorded_stride_one else config.stride
    assert saved["hf_config"]["patch_stride"] == expected_stride
    assert config.hf_config is None  # Training/version hash is not mutated.
    storage.promote_version("vother")
    loaded = storage.load_version_artifacts("vold")
    assert loaded.config.to_hf_config().patch_stride == expected_stride
    assert storage.read_current_version() == "vother"
    x = torch.zeros((2, 60, config.num_input_channels))
    with torch.no_grad():
        torch.testing.assert_close(
            model(past_values=x).prediction_outputs,
            loaded.model(past_values=x).prediction_outputs,
        )


def test_unpinned_old_five_channel_geometry_is_not_reconstructed(tmp_path):
    """Retired artifacts fail strict weight loading instead of guessing geometry."""
    config = PatchTSTConfig(
        num_input_channels=5,
        patch_length=16,
        stride=8,
        d_model=8,
        num_attention_heads=2,
        ffn_dim=16,
        feature_names=["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"],
    )
    actual = config.to_hf_config()
    actual.patch_stride = 1
    model = PatchTSTForPrediction(actual)
    storage = PatchTSTModelStorage(tmp_path)
    storage.write_artifacts(
        "vold", model, StandardScaler(), config, {"version": "vold"}
    )
    config_path = storage._version_path("vold") / "config.json"
    saved = json.loads(config_path.read_text())
    saved.pop("hf_config")
    config_path.write_text(json.dumps(saved))

    with pytest.raises(RuntimeError, match="size mismatch"):
        storage.load_version_artifacts("vold")


def test_explicit_hf_cache_returns_requested_version_without_network_or_pointer_change(
    tmp_path,
):
    storage = PatchTSTModelStorage(tmp_path)
    config = PatchTSTConfig(d_model=8, num_attention_heads=2, ffn_dim=16)
    model = PatchTSTForPrediction(config.to_hf_config())
    storage.write_artifacts(
        "vold", model, StandardScaler(), config, {"version": "vold"}
    )
    storage.promote_version("vcurrent")
    hf = PatchTSTHuggingFaceModelStorage(
        repo_id="test/repo", token="test", local_cache=storage
    )
    with patch("brain_api.storage.base_huggingface.snapshot_download") as download:
        loaded = hf.download_model(version="vold")
    download.assert_not_called()
    assert loaded.version == "vold"
    assert storage.read_current_version() == "vcurrent"
    assert hf._artifact_config(config, model)["hf_config"]["patch_stride"] == 5


def test_hf_upload_and_snapshot_pin_effective_architecture(tmp_path):
    config = PatchTSTConfig(d_model=8, num_attention_heads=2, ffn_dim=16)
    actual = config.to_hf_config()
    actual.patch_stride = 1
    model = PatchTSTForPrediction(actual)
    scaler = StandardScaler()
    hf = PatchTSTHuggingFaceModelStorage(
        repo_id="test/repo", token="test", local_cache=PatchTSTModelStorage(tmp_path)
    )
    hf.api = Mock()
    uploaded = []

    def capture_upload(**kwargs):
        from pathlib import Path

        uploaded.append(
            json.loads((Path(kwargs["folder_path"]) / "config.json").read_text())
        )

    hf.api.upload_folder.side_effect = capture_upload
    hf.upload_model("vtest", model, scaler, config, {})
    assert uploaded[0]["hf_config"]["patch_stride"] == 1
    snapshots = SnapshotLocalStorage("patchtst", base_path=tmp_path)
    snapshots.write_snapshot(
        cutoff_date=date(2025, 12, 31),
        snapshot_digest="aaaaaaaaaaaa",
        model=model,
        feature_scaler=scaler,
        config=config,
        metadata={},
    )
    loaded = snapshots.load_snapshot(date(2025, 12, 31))
    assert loaded.model.config.patch_stride == 1
    assert config.hf_config is None
