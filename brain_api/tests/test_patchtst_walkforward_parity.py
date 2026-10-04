"""SAC historical PatchTST must see the same completed closes as live inference."""

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from brain_api.core.inference_utils import compute_week_boundaries
from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.core.patchtst.inference import build_inference_features, run_inference
from brain_api.core.patchtst.price_history import completed_context_dates
from brain_api.core.portfolio_rl.walkforward import (
    SnapshotInferenceError,
    _predict_single_week_patchtst,
    _run_patchtst_snapshot_inference,
)


class ContextModel(torch.nn.Module):
    """Capture the actual input and make predictions depend on its last return."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.inputs = []

    def forward(self, past_values):
        assert not torch.is_grad_enabled()
        assert not self.training
        self.inputs.append(past_values.detach().clone())
        return SimpleNamespace(
            prediction_outputs=past_values[:, -1:, :].repeat(1, 5, 1)
        )


def history_before(decision, context_length=5):
    index = completed_context_dates(decision, context_length + 5)
    # Nonconstant returns make dropped/shifted closes observable in the tensor.
    return pd.DataFrame(
        {"close": 100 * np.exp(np.arange(len(index)) ** 2 * 0.001)}, index=index
    )


def predict(model, frame, config, actor_cutoff):
    return _predict_single_week_patchtst(
        model=model,
        scaler=Mock(side_effect=AssertionError("scaler must not be used")),
        config=config,
        weekly_idx=0,
        weekly_dates=pd.DatetimeIndex([actor_cutoff]),
        daily_ohlcv=frame,
        symbol="AAA",
    )


@pytest.mark.parametrize(
    "decision,actor_cutoff",
    [
        (date(2026, 9, 28), date(2026, 9, 25)),  # normal week
        (date(2024, 4, 1), date(2024, 3, 29)),  # calendar Good Friday cutoff
        (date(2024, 4, 1), date(2024, 3, 28)),  # last completed session
        (date(2026, 9, 7), date(2026, 9, 4)),  # Labor Day Monday
        (date(2026, 9, 8), date(2026, 9, 4)),  # first session after Labor Day
        (date(2026, 1, 5), date(2026, 1, 2)),  # New Year in context
    ],
)
@pytest.mark.parametrize("timezone", [None, "America/New_York"])
@pytest.mark.parametrize("context_length", [5, 60])
def test_historical_inputs_and_returns_match_live(
    decision, actor_cutoff, timezone, context_length
):
    config = PatchTSTConfig(context_length=context_length)
    frame = history_before(decision, context_length)
    # Future daily bars must never enter historical inference.
    frame.loc[pd.Timestamp(decision), "close"] = 999999
    if timezone:
        frame = frame.tz_localize(timezone)
    model = ContextModel()
    historical = predict(model, frame, config, actor_cutoff)
    live_features = build_inference_features("AAA", frame, config, decision)
    live = run_inference(
        model, None, [live_features], compute_week_boundaries(decision), config
    )[0]

    torch.testing.assert_close(model.inputs[0], model.inputs[1], rtol=0, atol=0)
    # Independent oracle: exactly context_length + 1 completed closes.
    closes = (
        frame.loc[frame.index.date <= actor_cutoff, "close"]
        .iloc[-context_length - 1 :]
        .values
    )
    expected = np.diff(np.log(closes)).reshape(1, context_length, 1)
    np.testing.assert_allclose(model.inputs[0].numpy(), expected, rtol=1e-6)
    assert historical == pytest.approx(np.exp(5 * expected[0, -1, 0]) - 1)
    # Live response intentionally rounds percentage values to four decimals.
    assert historical == pytest.approx(live.predicted_weekly_return_pct / 100, abs=5e-7)


@pytest.mark.parametrize(
    "corruption", ["stale", "gap", "zero", "nan", "inf", "duplicate", "reverse"]
)
def test_bad_context_never_reaches_snapshot_model(corruption):
    decision = date(2026, 9, 28)
    config = PatchTSTConfig(context_length=5)
    frame = history_before(decision)
    if corruption == "stale":
        frame = frame.iloc[:-1]
    elif corruption == "gap":
        frame = frame.drop(frame.index[-3])
    elif corruption in {"zero", "nan", "inf"}:
        frame.iloc[-3, 0] = {"zero": 0, "nan": np.nan, "inf": np.inf}[corruption]
    elif corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[-1:]])
    else:
        frame = frame.iloc[::-1]
    model = ContextModel()

    with pytest.raises(SnapshotInferenceError, match="completed-session context"):
        predict(model, frame, config, date(2026, 9, 25))
    assert not model.inputs


@pytest.mark.parametrize("bad_output", [np.nan, np.inf, 1000.0])
def test_nonfinite_compounded_prediction_raises(bad_output):
    model = ContextModel()
    model.forward = lambda past_values: SimpleNamespace(
        prediction_outputs=torch.full((1, 5, 1), bad_output)
    )
    with (
        np.errstate(over="ignore"),
        pytest.raises(SnapshotInferenceError, match="Non-finite"),
    ):
        predict(
            model,
            history_before(date(2026, 9, 28)),
            PatchTSTConfig(context_length=5),
            date(2026, 9, 25),
        )


def test_snapshot_runner_retains_close_with_missing_other_yahoo_fields(monkeypatch):
    """Cover real loader + snapshot runner; no unrelated OHLCV row filtering."""
    decision = date(2026, 9, 28)
    frame = history_before(decision).rename(columns={"close": "Close"})
    for field in ("Open", "High", "Low", "Volume"):
        frame[field] = np.nan
    download = Mock(return_value=frame)
    monkeypatch.setattr("brain_api.core.prices.yf.download", download)
    model = ContextModel()
    config = PatchTSTConfig(context_length=5)
    artifacts = SimpleNamespace(model=model, config=config, feature_scaler=None)

    predictions = _run_patchtst_snapshot_inference(
        artifacts, [0], pd.DatetimeIndex(["2026-09-25"]), "AAA"
    )

    assert len(predictions) == 1
    assert np.isfinite(predictions[0])
    assert len(model.inputs) == 1
    np.testing.assert_allclose(
        model.inputs[0].numpy()[0, :, 0],
        np.diff(np.log(frame.Close.values[-6:])),
        rtol=1e-6,
    )
    # Shared price loader handles the inclusive actor cutoff exactly once.
    assert download.call_args.kwargs["end"] == str(
        date(2026, 9, 25) + timedelta(days=1)
    )
