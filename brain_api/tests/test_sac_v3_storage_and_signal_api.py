"""SAC v3 artifact compatibility and raw-input API contracts."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from brain_api.core.portfolio_rl.sac_networks import GaussianActor, TwinCritic
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.portfolio_rl.state import LEARNED_STATE_DIM, STATE_DIM
from brain_api.core.sac import DEFAULT_SAC_CONFIG
from brain_api.core.sac.market_sessions import xnys_session_dates
from brain_api.core.sac.regime_hmm import RegimeHMMArtifact
from brain_api.main import app
from brain_api.routes.training.sac._market_history import (
    extract_aligned_market_history,
)
from brain_api.storage.sac.artifacts import (
    SACArtifactCompatibilityError,
    SACV3AuxiliaryArtifacts,
    validate_sac_v3_metadata,
)
from brain_api.storage.sac.huggingface import SACHuggingFaceModelStorage
from brain_api.storage.sac.local import (
    SACHalalFilteredModelStorage,
    create_sac_metadata,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(to_dict=lambda: {"seed": 42})


def _metadata(symbols: list[str]) -> dict[str, object]:
    return create_sac_metadata(
        version="v2026-08-09-v3",
        data_window_start="2020-01-01",
        data_window_end="2026-08-07",
        symbols=symbols,
        config=_config(),
        promoted=False,
        prior_version=None,
        actor_loss=0.1,
        critic_loss=0.2,
        avg_episode_return=0.0,
        avg_episode_sharpe=0.0,
        eval_sharpe=0.0,
        eval_cagr=0.0,
        eval_max_drawdown=0.0,
    )


def test_v3_metadata_rejects_legacy_and_noncanonical_slots():
    with pytest.raises(SACArtifactCompatibilityError, match="Legacy SAC artifact"):
        validate_sac_v3_metadata({"version": "legacy"}, ["AAPL"])

    metadata = _metadata(["AAPL", "MSFT"])
    metadata["symbol_to_slot"] = {"AAPL": 1, "MSFT": 0}
    with pytest.raises(SACArtifactCompatibilityError, match="symbol_to_slot"):
        validate_sac_v3_metadata(metadata, ["AAPL", "MSFT"])


def test_filesystem_loader_rejects_legacy_before_loading_weights(tmp_path):
    storage = SACHalalFilteredModelStorage(base_path=tmp_path)
    version = "legacy-flat"
    version_dir = tmp_path / "models" / "sac_halal_filtered" / version
    version_dir.mkdir(parents=True)
    (version_dir / "metadata.json").write_text('{"version":"legacy-flat"}')
    (version_dir / "symbol_order.json").write_text('["AAPL"]')

    assert storage.version_exists(version) is False
    with pytest.raises(SACArtifactCompatibilityError, match="Legacy SAC artifact"):
        storage.load_artifacts(version)


def test_v3_auxiliary_round_trip_and_missing_hmm_rejection():
    hmm = _regime_hmm()
    auxiliary = SACV3AuxiliaryArtifacts(
        regime_hmm=hmm,
        median_patchtst_scaler={"mean": 0.01, "scale": 0.02},
        audit_metadata={"fit_window": {"start": "2020-01-01", "end": "2026-08-07"}},
    )
    loaded = SACV3AuxiliaryArtifacts.from_dict(auxiliary.to_dict())
    assert loaded.training_cutoff_date == "2026-08-07"
    assert loaded.training_cutoff_posterior == pytest.approx((0.7, 0.2, 0.1))
    assert loaded.median_patchtst_scaler == auxiliary.median_patchtst_scaler

    incomplete = auxiliary.to_dict()
    incomplete.pop("regime_hmm")
    with pytest.raises(SACArtifactCompatibilityError, match="regime_hmm"):
        SACV3AuxiliaryArtifacts.from_dict(incomplete)


def _regime_hmm() -> RegimeHMMArtifact:
    return RegimeHMMArtifact(
        start_probability=np.asarray([0.4, 0.3, 0.3]),
        transition=np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        means=np.zeros((3, 4)),
        variances=np.ones((3, 4)),
        scaler_mean=np.zeros(4),
        scaler_scale=np.ones(4),
        label_map={"calm": 0, "transition": 1, "stress": 2},
        terminal_posterior=np.asarray([0.7, 0.2, 0.1]),
        training_cutoff_date=date(2026, 8, 7),
        fit_start_date=date(2020, 1, 1),
        iterations=10,
        log_likelihood=-1.0,
        spy_tail=np.linspace(600.0, 620.0, 21),
        vix_tail=np.linspace(15.0, 17.0, 21),
        tail_dates=list(pd.bdate_range(end="2026-08-07", periods=21).date),
    )


def _write_v3_artifact(
    storage: SACHalalFilteredModelStorage, version: str
) -> list[str]:
    symbols = ["AAPL", "MSFT"]
    actor = GaussianActor(hidden_sizes=DEFAULT_SAC_CONFIG.hidden_sizes)
    critic = TwinCritic(hidden_sizes=DEFAULT_SAC_CONFIG.hidden_sizes)
    critic_target = TwinCritic(hidden_sizes=DEFAULT_SAC_CONFIG.hidden_sizes)
    states = np.zeros((2, STATE_DIM), dtype=float)
    states[:, LEARNED_STATE_DIM : LEARNED_STATE_DIM + len(symbols)] = 1.0
    states[:, 210] = [0.01, 0.03]
    scaler = PortfolioScaler.create().fit(states)
    metadata = create_sac_metadata(
        version=version,
        data_window_start="2020-01-01",
        data_window_end="2026-08-07",
        symbols=symbols,
        config=DEFAULT_SAC_CONFIG,
        promoted=False,
        prior_version=None,
        actor_loss=0.1,
        critic_loss=0.2,
        avg_episode_return=0.0,
        avg_episode_sharpe=0.0,
        eval_sharpe=0.0,
        eval_cagr=0.0,
        eval_max_drawdown=0.0,
    )
    auxiliary = SACV3AuxiliaryArtifacts(
        regime_hmm=_regime_hmm(),
        median_patchtst_scaler={
            "mean": scaler.median_mean,
            "scale": scaler.median_scale,
        },
        audit_metadata={"data_sources": ["yfinance", "news", "patchtst"]},
    )
    storage.write_artifacts(
        version,
        actor,
        critic,
        critic_target,
        torch.tensor(0.0),
        scaler,
        DEFAULT_SAC_CONFIG,
        symbols,
        metadata,
        auxiliary,
    )
    return symbols


def test_v3_filesystem_round_trip_does_not_mutate_current_pointer(tmp_path):
    storage = SACHalalFilteredModelStorage(base_path=tmp_path)
    version = "v2026-08-09-roundtrip"
    symbols = _write_v3_artifact(storage, version)

    assert storage.read_current_version() is None
    loaded = storage.load_artifacts(version)
    assert loaded.metadata["sac_schema_version"] == 3
    assert loaded.v3_auxiliary.training_cutoff_date == "2026-08-07"
    assert loaded.symbol_order == symbols


def test_huggingface_v3_download_round_trip_uses_fixed_network_dimensions(tmp_path):
    storage = SACHalalFilteredModelStorage(base_path=tmp_path)
    version = "v2026-08-09-hf-roundtrip"
    symbols = _write_v3_artifact(storage, version)
    snapshot_path = tmp_path / "models" / storage.bucket_name / version
    hf_storage = SACHuggingFaceModelStorage(
        repo_id="test-owner/test-sac-v3",
        token="test-token",
        local_cache=storage,
    )

    with patch(
        "brain_api.storage.sac.huggingface.snapshot_download",
        return_value=str(snapshot_path),
    ):
        loaded = hf_storage.download_model(version=version, use_cache=False)

    assert loaded.symbol_order == symbols
    assert loaded.actor.state_dim == STATE_DIM
    assert loaded.actor.action_dim == 31


def _price_frame(values: list[float], dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": [100.0] * len(values),
        },
        index=dates,
    )


@pytest.mark.parametrize("missing_symbol", ["SPY", "^VIX"])
def test_training_market_history_rejects_missing_required_session(
    missing_symbol: str,
):
    dates = pd.bdate_range(end="2026-08-07", periods=21)
    complete = {
        "SPY": _price_frame(list(np.linspace(600.0, 620.0, 21)), dates),
        "^VIX": _price_frame(list(np.linspace(15.0, 17.0, 21)), dates),
    }
    market_dates, spy, vix = extract_aligned_market_history(
        complete, start_date=date(2026, 7, 10), completed_through=date(2026, 8, 7)
    )
    assert market_dates[-1] == date(2026, 8, 7)
    assert spy[-1] == pytest.approx(620.0)
    assert vix[-1] == pytest.approx(17.0)

    missing_date = dates[-2]
    complete[missing_symbol] = complete[missing_symbol].drop(missing_date)
    with pytest.raises(ValueError) as error:
        extract_aligned_market_history(
            complete,
            start_date=date(2026, 7, 10),
            completed_through=date(2026, 8, 7),
        )
    assert missing_symbol in str(error.value)
    assert missing_date.date().isoformat() in str(error.value)


def test_training_market_history_ignores_vix_memorial_day_provider_row():
    expected_dates = xnys_session_dates(date(2026, 4, 27), date(2026, 6, 2))
    expected_index = pd.DatetimeIndex(expected_dates)
    memorial_day = pd.Timestamp("2026-05-25")
    vix_index = expected_index.union(pd.DatetimeIndex([memorial_day]))
    spy_frame = _price_frame(
        list(np.linspace(600.0, 625.0, len(expected_index))), expected_index
    )
    vix_frame = _price_frame(list(np.linspace(15.0, 18.0, len(vix_index))), vix_index)
    vix_frame["close"] = vix_frame["close"].astype(object)
    vix_frame.loc[memorial_day, "close"] = "ignored non-session value"

    market_dates, spy, vix = extract_aligned_market_history(
        {"SPY": spy_frame, "^VIX": vix_frame},
        start_date=date(2026, 4, 27),
        completed_through=date(2026, 6, 2),
    )

    assert memorial_day.date() not in expected_dates
    assert market_dates == expected_dates
    np.testing.assert_array_equal(
        spy, spy_frame.loc[expected_index, "close"].to_numpy()
    )
    np.testing.assert_array_equal(
        vix, vix_frame.loc[expected_index, "close"].astype(float).to_numpy()
    )


def test_training_market_history_rejects_joint_early_ending():
    dates = pd.bdate_range("2026-08-03", "2026-08-07")
    early = {
        "SPY": _price_frame([600.0] * 4, dates[:-1]),
        "^VIX": _price_frame([16.0] * 4, dates[:-1]),
    }
    with pytest.raises(ValueError, match="2026-08-07"):
        extract_aligned_market_history(
            early,
            start_date=date(2026, 8, 3),
            completed_through=date(2026, 8, 7),
        )


def test_prices_api_returns_adjusted_history_and_short_asset():
    dates = pd.bdate_range(end="2026-08-07", periods=253)
    loaded = {
        "AAPL": _price_frame([100.0 + i for i in range(253)], dates),
        "NEW": _price_frame([10.0, 11.0], dates[-2:]),
    }
    with patch(
        "brain_api.routes.signals.endpoints.load_prices_yfinance",
        return_value=loaded,
    ):
        response = TestClient(app).post(
            "/signals/prices",
            json={
                "symbols": ["AAPL", "NEW"],
                "as_of_date": "2026-08-07",
                "lookback_bars": 253,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["adjusted_closes"]["AAPL"]) == 253
    assert body["adjusted_closes"]["NEW"] == [10.0, 11.0]
    assert body["provenance"]["price_basis"] == "adjusted"
    assert "closes" not in body


def test_prices_api_enforces_fixed_policy_capacity():
    response = TestClient(app).post(
        "/signals/prices",
        json={
            "symbols": [f"S{i}" for i in range(31)],
            "as_of_date": "2026-08-07",
        },
    )
    assert response.status_code == 422


def test_sac_api_rejects_bundle_over_capacity_before_artifact_loading():
    symbols = [f"S{i}" for i in range(31)]
    response = TestClient(app).post(
        "/inference/sac?universe=halal_filtered",
        json={
            "as_of": "2026-08-10T09:00:00-04:00",
            "as_of_date": "2026-08-07",
            "portfolio": {"cash": 1_000.0, "positions": []},
            "news_window": {
                "start_exclusive": "2026-08-03T09:00:00-04:00",
                "end_inclusive": "2026-08-10T09:00:00-04:00",
                "coverage": [],
                "events": [],
            },
            "feature_bundle": {
                "symbols": symbols,
                "adjusted_closes": {},
                "patchtst_forecasts": {},
                "market_history": [],
                "provenance": {},
            },
        },
    )
    assert response.status_code == 422


def test_sac_api_rejects_nonpositive_market_evidence_at_boundary():
    response = TestClient(app).post(
        "/inference/sac?universe=halal_filtered",
        json={
            "as_of": "2026-08-10T09:00:00-04:00",
            "as_of_date": "2026-08-07",
            "portfolio": {"cash": 1_000.0, "positions": []},
            "news_window": {
                "start_exclusive": "2026-08-03T09:00:00-04:00",
                "end_inclusive": "2026-08-10T09:00:00-04:00",
                "coverage": [
                    {
                        "symbol": "AAPL",
                        "status": "verified_empty",
                        "event_count": 0,
                        "future_revision_excluded_count": 0,
                        "sentiment_model_revision": (
                            "4556d13015211d73dccd3fdd39d39232506f3e43"
                        ),
                    }
                ],
                "events": [],
            },
            "feature_bundle": {
                "symbols": ["AAPL"],
                "adjusted_closes": {"AAPL": []},
                "patchtst_forecasts": {},
                "market_history": [
                    {
                        "date": "2026-08-07",
                        "spy_adjusted_close": 0.0,
                        "vix_close": 17.0,
                    }
                ],
                "provenance": {},
            },
        },
    )
    assert response.status_code == 422


def test_market_history_api_returns_aligned_rows_and_rejects_gaps():
    dates = pd.bdate_range("2026-08-03", "2026-08-05")
    complete = {
        "SPY": _price_frame([630.0, 631.0, 632.0], dates),
        "^VIX": _price_frame([17.0, 18.0, 16.0], dates),
    }
    client = TestClient(app)
    with patch(
        "brain_api.routes.signals.endpoints.load_prices_yfinance",
        return_value=complete,
    ):
        response = client.post(
            "/signals/market-history",
            json={"start_date": "2026-08-03", "as_of_date": "2026-08-05"},
        )
    assert response.status_code == 200
    assert response.json()["rows"][-1] == {
        "date": "2026-08-04",
        "spy_adjusted_close": 631.0,
        "vix_close": 18.0,
    }

    incomplete = {**complete, "^VIX": complete["^VIX"].drop(dates[1])}
    with patch(
        "brain_api.routes.signals.endpoints.load_prices_yfinance",
        return_value=incomplete,
    ):
        response = client.post(
            "/signals/market-history",
            json={"start_date": "2026-08-03", "as_of_date": "2026-08-05"},
        )
    assert response.status_code == 422
    assert "2026-08-04" in response.json()["detail"]


def test_market_history_api_monday_preopen_ends_at_prior_friday():
    dates = pd.DatetimeIndex(["2026-08-07", "2026-08-10"])
    complete = {
        "SPY": _price_frame([630.0, 999.0], dates),
        "^VIX": _price_frame([17.0, 999.0], dates),
    }
    with patch(
        "brain_api.routes.signals.endpoints.load_prices_yfinance",
        return_value=complete,
    ):
        response = TestClient(app).post(
            "/signals/market-history",
            json={"start_date": "2026-08-07", "as_of_date": "2026-08-10"},
        )
    assert response.status_code == 200
    assert [row["date"] for row in response.json()["rows"]] == ["2026-08-07"]
