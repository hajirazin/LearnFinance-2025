"""API-level tests for PPO + PatchTST training and inference endpoints.

Tests focus on:
- Endpoint contract (status codes, response structure)
- Constraint enforcement (cash buffer, max position weight)
- Promotion gate behavior
"""

import os
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.core.ppo_lstm.model import PPOActorCritic
from brain_api.core.ppo_patchtst import (
    PPOPatchTSTConfig,
    PPOPatchTSTTrainingResult,
)
from brain_api.main import app
from brain_api.routes.inference import (
    get_ppo_patchtst_storage as get_inference_storage,
)
from brain_api.routes.training import (
    get_ppo_patchtst_config,
    get_ppo_patchtst_storage,
    get_top15_symbols,
)
from brain_api.storage.ppo_patchtst import PPOPatchTSTLocalStorage

# ============================================================================
# Test fixtures and mocks
# ============================================================================


def mock_symbols() -> list[str]:
    """Return a small fixed list of symbols for testing."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def mock_config() -> PPOPatchTSTConfig:
    """Return a minimal config for fast testing."""
    return PPOPatchTSTConfig(
        n_stocks=5,
        total_timesteps=100,  # Very small for fast tests
        rollout_steps=10,
        n_epochs=2,
        hidden_sizes=(16, 16),  # Small network
    )


def mock_price_loader(symbols, start_date, end_date):
    """Return mock price data for testing."""
    import pandas as pd

    dates = pd.date_range(start=start_date, end=end_date, freq="W-FRI")[:100]
    prices = {}
    for i, symbol in enumerate(symbols):
        base = 100 + i * 10
        prices[symbol] = pd.DataFrame(
            {
                "open": [base] * len(dates),
                "high": [base * 1.01] * len(dates),
                "low": [base * 0.99] * len(dates),
                "close": [base * 1.005] * len(dates),
                "volume": [1000000] * len(dates),
            },
            index=dates,
        )
    return prices


def create_mock_training_result(config: PPOPatchTSTConfig) -> PPOPatchTSTTrainingResult:
    """Create a mock training result for testing."""
    n_stocks = config.n_stocks
    state_dim = n_stocks * 7 + n_stocks + n_stocks + 1
    action_dim = n_stocks + 1

    model = PPOActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
    )

    scaler = PortfolioScaler.create(n_stocks=n_stocks)
    dummy_states = np.random.randn(10, state_dim)
    scaler.fit(dummy_states)

    return PPOPatchTSTTrainingResult(
        model=model,
        scaler=scaler,
        config=config,
        symbol_order=mock_symbols()[:n_stocks],
        final_policy_loss=0.1,
        final_value_loss=0.05,
        avg_episode_return=0.02,
        avg_episode_sharpe=0.5,
        eval_sharpe=0.6,
        eval_cagr=0.10,
        eval_max_drawdown=0.15,
    )


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield PPOPatchTSTLocalStorage(base_path=tmpdir)


@pytest.fixture
def client_with_mocks(temp_storage):
    """Create test client with mocked dependencies."""
    app.dependency_overrides.clear()

    app.dependency_overrides[get_ppo_patchtst_storage] = lambda: temp_storage
    app.dependency_overrides[get_top15_symbols] = mock_symbols
    app.dependency_overrides[get_ppo_patchtst_config] = mock_config
    app.dependency_overrides[get_inference_storage] = lambda: temp_storage

    os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "5"
    os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()
    os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
    os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


@pytest.fixture
def trained_model_storage():
    """Create storage with a pre-trained model for inference tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PPOPatchTSTLocalStorage(base_path=tmpdir)

        config = mock_config()
        result = create_mock_training_result(config)

        version = "v2025-01-01-test123"
        metadata = {
            "model_type": "ppo_patchtst",
            "version": version,
            "data_window": {"start": "2020-01-01", "end": "2025-01-01"},
            "symbols": result.symbol_order,
            "config": config.to_dict(),
            "metrics": {
                "eval_sharpe": result.eval_sharpe,
                "eval_cagr": result.eval_cagr,
            },
            "promoted": True,
            "prior_version": None,
        }

        storage.write_artifacts(
            version=version,
            model=result.model,
            scaler=result.scaler,
            config=config,
            symbol_order=result.symbol_order,
            metadata=metadata,
        )
        storage.promote_version(version)

        yield storage


@pytest.fixture
def client_for_inference(trained_model_storage):
    """Create test client with pre-trained model for inference and fine-tune tests."""
    app.dependency_overrides.clear()
    # Override both inference and training storage with same pre-trained storage
    app.dependency_overrides[get_inference_storage] = lambda: trained_model_storage
    app.dependency_overrides[get_ppo_patchtst_storage] = lambda: trained_model_storage
    app.dependency_overrides[get_top15_symbols] = mock_symbols

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


# ============================================================================
# Training endpoint tests
# ============================================================================


def test_train_ppo_patchtst_returns_200(client_with_mocks, monkeypatch):
    """POST /train/ppo_patchtst returns 200."""
    from brain_api.routes.training import ppo_patchtst

    monkeypatch.setattr(ppo_patchtst, "load_prices_yfinance", mock_price_loader)

    response = client_with_mocks.post("/train/ppo_patchtst/full", json={})
    assert response.status_code == 200


def test_train_ppo_patchtst_returns_required_fields(client_with_mocks, monkeypatch):
    """POST /train/ppo_patchtst returns all required response fields."""
    from brain_api.routes.training import ppo_patchtst

    monkeypatch.setattr(ppo_patchtst, "load_prices_yfinance", mock_price_loader)

    response = client_with_mocks.post("/train/ppo_patchtst/full", json={})
    assert response.status_code == 200

    data = response.json()
    assert "version" in data
    assert "data_window_start" in data
    assert "data_window_end" in data
    assert "metrics" in data
    assert "promoted" in data
    assert "symbols_used" in data


def test_train_ppo_patchtst_first_model_auto_promoted(client_with_mocks, monkeypatch):
    """First PPO + PatchTST model is automatically promoted."""
    from brain_api.routes.training import ppo_patchtst

    monkeypatch.setattr(ppo_patchtst, "load_prices_yfinance", mock_price_loader)

    response = client_with_mocks.post("/train/ppo_patchtst/full", json={})
    assert response.status_code == 200

    data = response.json()
    assert data["promoted"] is True
    assert data["prior_version"] is None


def test_train_ppo_patchtst_idempotent(client_with_mocks, monkeypatch):
    """Calling POST /train/ppo_patchtst twice returns the same version."""
    from brain_api.routes.training import ppo_patchtst

    monkeypatch.setattr(ppo_patchtst, "load_prices_yfinance", mock_price_loader)

    response1 = client_with_mocks.post("/train/ppo_patchtst/full", json={})
    assert response1.status_code == 200
    version1 = response1.json()["version"]

    response2 = client_with_mocks.post("/train/ppo_patchtst/full", json={})
    assert response2.status_code == 200
    version2 = response2.json()["version"]

    assert version1 == version2


# ============================================================================
# Inference endpoint tests
# ============================================================================


def test_infer_ppo_patchtst_returns_503_without_model(client_with_mocks):
    """POST /inference/ppo_patchtst returns 503 when no model is trained."""
    response = client_with_mocks.post(
        "/inference/ppo_patchtst",
        json={
            "portfolio": {
                "cash": 10000.0,
                "positions": [],
            }
        },
    )
    assert response.status_code == 503


def test_infer_ppo_patchtst_returns_200_with_model(client_for_inference):
    """POST /inference/ppo_patchtst returns 200 with valid request."""
    response = client_for_inference.post(
        "/inference/ppo_patchtst",
        json={
            "portfolio": {
                "cash": 10000.0,
                "positions": [],
            }
        },
    )
    assert response.status_code == 200


def test_infer_ppo_patchtst_returns_target_weights(client_for_inference):
    """POST /inference/ppo_patchtst returns target weights."""
    response = client_for_inference.post(
        "/inference/ppo_patchtst",
        json={
            "portfolio": {
                "cash": 10000.0,
                "positions": [],
            }
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert "target_weights" in data
    assert "turnover" in data
    assert "weight_changes" in data
    assert "CASH" in data["target_weights"]


def test_infer_ppo_patchtst_enforces_cash_buffer(client_for_inference):
    """Target weights maintain minimum 2% cash buffer."""
    response = client_for_inference.post(
        "/inference/ppo_patchtst",
        json={
            "portfolio": {
                "cash": 10000.0,
                "positions": [],
            }
        },
    )
    assert response.status_code == 200

    data = response.json()
    cash_weight = data["target_weights"]["CASH"]
    assert cash_weight >= 0.02, f"Cash weight {cash_weight} is below 2% buffer"


def test_infer_ppo_patchtst_enforces_max_position(client_for_inference):
    """No single position exceeds 20% of portfolio."""
    response = client_for_inference.post(
        "/inference/ppo_patchtst",
        json={
            "portfolio": {
                "cash": 10000.0,
                "positions": [],
            }
        },
    )
    assert response.status_code == 200

    data = response.json()

    for symbol, weight in data["target_weights"].items():
        if symbol != "CASH":
            assert weight <= 0.20 + 0.001, (
                f"Weight for {symbol} ({weight}) exceeds 20% max"
            )


def test_infer_ppo_patchtst_weights_sum_to_one(client_for_inference):
    """Target weights sum to 1.0."""
    response = client_for_inference.post(
        "/inference/ppo_patchtst",
        json={
            "portfolio": {
                "cash": 10000.0,
                "positions": [],
            }
        },
    )
    assert response.status_code == 200

    data = response.json()
    total_weight = sum(data["target_weights"].values())
    assert abs(total_weight - 1.0) < 0.001, f"Weights sum to {total_weight}, not 1.0"


def test_infer_ppo_patchtst_with_existing_positions(client_for_inference):
    """Inference works with existing portfolio positions."""
    response = client_for_inference.post(
        "/inference/ppo_patchtst",
        json={
            "portfolio": {
                "cash": 2000.0,
                "positions": [
                    {"symbol": "AAPL", "market_value": 4000.0},
                    {"symbol": "MSFT", "market_value": 4000.0},
                ],
            }
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert "target_weights" in data
    assert "turnover" in data
    assert isinstance(data["turnover"], float)
    assert data["turnover"] >= 0


def test_infer_ppo_patchtst_returns_week_boundaries(client_for_inference):
    """Response includes target week start and end dates."""
    response = client_for_inference.post(
        "/inference/ppo_patchtst",
        json={
            "portfolio": {
                "cash": 10000.0,
                "positions": [],
            }
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert "target_week_start" in data
    assert "target_week_end" in data
    assert "model_version" in data


# ============================================================================
# Fine-tuning endpoint tests
# ============================================================================


def test_finetune_ppo_patchtst_returns_400_without_prior_model(client_with_mocks):
    """POST /train/ppo_patchtst/finetune returns 400 when no prior model exists."""
    response = client_with_mocks.post("/train/ppo_patchtst/finetune", json={})
    assert response.status_code == 400
    assert "No prior PPO_PatchTST model" in response.json()["detail"]


def test_finetune_ppo_patchtst_returns_200_with_prior_model(
    client_for_inference, monkeypatch
):
    """POST /train/ppo_patchtst/finetune returns 200 when prior model exists."""
    from brain_api.routes.training import ppo_patchtst

    monkeypatch.setattr(ppo_patchtst, "load_prices_yfinance", mock_price_loader)

    response = client_for_inference.post("/train/ppo_patchtst/finetune", json={})
    assert response.status_code == 200


def test_finetune_ppo_patchtst_returns_required_fields(
    client_for_inference, monkeypatch
):
    """POST /train/ppo_patchtst/finetune returns all required response fields."""
    from brain_api.routes.training import ppo_patchtst

    monkeypatch.setattr(ppo_patchtst, "load_prices_yfinance", mock_price_loader)

    response = client_for_inference.post("/train/ppo_patchtst/finetune", json={})
    assert response.status_code == 200

    data = response.json()
    assert "version" in data
    assert "-ft" in data["version"]  # Fine-tune marker
    assert "data_window_start" in data
    assert "data_window_end" in data
    assert "metrics" in data
    assert "promoted" in data
    assert "prior_version" in data
    assert "symbols_used" in data


def test_finetune_ppo_patchtst_idempotent(client_for_inference, monkeypatch):
    """Calling POST /train/ppo_patchtst/finetune twice returns the same version."""
    from brain_api.routes.training import ppo_patchtst

    monkeypatch.setattr(ppo_patchtst, "load_prices_yfinance", mock_price_loader)

    response1 = client_for_inference.post("/train/ppo_patchtst/finetune", json={})
    assert response1.status_code == 200
    version1 = response1.json()["version"]

    response2 = client_for_inference.post("/train/ppo_patchtst/finetune", json={})
    assert response2.status_code == 200
    version2 = response2.json()["version"]

    assert version1 == version2
