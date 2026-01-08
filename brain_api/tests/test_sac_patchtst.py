"""API-level tests for SAC + PatchTST training and inference endpoints.

Tests focus on:
- Endpoint contract (status codes, response structure)
- Constraint enforcement (cash buffer, max position weight)
- Promotion gate behavior (CAGR-first)
"""

import os
import tempfile

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from brain_api.core.sac_patchtst import (
    SACPatchTSTConfig,
    SACPatchTSTTrainingResult,
)
from brain_api.core.portfolio_rl.sac_networks import GaussianActor, TwinCritic
from brain_api.core.portfolio_rl.scaler import PortfolioScaler
from brain_api.main import app
from brain_api.routes.training import (
    get_sac_patchtst_storage,
    get_sac_patchtst_config,
    get_top15_symbols,
)
from brain_api.routes.inference import get_sac_patchtst_storage as get_inference_storage
from brain_api.storage.sac_patchtst import SACPatchTSTLocalStorage, create_sac_patchtst_metadata


# ============================================================================
# Test fixtures and mocks
# ============================================================================


def mock_symbols() -> list[str]:
    """Return a small fixed list of symbols for testing."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def mock_config() -> SACPatchTSTConfig:
    """Return a minimal config for fast testing."""
    return SACPatchTSTConfig(
        n_stocks=5,
        total_timesteps=100,  # Very small for fast tests
        hidden_sizes=(16, 16),  # Small network
        batch_size=8,
        warmup_steps=10,
    )


def create_mock_training_result(config: SACPatchTSTConfig) -> SACPatchTSTTrainingResult:
    """Create a mock training result for testing."""
    n_stocks = config.n_stocks
    state_dim = n_stocks * 7 + n_stocks + n_stocks + 1  # signals + forecasts + weights
    action_dim = n_stocks + 1

    actor = GaussianActor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
    )

    critic = TwinCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
    )

    critic_target = TwinCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
    )

    log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32)

    scaler = PortfolioScaler.create(n_stocks=n_stocks)
    # Fit scaler on dummy data
    dummy_states = np.random.randn(10, state_dim)
    scaler.fit(dummy_states)

    return SACPatchTSTTrainingResult(
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        log_alpha=log_alpha,
        scaler=scaler,
        config=config,
        symbol_order=mock_symbols()[:n_stocks],
        final_actor_loss=0.1,
        final_critic_loss=0.05,
        avg_episode_return=0.02,
        avg_episode_sharpe=0.5,
        eval_sharpe=0.6,
        eval_cagr=0.10,  # Good CAGR for promotion
        eval_max_drawdown=0.15,
    )


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield SACPatchTSTLocalStorage(base_path=tmpdir)


@pytest.fixture
def trained_model_storage():
    """Create storage with a pre-trained model for inference tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SACPatchTSTLocalStorage(base_path=tmpdir)
        config = mock_config()
        result = create_mock_training_result(config)

        version = "v2025-01-01_test123"
        metadata = create_sac_patchtst_metadata(
            version=version,
            data_window_start="2020-01-01",
            data_window_end="2025-01-01",
            symbols=result.symbol_order,
            config=config,
            promoted=True,
            prior_version=None,
            actor_loss=result.final_actor_loss,
            critic_loss=result.final_critic_loss,
            avg_episode_return=result.avg_episode_return,
            avg_episode_sharpe=result.avg_episode_sharpe,
            eval_sharpe=result.eval_sharpe,
            eval_cagr=result.eval_cagr,
            eval_max_drawdown=result.eval_max_drawdown,
        )

        storage.write_artifacts(
            version=version,
            actor=result.actor,
            critic=result.critic,
            critic_target=result.critic_target,
            log_alpha=result.log_alpha,
            scaler=result.scaler,
            config=config,
            symbol_order=result.symbol_order,
            metadata=metadata,
        )
        storage.promote_version(version)

        yield storage


@pytest.fixture
def inference_client(trained_model_storage):
    """Create test client with trained model for inference tests."""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_inference_storage] = lambda: trained_model_storage

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


# ============================================================================
# Inference endpoint tests
# ============================================================================


class TestSACPatchTSTInference:
    """Tests for /inference/sac_patchtst endpoint."""

    def test_inference_returns_valid_weights(self, inference_client):
        """Test that inference returns weights summing to 1."""
        response = inference_client.post(
            "/inference/sac_patchtst",
            json={
                "portfolio": {
                    "cash": 10000.0,
                    "positions": [
                        {"symbol": "AAPL", "market_value": 2000.0},
                        {"symbol": "MSFT", "market_value": 2000.0},
                    ],
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "target_weights" in data
        assert "turnover" in data
        assert "model_version" in data

        # Check weights sum to ~1
        weights = data["target_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"

    def test_inference_respects_cash_buffer(self, inference_client):
        """Test that CASH weight >= cash_buffer (2%)."""
        response = inference_client.post(
            "/inference/sac_patchtst",
            json={
                "portfolio": {
                    "cash": 100.0,  # Small cash
                    "positions": [
                        {"symbol": "AAPL", "market_value": 9900.0},
                    ],
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        weights = data["target_weights"]

        # CASH weight should be at least 2%
        assert weights.get("CASH", 0) >= 0.02, f"CASH weight {weights.get('CASH')} < 2%"

    def test_inference_respects_max_position(self, inference_client):
        """Test that no single position > max_position_weight (20%)."""
        response = inference_client.post(
            "/inference/sac_patchtst",
            json={
                "portfolio": {
                    "cash": 10000.0,
                    "positions": [],
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        weights = data["target_weights"]

        # No position should exceed 20%
        for symbol, weight in weights.items():
            if symbol != "CASH":
                assert weight <= 0.20 + 0.01, f"{symbol} weight {weight} > 20%"

    def test_inference_without_model_returns_503(self, temp_storage):
        """Test that inference without a trained model returns 503."""
        app.dependency_overrides.clear()
        app.dependency_overrides[get_inference_storage] = lambda: temp_storage

        client = TestClient(app)
        response = client.post(
            "/inference/sac_patchtst",
            json={
                "portfolio": {
                    "cash": 10000.0,
                    "positions": [],
                },
            },
        )

        assert response.status_code == 503
        app.dependency_overrides.clear()

    def test_inference_invalid_payload_returns_422(self, inference_client):
        """Test that invalid request payload returns 422."""
        # Missing required portfolio field
        response = inference_client.post(
            "/inference/sac_patchtst",
            json={},
        )
        assert response.status_code == 422


# ============================================================================
# Finetune endpoint tests
# ============================================================================


class TestSACPatchTSTFinetune:
    """Tests for /train/sac_patchtst/finetune endpoint."""

    def test_finetune_without_prior_returns_400(self, temp_storage):
        """Test that finetune without prior model returns 400."""
        app.dependency_overrides.clear()
        app.dependency_overrides[get_sac_patchtst_storage] = lambda: temp_storage
        app.dependency_overrides[get_top15_symbols] = mock_symbols
        app.dependency_overrides[get_sac_patchtst_config] = mock_config

        os.environ["LSTM_TRAIN_LOOKBACK_YEARS"] = "5"
        os.environ["LSTM_TRAIN_WINDOW_END_DATE"] = "2025-01-01"

        client = TestClient(app)
        response = client.post("/train/sac_patchtst/finetune")

        assert response.status_code == 400
        assert "No prior SAC_PatchTST model" in response.json()["detail"]

        app.dependency_overrides.clear()
        os.environ.pop("LSTM_TRAIN_LOOKBACK_YEARS", None)
        os.environ.pop("LSTM_TRAIN_WINDOW_END_DATE", None)


# ============================================================================
# Storage and artifacts tests
# ============================================================================


class TestSACPatchTSTStorage:
    """Tests for SAC PatchTST storage functionality."""

    def test_write_and_load_artifacts(self, temp_storage):
        """Test that artifacts can be written and loaded correctly."""
        config = mock_config()
        result = create_mock_training_result(config)

        version = "v2025-01-01_storage_test"
        metadata = create_sac_patchtst_metadata(
            version=version,
            data_window_start="2020-01-01",
            data_window_end="2025-01-01",
            symbols=result.symbol_order,
            config=config,
            promoted=True,
            prior_version=None,
            actor_loss=result.final_actor_loss,
            critic_loss=result.final_critic_loss,
            avg_episode_return=result.avg_episode_return,
            avg_episode_sharpe=result.avg_episode_sharpe,
            eval_sharpe=result.eval_sharpe,
            eval_cagr=result.eval_cagr,
            eval_max_drawdown=result.eval_max_drawdown,
        )

        # Write artifacts
        temp_storage.write_artifacts(
            version=version,
            actor=result.actor,
            critic=result.critic,
            critic_target=result.critic_target,
            log_alpha=result.log_alpha,
            scaler=result.scaler,
            config=config,
            symbol_order=result.symbol_order,
            metadata=metadata,
        )

        # Verify version exists
        assert temp_storage.version_exists(version)

        # Load and verify artifacts
        loaded = temp_storage.load_artifacts(version)
        assert loaded.version == version
        assert loaded.symbol_order == result.symbol_order
        assert isinstance(loaded.actor, GaussianActor)
        assert isinstance(loaded.critic, TwinCritic)
        assert isinstance(loaded.critic_target, TwinCritic)

    def test_promote_version(self, temp_storage):
        """Test version promotion."""
        config = mock_config()
        result = create_mock_training_result(config)

        version = "v2025-01-01_promote_test"
        metadata = create_sac_patchtst_metadata(
            version=version,
            data_window_start="2020-01-01",
            data_window_end="2025-01-01",
            symbols=result.symbol_order,
            config=config,
            promoted=True,
            prior_version=None,
            actor_loss=0.1,
            critic_loss=0.05,
            avg_episode_return=0.02,
            avg_episode_sharpe=0.5,
            eval_sharpe=0.6,
            eval_cagr=0.10,
            eval_max_drawdown=0.15,
        )

        temp_storage.write_artifacts(
            version=version,
            actor=result.actor,
            critic=result.critic,
            critic_target=result.critic_target,
            log_alpha=result.log_alpha,
            scaler=result.scaler,
            config=config,
            symbol_order=result.symbol_order,
            metadata=metadata,
        )

        # Promote version
        temp_storage.promote_version(version)

        # Verify current version
        assert temp_storage.read_current_version() == version

        # Verify can load current artifacts
        loaded = temp_storage.load_current_artifacts()
        assert loaded.version == version

