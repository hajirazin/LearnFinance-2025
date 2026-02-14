"""Tests for experience storage and labeling endpoints.

This module tests:
- Full state experience storage (PPO and SAC)
- Experience labeling with actual execution
- Execution report updates
- Order comparison logic
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from brain_api.core.portfolio_rl.rewards import compute_reward_from_log_return
from brain_api.main import app
from brain_api.routes.experience import (
    ExperienceRecord,
    ExperienceStorage,
    _compute_reward_from_actual_weights,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage for testing."""
    return ExperienceStorage(base_path=tmp_path)


@pytest.fixture
def sample_full_state():
    """Sample full state for testing."""
    return {
        "signals": {
            "AAPL": {
                "news_sentiment": 0.3,
                "gross_margin": 0.42,
                "operating_margin": 0.30,
                "net_margin": 0.25,
                "current_ratio": 1.5,
                "debt_to_equity": 0.8,
            },
            "MSFT": {
                "news_sentiment": 0.5,
                "gross_margin": 0.68,
                "operating_margin": 0.42,
                "net_margin": 0.36,
                "current_ratio": 2.1,
                "debt_to_equity": 0.4,
            },
        },
        "lstm_forecasts": {"AAPL": 0.012, "MSFT": -0.005},
        "patchtst_forecasts": {"AAPL": 0.015, "MSFT": -0.003},
        "current_weights": {"AAPL": 0.10, "MSFT": 0.08, "CASH": 0.82},
    }


@pytest.fixture
def sample_intended_action():
    """Sample intended action for testing."""
    return {"AAPL": 0.15, "MSFT": 0.12, "CASH": 0.73}


class TestExperienceFullStatePPO:
    """Tests for PPO experience store with full state."""

    def test_store_ppo_with_signals_and_forecasts(
        self, client, sample_full_state, sample_intended_action
    ):
        """Test storing PPO experience with full state including signals and forecasts."""
        response = client.post(
            "/experience/store",
            json={
                "run_id": "paper:2026-01-20",
                "week_start": "2026-01-20",
                "week_end": "2026-01-24",
                "model_type": "ppo",
                "model_version": "v1.0.0",
                "state": sample_full_state,
                "intended_action": sample_intended_action,
                "intended_turnover": 0.12,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stored"] is True
        assert data["model_type"] == "ppo"
        assert "ppo" in data["record_id"]

    def test_store_ppo_validates_required_fields(self, client):
        """Test that required fields are validated."""
        # Missing run_id
        response = client.post(
            "/experience/store",
            json={
                "week_start": "2026-01-20",
                "week_end": "2026-01-24",
                "model_type": "ppo",
                "model_version": "v1.0.0",
                "state": {},
                "intended_action": {},
                "intended_turnover": 0.0,
            },
        )
        assert response.status_code == 422

    def test_ppo_state_includes_all_signal_types(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that stored PPO state includes all signal types."""
        record = ExperienceRecord(
            run_id="paper:2026-01-20:ppo",
            week_start="2026-01-20",
            week_end="2026-01-24",
            model_type="ppo",
            model_version="v1.0.0",
            state=sample_full_state,
            intended_action=sample_intended_action,
            intended_turnover=0.12,
        )
        temp_storage.store(record)
        loaded = temp_storage.load("paper:2026-01-20:ppo")

        assert loaded is not None
        state = loaded.state
        if isinstance(state, dict):
            assert "signals" in state
            assert "AAPL" in state["signals"]
            assert "news_sentiment" in state["signals"]["AAPL"]
            assert "gross_margin" in state["signals"]["AAPL"]

    def test_ppo_state_includes_both_forecasts(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that stored PPO state includes both LSTM and PatchTST forecasts."""
        record = ExperienceRecord(
            run_id="paper:2026-01-20:ppo",
            week_start="2026-01-20",
            week_end="2026-01-24",
            model_type="ppo",
            model_version="v1.0.0",
            state=sample_full_state,
            intended_action=sample_intended_action,
            intended_turnover=0.12,
        )
        temp_storage.store(record)
        loaded = temp_storage.load("paper:2026-01-20:ppo")

        state = loaded.state
        if isinstance(state, dict):
            assert "lstm_forecasts" in state
            assert "patchtst_forecasts" in state
            assert "AAPL" in state["lstm_forecasts"]
            assert "AAPL" in state["patchtst_forecasts"]

    def test_ppo_state_includes_current_weights(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that stored PPO state includes current portfolio weights."""
        record = ExperienceRecord(
            run_id="paper:2026-01-20:ppo",
            week_start="2026-01-20",
            week_end="2026-01-24",
            model_type="ppo",
            model_version="v1.0.0",
            state=sample_full_state,
            intended_action=sample_intended_action,
            intended_turnover=0.12,
        )
        temp_storage.store(record)
        loaded = temp_storage.load("paper:2026-01-20:ppo")

        state = loaded.state
        if isinstance(state, dict):
            assert "current_weights" in state
            assert "CASH" in state["current_weights"]


class TestExperienceFullStateSAC:
    """Tests for SAC experience store with full state."""

    def test_store_sac_with_signals_and_forecasts(
        self, client, sample_full_state, sample_intended_action
    ):
        """Test storing SAC experience with full state including signals and forecasts."""
        response = client.post(
            "/experience/store",
            json={
                "run_id": "paper:2026-01-20",
                "week_start": "2026-01-20",
                "week_end": "2026-01-24",
                "model_type": "sac",
                "model_version": "v1.0.0",
                "state": sample_full_state,
                "intended_action": sample_intended_action,
                "intended_turnover": 0.15,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stored"] is True
        assert data["model_type"] == "sac"
        assert "sac" in data["record_id"]

    def test_store_sac_validates_required_fields(self, client):
        """Test that required fields are validated for SAC."""
        # Missing model_type
        response = client.post(
            "/experience/store",
            json={
                "run_id": "paper:2026-01-20",
                "week_start": "2026-01-20",
                "week_end": "2026-01-24",
                "model_version": "v1.0.0",
                "state": {},
                "intended_action": {},
                "intended_turnover": 0.0,
            },
        )
        assert response.status_code == 422

    def test_sac_state_includes_all_signal_types(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that stored SAC state includes all signal types."""
        record = ExperienceRecord(
            run_id="paper:2026-01-20:sac",
            week_start="2026-01-20",
            week_end="2026-01-24",
            model_type="sac",
            model_version="v1.0.0",
            state=sample_full_state,
            intended_action=sample_intended_action,
            intended_turnover=0.15,
        )
        temp_storage.store(record)
        loaded = temp_storage.load("paper:2026-01-20:sac")

        assert loaded is not None
        state = loaded.state
        if isinstance(state, dict):
            assert "signals" in state
            assert "MSFT" in state["signals"]
            assert "operating_margin" in state["signals"]["MSFT"]

    def test_sac_state_includes_both_forecasts(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that stored SAC state includes both LSTM and PatchTST forecasts."""
        record = ExperienceRecord(
            run_id="paper:2026-01-20:sac",
            week_start="2026-01-20",
            week_end="2026-01-24",
            model_type="sac",
            model_version="v1.0.0",
            state=sample_full_state,
            intended_action=sample_intended_action,
            intended_turnover=0.15,
        )
        temp_storage.store(record)
        loaded = temp_storage.load("paper:2026-01-20:sac")

        state = loaded.state
        if isinstance(state, dict):
            assert "lstm_forecasts" in state
            assert "patchtst_forecasts" in state

    def test_sac_state_includes_current_weights(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that stored SAC state includes current portfolio weights."""
        record = ExperienceRecord(
            run_id="paper:2026-01-20:sac",
            week_start="2026-01-20",
            week_end="2026-01-24",
            model_type="sac",
            model_version="v1.0.0",
            state=sample_full_state,
            intended_action=sample_intended_action,
            intended_turnover=0.15,
        )
        temp_storage.store(record)
        loaded = temp_storage.load("paper:2026-01-20:sac")

        state = loaded.state
        if isinstance(state, dict):
            assert "current_weights" in state
            assert abs(sum(state["current_weights"].values()) - 1.0) < 0.01


class TestLabelPPOEndpoint:
    """Tests for /experience/label/ppo endpoint."""

    def test_label_ppo_endpoint_returns_200(self, client):
        """Test that label PPO endpoint returns 200."""
        with (
            patch("brain_api.core.alpaca_client.get_alpaca_client") as mock_get_client,
            patch("brain_api.core.lstm.load_prices_yfinance") as mock_prices,
        ):
            mock_client = MagicMock()
            mock_client.get_portfolio_weights.return_value = {"AAPL": 0.5, "CASH": 0.5}
            mock_get_client.return_value = mock_client
            mock_prices.return_value = {}

            response = client.post(
                "/experience/label/ppo",
                json={"run_id": None},
            )
            assert response.status_code == 200

    def test_label_ppo_fetches_from_ppo_account(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that label PPO fetches from PPO Alpaca account."""
        # Create a record first
        record = ExperienceRecord(
            run_id="paper:2026-01-01:ppo",
            week_start="2026-01-01",
            week_end="2026-01-05",
            model_type="ppo",
            model_version="v1.0.0",
            state=sample_full_state,
            intended_action=sample_intended_action,
            intended_turnover=0.1,
        )
        temp_storage.store(record)

        with (
            patch("brain_api.core.alpaca_client.get_alpaca_client") as mock_get_client,
            patch(
                "brain_api.routes.experience.get_experience_storage",
                return_value=temp_storage,
            ),
        ):
            mock_client = MagicMock()
            mock_client.get_portfolio_weights.return_value = {"AAPL": 0.5, "CASH": 0.5}
            mock_get_client.return_value = mock_client

            from brain_api.routes.experience import _label_experience_for_account

            # This should call the PPO client
            with patch("brain_api.core.lstm.load_prices_yfinance") as mock_prices:
                mock_prices.return_value = {}
                _label_experience_for_account("ppo", None, temp_storage)

            # Verify it was called with ppo account
            from brain_api.core.alpaca_client import AlpacaAccount

            mock_get_client.assert_called_with(AlpacaAccount.PPO)

    def test_label_ppo_uses_actual_weights_not_intended(self):
        """Test that labeling uses actual weights, not intended action."""
        # Test the reward computation function directly
        actual_weights = {"AAPL": 0.5, "MSFT": 0.3, "CASH": 0.2}
        symbol_returns = {"AAPL": 0.05, "MSFT": -0.02}  # 5% and -2%

        _reward, portfolio_return = _compute_reward_from_actual_weights(
            actual_weights=actual_weights,
            symbol_returns=symbol_returns,
        )

        # Expected: 0.5 * 0.05 + 0.3 * (-0.02) = 0.025 - 0.006 = 0.019
        expected_return = 0.5 * 0.05 + 0.3 * (-0.02)
        assert abs(portfolio_return - expected_return) < 0.001

    def test_label_ppo_calculates_log_return(self):
        """Test that reward uses log return."""
        import numpy as np

        actual_weights = {"AAPL": 1.0, "CASH": 0.0}
        symbol_returns = {"AAPL": 0.10}  # 10% return

        reward, _portfolio_return = _compute_reward_from_actual_weights(
            actual_weights=actual_weights,
            symbol_returns=symbol_returns,
        )

        # Log return for 10% = ln(1.10) ≈ 0.0953
        expected_log_return = np.log(1.10)
        # Reward = (log_return - tx_cost) * scale
        # With estimated 10% turnover and 10bps cost: tx_cost = 0.10 * 0.001 = 0.0001
        # scale = 100
        expected_reward = (expected_log_return - 0.0001) * 100

        # Check reward is in expected range (accounting for scale and cost)
        assert reward > 0  # Should be positive for positive return
        assert abs(reward - expected_reward) < 1.0

    def test_label_ppo_includes_transaction_cost(self):
        """Test that reward calculation includes transaction cost."""
        # Two identical portfolio returns but different turnover expectations
        actual_weights = {"AAPL": 1.0, "CASH": 0.0}
        symbol_returns = {"AAPL": 0.0}  # Zero return

        reward, _ = _compute_reward_from_actual_weights(
            actual_weights=actual_weights,
            symbol_returns=symbol_returns,
            cost_bps=10,
        )

        # With zero return but transaction cost, reward should be negative
        assert reward < 0

    def test_label_ppo_skips_if_week_not_ended(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that labeling skips records where week hasn't ended."""
        # Create a record with future week_end
        future_date = (date.today() + timedelta(days=7)).isoformat()
        record = ExperienceRecord(
            run_id="paper:future:ppo",
            week_start=date.today().isoformat(),
            week_end=future_date,
            model_type="ppo",
            model_version="v1.0.0",
            state=sample_full_state,
            intended_action=sample_intended_action,
            intended_turnover=0.1,
        )
        temp_storage.store(record)

        with (
            patch("brain_api.core.alpaca_client.get_alpaca_client") as mock_get_client,
            patch("brain_api.core.lstm.load_prices_yfinance"),
        ):
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            from brain_api.routes.experience import _label_experience_for_account

            result = _label_experience_for_account("ppo", None, temp_storage)

            # Should skip the record
            assert result.records_skipped >= 1


class TestLabelSACEndpoint:
    """Tests for /experience/label/sac endpoint."""

    def test_label_sac_endpoint_returns_200(self, client):
        """Test that label SAC endpoint returns 200."""
        with (
            patch("brain_api.core.alpaca_client.get_alpaca_client") as mock_get_client,
            patch("brain_api.core.lstm.load_prices_yfinance") as mock_prices,
        ):
            mock_client = MagicMock()
            mock_client.get_portfolio_weights.return_value = {"MSFT": 0.6, "CASH": 0.4}
            mock_get_client.return_value = mock_client
            mock_prices.return_value = {}

            response = client.post(
                "/experience/label/sac",
                json={"run_id": None},
            )
            assert response.status_code == 200

    def test_label_sac_fetches_from_sac_account(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that label SAC fetches from SAC Alpaca account."""
        record = ExperienceRecord(
            run_id="paper:2026-01-01:sac",
            week_start="2026-01-01",
            week_end="2026-01-05",
            model_type="sac",
            model_version="v1.0.0",
            state=sample_full_state,
            intended_action=sample_intended_action,
            intended_turnover=0.1,
        )
        temp_storage.store(record)

        with (
            patch("brain_api.core.alpaca_client.get_alpaca_client") as mock_get_client,
            patch(
                "brain_api.routes.experience.get_experience_storage",
                return_value=temp_storage,
            ),
        ):
            mock_client = MagicMock()
            mock_client.get_portfolio_weights.return_value = {"MSFT": 0.6, "CASH": 0.4}
            mock_get_client.return_value = mock_client

            from brain_api.routes.experience import _label_experience_for_account

            with patch("brain_api.core.lstm.load_prices_yfinance") as mock_prices:
                mock_prices.return_value = {}
                _label_experience_for_account("sac", None, temp_storage)

            from brain_api.core.alpaca_client import AlpacaAccount

            mock_get_client.assert_called_with(AlpacaAccount.SAC)

    def test_label_sac_uses_actual_weights_not_intended(self):
        """Test that SAC labeling uses actual weights."""
        actual_weights = {"GOOGL": 0.4, "AMZN": 0.4, "CASH": 0.2}
        symbol_returns = {"GOOGL": 0.03, "AMZN": 0.02}

        _reward, portfolio_return = _compute_reward_from_actual_weights(
            actual_weights=actual_weights,
            symbol_returns=symbol_returns,
        )

        expected_return = 0.4 * 0.03 + 0.4 * 0.02
        assert abs(portfolio_return - expected_return) < 0.001

    def test_label_sac_calculates_log_return(self):
        """Test that SAC reward uses log return."""
        actual_weights = {"NVDA": 1.0, "CASH": 0.0}
        symbol_returns = {"NVDA": 0.20}  # 20% return

        reward, _ = _compute_reward_from_actual_weights(
            actual_weights=actual_weights,
            symbol_returns=symbol_returns,
        )

        # Should be positive for positive return
        assert reward > 0

    def test_label_sac_includes_transaction_cost(self):
        """Test that SAC reward includes transaction cost."""
        actual_weights = {"TSLA": 1.0, "CASH": 0.0}
        symbol_returns = {"TSLA": 0.0}

        reward, _ = _compute_reward_from_actual_weights(
            actual_weights=actual_weights,
            symbol_returns=symbol_returns,
            cost_bps=20,  # Higher cost
        )

        assert reward < 0

    def test_label_sac_skips_if_week_not_ended(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that SAC labeling skips future weeks."""
        future_date = (date.today() + timedelta(days=7)).isoformat()
        record = ExperienceRecord(
            run_id="paper:future:sac",
            week_start=date.today().isoformat(),
            week_end=future_date,
            model_type="sac",
            model_version="v1.0.0",
            state=sample_full_state,
            intended_action=sample_intended_action,
            intended_turnover=0.1,
        )
        temp_storage.store(record)

        with (
            patch("brain_api.core.alpaca_client.get_alpaca_client") as mock_get_client,
            patch("brain_api.core.lstm.load_prices_yfinance"),
        ):
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            from brain_api.routes.experience import _label_experience_for_account

            result = _label_experience_for_account("sac", None, temp_storage)
            assert result.records_skipped >= 1


class TestUpdateExecution:
    """Tests for /experience/update-execution endpoint."""

    def test_update_execution_stores_report(
        self, client, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that execution report is stored."""
        # First create a record
        with patch(
            "brain_api.routes.experience.get_experience_storage",
            return_value=temp_storage,
        ):
            client.post(
                "/experience/store",
                json={
                    "run_id": "paper:2026-01-20",
                    "week_start": "2026-01-20",
                    "week_end": "2026-01-24",
                    "model_type": "ppo",
                    "model_version": "v1.0.0",
                    "state": sample_full_state,
                    "intended_action": sample_intended_action,
                    "intended_turnover": 0.1,
                },
            )

            # Now update with execution report
            response = client.post(
                "/experience/update-execution",
                json={
                    "run_id": "paper:2026-01-20",
                    "model_type": "ppo",
                    "execution_report": [
                        {
                            "symbol": "AAPL",
                            "side": "buy",
                            "intended_qty": 10.0,
                            "filled_qty": 10.0,
                            "filled_avg_price": 150.0,
                            "status": "filled",
                            "client_order_id": "paper:2026-01-20:attempt-1:AAPL:BUY",
                        }
                    ],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["updated"] is True
            assert data["orders_filled"] == 1

    def test_update_execution_matches_run_id(self, client):
        """Test that execution update matches by run_id."""
        response = client.post(
            "/experience/update-execution",
            json={
                "run_id": "nonexistent:2026-01-20",
                "model_type": "ppo",
                "execution_report": [],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is False

    def test_update_execution_handles_partial_fills(
        self, client, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that partial fills are correctly counted."""
        with patch(
            "brain_api.routes.experience.get_experience_storage",
            return_value=temp_storage,
        ):
            client.post(
                "/experience/store",
                json={
                    "run_id": "paper:2026-01-20",
                    "week_start": "2026-01-20",
                    "week_end": "2026-01-24",
                    "model_type": "sac",
                    "model_version": "v1.0.0",
                    "state": sample_full_state,
                    "intended_action": sample_intended_action,
                    "intended_turnover": 0.1,
                },
            )

            response = client.post(
                "/experience/update-execution",
                json={
                    "run_id": "paper:2026-01-20",
                    "model_type": "sac",
                    "execution_report": [
                        {"symbol": "AAPL", "status": "filled"},
                        {"symbol": "MSFT", "status": "partial"},
                        {"symbol": "GOOGL", "status": "expired"},
                    ],
                },
            )

            data = response.json()
            assert data["orders_filled"] == 1
            assert data["orders_partial"] == 1
            assert data["orders_expired"] == 1


class TestOrderComparison:
    """Tests for order comparison logic in execution reports."""

    def test_matches_by_client_order_id(self):
        """Test that orders are matched by client_order_id."""
        intended = [
            {
                "symbol": "AAPL",
                "side": "buy",
                "qty": 10,
                "client_order_id": "paper:2026-01-20:attempt-1:AAPL:BUY",
            }
        ]
        executed = [
            {
                "client_order_id": "paper:2026-01-20:attempt-1:AAPL:BUY",
                "status": "filled",
                "filled_qty": "10",
                "filled_avg_price": "150.00",
            }
        ]

        # Simulate the comparison logic
        matched = None
        for i in intended:
            for e in executed:
                if e["client_order_id"] == i["client_order_id"]:
                    matched = e
                    break

        assert matched is not None
        assert matched["status"] == "filled"

    def test_identifies_filled_orders(self):
        """Test identification of filled orders."""
        execution_report = [
            {"symbol": "AAPL", "status": "filled"},
            {"symbol": "MSFT", "status": "filled"},
        ]

        filled = [o for o in execution_report if o["status"] == "filled"]
        assert len(filled) == 2

    def test_identifies_expired_orders(self):
        """Test identification of expired orders."""
        execution_report = [
            {"symbol": "AAPL", "status": "filled"},
            {"symbol": "MSFT", "status": "expired"},
            {"symbol": "GOOGL", "status": "canceled"},
        ]

        expired = [
            o for o in execution_report if o["status"] in ("expired", "canceled")
        ]
        assert len(expired) == 2

    def test_identifies_partial_fills(self):
        """Test identification of partial fills."""
        execution_report = [
            {"symbol": "AAPL", "status": "partially_filled", "filled_qty": 5},
            {"symbol": "MSFT", "status": "partial", "filled_qty": 3},
        ]

        partial = [
            o
            for o in execution_report
            if o["status"] in ("partial", "partially_filled")
        ]
        assert len(partial) == 2

    def test_handles_missing_orders(self):
        """Test handling of orders not found in Alpaca history."""
        intended = [
            {"symbol": "AAPL", "client_order_id": "id1"},
            {"symbol": "MSFT", "client_order_id": "id2"},
        ]
        executed = [
            {"client_order_id": "id1", "status": "filled"},
            # id2 is missing
        ]

        report = []
        for i in intended:
            found = next(
                (e for e in executed if e["client_order_id"] == i["client_order_id"]),
                None,
            )
            report.append(
                {
                    "symbol": i["symbol"],
                    "status": found["status"] if found else "not_found",
                }
            )

        assert report[0]["status"] == "filled"
        assert report[1]["status"] == "not_found"


class TestMatchOrdersFunction:
    """Tests for match_orders() helper function in experience module."""

    def test_match_orders_all_matched(self):
        """Test match_orders with all orders successfully matched."""
        from brain_api.routes.experience import match_orders

        intended = [
            {
                "symbol": "AAPL",
                "side": "buy",
                "qty": 10,
                "client_order_id": "paper:2026-02-05:attempt-1:AAPL:BUY",
            },
            {
                "symbol": "MSFT",
                "side": "sell",
                "qty": 5,
                "client_order_id": "paper:2026-02-05:attempt-1:MSFT:SELL",
            },
        ]
        executed = [
            {
                "client_order_id": "paper:2026-02-05:attempt-1:AAPL:BUY",
                "status": "filled",
                "filled_qty": "10",
                "filled_avg_price": "175.50",
            },
            {
                "client_order_id": "paper:2026-02-05:attempt-1:MSFT:SELL",
                "status": "filled",
                "filled_qty": "5",
                "filled_avg_price": "400.25",
            },
        ]

        report = match_orders(intended, executed)

        assert len(report) == 2
        # Check AAPL order
        assert report[0]["symbol"] == "AAPL"
        assert report[0]["side"] == "buy"
        assert report[0]["intended_qty"] == 10
        assert report[0]["filled_qty"] == 10.0
        assert report[0]["filled_avg_price"] == 175.50
        assert report[0]["status"] == "filled"
        assert report[0]["client_order_id"] == "paper:2026-02-05:attempt-1:AAPL:BUY"
        # Check MSFT order
        assert report[1]["symbol"] == "MSFT"
        assert report[1]["side"] == "sell"
        assert report[1]["filled_qty"] == 5.0
        assert report[1]["filled_avg_price"] == 400.25
        assert report[1]["status"] == "filled"

    def test_match_orders_some_not_found(self):
        """Test match_orders with some orders not found in Alpaca history."""
        from brain_api.routes.experience import match_orders

        intended = [
            {
                "symbol": "AAPL",
                "side": "buy",
                "qty": 10,
                "client_order_id": "paper:2026-02-05:attempt-1:AAPL:BUY",
            },
            {
                "symbol": "GOOGL",
                "side": "buy",
                "qty": 3,
                "client_order_id": "paper:2026-02-05:attempt-1:GOOGL:BUY",
            },
            {
                "symbol": "MSFT",
                "side": "sell",
                "qty": 5,
                "client_order_id": "paper:2026-02-05:attempt-1:MSFT:SELL",
            },
        ]
        executed = [
            {
                "client_order_id": "paper:2026-02-05:attempt-1:AAPL:BUY",
                "status": "filled",
                "filled_qty": "10",
                "filled_avg_price": "175.50",
            },
            # GOOGL not present - order might have been rejected before reaching Alpaca
            {
                "client_order_id": "paper:2026-02-05:attempt-1:MSFT:SELL",
                "status": "canceled",
                "filled_qty": "0",
                "filled_avg_price": None,
            },
        ]

        report = match_orders(intended, executed)

        assert len(report) == 3
        # AAPL matched
        assert report[0]["status"] == "filled"
        assert report[0]["filled_qty"] == 10.0
        # GOOGL not found
        assert report[1]["symbol"] == "GOOGL"
        assert report[1]["status"] == "not_found"
        assert report[1]["filled_qty"] == 0.0
        assert report[1]["filled_avg_price"] is None
        # MSFT canceled
        assert report[2]["status"] == "canceled"
        assert report[2]["filled_qty"] == 0.0

    def test_match_orders_partial_fills(self):
        """Test match_orders with partial fills."""
        from brain_api.routes.experience import match_orders

        intended = [
            {
                "symbol": "AAPL",
                "side": "buy",
                "qty": 100,
                "client_order_id": "paper:2026-02-05:attempt-1:AAPL:BUY",
            },
            {
                "symbol": "TSLA",
                "side": "buy",
                "qty": 50,
                "client_order_id": "paper:2026-02-05:attempt-1:TSLA:BUY",
            },
        ]
        executed = [
            {
                "client_order_id": "paper:2026-02-05:attempt-1:AAPL:BUY",
                "status": "partially_filled",
                "filled_qty": "75",
                "filled_avg_price": "175.00",
            },
            {
                "client_order_id": "paper:2026-02-05:attempt-1:TSLA:BUY",
                "status": "filled",
                "filled_qty": "50",
                "filled_avg_price": "250.00",
            },
        ]

        report = match_orders(intended, executed)

        assert len(report) == 2
        # AAPL partial fill
        assert report[0]["symbol"] == "AAPL"
        assert report[0]["intended_qty"] == 100
        assert report[0]["filled_qty"] == 75.0
        assert report[0]["status"] == "partially_filled"
        # TSLA full fill
        assert report[1]["filled_qty"] == 50.0
        assert report[1]["status"] == "filled"

    def test_match_orders_empty_lists(self):
        """Test match_orders with empty input lists."""
        from brain_api.routes.experience import match_orders

        # Empty intended
        report = match_orders([], [])
        assert len(report) == 0

        # Empty executed - all should be not_found
        intended = [
            {"symbol": "AAPL", "side": "buy", "qty": 10, "client_order_id": "id1"}
        ]
        report = match_orders(intended, [])
        assert len(report) == 1
        assert report[0]["status"] == "not_found"

    def test_match_orders_handles_string_quantities(self):
        """Test that match_orders properly parses string quantities from Alpaca."""
        from brain_api.routes.experience import match_orders

        intended = [
            {"symbol": "AAPL", "side": "buy", "qty": 10, "client_order_id": "id1"}
        ]
        executed = [
            {
                "client_order_id": "id1",
                "status": "filled",
                "filled_qty": "10.5",  # String with decimal
                "filled_avg_price": "175.123456",  # String with precision
            }
        ]

        report = match_orders(intended, executed)

        assert report[0]["filled_qty"] == 10.5
        assert report[0]["filled_avg_price"] == 175.123456


class TestUpdateExecutionWithMatching:
    """Tests for /experience/update-execution with raw order matching."""

    def test_update_execution_with_raw_orders(
        self, client, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test update-execution accepts intended_orders and executed_orders."""
        with patch(
            "brain_api.routes.experience.get_experience_storage",
            return_value=temp_storage,
        ):
            # First create a record
            client.post(
                "/experience/store",
                json={
                    "run_id": "paper:2026-02-05",
                    "week_start": "2026-02-05",
                    "week_end": "2026-02-09",
                    "model_type": "ppo",
                    "model_version": "v1.0.0",
                    "state": sample_full_state,
                    "intended_action": sample_intended_action,
                    "intended_turnover": 0.1,
                },
            )

            # Update with raw orders instead of pre-matched report
            response = client.post(
                "/experience/update-execution",
                json={
                    "run_id": "paper:2026-02-05",
                    "model_type": "ppo",
                    "intended_orders": [
                        {
                            "symbol": "AAPL",
                            "side": "buy",
                            "qty": 10,
                            "client_order_id": "paper:2026-02-05:attempt-1:AAPL:BUY",
                        },
                        {
                            "symbol": "MSFT",
                            "side": "sell",
                            "qty": 5,
                            "client_order_id": "paper:2026-02-05:attempt-1:MSFT:SELL",
                        },
                    ],
                    "executed_orders": [
                        {
                            "client_order_id": "paper:2026-02-05:attempt-1:AAPL:BUY",
                            "status": "filled",
                            "filled_qty": "10",
                            "filled_avg_price": "175.50",
                        },
                        {
                            "client_order_id": "paper:2026-02-05:attempt-1:MSFT:SELL",
                            "status": "filled",
                            "filled_qty": "5",
                            "filled_avg_price": "400.00",
                        },
                    ],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["updated"] is True
            assert data["orders_filled"] == 2

    def test_update_execution_raw_orders_with_not_found(
        self, client, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that raw orders matching handles missing orders correctly."""
        with patch(
            "brain_api.routes.experience.get_experience_storage",
            return_value=temp_storage,
        ):
            client.post(
                "/experience/store",
                json={
                    "run_id": "paper:2026-02-05",
                    "week_start": "2026-02-05",
                    "week_end": "2026-02-09",
                    "model_type": "sac",
                    "model_version": "v1.0.0",
                    "state": sample_full_state,
                    "intended_action": sample_intended_action,
                    "intended_turnover": 0.1,
                },
            )

            # One order has no match in executed
            response = client.post(
                "/experience/update-execution",
                json={
                    "run_id": "paper:2026-02-05",
                    "model_type": "sac",
                    "intended_orders": [
                        {
                            "symbol": "AAPL",
                            "side": "buy",
                            "qty": 10,
                            "client_order_id": "id1",
                        },
                        {
                            "symbol": "GOOGL",
                            "side": "buy",
                            "qty": 5,
                            "client_order_id": "id2",
                        },
                    ],
                    "executed_orders": [
                        {
                            "client_order_id": "id1",
                            "status": "filled",
                            "filled_qty": "10",
                            "filled_avg_price": "175.00",
                        },
                        # id2 missing - order was rejected before Alpaca
                    ],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["updated"] is True
            assert data["orders_filled"] == 1
            # The not_found order shouldn't count as filled
            assert data.get("orders_expired", 0) + data.get("orders_partial", 0) <= 1


class TestRewardLogSpaceConsistency:
    """Tests verifying reward function uses log-space for both return and cost."""

    def test_reward_log_space_consistency(self):
        """Verify reward = (log(1+r) - log(1+tc)) * scale equals log((1+r)/(1+tc)) * scale."""
        import numpy as np

        from brain_api.core.portfolio_rl.config import PPOBaseConfig

        config = PPOBaseConfig(cost_bps=10, reward_scale=100.0)
        r = 0.02  # 2% weekly return
        turnover = 0.5  # 50% turnover
        tc = turnover * (config.cost_bps / 10_000)  # 0.0005

        portfolio_log_return = np.log(1 + r)
        reward = compute_reward_from_log_return(portfolio_log_return, turnover, config)

        # The reward should equal log((1+r)/(1+tc)) * scale
        expected = np.log((1 + r) / (1 + tc)) * config.reward_scale
        assert abs(reward - expected) < 1e-10

    def test_reward_zero_return_with_cost(self):
        """Zero return with nonzero cost gives exactly -log(1+tc) * scale."""
        import numpy as np

        from brain_api.core.portfolio_rl.config import PPOBaseConfig

        config = PPOBaseConfig(cost_bps=10, reward_scale=100.0)
        turnover = 0.5
        tc = turnover * (config.cost_bps / 10_000)

        portfolio_log_return = 0.0  # log(1 + 0) = 0
        reward = compute_reward_from_log_return(portfolio_log_return, turnover, config)

        expected = -np.log(1 + tc) * config.reward_scale
        assert abs(reward - expected) < 1e-10
        # Also verify it differs from the old (incorrect) formula
        incorrect_reward = (0.0 - tc) * config.reward_scale
        assert abs(reward - incorrect_reward) > 1e-10

    def test_reward_cost_is_log_transformed(self):
        """Verify the transaction cost term is log(1+tc), not raw tc."""
        import numpy as np

        from brain_api.core.portfolio_rl.config import PPOBaseConfig

        config = PPOBaseConfig(
            cost_bps=100, reward_scale=1.0
        )  # 1% cost for larger diff
        turnover = 1.0  # 100% turnover for maximum cost
        tc = turnover * (config.cost_bps / 10_000)  # 0.01

        portfolio_log_return = 0.0
        reward = compute_reward_from_log_return(portfolio_log_return, turnover, config)

        # With log transform: reward = -log(1.01) * 1.0 = -0.00995...
        # Without log transform (old bug): reward = -0.01 * 1.0 = -0.01
        log_tc = np.log(1 + tc)
        assert abs(reward - (-log_tc)) < 1e-10
        assert abs(reward - (-tc)) > 1e-6  # Must differ from raw tc
