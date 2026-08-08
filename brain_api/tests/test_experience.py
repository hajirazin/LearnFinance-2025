"""Tests for experience storage and labeling endpoints.

This module tests:
- Full state experience storage (SAC)
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
                "news_coverage": 1.0,
                "gross_margin": 0.42,
                "operating_margin": 0.30,
                "current_ratio": 1.5,
                "debt_to_equity": 0.8,
                "fundamental_age": 7.0,
            },
            "MSFT": {
                "news_sentiment": 0.5,
                "news_coverage": 0.67,
                "gross_margin": 0.68,
                "operating_margin": 0.42,
                "current_ratio": 2.1,
                "debt_to_equity": 0.4,
                "fundamental_age": 14.0,
            },
        },
        "patchtst_forecasts": {"AAPL": 0.015, "MSFT": -0.003},
        "current_weights": {"AAPL": 0.10, "MSFT": 0.08, "CASH": 0.82},
    }


@pytest.fixture
def sample_intended_action():
    """Sample intended action for testing."""
    return {"AAPL": 0.15, "MSFT": 0.12, "CASH": 0.73}


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

    def test_store_sac_persists_universe_field(
        self, client, temp_storage, sample_full_state, sample_intended_action
    ):
        """POST /experience/store with universe=halal round-trips the field.

        The labeller reads ``universe`` back to route each record to the
        correct Alpaca account; a missing or stripped field would
        silently re-introduce the bug this ticket fixed.
        """
        from brain_api.routes.experience import get_experience_storage

        app.dependency_overrides[get_experience_storage] = lambda: temp_storage
        try:
            response = client.post(
                "/experience/store",
                json={
                    "run_id": "paper:halal:2026-05-04",
                    "week_start": "2026-05-04",
                    "week_end": "2026-05-08",
                    "model_type": "sac",
                    "model_version": "v1.0.0",
                    "universe": "halal",
                    "state": sample_full_state,
                    "intended_action": sample_intended_action,
                    "intended_turnover": 0.1,
                },
            )
        finally:
            app.dependency_overrides.pop(get_experience_storage, None)
        assert response.status_code == 200

        loaded = temp_storage.load("paper:halal:2026-05-04:sac")
        assert loaded is not None
        assert loaded.universe == "halal"

    def test_store_sac_persists_and_validates_canonical_state_digest(
        self, client, temp_storage, sample_intended_action
    ):
        from brain_api.routes.experience import get_experience_storage

        state = {"vector": [0.1, 0.9], "digest": "abc123"}
        body = {
            "run_id": "paper:2026-05-04",
            "week_start": "2026-05-04",
            "week_end": "2026-05-08",
            "model_type": "sac",
            "model_version": "v2",
            "state": state,
            "state_digest": "abc123",
            "intended_action": sample_intended_action,
        }
        app.dependency_overrides[get_experience_storage] = lambda: temp_storage
        try:
            response = client.post("/experience/store", json=body)
            mismatch = client.post(
                "/experience/store", json={**body, "state_digest": "wrong"}
            )
        finally:
            app.dependency_overrides.pop(get_experience_storage, None)

        assert response.status_code == 200
        assert mismatch.status_code == 422
        loaded = temp_storage.load("paper:2026-05-04:sac")
        assert loaded is not None
        assert loaded.state_digest == "abc123"

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

    def test_sac_state_includes_patchtst_forecasts(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """Test that stored SAC state includes PatchTST forecasts only."""
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
            assert "lstm_forecasts" not in state
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


class TestLabelSACEndpoint:
    """Tests for /experience/label/sac endpoint."""

    def test_label_sac_endpoint_returns_200(self, client):
        """Test that label SAC endpoint returns 200."""
        with (
            patch("brain_api.core.alpaca_client.get_alpaca_client") as mock_get_client,
            patch("brain_api.core.lstm.load_prices_yfinance") as mock_prices,
        ):
            mock_client = MagicMock()
            mock_client.account = MagicMock(value="sac")
            mock_client.get_portfolio_weights.return_value = {"MSFT": 0.6, "CASH": 0.4}
            mock_get_client.return_value = mock_client
            mock_prices.return_value = {}

            response = client.post(
                "/experience/label/sac",
                json={"run_id": None},
            )
            assert response.status_code == 200

    def test_label_sac_halal_filtered_routes_to_sac_account(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """halal_filtered universe -> AlpacaAccount.SAC."""
        past_monday = date.today() - timedelta(days=date.today().weekday() + 14)
        past_week_start = past_monday.isoformat()
        past_week_end = (past_monday + timedelta(days=7)).isoformat()
        record = ExperienceRecord(
            run_id="paper:2026-04-13:sac",
            week_start=past_week_start,
            week_end=past_week_end,
            model_type="sac",
            model_version="v1.0.0",
            universe="halal_filtered",
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
            mock_client.account = MagicMock(value="sac")
            mock_client.get_portfolio_weights.return_value = {"MSFT": 0.6, "CASH": 0.4}
            mock_get_client.return_value = mock_client

            from brain_api.routes.experience import _label_experience_for_account

            with patch("brain_api.core.lstm.load_prices_yfinance") as mock_prices:
                mock_prices.return_value = {}
                _label_experience_for_account("sac", None, temp_storage)

            from brain_api.core.alpaca_client import AlpacaAccount

            mock_get_client.assert_called_with(AlpacaAccount.SAC)

    def test_label_sac_halal_skips_when_no_actual_weights(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """halal records have NO Alpaca account post IBKR migration.

        The halal SAC variant trades through IBKR (see
        ``brain_api.routes.ibkr``), so ``resolve_alpaca_account('sac',
        'halal')`` raises by design (AGENTS.md rule #1). A halal
        record reaching the labeller without ``actual_weights`` plumbed
        in MUST be skipped with an error rather than silently labelled
        against an Alpaca account that never held the IBKR positions.
        """
        past_week_start = (date.today() - timedelta(days=14)).isoformat()
        past_week_end = (date.today() - timedelta(days=10)).isoformat()
        record = ExperienceRecord(
            run_id="paper:halal:2026-04-13:sac",
            week_start=past_week_start,
            week_end=past_week_end,
            model_type="sac",
            model_version="v1.0.0",
            universe="halal",
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
            patch("brain_api.core.lstm.load_prices_yfinance", return_value={}),
        ):
            from brain_api.routes.experience import _label_experience_for_account

            response = _label_experience_for_account("sac", None, temp_storage)

        # No Alpaca client should ever be constructed for a halal record.
        mock_get_client.assert_not_called()
        # The labeller must surface the routing failure rather than
        # silently labelling -- AGENTS.md rule #1.
        assert response.records_labeled == 0
        assert response.records_skipped >= 1 or response.errors

    def test_label_sac_halal_with_actual_weights_skips_alpaca_lookup(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """halal records with actual_weights labelled without touching Alpaca.

        The IBKR-routed workflow plumbs the post-trade IBKR portfolio
        snapshot onto the experience record at write time; the labeller
        consumes that and never falls through to the Alpaca-account
        lookup that no longer has a halal entry.
        """
        past_week_start = (date.today() - timedelta(days=14)).isoformat()
        past_week_end = (date.today() - timedelta(days=10)).isoformat()
        record = ExperienceRecord(
            run_id="paper:halal:2026-04-13:sac",
            week_start=past_week_start,
            week_end=past_week_end,
            model_type="sac",
            model_version="v1.0.0",
            universe="halal",
            state=sample_full_state,
            intended_action=sample_intended_action,
            intended_turnover=0.1,
            actual_weights={"AAPL": 0.5, "CASH": 0.5},
        )
        temp_storage.store(record)

        with (
            patch("brain_api.core.alpaca_client.get_alpaca_client") as mock_get_client,
            patch(
                "brain_api.routes.experience.get_experience_storage",
                return_value=temp_storage,
            ),
            patch("brain_api.core.lstm.load_prices_yfinance", return_value={}),
        ):
            from brain_api.routes.experience import _label_experience_for_account

            _label_experience_for_account("sac", None, temp_storage)

        # The post-trade snapshot already provides actual_weights, so
        # the labeller never even attempts to construct an Alpaca client.
        mock_get_client.assert_not_called()

    def test_label_sac_routes_halal_filtered_unaffected_by_halal_drop(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """halal_filtered records still route to AlpacaAccount.SAC.

        Surgical edit guarantee: dropping the halal entry from
        ``_SAC_UNIVERSE_TO_ACCOUNT`` MUST NOT affect halal_filtered
        labelling. This test pins that guarantee so a future cleanup
        can't accidentally take SAC down with sac_halal.
        """
        from brain_api.core.alpaca_client import AlpacaAccount

        past_week_start = (date.today() - timedelta(days=14)).isoformat()
        past_week_end = (date.today() - timedelta(days=10)).isoformat()

        temp_storage.store(
            ExperienceRecord(
                run_id="paper:2026-01-01:sac",
                week_start=past_week_start,
                week_end=past_week_end,
                model_type="sac",
                model_version="v1.0.0",
                universe="halal_filtered",
                state=sample_full_state,
                intended_action=sample_intended_action,
                intended_turnover=0.1,
            )
        )

        constructed: list[AlpacaAccount] = []

        def fake_get_client(account: AlpacaAccount):
            constructed.append(account)
            mc = MagicMock()
            mc.account = MagicMock(value=account.value)
            mc.get_portfolio_weights.return_value = {"AAPL": 0.5, "CASH": 0.5}
            return mc

        with (
            patch(
                "brain_api.core.alpaca_client.get_alpaca_client",
                side_effect=fake_get_client,
            ),
            patch("brain_api.core.lstm.load_prices_yfinance", return_value={}),
        ):
            from brain_api.routes.experience import _label_experience_for_account

            _label_experience_for_account("sac", None, temp_storage)

        assert constructed == [AlpacaAccount.SAC]

    def test_label_sac_legacy_halal_run_id_skips_safely(
        self, temp_storage, sample_full_state, sample_intended_action
    ):
        """Legacy halal records (no universe field) skip safely.

        A pre-IBKR-migration record with run_id ``paper:halal:...`` and
        no ``universe`` field would have its universe inferred as
        ``halal`` from the run_id prefix. Post-migration that no
        longer maps to an Alpaca account, so the labeller surfaces the
        routing failure and skips rather than silently labelling.
        """
        past_week_start = (date.today() - timedelta(days=14)).isoformat()
        past_week_end = (date.today() - timedelta(days=10)).isoformat()
        record = ExperienceRecord(
            run_id="paper:halal:2026-04-13:sac",
            week_start=past_week_start,
            week_end=past_week_end,
            model_type="sac",
            model_version="v1.0.0",
            universe=None,  # legacy record predating the universe field
            state=sample_full_state,
            intended_action=sample_intended_action,
            intended_turnover=0.1,
        )
        temp_storage.store(record)

        with (
            patch("brain_api.core.alpaca_client.get_alpaca_client") as mock_get_client,
            patch("brain_api.core.lstm.load_prices_yfinance", return_value={}),
        ):
            from brain_api.routes.experience import _label_experience_for_account

            response = _label_experience_for_account("sac", None, temp_storage)

        mock_get_client.assert_not_called()
        assert response.records_labeled == 0

    def test_label_sac_uses_actual_weights_not_intended(self):
        """Test that SAC labeling uses actual weights for the return.

        Pass a no-trade rebalance (prior == actual) so the IBKR cost
        leg is zero and we can isolate the portfolio-return part.
        """
        actual_weights = {"GOOGL": 0.4, "AMZN": 0.4, "CASH": 0.2}
        symbol_returns = {"GOOGL": 0.03, "AMZN": 0.02}

        _reward, portfolio_return = _compute_reward_from_actual_weights(
            actual_weights=actual_weights,
            symbol_returns=symbol_returns,
            prior_weights=actual_weights,  # no rebalance -> zero cost
            symbol_prices={"GOOGL": 150.0, "AMZN": 180.0},
            nav_usd=10_000.0,
        )

        expected_return = 0.4 * 0.03 + 0.4 * 0.02
        assert abs(portfolio_return - expected_return) < 0.001

    def test_label_sac_calculates_log_return(self):
        """Test that SAC reward uses log return.

        With a no-trade rebalance the IBKR cost is zero, so the only
        thing pushing the reward away from zero is the price return.
        """
        actual_weights = {"NVDA": 1.0, "CASH": 0.0}
        symbol_returns = {"NVDA": 0.20}  # 20% return

        reward, _ = _compute_reward_from_actual_weights(
            actual_weights=actual_weights,
            symbol_returns=symbol_returns,
            prior_weights=actual_weights,  # no rebalance
            symbol_prices={"NVDA": 500.0},
            nav_usd=10_000.0,
        )

        # Should be positive for positive return
        assert reward > 0

    def test_label_sac_includes_transaction_cost(self):
        """A 100% open-from-cash rebalance with zero return must produce
        a negative reward (the IBKR commission floor on a single leg
        binds and cuts into a flat-return portfolio)."""
        actual_weights = {"TSLA": 1.0, "CASH": 0.0}
        symbol_returns = {"TSLA": 0.0}

        reward, _ = _compute_reward_from_actual_weights(
            actual_weights=actual_weights,
            symbol_returns=symbol_returns,
            prior_weights={"CASH": 1.0},  # 100% buy from cash
            symbol_prices={"TSLA": 250.0},
            nav_usd=10_000.0,
        )

        assert reward < 0

    def test_label_sac_rejects_missing_held_symbol_return(self):
        """An unobserved held-symbol return is not a zero-return observation."""
        with pytest.raises(ValueError, match=r"Missing realized return.*MSFT"):
            _compute_reward_from_actual_weights(
                actual_weights={"AAPL": 0.4, "MSFT": 0.4, "CASH": 0.2},
                symbol_returns={"AAPL": 0.01},
                prior_weights={"AAPL": 0.4, "MSFT": 0.4, "CASH": 0.2},
                symbol_prices={"AAPL": 100.0, "MSFT": 200.0},
            )

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
                    "model_type": "sac",
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
                    "model_type": "sac",
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
                "model_type": "sac",
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
                    "model_type": "sac",
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
                    "model_type": "sac",
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


class TestLabelSACUsesIBKRCostModel:
    """End-to-end check that the labeller's realised reward uses the IBKR-SG cost model.

    Stubs ``load_prices_yfinance`` to return a known price series and
    pre-populates the experience record with prior_weights + actual_weights
    + nav_usd. The reward must then equal the analytic value produced by
    ``compute_ibkr_rebalance_cost`` -- not the legacy flat 10 bps formula.
    """

    def test_label_sac_realised_reward_matches_ibkr_cost_model(self, temp_storage):
        """Realised reward equals exact log(1 + r - tc) with IBKR-SG costs."""
        import numpy as np
        import pandas as pd

        from brain_api.core.portfolio_rl.broker_costs import (
            IBKRSingaporeCostConfig,
            compute_ibkr_rebalance_cost,
        )

        past_monday = date.today() - timedelta(days=date.today().weekday() + 14)
        past_week_start = past_monday.isoformat()
        past_week_end = (past_monday + timedelta(days=7)).isoformat()
        # Prior was 100% cash; actual is 50% AAPL / 50% MSFT.
        actual_weights = {"AAPL": 0.5, "MSFT": 0.5, "CASH": 0.0}
        prior_state = {
            "signals": {},
            "patchtst_forecasts": {},
            "current_weights": {"CASH": 1.0},
        }
        nav_usd = 10_000.0
        record = ExperienceRecord(
            run_id="paper:halal:2026-04-13:sac",
            week_start=past_week_start,
            week_end=past_week_end,
            model_type="sac",
            model_version="v1.0.0",
            universe="halal",
            state=prior_state,
            intended_action=actual_weights,
            intended_turnover=0.5,
            actual_weights=actual_weights,
            nav_usd=nav_usd,
        )
        temp_storage.store(record)

        # Stub yfinance: AAPL +2%, MSFT +1% from one weekly XNYS open
        # to the next.
        def _fake_prices(symbols, start, end):
            idx = pd.to_datetime([past_week_start, past_week_end])
            return {
                "AAPL": pd.DataFrame({"open": [200.0, 204.0]}, index=idx),
                "MSFT": pd.DataFrame({"open": [300.0, 303.0]}, index=idx),
            }

        with (
            patch("brain_api.core.lstm.load_prices_yfinance", side_effect=_fake_prices),
            patch(
                "brain_api.routes.experience.get_experience_storage",
                return_value=temp_storage,
            ),
        ):
            from brain_api.routes.experience import _label_experience_for_account

            response = _label_experience_for_account("sac", None, temp_storage)

        assert response.records_labeled == 1, response.errors

        labelled = temp_storage.load("paper:halal:2026-04-13:sac")
        assert labelled.reward is not None

        # Analytic check using the same IBKR cost model the labeller
        # invokes. Two open-from-cash buy legs (50% each), $5000 each.
        cfg = IBKRSingaporeCostConfig.default().with_nav(nav_usd)
        prior_arr = np.array([0.0, 0.0, 1.0])  # AAPL, MSFT, CASH (sorted)
        target_arr = np.array([0.5, 0.5, 0.0])
        prices = np.array([200.0, 300.0])  # transition trade opens, alphabetical
        expected_cost = compute_ibkr_rebalance_cost(
            symbol_order=["AAPL", "MSFT"],
            current_weights=prior_arr,
            target_weights=target_arr,
            prices=prices,
            cfg=cfg,
        )
        expected_tc = expected_cost.total_fraction
        # AAPL +2%, MSFT +1% -> portfolio simple return = 0.5*0.02 + 0.5*0.01 = 0.015
        expected_return = 0.5 * 0.02 + 0.5 * 0.01
        expected_reward = np.log(1 + expected_return - expected_tc) * 100.0

        assert labelled.reward == pytest.approx(expected_reward, rel=1e-6)
        assert labelled.realized_return == pytest.approx(expected_return, rel=1e-6)


class TestRewardLogSpaceConsistency:
    """Tests verifying reward uses exact net wealth after transaction costs.

    The cost source moved from a flat ``cost_bps * turnover`` formula
    to the IBKR-SG per-leg model (see broker_costs.py); these tests
    pass the precomputed ``transaction_cost_fraction`` directly so
    they exercise only the reward-shape invariant.
    """

    def test_reward_log_space_consistency(self):
        """Reward is exact log(1 + gross return - cost fraction)."""
        import numpy as np

        from brain_api.core.portfolio_rl.config import RLBaseConfig

        config = RLBaseConfig(reward_scale=100.0)
        r = 0.02  # 2% weekly return
        tc = 0.0005  # 5 bps fraction of NAV

        portfolio_log_return = np.log(1 + r)
        reward = compute_reward_from_log_return(portfolio_log_return, tc, config)

        expected = np.log(1 + r - tc) * config.reward_scale
        assert abs(reward - expected) < 1e-10

    def test_reward_zero_return_with_cost(self):
        """Zero return with cost gives exactly log(1-tc) * scale."""
        import numpy as np

        from brain_api.core.portfolio_rl.config import RLBaseConfig

        config = RLBaseConfig(reward_scale=100.0)
        tc = 0.0005

        portfolio_log_return = 0.0  # log(1 + 0) = 0
        reward = compute_reward_from_log_return(portfolio_log_return, tc, config)

        expected = np.log(1 - tc) * config.reward_scale
        assert abs(reward - expected) < 1e-10
        # Also verify it differs from the old (incorrect) formula
        incorrect_reward = (0.0 - tc) * config.reward_scale
        assert abs(reward - incorrect_reward) > 1e-10

    def test_reward_cost_is_log_transformed(self):
        """Verify cost is deducted from wealth before taking the logarithm."""
        import numpy as np

        from brain_api.core.portfolio_rl.config import RLBaseConfig

        config = RLBaseConfig(reward_scale=1.0)
        tc = 0.01  # 1% of NAV cost for a clearly visible log-vs-linear gap

        portfolio_log_return = 0.0
        reward = compute_reward_from_log_return(portfolio_log_return, tc, config)

        exact = np.log(1 - tc)
        assert abs(reward - exact) < 1e-10
        assert abs(reward - (-tc)) > 1e-6  # Must differ from raw tc
