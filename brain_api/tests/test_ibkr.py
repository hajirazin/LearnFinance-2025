"""Router-level tests for ``/ibkr/*``.

Per repo policy (AGENTS.md "Testing policy"), the test suite covers
the API surface end-to-end through FastAPI's ``TestClient`` with the
``ib_async``-touching layer mocked at the
``brain_api.core.ibkr_client`` boundary. Schema validation is not
unit-tested in isolation -- the assertions here exercise the schemas
implicitly through real route invocations.

Coverage targets:

* ``GET /ibkr/portfolio`` -- happy path returns the broker-agnostic
  ``PortfolioResponse`` shape, missing env vars surface as 500, and a
  failed gateway connection surfaces as 503.
* ``POST /ibkr/submit-orders`` -- submission flows through the
  pre-submit dedup gate (local ledger + open-trades scan), records
  every placement into the local ledger, and surfaces broker-side
  errors as ``status='error'`` rows without crashing the loop.
* ``GET /ibkr/order-history`` -- backed by the local ledger
  (NOT the gateway), respects the ``after`` filter, and returns the
  Alpaca-shaped ``OrderHistoryItem`` payload that the Temporal
  ``check_order_statuses`` regex depends on.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain_api.core.ibkr_client import (
    IBKRConnectionConfig,
    IBKRPortfolio,
    IBKRPosition,
    IBKRSubmitResult,
)
from brain_api.main import app
from brain_api.routes.ibkr import get_ibkr_order_ledger
from brain_api.storage.ibkr_orders import IBKROrderLedger, SubmittedOrderRow


@pytest.fixture
def ibkr_env(monkeypatch):
    """Set the minimal IBKR_SAC_HALAL_* env block for routing."""
    monkeypatch.setenv("IBKR_SAC_HALAL_HOST", "localhost")
    monkeypatch.setenv("IBKR_SAC_HALAL_PORT", "4002")
    monkeypatch.setenv("IBKR_SAC_HALAL_CLIENT_ID", "11")
    monkeypatch.setenv("IBKR_SAC_HALAL_ACCOUNT_CODE", "DU1234567")


@pytest.fixture
def temp_ledger(tmp_path: Path):
    """Provide a fresh sqlite ledger pointed at a temp file.

    Overrides the FastAPI dependency factory so route handlers get
    this temp-backed repo instead of the production
    ``data/ibkr/submitted_orders.db`` path.
    """
    db_path = tmp_path / "ledger.db"
    ledger = IBKROrderLedger(db_path=db_path)
    app.dependency_overrides[get_ibkr_order_ledger] = lambda: ledger
    try:
        yield ledger
    finally:
        app.dependency_overrides.pop(get_ibkr_order_ledger, None)


@pytest.fixture
def client(ibkr_env, temp_ledger):
    return TestClient(app)


# ============================================================================
# GET /ibkr/portfolio
# ============================================================================


class TestIBKRGetPortfolio:
    """Cover the IBKR portfolio route end-to-end with ib_async mocked."""

    def test_portfolio_returns_broker_agnostic_shape(self, client):
        """Happy path: cash + positions + open_orders_count from gateway."""
        fake_portfolio = IBKRPortfolio(
            cash=12_345.67,
            cash_balances={"USD": 12_345.67},
            positions=[
                IBKRPosition(symbol="AAPL", qty=10.0, market_value=1750.0),
                IBKRPosition(symbol="MSFT", qty=2.0, market_value=815.50),
            ],
            open_orders_count=1,
        )
        with patch(
            "brain_api.routes.ibkr.get_portfolio", return_value=fake_portfolio
        ) as mock_portfolio:
            response = client.get("/ibkr/portfolio", params={"account": "sac_halal"})

        assert response.status_code == 200
        data = response.json()
        assert data["cash"] == pytest.approx(12_345.67)
        assert data["open_orders_count"] == 1
        assert {p["symbol"] for p in data["positions"]} == {"AAPL", "MSFT"}

        # Ensure the route resolved the env triple into a real config
        # (account code, port, etc.) before calling get_portfolio.
        config = mock_portfolio.call_args.args[0]
        assert isinstance(config, IBKRConnectionConfig)
        assert config.account == "sac_halal"
        assert config.port == 4002
        assert config.account_code == "DU1234567"

    def test_portfolio_invalid_account_returns_422(self, client):
        """Unknown account value is rejected by the IBKRAccount enum."""
        response = client.get("/ibkr/portfolio", params={"account": "unknown"})
        assert response.status_code == 422

    def test_portfolio_missing_env_returns_500(self, client, monkeypatch):
        """Missing IBKR_SAC_HALAL_CLIENT_ID -> 500 with explicit message."""
        monkeypatch.delenv("IBKR_SAC_HALAL_CLIENT_ID", raising=False)
        response = client.get("/ibkr/portfolio", params={"account": "sac_halal"})
        assert response.status_code == 500
        assert "IBKR_SAC_HALAL_CLIENT_ID" in response.json()["detail"]

    def test_portfolio_gateway_unreachable_returns_503(self, client):
        """ConnectionError from the gateway surfaces as 503."""
        with patch(
            "brain_api.routes.ibkr.get_portfolio",
            side_effect=ConnectionError("connection refused"),
        ):
            response = client.get("/ibkr/portfolio", params={"account": "sac_halal"})
        assert response.status_code == 503
        assert "unreachable" in response.json()["detail"].lower()


# ============================================================================
# POST /ibkr/submit-orders
# ============================================================================


def _order(symbol: str, side: str, client_order_id: str) -> dict:
    """Build a minimal valid OrderToSubmit payload."""
    return {
        "symbol": symbol,
        "qty": 1.0,
        "side": side,
        "type": "limit",
        "time_in_force": "day",
        "limit_price": 100.0,
        "client_order_id": client_order_id,
    }


class TestIBKRSubmitOrders:
    """Cover the dedup gate + ledger persistence + broker error paths."""

    def test_submit_records_to_ledger_on_success(self, client, temp_ledger):
        """Each successful placement upserts a row into the local ledger."""
        coid = "paper:halal:2026-05-04:attempt-1:AAPL:BUY"
        with (
            patch(
                "brain_api.routes.ibkr.get_session_status",
                return_value=True,
            ),
            patch("brain_api.routes.ibkr.list_open_order_refs", return_value=set()),
            patch(
                "brain_api.routes.ibkr.submit_order",
                return_value=IBKRSubmitResult(
                    client_order_id=coid,
                    perm_id=4242,
                    status="Submitted",
                    error=None,
                ),
            ),
        ):
            response = client.post(
                "/ibkr/submit-orders",
                json={
                    "account": "sac_halal",
                    "orders": [_order("AAPL", "buy", coid)],
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["account"] == "sac_halal"
        assert body["orders_submitted"] == 1
        assert body["orders_failed"] == 0
        assert body["results"][0]["status"] == "Submitted"
        # Ledger row materialised.
        assert temp_ledger.has_order_ref(coid)

    def test_submit_dedupes_on_existing_ledger_row(self, client, temp_ledger):
        """A previously-submitted order_ref short-circuits to status=deduped."""
        coid = "paper:halal:2026-05-04:attempt-1:AAPL:BUY"
        temp_ledger.record_submission(
            SubmittedOrderRow(
                account="sac_halal",
                run_id="paper:halal:2026-05-04",
                attempt=1,
                symbol="AAPL",
                side="buy",
                qty=1.0,
                limit_price=100.0,
                order_ref=coid,
                ibkr_perm_id=99,
                status="Filled",
                filled_qty=1.0,
                filled_avg_price=100.0,
            )
        )
        with (
            patch(
                "brain_api.routes.ibkr.get_session_status",
                return_value=True,
            ),
            patch("brain_api.routes.ibkr.list_open_order_refs", return_value=set()),
            patch("brain_api.routes.ibkr.submit_order") as mock_submit,
        ):
            response = client.post(
                "/ibkr/submit-orders",
                json={
                    "account": "sac_halal",
                    "orders": [_order("AAPL", "buy", coid)],
                },
            )

        assert response.status_code == 200
        body = response.json()
        # Dedup short-circuit: order was NOT placed and NOT counted.
        mock_submit.assert_not_called()
        assert body["orders_submitted"] == 0
        assert body["orders_failed"] == 0
        assert body["results"][0]["status"] == "deduped"

    def test_submit_dedupes_on_open_trades_book(self, client, temp_ledger):
        """An open_ref already on the gateway short-circuits placement.

        Catches the gap where a previous attempt placed an order on the
        gateway but crashed before the ledger insert. The open-trades
        scan picks it up before we re-submit.
        """
        coid = "paper:halal:2026-05-04:attempt-1:AAPL:BUY"
        with (
            patch(
                "brain_api.routes.ibkr.get_session_status",
                return_value=True,
            ),
            patch(
                "brain_api.routes.ibkr.list_open_order_refs",
                return_value={coid},
            ),
            patch("brain_api.routes.ibkr.submit_order") as mock_submit,
        ):
            response = client.post(
                "/ibkr/submit-orders",
                json={
                    "account": "sac_halal",
                    "orders": [_order("AAPL", "buy", coid)],
                },
            )

        mock_submit.assert_not_called()
        assert response.json()["results"][0]["status"] == "deduped"

    def test_submit_partial_failure_keeps_loop_alive(self, client, temp_ledger):
        """One bad order does not stop the others from being submitted."""
        good_coid = "paper:halal:2026-05-04:attempt-1:AAPL:SELL"
        bad_coid = "paper:halal:2026-05-04:attempt-1:MSFT:SELL"

        def fake_submit(_config, spec):
            if spec.client_order_id == bad_coid:
                return IBKRSubmitResult(
                    client_order_id=spec.client_order_id,
                    perm_id=None,
                    status="error",
                    error="margin check failed",
                )
            return IBKRSubmitResult(
                client_order_id=spec.client_order_id,
                perm_id=7,
                status="Submitted",
                error=None,
            )

        with (
            patch(
                "brain_api.routes.ibkr.get_session_status",
                return_value=True,
            ),
            patch("brain_api.routes.ibkr.list_open_order_refs", return_value=set()),
            patch(
                "brain_api.routes.ibkr.submit_order",
                side_effect=fake_submit,
            ),
        ):
            response = client.post(
                "/ibkr/submit-orders",
                json={
                    "account": "sac_halal",
                    "orders": [
                        _order("AAPL", "sell", good_coid),
                        _order("MSFT", "sell", bad_coid),
                    ],
                },
            )

        body = response.json()
        assert body["orders_submitted"] == 1
        assert body["orders_failed"] == 1
        statuses = {r["client_order_id"]: r for r in body["results"]}
        assert statuses[good_coid]["status"] == "Submitted"
        assert statuses[bad_coid]["status"] == "error"

    def test_submit_empty_orders_is_a_noop(self, client, temp_ledger):
        """An empty orders list returns 0/0 without touching the gateway."""
        with patch("brain_api.routes.ibkr.get_session_status") as mock_get_status:
            response = client.post(
                "/ibkr/submit-orders",
                json={"account": "sac_halal", "orders": []},
            )
        mock_get_status.assert_not_called()
        assert response.status_code == 200
        assert response.json()["orders_submitted"] == 0


# ============================================================================
# GET /ibkr/order-history
# ============================================================================


class TestIBKROrderHistory:
    """Cover the local-ledger backed history endpoint."""

    def test_history_serves_from_local_ledger(self, client, temp_ledger):
        """Mirrors Alpaca's order-history shape with client_order_id == order_ref."""
        coid = "paper:halal:2026-05-04:attempt-1:AAPL:BUY"
        temp_ledger.record_submission(
            SubmittedOrderRow(
                account="sac_halal",
                run_id="paper:halal:2026-05-04",
                attempt=1,
                symbol="AAPL",
                side="buy",
                qty=2.0,
                limit_price=175.0,
                order_ref=coid,
                ibkr_perm_id=12345,
                status="Filled",
                filled_qty=2.0,
                filled_avg_price=174.50,
            )
        )

        response = client.get(
            "/ibkr/order-history",
            params={"account": "sac_halal", "after": "2026-01-01"},
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        item = body[0]
        # The Temporal regex on client_order_id depends on this contract.
        assert item["client_order_id"] == coid
        assert item["status"] == "Filled"
        assert item["filled_qty"] == "2.0"
        assert item["filled_avg_price"] == "174.5"

    def test_history_after_filter_excludes_older_rows(self, client, temp_ledger):
        """The ``after`` query param excludes pre-cutoff rows."""
        # Manufacture a row dated way in the past by writing it through
        # the repo and then mutating submitted_at via a direct sqlite update.
        coid = "paper:halal:2020-01-01:attempt-1:AAPL:BUY"
        temp_ledger.record_submission(
            SubmittedOrderRow(
                account="sac_halal",
                run_id="paper:halal:2020-01-01",
                attempt=1,
                symbol="AAPL",
                side="buy",
                qty=1.0,
                limit_price=100.0,
                order_ref=coid,
                ibkr_perm_id=1,
                status="Filled",
                filled_qty=1.0,
                filled_avg_price=100.0,
            )
        )
        # Backdate the row.
        with temp_ledger._connect() as conn:
            conn.execute(
                "UPDATE ibkr_submitted_orders SET submitted_at='2020-01-01 00:00:00' WHERE order_ref=?",
                (coid,),
            )
            conn.commit()

        response = client.get(
            "/ibkr/order-history",
            params={"account": "sac_halal", "after": "2026-01-01"},
        )
        assert response.status_code == 200
        assert response.json() == []

    def test_history_invalid_account_returns_422(self, client):
        """Unknown account value is rejected by the IBKRAccount enum."""
        response = client.get(
            "/ibkr/order-history",
            params={"account": "unknown", "after": "2026-01-01"},
        )
        assert response.status_code == 422

    def test_history_missing_after_returns_422(self, client):
        """Missing ``after`` query param fails validation."""
        response = client.get("/ibkr/order-history", params={"account": "sac_halal"})
        assert response.status_code == 422
