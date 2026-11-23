"""IBKR cancel confirmations should end the sell wait without hiding fills."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from activities import portfolio


def test_ibkr_poll_normalizes_only_confirmed_cancellations(monkeypatch):
    orders = [
        {"client_order_id": str(i), "status": status, "filled_qty": "0.5"}
        for i, status in enumerate(
            ["Cancelled", "ApiCancelled", "PendingCancel", "PreCancelled", "Filled"]
        )
    ]
    client = MagicMock()
    client.__enter__.return_value.get.return_value.json.return_value = orders
    monkeypatch.setattr(portfolio, "get_client", lambda: client)
    monkeypatch.setattr(
        portfolio.activity,
        "info",
        lambda: SimpleNamespace(
            current_attempt_scheduled_time=datetime(2026, 11, 23, tzinfo=UTC)
        ),
    )

    results = portfolio.check_order_statuses_ibkr("sac_halal", ["0", "1", "2", "3"])

    assert [order["status"] for order in results] == [
        "canceled",
        "canceled",
        "PendingCancel",
        "PreCancelled",
    ]
    assert [order["filled_qty"] for order in results] == ["0.5"] * 4
    client.__enter__.return_value.get.assert_called_once_with(
        "/ibkr/order-history",
        params={"account": "sac_halal", "after": "2026-11-16", "sync_broker": "true"},
    )
