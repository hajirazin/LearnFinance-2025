"""Regression tests for broker-history date boundaries."""

from activities.portfolio import _ibkr_ledger_after


def test_ibkr_ledger_lookup_crosses_utc_midnight_safely() -> None:
    """A Sep 8 workflow must still see orders persisted late on Sep 7 UTC."""
    assert _ibkr_ledger_after("2026-09-08") == "2026-09-01"
