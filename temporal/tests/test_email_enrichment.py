"""Tests for the pure helpers in ``activities.email_enrichment``.

Split out of ``test_alpha_hrp_activities.py`` so each file stays under
the 600-line limit (AGENTS.md). These helpers don't make HTTP calls;
they're called inline inside workflow code to assemble the email
payload.

Stop-loss math is now centralised in
``brain_api.core.stop_loss.compute_stop_loss`` and populated on each
``OrderModel`` by brain_api's ``/orders/generate``. ``build_order_details``
reads those fields verbatim, so these tests just assert the wiring
(stop-loss values round-trip onto each ``OrderDetail``) rather than
re-asserting the math.
"""

from __future__ import annotations

import pytest

from activities import email_enrichment
from models import (
    GenerateOrdersResponse,
    OrderModel,
    OrderSummary,
    PortfolioResponse,
    PositionModel,
    SkippedOrdersResponse,
    SubmitOrdersResponse,
)
from models.forecast_email import OrderSubmitResult


def _orders_response(orders: list[OrderModel]) -> GenerateOrdersResponse:
    """Build a minimal ``GenerateOrdersResponse`` around an order list."""
    return GenerateOrdersResponse(
        orders=orders,
        summary=OrderSummary(
            buys=sum(1 for o in orders if o.side == "buy"),
            sells=sum(1 for o in orders if o.side == "sell"),
            total_buy_value=0.0,
            total_sell_value=0.0,
            turnover_pct=0.0,
            skipped_small_orders=0,
            skipped_below_threshold=0,
        ),
        prices_used={o.symbol: 100.0 for o in orders},
    )


class TestBuildOrderDetails:
    """Stop-loss is sourced from the OrderModel itself; the helper just wires."""

    def test_skipped_orders_returns_empty_list(self):
        skipped = SkippedOrdersResponse(skipped=True, algorithm="alpha_hrp")
        result = email_enrichment.build_order_details(skipped, None)
        assert result == []

    def test_none_orders_returns_empty_list(self):
        assert email_enrichment.build_order_details(None, None) == []

    def test_buy_with_atr14_stop_round_trips(self):
        orders = _orders_response(
            [
                OrderModel(
                    client_order_id="cid:A:buy",
                    symbol="A",
                    side="buy",
                    qty=10.0,
                    type="limit",
                    time_in_force="day",
                    stop_loss_price=94.0,
                    stop_loss_distance_pct=0.06,
                    stop_loss_reason="atr14",
                ),
            ]
        )
        details = email_enrichment.build_order_details(orders, None)
        assert len(details) == 1
        d = details[0]
        assert d.stop_loss_reason == "atr14"
        assert d.stop_loss_price == pytest.approx(94.0)
        assert d.stop_loss_distance_pct == pytest.approx(0.06)
        assert d.trade_value == pytest.approx(1000.0)

    def test_sell_no_stop_sentinel_round_trips(self):
        orders = _orders_response(
            [
                OrderModel(
                    client_order_id="cid:A:sell",
                    symbol="A",
                    side="sell",
                    qty=2.0,
                    type="limit",
                    time_in_force="day",
                    stop_loss_reason="sell_no_stop",
                ),
            ]
        )
        details = email_enrichment.build_order_details(orders, None)
        assert details[0].stop_loss_reason == "sell_no_stop"
        assert details[0].stop_loss_price is None

    def test_atr_unavailable_sentinel_round_trips(self):
        # AGENTS.md rule #1: missing ATR surfaces as the literal sentinel,
        # never as a flat-percent fallback.
        orders = _orders_response(
            [
                OrderModel(
                    client_order_id="cid:Z:buy",
                    symbol="Z",
                    side="buy",
                    qty=1.0,
                    type="limit",
                    time_in_force="day",
                    stop_loss_reason="atr_unavailable",
                ),
            ]
        )
        details = email_enrichment.build_order_details(orders, None)
        assert details[0].stop_loss_reason == "atr_unavailable"
        assert details[0].stop_loss_price is None

    def test_submission_status_round_trips_failure(self):
        orders = _orders_response(
            [
                OrderModel(
                    client_order_id="cid:A:buy",
                    symbol="A",
                    side="buy",
                    qty=1.0,
                    type="limit",
                    time_in_force="day",
                    stop_loss_reason="atr14",
                    stop_loss_price=94.0,
                    stop_loss_distance_pct=0.06,
                ),
                OrderModel(
                    client_order_id="cid:B:buy",
                    symbol="B",
                    side="buy",
                    qty=1.0,
                    type="limit",
                    time_in_force="day",
                    stop_loss_reason="atr14",
                    stop_loss_price=94.0,
                    stop_loss_distance_pct=0.06,
                ),
            ]
        )
        submit = SubmitOrdersResponse(
            account="hrp",
            orders_submitted=1,
            orders_failed=1,
            skipped=False,
            results=[
                OrderSubmitResult(
                    client_order_id="cid:A:buy",
                    symbol="A",
                    status="submitted",
                ),
                OrderSubmitResult(
                    client_order_id="cid:B:buy",
                    symbol="B",
                    status="failed",
                    error="rejected by venue",
                ),
            ],
        )
        details = email_enrichment.build_order_details(orders, submit)
        statuses = {d.symbol: d.submission_status for d in details}
        assert statuses == {"A": "submitted", "B": "failed"}


class TestBuildPriorAllocationFromPortfolio:
    def test_converts_positions_to_weights_with_cash_slot(self):
        portfolio = PortfolioResponse(
            cash=200.0,
            positions=[
                PositionModel(symbol="A", qty=1.0, market_value=600.0),
                PositionModel(symbol="B", qty=2.0, market_value=200.0),
            ],
            open_orders_count=0,
        )
        prior = email_enrichment.build_prior_allocation_from_portfolio(
            portfolio,
            source_label="live Alpaca account: hrp",
            as_of="2026-04-21",
        )
        assert prior.weights == {"A": 0.6, "B": 0.2, "CASH": 0.2}
        assert prior.source_label == "live Alpaca account: hrp"

    def test_zero_nav_yields_empty_weights(self):
        # Empty account -> empty weights so the partial template's
        # ``{% if prior_allocation.weights %}`` gate hides the block,
        # rather than rendering a misleading "100% cash" of $0.
        portfolio = PortfolioResponse(
            cash=0.0,
            positions=[],
            open_orders_count=0,
        )
        prior = email_enrichment.build_prior_allocation_from_portfolio(
            portfolio,
            source_label="live Alpaca account: dhrp",
        )
        assert prior.weights == {}
        assert prior.source_label == "live Alpaca account: dhrp"


class TestBuildPriorAllocationFromDb:
    def test_converts_pct_to_fractions(self):
        prior = email_enrichment.build_prior_allocation_from_db(
            {"S001.NS": 50.0, "S002.NS": 30.0},
            source_label="recorded last week (202608)",
            as_of="202608",
        )
        assert prior.weights == {"S001.NS": 0.5, "S002.NS": 0.3}
        assert prior.source_label == "recorded last week (202608)"

    def test_empty_db_yields_empty_weights(self):
        prior = email_enrichment.build_prior_allocation_from_db(
            {},
            source_label="recorded last week (cold start)",
        )
        assert prior.weights == {}
