"""Pure helpers for assembling US weekly email payloads.

Extracted from :mod:`activities.reporting` so the activity module
stays focused on ``@activity.defn``-decorated HTTP wrappers and stays
under the 600-line file-size limit (AGENTS.md).

These helpers run inline inside workflows; they don't make brain_api
calls and don't need their own activity round-trips. The same routine
is shared by every US weekly workflow (SAC halal_filtered, SAC halal,
US Alpha-HRP, US Double HRP) -- one renderer, four callers.

Stop-loss math lives in brain_api (``brain_api.core.stop_loss``) and
is populated on each ``OrderModel`` by ``/orders/generate``. This
module reads those fields verbatim instead of recomputing them, so
there is exactly one math implementation across the two services
(AGENTS.md rule #2).
"""

from __future__ import annotations

from models import (
    GenerateOrdersResponse,
    OrderDetail,
    PortfolioResponse,
    PriorAllocation,
    SkippedOrdersResponse,
    SkippedSubmitResponse,
    SubmitOrdersResponse,
)


def _submission_status_map(
    submit: SubmitOrdersResponse | SkippedSubmitResponse | None,
) -> dict[str, str]:
    """Index broker submission outcomes by ``client_order_id``.

    Returns ``{}`` for skipped/None payloads so the caller falls back
    to ``"submitted"`` (default optimistic state) only when no broker
    submission happened. When a real submission occurred, the map
    holds the real outcome for every order so the email can show
    ``failed`` / ``deduped`` honestly.
    """
    if submit is None:
        return {}
    if isinstance(submit, SkippedSubmitResponse) or getattr(submit, "skipped", False):
        return {}
    return {r.client_order_id: r.status for r in submit.results}


def build_order_details(
    orders: GenerateOrdersResponse | SkippedOrdersResponse | None,
    submit: SubmitOrdersResponse | SkippedSubmitResponse | None,
) -> list[OrderDetail]:
    """Combine generated orders + submission outcomes into render rows.

    Stop-loss fields are read directly from the ``OrderModel`` (set
    upstream by brain_api's ``/orders/generate``), so this helper
    contains zero math -- it just wires the broker submission status
    onto each row. Returns ``[]`` when the upstream skip path fired,
    which keeps the email partial silent (gated by
    ``{% if order_results.orders %}``).
    """
    if orders is None:
        return []
    if isinstance(orders, SkippedOrdersResponse) or getattr(orders, "skipped", False):
        return []

    prices = orders.prices_used
    status_by_id = _submission_status_map(submit)

    details: list[OrderDetail] = []
    for o in orders.orders:
        price = prices.get(o.symbol, 0.0)
        status = status_by_id.get(o.client_order_id, "submitted")
        details.append(
            OrderDetail(
                symbol=o.symbol,
                side=o.side,
                qty=o.qty,
                current_price=price,
                trade_value=o.qty * price,
                stop_loss_price=o.stop_loss_price,
                stop_loss_distance_pct=o.stop_loss_distance_pct,
                stop_loss_reason=o.stop_loss_reason,
                client_order_id=o.client_order_id,
                submission_status=status,
            )
        )
    return details


def build_prior_allocation_from_portfolio(
    portfolio: PortfolioResponse,
    source_label: str,
    as_of: str | None = None,
) -> PriorAllocation:
    """Convert a live broker snapshot into a "going into this week" view.

    US emails source the prior allocation from the live broker (so a
    failed order surfaces as a missing position). The conversion math
    -- ``market_value / total_value`` for each position, plus a
    ``CASH`` slot -- mirrors the experience-update path used for
    SAC's ``actual_weights`` so the two stay unit-consistent.

    Zero-NAV portfolios (cash=0, no positions) yield empty weights
    rather than the misleading ``{"CASH": 1.0}`` (which would render
    "100% cash" of $0). Empty weights cause the partial template's
    ``{% if prior_allocation.weights %}`` gate to hide the block,
    matching the cold-start semantics on the India side.
    """
    total_value = portfolio.cash + sum(p.market_value for p in portfolio.positions)
    if total_value <= 0:
        return PriorAllocation(
            weights={},
            source_label=source_label,
            as_of=as_of,
        )
    weights: dict[str, float] = {
        p.symbol: p.market_value / total_value for p in portfolio.positions
    }
    weights["CASH"] = portfolio.cash / total_value
    return PriorAllocation(
        weights=weights,
        source_label=source_label,
        as_of=as_of,
    )


def build_prior_allocation_from_db(
    final_weights_pct: dict[str, float],
    source_label: str,
    as_of: str | None = None,
) -> PriorAllocation:
    """Convert prior-week DB weights (in %) to the email partial's 0..1 form.

    Used by the India workflows: ``read_previous_final_set`` returns
    ``final_allocation_pct`` keyed by stock; the partial expects 0..1
    fractions so the delta-vs-target arithmetic stays unit-consistent
    with US (where the live-broker conversion is already 0..1).

    Empty input -> empty PriorAllocation (cold-start week renders no
    block; the partial's ``{% if prior_allocation.weights %}`` gate
    keeps the header from showing as a hollow widget).
    """
    weights = {sym: pct / 100.0 for sym, pct in final_weights_pct.items()}
    return PriorAllocation(
        weights=weights,
        source_label=source_label,
        as_of=as_of,
    )
