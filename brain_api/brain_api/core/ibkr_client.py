"""Interactive Brokers TWS API client (paper by default, live via gateway port).

This module wraps the ``ib_async`` library so the brain_api ``/ibkr/*``
routes can talk to a local IB Gateway daemon. It is the **only**
stateful piece in brain_api -- ib_async maintains a long-lived TCP
socket to the gateway, which is intentionally cached at process scope
because:

1. Reconnecting on every request adds 1-2 seconds of TLS + login
   overhead per call.
2. The gateway throttles aggressive reconnects.
3. ib_async's ``IB`` object is the canonical place for the "managed
   accounts" handshake and the "open orders" event subscription.

The brain_api "stateless endpoint" rule (AGENTS.md "API design rules"
#3) is preserved in spirit: this client holds **only** a connection
pool, not model state or per-request data.

Account routing
---------------
Each logical account (currently only ``sac_halal``) maps to an env
triple ``IBKR_{ACCOUNT}_HOST``, ``IBKR_{ACCOUNT}_PORT``,
``IBKR_{ACCOUNT}_CLIENT_ID``, plus ``IBKR_{ACCOUNT}_ACCOUNT_CODE``
(the ``DU…`` paper or ``U…`` live IBKR account ID set on
``Order.account``).

There is no ``IBKR_{ACCOUNT}_URL`` analogue to Alpaca's URL override:
paper vs live is selected at IB Gateway login (``TRADING_MODE`` env
in the gnzsnz image), and brain_api just connects to whichever
gateway port is configured. Stand up a second gateway container on
port 4001 with live credentials and flip ``IBKR_{ACCOUNT}_PORT`` to
go live for a single account.

Order field mapping (OrderToSubmit -> ib_async.Order)
-----------------------------------------------------
This is the only place that owns IBKR-specific field names; the route
handler stays in shape parity with the Alpaca route. Mapping table:

================  ===========================================
Our schema         IBKR schema
================  ===========================================
``side``           ``Order.action`` ("BUY" / "SELL")
``qty``            ``Order.totalQuantity``
``type=market``    ``Order.orderType="MKT"``
``type=limit``     ``Order.orderType="LMT"`` + ``lmtPrice``
``time_in_force``  ``Order.tif`` ("DAY", "GTC", ...)
``client_order_id``  ``Order.orderRef`` (free-text tag)
``IBKR_{ACCT}_ACCOUNT_CODE``  ``Order.account``
================  ===========================================

IBKR will NOT auto-reject duplicate ``orderRef`` values (unlike
Alpaca's per-account ``client_order_id`` dedup). The pre-submit dedup
guardrail is therefore the responsibility of brain_api itself; see
``ibkr.py`` for the open-trades + Postgres ledger lookup that fires
before this module ever calls ``ib.placeOrder``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Default IB Gateway connection params (paper).
DEFAULT_HOST = "localhost"
DEFAULT_PAPER_PORT = 4002
DEFAULT_LIVE_PORT = 4001  # documented but not selected here; flip via IBKR_{ACCT}_PORT


@dataclass(frozen=True)
class IBKRConnectionConfig:
    """Per-account IB Gateway connection parameters.

    Resolved from ``IBKR_{ACCOUNT}_HOST`` / ``_PORT`` / ``_CLIENT_ID`` /
    ``_ACCOUNT_CODE`` env vars by :func:`get_connection_config`. The
    config is hashable so it can key the connection pool.
    """

    account: str
    host: str
    port: int
    client_id: int
    account_code: str


def get_connection_config(account: str) -> IBKRConnectionConfig:
    """Resolve IB Gateway connection params for a logical account.

    Reads:

    - ``IBKR_{ACCOUNT}_HOST`` (default ``localhost``)
    - ``IBKR_{ACCOUNT}_PORT`` (default 4002 -- paper gateway)
    - ``IBKR_{ACCOUNT}_CLIENT_ID`` (mandatory; each account needs a
      unique integer so two simultaneous connections don't fight over
      the same ID at the gateway)
    - ``IBKR_{ACCOUNT}_ACCOUNT_CODE`` (mandatory; the ``DU…`` paper or
      ``U…`` live IBKR account ID; threaded onto every order's
      ``Order.account`` field)

    Raises:
        ValueError: if a mandatory env var is missing or malformed. Per
            AGENTS.md rule #1, callers must surface this rather than
            silently falling back to a default.
    """
    upper = account.upper()
    host = os.environ.get(f"IBKR_{upper}_HOST", "").strip() or DEFAULT_HOST

    port_raw = os.environ.get(f"IBKR_{upper}_PORT", "").strip()
    if not port_raw:
        port = DEFAULT_PAPER_PORT
    else:
        try:
            port = int(port_raw)
        except ValueError as e:
            raise ValueError(
                f"IBKR_{upper}_PORT must be an integer; got {port_raw!r}"
            ) from e

    client_id_raw = os.environ.get(f"IBKR_{upper}_CLIENT_ID", "").strip()
    if not client_id_raw:
        raise ValueError(
            f"IBKR_{upper}_CLIENT_ID is required (each account needs a unique int)"
        )
    try:
        client_id = int(client_id_raw)
    except ValueError as e:
        raise ValueError(
            f"IBKR_{upper}_CLIENT_ID must be an integer; got {client_id_raw!r}"
        ) from e

    account_code = os.environ.get(f"IBKR_{upper}_ACCOUNT_CODE", "").strip()
    if not account_code:
        raise ValueError(
            f"IBKR_{upper}_ACCOUNT_CODE is required (the DU.../U... IBKR account ID)"
        )

    return IBKRConnectionConfig(
        account=account,
        host=host,
        port=port,
        client_id=client_id,
        account_code=account_code,
    )


# ---------------------------------------------------------------------------
# Connection pool. One ib_async.IB instance per (host, port, client_id).
# ---------------------------------------------------------------------------


_CONNECTIONS: dict[tuple[str, int, int], object] = {}
_CONNECTION_LOCK = threading.Lock()

# Reconnect backoff (seconds). Exponential: 1, 2, 4, 8, 16, 32 -- caps below
# the gateway's typical disconnect-recovery window.
_BACKOFF_SCHEDULE = (1, 2, 4, 8, 16, 32)


def _import_ib_async():
    """Lazy-import ib_async so test environments without the gateway
    can mock this module without installing the dependency."""
    from ib_async import IB

    return IB


def get_ib_connection(config: IBKRConnectionConfig):
    """Return the cached ``ib_async.IB`` for ``config``, connecting if needed.

    Reconnect strategy: if the cached connection reports
    ``isConnected() == False`` we drop it and reconnect with bounded
    exponential backoff (1, 2, 4, 8, 16, 32s). Per AGENTS.md rule #1,
    a final failure raises ``ConnectionError`` rather than returning a
    dead handle.

    Thread-safety: a coarse module-level lock guards the dict. Per-call
    overhead is negligible because the slow path (TCP connect) only
    runs on first use or after a disconnect.
    """
    key = (config.host, config.port, config.client_id)
    with _CONNECTION_LOCK:
        ib = _CONNECTIONS.get(key)
        if ib is not None and ib.isConnected():
            return ib

        # Drop a dead handle before reconnecting; keep the slot so
        # concurrent callers wait on the same lock instead of racing
        # to open multiple sockets to the same client_id.
        if ib is not None:
            try:
                ib.disconnect()
            except Exception:
                logger.debug("Ignoring error during stale IB disconnect", exc_info=True)

        ib_class = _import_ib_async()
        ib = ib_class()

        last_err: Exception | None = None
        for delay in _BACKOFF_SCHEDULE:
            try:
                ib.connect(
                    config.host,
                    config.port,
                    clientId=config.client_id,
                    readonly=False,
                )
                logger.info(
                    f"[IBKR] Connected to gateway {config.host}:{config.port} "
                    f"clientId={config.client_id} for account={config.account}"
                )
                _CONNECTIONS[key] = ib
                return ib
            except Exception as e:
                last_err = e
                logger.warning(
                    f"[IBKR] Connect to {config.host}:{config.port} "
                    f"clientId={config.client_id} failed: {e}; retrying in {delay}s"
                )
                time.sleep(delay)

        raise ConnectionError(
            f"Could not connect to IB Gateway at {config.host}:{config.port} "
            f"clientId={config.client_id} after {len(_BACKOFF_SCHEDULE)} attempts: "
            f"{last_err}"
        )


def reset_connections() -> None:
    """Drop all cached gateway connections. Test-helper / shutdown hook."""
    with _CONNECTION_LOCK:
        for ib in _CONNECTIONS.values():
            try:
                ib.disconnect()
            except Exception:
                logger.debug("Ignoring error during reset_connections", exc_info=True)
        _CONNECTIONS.clear()


# ---------------------------------------------------------------------------
# Domain objects (broker-agnostic at the route boundary).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IBKRPosition:
    """A position in an IBKR account."""

    symbol: str
    qty: float
    market_value: float


@dataclass(frozen=True)
class IBKRPortfolio:
    """Portfolio state from an IBKR account.

    Field set deliberately matches the Alpaca ``PortfolioResponse``
    schema (``cash``, ``positions``, ``open_orders_count``) so the
    /ibkr/portfolio route can return the shared ``PortfolioResponse``
    Pydantic model without per-broker branching.
    """

    cash: float
    positions: list[IBKRPosition]
    open_orders_count: int


# ---------------------------------------------------------------------------
# Order mapping. Single source of truth for OrderToSubmit -> ib_async.Order.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IBKROrderSpec:
    """Broker-agnostic intent passed to :func:`submit_order`.

    Mirrors the route-level ``OrderToSubmit`` shape so the route handler
    can construct one of these without importing ib_async at all.
    """

    symbol: str
    qty: float
    side: str  # "buy" | "sell"
    order_type: str  # "market" | "limit"
    time_in_force: str  # "day" | "gtc" | ...
    limit_price: float | None
    client_order_id: str  # mapped to Order.orderRef


def _build_ib_objects(config: IBKRConnectionConfig, spec: IBKROrderSpec):
    """Build ``(Stock, Order)`` for ``spec`` using ib_async types.

    Lazy import so test environments without ib_async installed can
    monkeypatch this function.
    """
    from ib_async import Order, Stock

    contract = Stock(spec.symbol, "SMART", "USD")

    side_upper = spec.side.upper()
    if side_upper not in ("BUY", "SELL"):
        raise ValueError(f"Unsupported order side {spec.side!r}; expected buy or sell")

    order_type_upper = spec.order_type.upper()
    if order_type_upper == "MARKET":
        ib_order_type = "MKT"
    elif order_type_upper == "LIMIT":
        ib_order_type = "LMT"
    else:
        raise ValueError(
            f"Unsupported order type {spec.order_type!r}; expected market or limit"
        )

    if ib_order_type == "LMT" and spec.limit_price is None:
        raise ValueError(
            f"limit_price is required for limit orders (client_order_id={spec.client_order_id})"
        )

    order = Order()
    order.action = side_upper
    order.totalQuantity = spec.qty
    order.orderType = ib_order_type
    order.tif = spec.time_in_force.upper()
    order.orderRef = spec.client_order_id
    order.account = config.account_code
    if ib_order_type == "LMT":
        order.lmtPrice = spec.limit_price

    return contract, order


# ---------------------------------------------------------------------------
# Public API used by routes/ibkr.py.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IBKRSubmitResult:
    """Outcome of one ``submit_order`` call.

    ``perm_id`` is IBKR's broker-side permanent order id (returned in
    the ``Trade.order.permId`` field once the gateway has acknowledged
    the placement). ``status`` mirrors IBKR's lifecycle states
    (``Submitted``, ``Filled``, ``Cancelled``, ...). Either
    ``perm_id`` is set OR ``error`` is set, never both.
    """

    client_order_id: str
    perm_id: int | None
    status: str
    error: str | None = None


def get_portfolio(config: IBKRConnectionConfig) -> IBKRPortfolio:
    """Fetch cash + positions + open orders count for ``config.account``.

    Uses ``ib.accountSummary(account_code)`` for cash and
    ``ib.portfolio(account_code)`` for positions to get IBKR's
    pre-computed market values (avoids a separate market-data
    subscription that would otherwise be required to re-mark each
    position).

    The "open orders count" is the number of trades the gateway
    currently considers open for this account -- mirrors the Alpaca
    route's ``open_orders_count`` so the workflow's "skip if pending
    orders" check is broker-agnostic.
    """
    ib = get_ib_connection(config)

    # accountSummary returns multiple "tag" rows; we only want TotalCashValue
    # for the account_code we're trading on.
    cash = 0.0
    for row in ib.accountSummary(config.account_code):
        if getattr(row, "tag", "") == "TotalCashValue":
            try:
                cash = float(row.value)
            except (TypeError, ValueError):
                logger.warning(
                    f"[IBKR] Could not parse TotalCashValue={row.value!r} as float"
                )
            break

    positions: list[IBKRPosition] = []
    for item in ib.portfolio(config.account_code):
        contract = item.contract
        symbol = getattr(contract, "symbol", None)
        if not symbol:
            continue
        positions.append(
            IBKRPosition(
                symbol=symbol,
                qty=float(getattr(item, "position", 0.0)),
                market_value=float(getattr(item, "marketValue", 0.0)),
            )
        )

    open_trades = [
        t
        for t in ib.openTrades()
        if getattr(t.order, "account", "") == config.account_code
    ]

    return IBKRPortfolio(
        cash=cash,
        positions=positions,
        open_orders_count=len(open_trades),
    )


def submit_order(config: IBKRConnectionConfig, spec: IBKROrderSpec) -> IBKRSubmitResult:
    """Place a single order on the IB Gateway.

    Returns an :class:`IBKRSubmitResult` with the IBKR ``permId`` and
    initial status. Does NOT wait for fill -- the workflow's
    sell-wait-buy loop is responsible for polling lifecycle status via
    the order-history route.

    Pre-submit dedup is the route handler's responsibility (see
    ``routes/ibkr.py``) -- this function just translates the spec and
    calls ``ib.placeOrder``.
    """
    ib = get_ib_connection(config)
    contract, order = _build_ib_objects(config, spec)
    try:
        trade = ib.placeOrder(contract, order)
    except Exception as e:
        logger.error(
            f"[IBKR] placeOrder failed for {spec.client_order_id}: {e}", exc_info=True
        )
        return IBKRSubmitResult(
            client_order_id=spec.client_order_id,
            perm_id=None,
            status="error",
            error=str(e),
        )

    perm_id = getattr(trade.order, "permId", None) or None
    status = getattr(trade.orderStatus, "status", "Submitted")
    return IBKRSubmitResult(
        client_order_id=spec.client_order_id,
        perm_id=perm_id,
        status=status,
        error=None,
    )


def list_open_order_refs(config: IBKRConnectionConfig) -> set[str]:
    """Return the ``orderRef`` set currently considered open by IBKR.

    Used by the pre-submit dedup gate alongside the local Postgres
    ledger -- if a previous submit attempt already lives on the
    gateway's open-orders book we MUST NOT re-submit (IBKR will not
    reject the duplicate by orderRef on its own).
    """
    ib = get_ib_connection(config)
    refs: set[str] = set()
    for trade in ib.openTrades():
        order_ref = getattr(trade.order, "orderRef", "") or ""
        order_account = getattr(trade.order, "account", "")
        if order_ref and order_account == config.account_code:
            refs.add(order_ref)
    return refs
