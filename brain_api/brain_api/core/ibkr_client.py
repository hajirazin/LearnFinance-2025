"""Interactive Brokers Client Portal REST API client.

This module provides a stateless httpx client for the IBKR Client Portal Web API
Gateway (managed by voyz/ibeam). The previous ib_async socket implementation
has been replaced with REST calls to the gateway running on localhost:5000.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5000


@dataclass(frozen=True)
class IBKRConnectionConfig:
    """Per-account IB Gateway connection parameters."""

    account: str
    host: str
    port: int
    client_id: int
    account_code: str
    base_url: str


def get_connection_config(account: str) -> IBKRConnectionConfig:
    upper = account.upper()
    host = os.environ.get(f"IBKR_{upper}_HOST", "").strip() or DEFAULT_HOST

    port_raw = os.environ.get(f"IBKR_{upper}_PORT", "").strip()
    if not port_raw:
        port = DEFAULT_PORT
    else:
        try:
            port = int(port_raw)
        except ValueError as e:
            raise ValueError(
                f"IBKR_{upper}_PORT must be an integer; got {port_raw!r}"
            ) from e

    client_id_raw = os.environ.get(f"IBKR_{upper}_CLIENT_ID", "").strip()
    if not client_id_raw:
        raise ValueError(f"IBKR_{upper}_CLIENT_ID is required")
    try:
        client_id = int(client_id_raw)
    except ValueError as e:
        raise ValueError(
            f"IBKR_{upper}_CLIENT_ID must be an integer; got {client_id_raw!r}"
        ) from e

    account_code = os.environ.get(f"IBKR_{upper}_ACCOUNT_CODE", "").strip()
    if not account_code:
        raise ValueError(f"IBKR_{upper}_ACCOUNT_CODE is required")

    base_url = f"https://{host}:{port}/v1/api"

    return IBKRConnectionConfig(
        account=account,
        host=host,
        port=port,
        client_id=client_id,
        account_code=account_code,
        base_url=base_url,
    )


def _get_client() -> httpx.Client:
    """Return a configured httpx Client ignoring SSL errors (self-signed)."""
    return httpx.Client(verify=False, timeout=30.0)


def get_session_status(config: IBKRConnectionConfig) -> bool:
    """Check auth status and tickle session."""
    with _get_client() as client:
        try:
            resp = client.post(f"{config.base_url}/iserver/auth/status")
            resp.raise_for_status()
            data = resp.json()
            authenticated = data.get("authenticated", False)

            # Send a tickle request
            client.post(f"{config.base_url}/tickle")

            return authenticated
        except Exception as e:
            logger.error(f"[IBKR] Failed to get session status: {e}")
            return False


@dataclass(frozen=True)
class IBKRPosition:
    symbol: str
    qty: float
    market_value: float


@dataclass(frozen=True)
class IBKRPortfolio:
    cash: float
    currency: str
    cash_balances: dict[str, float]
    positions: list[IBKRPosition]
    open_orders_count: int


def get_portfolio(
    config: IBKRConnectionConfig, target_currency: str = "USD"
) -> IBKRPortfolio:
    """Fetch cash, positions, and open orders count."""
    with _get_client() as client:
        # 1. Initialize context
        try:
            client.get(f"{config.base_url}/iserver/accounts").raise_for_status()
            client.get(f"{config.base_url}/portfolio/accounts").raise_for_status()
        except Exception as e:
            logger.error(f"[IBKR] Failed to initialize account context: {e}")
            raise ConnectionError(f"Gateway unreachable: {e}") from e

        # 2. Get Ledger for cash
        cash_balances = {}
        cash = 0.0
        currency = target_currency
        try:
            resp = client.get(
                f"{config.base_url}/portfolio/{config.account_code}/ledger"
            )
            resp.raise_for_status()
            ledger_data = resp.json()

            # Fetch target currency balance if available, otherwise fallback to BASE
            if target_currency in ledger_data:
                cash = float(ledger_data[target_currency].get("cashbalance", 0.0))
                currency = target_currency
            elif "BASE" in ledger_data:
                cash = float(ledger_data["BASE"].get("cashbalance", 0.0))
                currency = ledger_data["BASE"].get("currency", "BASE")

            # Populate cash balances dict
            for curr, data in ledger_data.items():
                if isinstance(data, dict) and "cashbalance" in data:
                    cash_balances[curr] = float(data.get("cashbalance", 0.0))
        except Exception as e:
            logger.error(f"[IBKR] Failed to fetch ledger: {e}")

        # 3. Get Positions
        positions = []
        try:
            # page 0
            resp = client.get(
                f"{config.base_url}/portfolio/{config.account_code}/positions/0"
            )
            resp.raise_for_status()
            pos_data = resp.json()
            for item in pos_data:
                symbol = item.get("contractDesc")
                qty = item.get("position", 0.0)
                mkt_val = item.get("mktValue", 0.0)
                if symbol:
                    positions.append(
                        IBKRPosition(
                            symbol=symbol, qty=float(qty), market_value=float(mkt_val)
                        )
                    )
        except Exception as e:
            logger.error(f"[IBKR] Failed to fetch positions: {e}")

        # 4. Get Open Orders Count
        open_count = 0
        try:
            # Need to call twice for snapshot initialization
            client.get(f"{config.base_url}/iserver/account/orders")
            resp = client.get(f"{config.base_url}/iserver/account/orders")
            resp.raise_for_status()
            orders_data = resp.json()
            if "orders" in orders_data:
                # filter by account and open status
                open_trades = [
                    o
                    for o in orders_data["orders"]
                    if o.get("acct") == config.account_code
                    and o.get("status")
                    in ("PreSubmitted", "Submitted", "PendingSubmit", "PendingCancel")
                ]
                open_count = len(open_trades)
        except Exception as e:
            logger.error(f"[IBKR] Failed to fetch open orders: {e}")

        return IBKRPortfolio(
            cash=cash,
            currency=currency,
            cash_balances=cash_balances,
            positions=positions,
            open_orders_count=open_count,
        )


@dataclass(frozen=True)
class IBKROrderSpec:
    symbol: str
    qty: float
    side: str
    order_type: str
    time_in_force: str
    limit_price: float | None
    client_order_id: str
    currency: str
    cash_qty: float | None = None


@dataclass(frozen=True)
class IBKRSubmitResult:
    client_order_id: str
    perm_id: int | None
    status: str
    error: str | None = None


def submit_order(config: IBKRConnectionConfig, spec: IBKROrderSpec) -> IBKRSubmitResult:
    with _get_client() as client:
        try:
            # 1. Resolve conid
            search_payload = {"symbol": spec.symbol, "secType": "STK", "name": False}
            search_resp = client.post(
                f"{config.base_url}/iserver/secdef/search", json=search_payload
            )
            search_resp.raise_for_status()
            search_data = search_resp.json()
            if not search_data:
                return IBKRSubmitResult(
                    spec.client_order_id,
                    None,
                    "error",
                    f"Could not find conid for {spec.symbol}",
                )
            conid = search_data[0].get("conid")

            # 2. Prepare Order
            order_payload = {
                "conid": int(conid),
                "orderType": "LMT" if spec.order_type.upper() == "LIMIT" else "MKT",
                "side": spec.side.upper(),
                "tif": spec.time_in_force.upper(),
                "cOID": spec.client_order_id,
            }
            is_fractional = not float(spec.qty).is_integer()
            if is_fractional and spec.cash_qty is not None:
                order_payload["cashQty"] = spec.cash_qty
            else:
                order_payload["quantity"] = (
                    int(spec.qty) if float(spec.qty).is_integer() else spec.qty
                )

            if spec.limit_price is not None:
                order_payload["price"] = spec.limit_price

            # 3. Submit with Reply Loop
            logger.info(f"[IBKR] order_payload before submit: {order_payload}")
            resp = client.post(
                f"{config.base_url}/iserver/account/{config.account_code}/orders",
                json={"orders": [order_payload]},
            )
            resp.raise_for_status()
            data = resp.json()

            # Loop for precautions
            while isinstance(data, list) and len(data) > 0 and "id" in data[0]:
                reply_id = data[0]["id"]
                logger.info(
                    f"[IBKR] Confirming precaution for {spec.client_order_id}: {reply_id}"
                )
                resp = client.post(
                    f"{config.base_url}/iserver/reply/{reply_id}",
                    json={"confirmed": True},
                )
                resp.raise_for_status()
                data = resp.json()

            if isinstance(data, list) and len(data) > 0 and "order_id" in data[0]:
                # Array response on success after reply
                perm_id = int(data[0].get("order_id", 0))
                status = data[0].get("order_status", "Submitted")
                return IBKRSubmitResult(spec.client_order_id, perm_id, status)

            return IBKRSubmitResult(
                spec.client_order_id, None, "error", f"Unexpected response: {data}"
            )

        except Exception as e:
            logger.error(
                f"[IBKR] Error submitting order {spec.client_order_id}: {e}",
                exc_info=True,
            )
            return IBKRSubmitResult(spec.client_order_id, None, "error", str(e))


@dataclass(frozen=True)
class IBKROrderStatus:
    status: str
    filled_qty: float
    filled_avg_price: float


def get_order_status(
    config: IBKRConnectionConfig, perm_id: int
) -> IBKROrderStatus | None:
    """Fetch live order status from the gateway."""
    with _get_client() as client:
        try:
            resp = client.get(
                f"{config.base_url}/iserver/account/order/status/{perm_id}"
            )
            resp.raise_for_status()
            data = resp.json()

            status = data.get("order_status", "Unknown")
            filled_qty = float(data.get("cum_fill", 0.0))
            # Average price can be under 'avg_price' or 'price' depending on endpoint behavior,
            # but usually 'avg_price' for fills.
            filled_avg_price = float(data.get("avg_price", 0.0))

            return IBKROrderStatus(status, filled_qty, filled_avg_price)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"[IBKR] Failed to get status for {perm_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"[IBKR] Failed to get status for {perm_id}: {e}")
            raise


def list_open_order_refs(config: IBKRConnectionConfig) -> set[str]:
    with _get_client() as client:
        try:
            # Need to call twice for snapshot
            client.get(f"{config.base_url}/iserver/account/orders")
            resp = client.get(f"{config.base_url}/iserver/account/orders")
            resp.raise_for_status()
            orders_data = resp.json()
            refs = set()
            if "orders" in orders_data:
                for o in orders_data["orders"]:
                    status = o.get("status")
                    if status not in ("Filled", "Cancelled", "Inactive"):
                        coid = o.get("cOID")
                        if coid:
                            refs.add(coid)
            return refs
        except Exception as e:
            logger.error(f"[IBKR] Failed to fetch open order refs: {e}")
            return set()
