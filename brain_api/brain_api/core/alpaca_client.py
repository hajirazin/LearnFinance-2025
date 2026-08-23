"""Alpaca API client + shared account/URL/credential helpers.

This module is the single source of truth for:

* The :class:`AlpacaAccount` enum (``sac``, ``hrp``, ``dhrp``).
* :func:`get_alpaca_base_url` -- per-account URL resolution that honours
  the ``ALPACA_{ACCOUNT}_URL`` env override (paper by default, live opt-in
  per AGENTS.md "Trading mode").
* :func:`get_alpaca_credentials` -- FastAPI-free credential lookup that
  raises :class:`ValueError` on missing creds (per AGENTS.md rule #1, no
  silent ``None`` fallback).
* :class:`AlpacaClient` -- direct Alpaca HTTP client used by callers that
  need read-only portfolio data outside a FastAPI request (e.g. the
  experience labeller). Honours the per-account URL override via
  :func:`get_alpaca_base_url`, so a live-mode flip works for both the
  trading routes and the labeller.
* :func:`resolve_alpaca_account` -- maps ``(model_type, universe)`` to the
  right Alpaca account so the labeller routes per-record instead of
  hardcoding the legacy ``sac`` account.

The route module ``brain_api.routes.alpaca`` imports the enum + URL/cred
helpers from here and wraps :class:`ValueError` in
:class:`fastapi.HTTPException` at the route boundary -- core stays
FastAPI-free per AGENTS.md "API design rules" #3.

Note on the dropped ``SAC_HALAL`` entry: the ``sac_halal`` SAC variant
(legacy ``halal`` universe) used to live here as
``AlpacaAccount.SAC_HALAL`` but was migrated wholesale to IBKR; see
``brain_api.routes.ibkr`` and ``brain_api.core.ibkr_client``. The
``halal`` universe therefore has no Alpaca routing target -- a record
reaching the labeller without ``actual_weights`` will raise from
:func:`resolve_alpaca_account` rather than silently labelling against
the wrong account (AGENTS.md rule #1).
"""

import logging
import os
from dataclasses import dataclass
from enum import StrEnum

import httpx

logger = logging.getLogger(__name__)

# Default Alpaca host (used when ALPACA_{ACCOUNT}_URL env var is unset/blank).
PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class AlpacaAccount(StrEnum):
    """Supported Alpaca trading accounts (paper by default; live opt-in per-account).

    - ``sac``: SAC RL allocator (US, ``halal_filtered`` universe -- sticky-15
      from PatchTST).
    - ``hrp``: HRP baseline allocator (US, ``halal_new`` universe via the
      Alpha-HRP workflow).
    - ``dhrp``: Double HRP allocator (US, ``halal_new`` universe,
      sticky-selected).

    The ``halal`` SAC variant (formerly ``sac_halal``) trades through
    IBKR rather than Alpaca; see :mod:`brain_api.routes.ibkr`.
    """

    SAC = "sac"
    HRP = "hrp"
    DHRP = "dhrp"


# ---------------------------------------------------------------------------
# Per-account URL + credential helpers (FastAPI-free).
# ---------------------------------------------------------------------------


def get_alpaca_base_url(account: AlpacaAccount) -> str:
    """Resolve Alpaca base URL for ``account``.

    Reads ``ALPACA_{ACCOUNT}_URL``; returns the paper host when unset,
    empty, or whitespace. Setting it to ``https://api.alpaca.markets``
    (with matching live API key + secret) flips that single account to
    live without affecting the others.
    """
    raw = os.environ.get(f"ALPACA_{account.value.upper()}_URL", "")
    return raw.strip() or PAPER_BASE_URL


def get_alpaca_credentials(account: AlpacaAccount) -> tuple[str, str]:
    """Get ``(api_key, api_secret)`` for ``account`` from env vars.

    Environment variables expected:

    - ``ALPACA_SAC_KEY``,   ``ALPACA_SAC_SECRET``
    - ``ALPACA_HRP_KEY``,   ``ALPACA_HRP_SECRET``
    - ``ALPACA_DHRP_KEY``,  ``ALPACA_DHRP_SECRET``

    Raises:
        ValueError: if either env var is missing or empty. Per AGENTS.md
            rule #1, callers must surface this rather than silently
            falling back to a default account.
    """
    account_upper = account.value.upper()
    key_var = f"ALPACA_{account_upper}_KEY"
    secret_var = f"ALPACA_{account_upper}_SECRET"

    api_key = os.environ.get(key_var)
    api_secret = os.environ.get(secret_var)

    if not api_key or not api_secret:
        raise ValueError(
            f"Alpaca credentials not configured for account {account.value}. "
            f"Set {key_var} and {secret_var} environment variables."
        )

    return api_key, api_secret


# ---------------------------------------------------------------------------
# Direct Alpaca HTTP client (used by the experience labeller).
# ---------------------------------------------------------------------------


@dataclass
class AlpacaCredentials:
    """Alpaca API credentials for an account."""

    api_key: str
    api_secret: str


@dataclass
class AlpacaPosition:
    """A position in an Alpaca account."""

    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float


@dataclass
class AlpacaPortfolio:
    """Portfolio state from Alpaca account."""

    cash: float
    positions: list[AlpacaPosition]
    equity: float
    buying_power: float


class AlpacaClient:
    """Client for Alpaca trading API with multi-account support.

    Honours :func:`get_alpaca_base_url` so a per-account live override
    (``ALPACA_{ACCOUNT}_URL``) flips this client too -- not just the
    FastAPI routes. Currently only used by the experience labeller for
    read-only portfolio queries.
    """

    def __init__(
        self,
        account: AlpacaAccount,
        credentials: AlpacaCredentials | None = None,
        timeout: float = 30.0,
    ):
        """Initialize Alpaca client for a specific account.

        Args:
            account: Which account to use.
            credentials: Optional pre-resolved credentials. If ``None``,
                loaded from env vars via :func:`get_alpaca_credentials`.
            timeout: HTTP request timeout in seconds.

        Raises:
            ValueError: if ``credentials`` is ``None`` and env-var
                lookup fails.
        """
        self.account = account
        if credentials is None:
            api_key, api_secret = get_alpaca_credentials(account)
            credentials = AlpacaCredentials(api_key=api_key, api_secret=api_secret)
        self.credentials = credentials
        self.timeout = timeout
        self.base_url = get_alpaca_base_url(account)

    def _headers(self) -> dict[str, str]:
        """Get headers for Alpaca API requests."""
        return {
            "APCA-API-KEY-ID": self.credentials.api_key,
            "APCA-API-SECRET-KEY": self.credentials.api_secret,
            "Content-Type": "application/json",
        }

    def get_account(self) -> dict:
        """Get account information."""
        url = f"{self.base_url}/v2/account"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=self._headers())
            response.raise_for_status()
            return response.json()

    def get_positions(self) -> list[dict]:
        """Get all positions in the account."""
        url = f"{self.base_url}/v2/positions"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=self._headers())
            response.raise_for_status()
            return response.json()

    def get_orders(
        self,
        status: str = "all",
        after: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get orders from the account."""
        url = f"{self.base_url}/v2/orders"
        params: dict[str, str | int] = {"status": status, "limit": limit}
        if after:
            params["after"] = after

        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=self._headers(), params=params)
            response.raise_for_status()
            return response.json()

    def get_portfolio(self) -> AlpacaPortfolio:
        """Get full portfolio state including cash and positions."""
        account_info = self.get_account()
        positions_data = self.get_positions()

        positions = [
            AlpacaPosition(
                symbol=p["symbol"],
                qty=float(p["qty"]),
                market_value=float(p["market_value"]),
                avg_entry_price=float(p["avg_entry_price"]),
                current_price=float(p["current_price"]),
                unrealized_pl=float(p["unrealized_pl"]),
            )
            for p in positions_data
        ]

        return AlpacaPortfolio(
            cash=float(account_info["cash"]),
            positions=positions,
            equity=float(account_info["equity"]),
            buying_power=float(account_info["buying_power"]),
        )

    def get_portfolio_weights(self) -> dict[str, float]:
        """Get current portfolio weights including ``CASH``."""
        portfolio = self.get_portfolio()
        total_value = portfolio.equity

        if total_value <= 0:
            return {"CASH": 1.0}

        weights = {}
        for pos in portfolio.positions:
            weights[pos.symbol] = pos.market_value / total_value

        weights["CASH"] = portfolio.cash / total_value
        return weights


def get_alpaca_client(account: str | AlpacaAccount) -> AlpacaClient:
    """Factory function to get an Alpaca client for a specific account."""
    if isinstance(account, str):
        account = AlpacaAccount(account.lower())
    return AlpacaClient(account)


def get_sac_client() -> AlpacaClient:
    """Get Alpaca client for SAC account (``halal_filtered`` universe)."""
    return get_alpaca_client(AlpacaAccount.SAC)


def get_hrp_client() -> AlpacaClient:
    """Get Alpaca client for HRP account."""
    return get_alpaca_client(AlpacaAccount.HRP)


def get_dhrp_client() -> AlpacaClient:
    """Get Alpaca client for Double HRP account."""
    return get_alpaca_client(AlpacaAccount.DHRP)


# ---------------------------------------------------------------------------
# (model_type, universe) -> AlpacaAccount routing.
# ---------------------------------------------------------------------------


# Single source of truth for the SAC universe -> Alpaca account mapping.
# Currently only ``halal_filtered`` is Alpaca-routed; ``halal`` was
# migrated to IBKR (see ``brain_api.routes.ibkr``) and intentionally
# has NO entry here so a misrouted record fails loud.
_SAC_UNIVERSE_TO_ACCOUNT: dict[str, AlpacaAccount] = {
    "halal_filtered": AlpacaAccount.SAC,
}


def resolve_alpaca_account(model_type: str, universe: str) -> AlpacaAccount:
    """Resolve the Alpaca account for an ``(model_type, universe)`` pair.

    Currently only ``('sac', 'halal_filtered')`` has a mapping. The
    ``('sac', 'halal')`` pair has NO Alpaca account by design: that
    bucket trades through IBKR (see :mod:`brain_api.routes.ibkr`) so
    any halal experience record that reaches the labeller without
    ``actual_weights`` already plumbed in MUST surface as an error
    rather than silently labelling against an Alpaca account that
    never held the IBKR positions. Per AGENTS.md rule #1 the function
    raises on any unknown pair instead of falling back.

    Args:
        model_type: ``"sac"`` (currently the only routable model type;
            extend this when other RL allocators get an Alpaca-backed
            labelling story).
        universe: SAC bucket universe -- only ``"halal_filtered"`` is
            currently routable (``"halal"`` lives on IBKR).

    Raises:
        ValueError: if no Alpaca account is mapped for the given pair.
    """
    if model_type == "sac":
        try:
            return _SAC_UNIVERSE_TO_ACCOUNT[universe]
        except KeyError as e:
            valid = sorted(_SAC_UNIVERSE_TO_ACCOUNT)
            raise ValueError(
                f"No Alpaca account mapped for SAC universe {universe!r}. "
                f"Known SAC universes: {valid}."
            ) from e
    raise ValueError(
        f"No Alpaca account mapped for model_type {model_type!r}. "
        f"Only model_type='sac' is currently routable."
    )
