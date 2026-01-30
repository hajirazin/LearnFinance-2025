"""Alpaca API client with multi-account support.

This module provides a client for interacting with Alpaca's paper trading API
with support for multiple accounts (PPO, SAC, HRP).

Each account has its own API credentials and is used for separate RL strategies.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum

import httpx

logger = logging.getLogger(__name__)

# Alpaca Paper Trading API base URL
ALPACA_PAPER_API_URL = "https://paper-api.alpaca.markets"


class AlpacaAccount(str, Enum):
    """Alpaca account identifiers."""

    PPO = "ppo"
    SAC = "sac"
    HRP = "hrp"


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


def get_account_credentials(account: AlpacaAccount) -> AlpacaCredentials | None:
    """Get Alpaca API credentials for an account from environment variables.

    Environment variables expected:
    - PPO: ALPACA_PPO_API_KEY, ALPACA_PPO_API_SECRET
    - SAC: ALPACA_SAC_API_KEY, ALPACA_SAC_API_SECRET
    - HRP: ALPACA_HRP_API_KEY, ALPACA_HRP_API_SECRET

    Returns:
        AlpacaCredentials if found, None otherwise.
    """
    account_upper = account.value.upper()
    api_key = os.environ.get(f"ALPACA_{account_upper}_API_KEY")
    api_secret = os.environ.get(f"ALPACA_{account_upper}_API_SECRET")

    if not api_key or not api_secret:
        logger.warning(
            f"[Alpaca] Missing credentials for {account_upper} account. "
            f"Set ALPACA_{account_upper}_API_KEY and ALPACA_{account_upper}_API_SECRET."
        )
        return None

    return AlpacaCredentials(api_key=api_key, api_secret=api_secret)


class AlpacaClient:
    """Client for Alpaca paper trading API with multi-account support."""

    def __init__(
        self,
        account: AlpacaAccount,
        credentials: AlpacaCredentials | None = None,
        timeout: float = 30.0,
    ):
        """Initialize Alpaca client for a specific account.

        Args:
            account: Which account to use (PPO, SAC, HRP).
            credentials: API credentials. If None, loaded from env vars.
            timeout: HTTP request timeout in seconds.
        """
        self.account = account
        self.credentials = credentials or get_account_credentials(account)
        self.timeout = timeout
        self.base_url = ALPACA_PAPER_API_URL

        if self.credentials is None:
            raise ValueError(
                f"No credentials available for account {account.value}. "
                f"Set ALPACA_{account.value.upper()}_API_KEY and "
                f"ALPACA_{account.value.upper()}_API_SECRET environment variables."
            )

    def _headers(self) -> dict[str, str]:
        """Get headers for Alpaca API requests."""
        return {
            "APCA-API-KEY-ID": self.credentials.api_key,
            "APCA-API-SECRET-KEY": self.credentials.api_secret,
            "Content-Type": "application/json",
        }

    def get_account(self) -> dict:
        """Get account information.

        Returns:
            Account info dict with cash, equity, buying_power, etc.
        """
        url = f"{self.base_url}/v2/account"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=self._headers())
            response.raise_for_status()
            return response.json()

    def get_positions(self) -> list[dict]:
        """Get all positions in the account.

        Returns:
            List of position dicts with symbol, qty, market_value, etc.
        """
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
        """Get orders from the account.

        Args:
            status: Order status filter ("open", "closed", "all").
            after: Only return orders after this timestamp (ISO format).
            limit: Maximum number of orders to return.

        Returns:
            List of order dicts.
        """
        url = f"{self.base_url}/v2/orders"
        params = {"status": status, "limit": limit}
        if after:
            params["after"] = after

        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=self._headers(), params=params)
            response.raise_for_status()
            return response.json()

    def get_portfolio(self) -> AlpacaPortfolio:
        """Get full portfolio state including cash and positions.

        Returns:
            AlpacaPortfolio with cash and positions.
        """
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
        """Get current portfolio weights.

        Returns:
            Dict of symbol -> weight, including CASH.
        """
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
    """Factory function to get an Alpaca client for a specific account.

    Args:
        account: Account name ("ppo", "sac", "hrp") or AlpacaAccount enum.

    Returns:
        AlpacaClient instance.
    """
    if isinstance(account, str):
        account = AlpacaAccount(account.lower())
    return AlpacaClient(account)


def get_ppo_client() -> AlpacaClient:
    """Get Alpaca client for PPO account."""
    return get_alpaca_client(AlpacaAccount.PPO)


def get_sac_client() -> AlpacaClient:
    """Get Alpaca client for SAC account."""
    return get_alpaca_client(AlpacaAccount.SAC)


def get_hrp_client() -> AlpacaClient:
    """Get Alpaca client for HRP account."""
    return get_alpaca_client(AlpacaAccount.HRP)
