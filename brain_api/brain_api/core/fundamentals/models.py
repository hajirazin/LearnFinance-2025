"""Data models for fundamentals module."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass
class QuarterlyStatement:
    """A single quarterly financial statement (income or balance sheet)."""

    symbol: str
    statement_type: str  # "income_statement" or "balance_sheet"
    fiscal_date_ending: str  # YYYY-MM-DD
    reported_currency: str
    raw_data: dict[str, Any]  # All fields from API

    def get_value(self, field: str) -> Decimal | None:
        """Get a numeric value from the statement, handling 'None' strings."""
        val = self.raw_data.get(field)
        if val is None or val == "None" or val == "":
            return None
        try:
            return Decimal(str(val))
        except Exception:
            return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "statement_type": self.statement_type,
            "fiscal_date_ending": self.fiscal_date_ending,
            "reported_currency": self.reported_currency,
            "raw_data": self.raw_data,
        }


@dataclass
class FundamentalRatios:
    """Computed financial ratios for a symbol at a point in time.

    5 core ratios for PPO:
    - Profitability: gross_margin, operating_margin, net_margin
    - Liquidity: current_ratio
    - Leverage: debt_to_equity
    """

    symbol: str
    as_of_date: str  # The fiscal_date_ending used

    # Profitability
    gross_margin: float | None  # grossProfit / totalRevenue
    operating_margin: float | None  # operatingIncome / totalRevenue
    net_margin: float | None  # netIncome / totalRevenue

    # Liquidity
    current_ratio: float | None  # totalCurrentAssets / totalCurrentLiabilities

    # Leverage
    debt_to_equity: float | None  # shortLongTermDebtTotal / totalShareholderEquity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "as_of_date": self.as_of_date,
            "gross_margin": self.gross_margin,
            "operating_margin": self.operating_margin,
            "net_margin": self.net_margin,
            "current_ratio": self.current_ratio,
            "debt_to_equity": self.debt_to_equity,
        }


@dataclass
class FetchRecord:
    """Record of a fetched statement file."""

    symbol: str
    endpoint: str  # "income_statement" or "balance_sheet"
    file_path: str
    fetched_at: str  # ISO timestamp
    latest_annual_date: str | None
    latest_quarterly_date: str | None


@dataclass
class FundamentalsResult:
    """Result of fundamentals fetch operation."""

    symbol: str
    income_statements: list[QuarterlyStatement]
    balance_sheets: list[QuarterlyStatement]
    from_cache: bool
    api_calls_made: int
    api_calls_remaining: int
