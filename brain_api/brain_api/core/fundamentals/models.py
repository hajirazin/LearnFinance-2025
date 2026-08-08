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
    filing_available_date: str | None = None
    filing_accession_number: str | None = None
    filing_form: str | None = None
    filing_source: str | None = None

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
            "filing_available_date": self.filing_available_date,
            "filing_accession_number": self.filing_accession_number,
            "filing_form": self.filing_form,
            "filing_source": self.filing_source,
            "raw_data": self.raw_data,
        }


@dataclass
class FundamentalRatios:
    """Computed financial ratios for a symbol at a point in time.

    3 core ratios for RL allocators:
    - Profitability: gross_margin
    - Leverage: debt_to_equity
    """

    symbol: str
    as_of_date: str  # The fiscal_date_ending used

    # Profitability
    gross_margin: float | None  # grossProfit / totalRevenue

    # Leverage
    debt_to_equity: float | None  # shortLongTermDebtTotal / totalShareholderEquity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "as_of_date": self.as_of_date,
            "gross_margin": self.gross_margin,
            "debt_to_equity": self.debt_to_equity,
        }


@dataclass(frozen=True)
class PointInTimeFundamental:
    """Ratios from a filing that was publicly available by a decision date."""

    symbol: str
    fiscal_period_end: str
    filing_available_date: str
    filing_accession_number: str
    filing_form: str
    filing_source: str
    gross_margin: float
    debt_to_equity: float

    def to_dict(self) -> dict[str, Any]:
        """Return an audit-friendly JSON representation."""
        return {
            "symbol": self.symbol,
            "fiscal_period_end": self.fiscal_period_end,
            "filing_available_date": self.filing_available_date,
            "filing_accession_number": self.filing_accession_number,
            "filing_form": self.filing_form,
            "filing_source": self.filing_source,
            "gross_margin": self.gross_margin,
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
