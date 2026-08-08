"""Statement parsing and ratio computation."""

from __future__ import annotations

from typing import Any

from brain_api.core.fundamentals.models import FundamentalRatios, QuarterlyStatement


def parse_quarterly_statements(
    symbol: str,
    endpoint: str,
    raw_response: dict[str, Any],
) -> list[QuarterlyStatement]:
    """Parse quarterly statements from raw API response.

    Args:
        symbol: Stock ticker
        endpoint: "income_statement" or "balance_sheet"
        raw_response: Raw API response dict

    Returns:
        List of QuarterlyStatement objects sorted by date (newest first)
    """
    api_data = raw_response.get("response", raw_response)
    quarterly_reports = api_data.get("quarterlyReports", [])

    statements = []
    for report in quarterly_reports:
        fiscal_date = report.get("fiscalDateEnding", "")
        currency = report.get("reportedCurrency", "USD")

        if fiscal_date:
            statements.append(
                QuarterlyStatement(
                    symbol=symbol,
                    statement_type=endpoint,
                    fiscal_date_ending=fiscal_date,
                    reported_currency=currency,
                    raw_data=report,
                    filing_available_date=report.get("filingDate"),
                    filing_accession_number=report.get("accessionNumber"),
                    filing_form=report.get("filingForm"),
                    filing_source=report.get("filingSource"),
                )
            )

    # Sort by date descending (newest first)
    statements.sort(key=lambda s: s.fiscal_date_ending, reverse=True)
    return statements


def get_statement_as_of(
    statements: list[QuarterlyStatement],
    as_of_date: str,
) -> QuarterlyStatement | None:
    """Get the most recent statement as of a given date.

    This ensures point-in-time correctness - we only use data that
    would have been available on the as_of_date.

    Args:
        statements: List of statements sorted by date descending
        as_of_date: YYYY-MM-DD date string

    Returns:
        Most recent statement with fiscal_date_ending <= as_of_date
    """
    available = [
        stmt
        for stmt in statements
        if stmt.filing_available_date is not None
        and stmt.filing_available_date <= as_of_date
    ]
    if available:
        return max(
            available,
            key=lambda stmt: (
                stmt.filing_available_date or "",
                stmt.fiscal_date_ending,
            ),
        )
    return None


def compute_ratios(
    income: QuarterlyStatement | None,
    balance: QuarterlyStatement | None,
) -> FundamentalRatios | None:
    """Compute financial ratios from income statement and balance sheet.

    Args:
        income: Quarterly income statement
        balance: Quarterly balance sheet (should be same fiscal period)

    Returns:
        FundamentalRatios or None if insufficient data
    """
    if income is None and balance is None:
        return None

    symbol = income.symbol if income else balance.symbol  # type: ignore
    as_of_date = income.fiscal_date_ending if income else balance.fiscal_date_ending  # type: ignore

    # Initialize all ratios as None
    gross_margin = None
    debt_to_equity = None
    eps_diluted = None

    # Compute profitability ratios from income statement
    if income:
        total_revenue = income.get_value("totalRevenue")
        gross_profit = income.get_value("grossProfit")

        if total_revenue and total_revenue > 0 and gross_profit is not None:
            gross_margin = float(gross_profit / total_revenue)

        # Raw per-share figure (not a ratio) for SAC's earnings_yield.
        # Optional -- a filing lacking EPS must not block gross_margin.
        eps_diluted_raw = income.get_value("epsDiluted")
        if eps_diluted_raw is not None:
            eps_diluted = float(eps_diluted_raw)

    # Compute leverage from balance sheet
    if balance:
        total_debt = balance.get_value("shortLongTermDebtTotal")
        shareholder_equity = balance.get_value("totalShareholderEquity")

        if total_debt is not None and shareholder_equity and shareholder_equity > 0:
            debt_to_equity = float(total_debt / shareholder_equity)

    return FundamentalRatios(
        symbol=symbol,
        as_of_date=as_of_date,
        gross_margin=round(gross_margin, 4) if gross_margin is not None else None,
        debt_to_equity=round(debt_to_equity, 4) if debt_to_equity is not None else None,
        eps_diluted=round(eps_diluted, 4) if eps_diluted is not None else None,
    )
