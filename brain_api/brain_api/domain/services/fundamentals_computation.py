"""Fundamentals computation domain service.

Pure functions for parsing statements and computing financial ratios.
"""

from datetime import date
from decimal import Decimal
from typing import Any

from brain_api.domain.entities.fundamentals import (
    FundamentalRatios,
    QuarterlyStatement,
)


def parse_quarterly_statements(
    symbol: str,
    statement_type: str,
    raw_data: dict[str, Any],
) -> list[QuarterlyStatement]:
    """Parse raw API response into QuarterlyStatement objects.

    Args:
        symbol: Stock ticker
        statement_type: "income_statement" or "balance_sheet"
        raw_data: Raw JSON response from Alpha Vantage

    Returns:
        List of QuarterlyStatement objects
    """
    statements = []

    # Alpha Vantage returns "quarterlyReports" and "annualReports"
    quarterly_key = "quarterlyReports"
    reports = raw_data.get(quarterly_key, [])

    for report in reports:
        fiscal_date = report.get("fiscalDateEnding", "")
        currency = report.get("reportedCurrency", "USD")

        if not fiscal_date:
            continue

        statements.append(
            QuarterlyStatement(
                symbol=symbol,
                statement_type=statement_type,
                fiscal_date_ending=fiscal_date,
                reported_currency=currency,
                raw_data=report,
            )
        )

    return statements


def get_statement_as_of(
    statements: list[QuarterlyStatement],
    target_date: date,
) -> QuarterlyStatement | None:
    """Get the most recent statement as of a given date.

    Returns the statement with the latest fiscal_date_ending that is
    on or before target_date.

    Args:
        statements: List of statements to search
        target_date: Point-in-time date

    Returns:
        Most recent QuarterlyStatement, or None if none found
    """
    if not statements:
        return None

    target_str = target_date.isoformat()

    valid_statements = [
        s for s in statements if s.fiscal_date_ending <= target_str
    ]

    if not valid_statements:
        return None

    # Sort by date descending and return most recent
    valid_statements.sort(key=lambda s: s.fiscal_date_ending, reverse=True)
    return valid_statements[0]


def compute_ratios(
    symbol: str,
    income_statement: QuarterlyStatement | None,
    balance_sheet: QuarterlyStatement | None,
) -> FundamentalRatios:
    """Compute financial ratios from income and balance sheet statements.

    5 core ratios:
    - Profitability: gross_margin, operating_margin, net_margin
    - Liquidity: current_ratio
    - Leverage: debt_to_equity

    Args:
        symbol: Stock ticker
        income_statement: Most recent income statement
        balance_sheet: Most recent balance sheet

    Returns:
        FundamentalRatios with computed values
    """
    # Determine as_of_date (use income statement date or balance sheet date)
    as_of_date = ""
    if income_statement:
        as_of_date = income_statement.fiscal_date_ending
    elif balance_sheet:
        as_of_date = balance_sheet.fiscal_date_ending

    def safe_divide(numerator: Decimal | None, denominator: Decimal | None) -> float | None:
        """Safely divide two Decimals, handling None and zero."""
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            return None
        return float(numerator / denominator)

    # Profitability ratios (from income statement)
    gross_margin = None
    operating_margin = None
    net_margin = None

    if income_statement:
        revenue = income_statement.get_value("totalRevenue")
        gross_profit = income_statement.get_value("grossProfit")
        operating_income = income_statement.get_value("operatingIncome")
        net_income = income_statement.get_value("netIncome")

        gross_margin = safe_divide(gross_profit, revenue)
        operating_margin = safe_divide(operating_income, revenue)
        net_margin = safe_divide(net_income, revenue)

    # Liquidity and leverage ratios (from balance sheet)
    current_ratio = None
    debt_to_equity = None

    if balance_sheet:
        current_assets = balance_sheet.get_value("totalCurrentAssets")
        current_liabilities = balance_sheet.get_value("totalCurrentLiabilities")
        total_debt = balance_sheet.get_value("shortLongTermDebtTotal")
        equity = balance_sheet.get_value("totalShareholderEquity")

        current_ratio = safe_divide(current_assets, current_liabilities)
        debt_to_equity = safe_divide(total_debt, equity)

    return FundamentalRatios(
        symbol=symbol,
        as_of_date=as_of_date,
        gross_margin=round(gross_margin, 4) if gross_margin else None,
        operating_margin=round(operating_margin, 4) if operating_margin else None,
        net_margin=round(net_margin, 4) if net_margin else None,
        current_ratio=round(current_ratio, 4) if current_ratio else None,
        debt_to_equity=round(debt_to_equity, 4) if debt_to_equity else None,
    )

