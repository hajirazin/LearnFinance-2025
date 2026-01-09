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
            statements.append(QuarterlyStatement(
                symbol=symbol,
                statement_type=endpoint,
                fiscal_date_ending=fiscal_date,
                reported_currency=currency,
                raw_data=report,
            ))

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
    for stmt in statements:
        if stmt.fiscal_date_ending <= as_of_date:
            return stmt
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
    operating_margin = None
    net_margin = None
    current_ratio = None
    debt_to_equity = None

    # Compute profitability ratios from income statement
    if income:
        total_revenue = income.get_value("totalRevenue")
        gross_profit = income.get_value("grossProfit")
        operating_income = income.get_value("operatingIncome")
        net_income = income.get_value("netIncome")

        if total_revenue and total_revenue > 0:
            if gross_profit is not None:
                gross_margin = float(gross_profit / total_revenue)
            if operating_income is not None:
                operating_margin = float(operating_income / total_revenue)
            if net_income is not None:
                net_margin = float(net_income / total_revenue)

    # Compute liquidity and leverage from balance sheet
    if balance:
        total_current_assets = balance.get_value("totalCurrentAssets")
        total_current_liabilities = balance.get_value("totalCurrentLiabilities")
        total_debt = balance.get_value("shortLongTermDebtTotal")
        shareholder_equity = balance.get_value("totalShareholderEquity")

        if total_current_assets and total_current_liabilities and total_current_liabilities > 0:
            current_ratio = float(total_current_assets / total_current_liabilities)

        if total_debt is not None and shareholder_equity and shareholder_equity > 0:
            debt_to_equity = float(total_debt / shareholder_equity)

    return FundamentalRatios(
        symbol=symbol,
        as_of_date=as_of_date,
        gross_margin=round(gross_margin, 4) if gross_margin is not None else None,
        operating_margin=round(operating_margin, 4) if operating_margin is not None else None,
        net_margin=round(net_margin, 4) if net_margin is not None else None,
        current_ratio=round(current_ratio, 4) if current_ratio is not None else None,
        debt_to_equity=round(debt_to_equity, 4) if debt_to_equity is not None else None,
    )


