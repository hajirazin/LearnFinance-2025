"""Fundamentals signal endpoints."""

import os
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from brain_api.core.fundamentals import (
    FundamentalRatios,
    FundamentalsFetcher,
    compute_ratios,
    load_raw_response,
    parse_quarterly_statements,
)

router = APIRouter()


# ============================================================================
# Configuration constants
# ============================================================================

MAX_FUNDAMENTALS_SYMBOLS = 20


# ============================================================================
# Request / Response models
# ============================================================================


class FundamentalsRequest(BaseModel):
    """Request model for current fundamentals endpoint (inference via yfinance)."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_FUNDAMENTALS_SYMBOLS,
        description=f"List of ticker symbols (1-{MAX_FUNDAMENTALS_SYMBOLS})",
    )


class HistoricalFundamentalsRequest(BaseModel):
    """Request model for historical fundamentals endpoint (training via Alpha Vantage)."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_FUNDAMENTALS_SYMBOLS,
        description=f"List of ticker symbols (1-{MAX_FUNDAMENTALS_SYMBOLS})",
    )
    start_date: str = Field(
        ...,
        description="Start date for historical range (YYYY-MM-DD)",
    )
    end_date: str = Field(
        ...,
        description="End date for historical range (YYYY-MM-DD)",
    )
    force_refresh: bool = Field(
        False,
        description="If True, ignore cache and re-fetch from API (uses API quota)",
    )


class RatiosResponse(BaseModel):
    """Financial ratios for a symbol at a point in time.

    5 core ratios for PPO:
    - Profitability: gross_margin, operating_margin, net_margin
    - Liquidity: current_ratio
    - Leverage: debt_to_equity
    """

    symbol: str
    as_of_date: str
    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None
    current_ratio: float | None
    debt_to_equity: float | None


class CurrentRatiosResponse(BaseModel):
    """Per-symbol current fundamentals response (from yfinance)."""

    symbol: str
    ratios: RatiosResponse | None
    error: str | None = None


class ApiStatusResponse(BaseModel):
    """API usage status (for Alpha Vantage historical endpoint)."""

    calls_today: int
    daily_limit: int
    remaining: int


class FundamentalsResponse(BaseModel):
    """Response model for current fundamentals endpoint (yfinance)."""

    as_of_date: str
    per_symbol: list[CurrentRatiosResponse]


class HistoricalFundamentalsResponse(BaseModel):
    """Response model for historical fundamentals endpoint (n symbols × m dates)."""

    start_date: str
    end_date: str
    api_status: ApiStatusResponse
    data: list[RatiosResponse]  # Flat list: n symbols × m quarterly periods


# ============================================================================
# Dependency injection
# ============================================================================


def get_data_base_path() -> Path:
    """Get the base path for data storage."""
    return Path("data")


def get_alpha_vantage_api_key() -> str:
    """Get Alpha Vantage API key from environment."""
    return os.environ.get("ALPHA_VANTAGE_API_KEY", "")


def get_fundamentals_fetcher(
    api_key: Annotated[str, Depends(get_alpha_vantage_api_key)],
    base_path: Annotated[Path, Depends(get_data_base_path)],
) -> FundamentalsFetcher:
    """Get the fundamentals fetcher with injected dependencies."""
    return FundamentalsFetcher(
        api_key=api_key,
        base_path=base_path,
        daily_limit=25,  # Alpha Vantage free tier
    )


# ============================================================================
# Helper functions
# ============================================================================


def _get_yfinance_ratios(symbol: str, as_of_date: str) -> RatiosResponse | None:
    """Fetch current fundamental ratios from yfinance.

    yfinance ticker.info contains pre-computed ratios from the latest filings.
    No rate limits, no caching needed.
    """
    import yfinance as yf

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        # yfinance provides these as decimals (e.g., 0.45 for 45%)
        gross_margin = info.get("grossMargins")
        operating_margin = info.get("operatingMargins")
        net_margin = info.get("profitMargins")
        current_ratio = info.get("currentRatio")
        debt_to_equity = info.get("debtToEquity")

        # debtToEquity from yfinance is as percentage (e.g., 150 for 1.5x)
        # Normalize to ratio
        if debt_to_equity is not None and debt_to_equity > 10:
            debt_to_equity = debt_to_equity / 100

        return RatiosResponse(
            symbol=symbol,
            as_of_date=as_of_date,
            gross_margin=round(gross_margin, 4) if gross_margin else None,
            operating_margin=round(operating_margin, 4) if operating_margin else None,
            net_margin=round(net_margin, 4) if net_margin else None,
            current_ratio=round(current_ratio, 4) if current_ratio else None,
            debt_to_equity=round(debt_to_equity, 4) if debt_to_equity else None,
        )
    except Exception:
        return None


def _ratios_to_response(
    ratios: FundamentalRatios | None,
) -> RatiosResponse | None:
    """Convert internal FundamentalRatios to API response."""
    if ratios is None:
        return None
    return RatiosResponse(
        symbol=ratios.symbol,
        as_of_date=ratios.as_of_date,
        gross_margin=ratios.gross_margin,
        operating_margin=ratios.operating_margin,
        net_margin=ratios.net_margin,
        current_ratio=ratios.current_ratio,
        debt_to_equity=ratios.debt_to_equity,
    )


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/fundamentals", response_model=FundamentalsResponse)
def get_fundamentals(
    request: FundamentalsRequest,
) -> FundamentalsResponse:
    """Get CURRENT fundamental ratios for inference.

    This endpoint fetches the most recent available fundamentals for each symbol
    using yfinance. No rate limits, no caching needed.

    Data source: yfinance (ticker.info)
    """
    as_of = date.today().isoformat()
    results: list[CurrentRatiosResponse] = []

    for symbol in request.symbols:
        try:
            ratios = _get_yfinance_ratios(symbol, as_of)
            results.append(CurrentRatiosResponse(
                symbol=symbol,
                ratios=ratios,
                error=None if ratios else "No data available",
            ))
        except Exception as e:
            results.append(CurrentRatiosResponse(
                symbol=symbol,
                ratios=None,
                error=str(e),
            ))

    return FundamentalsResponse(
        as_of_date=as_of,
        per_symbol=results,
    )


@router.post("/fundamentals/historical", response_model=HistoricalFundamentalsResponse)
def get_historical_fundamentals(
    request: HistoricalFundamentalsRequest,
    fetcher: Annotated[FundamentalsFetcher, Depends(get_fundamentals_fetcher)],
) -> HistoricalFundamentalsResponse:
    """Get HISTORICAL fundamental ratios for training (date range).

    Returns n symbols × m quarterly periods as a flat list. Each entry represents
    the financial ratios that would have been available at that point in time
    (point-in-time correctness for avoiding look-ahead bias in training).

    Data source: Alpha Vantage (INCOME_STATEMENT + BALANCE_SHEET)

    Rate limits:
    - Free tier: 25 API calls/day
    - Each symbol requires 2 calls (income + balance) on first fetch
    - Cached data is reused until force_refresh is True
    """
    try:
        all_ratios: list[RatiosResponse] = []

        for symbol in request.symbols:
            try:
                # Fetch data (uses cache if available)
                fetcher.fetch_symbol(
                    symbol=symbol,
                    force_refresh=request.force_refresh,
                )

                income_data = load_raw_response(
                    fetcher.base_path, symbol, "income_statement"
                )
                balance_data = load_raw_response(
                    fetcher.base_path, symbol, "balance_sheet"
                )

                if income_data is None and balance_data is None:
                    continue

                # Parse statements
                income_stmts = []
                balance_stmts = []

                if income_data:
                    income_stmts = parse_quarterly_statements(
                        symbol, "income_statement", income_data
                    )
                if balance_data:
                    balance_stmts = parse_quarterly_statements(
                        symbol, "balance_sheet", balance_data
                    )

                # Get all unique fiscal dates in range
                fiscal_dates = set()
                for stmt in income_stmts:
                    if request.start_date <= stmt.fiscal_date_ending <= request.end_date:
                        fiscal_dates.add(stmt.fiscal_date_ending)
                for stmt in balance_stmts:
                    if request.start_date <= stmt.fiscal_date_ending <= request.end_date:
                        fiscal_dates.add(stmt.fiscal_date_ending)

                # Compute ratios for each fiscal date
                for fiscal_date in sorted(fiscal_dates):
                    income_stmt = next(
                        (s for s in income_stmts if s.fiscal_date_ending == fiscal_date),
                        None,
                    )
                    balance_stmt = next(
                        (s for s in balance_stmts if s.fiscal_date_ending == fiscal_date),
                        None,
                    )

                    ratios = compute_ratios(income_stmt, balance_stmt)
                    if ratios:
                        all_ratios.append(RatiosResponse(
                            symbol=ratios.symbol,
                            as_of_date=ratios.as_of_date,
                            gross_margin=ratios.gross_margin,
                            operating_margin=ratios.operating_margin,
                            net_margin=ratios.net_margin,
                            current_ratio=ratios.current_ratio,
                            debt_to_equity=ratios.debt_to_equity,
                        ))

            except Exception:
                # Skip symbols with errors, continue with others
                continue

        # Get API status
        api_status = fetcher.get_api_status()

        return HistoricalFundamentalsResponse(
            start_date=request.start_date,
            end_date=request.end_date,
            api_status=ApiStatusResponse(**api_status),
            data=all_ratios,
        )
    finally:
        fetcher.close()

