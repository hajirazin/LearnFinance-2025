"""Shared loader for historical fundamentals from cache.

This module provides the ONE function for loading historical fundamentals
from the local cache. All consumers should use this function:
- POST /signals/fundamentals/historical endpoint
- PatchTST training
- SAC training/finetune
- SAC training/finetune
"""

from datetime import date
from pathlib import Path

import pandas as pd

from brain_api.core.fundamentals.models import PointInTimeFundamental
from brain_api.core.fundamentals.parser import (
    compute_ratios,
    parse_quarterly_statements,
)
from brain_api.core.fundamentals.storage import load_raw_response


class FundamentalsCacheError(RuntimeError):
    """Raised when cached fundamentals exist but cannot be parsed safely."""


def get_default_data_path() -> Path:
    """Get the default data path for brain_api."""
    return Path(__file__).parent.parent.parent.parent / "data"


def load_historical_fundamentals_from_cache(
    symbols: list[str],
    start_date: date,
    end_date: date,
    base_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load historical fundamentals from cache.

    THE ONE function used by ALL:
    - POST /signals/fundamentals/historical endpoint
    - PatchTST training
    - SAC training/finetune
    - SAC training/finetune

    Fundamentals are quarterly data that should be forward-filled to daily.
    Reads from cached JSON files from Alpha Vantage.

    Args:
        symbols: List of ticker symbols
        start_date: Start of data window
        end_date: End of data window
        base_path: Base path for fundamentals cache (defaults to brain_api/data/)

    Returns:
        Dict mapping symbol -> DataFrame with fundamental ratio columns
        and DatetimeIndex (quarterly dates, to be forward-filled later)
    """
    if base_path is None:
        base_path = get_default_data_path()

    fundamentals: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            # Load cached responses (base_path is the data dir, load_raw_response adds raw/fundamentals/)
            income_data = load_raw_response(base_path, symbol, "income_statement")
            balance_data = load_raw_response(base_path, symbol, "balance_sheet")

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

            # Collect ratios for filings that were publicly available in-range.
            fiscal_dates: set[str] = set()

            for stmt in income_stmts:
                if stmt.filing_available_date is not None:
                    fiscal_dates.add(stmt.fiscal_date_ending)
            for stmt in balance_stmts:
                if stmt.filing_available_date is not None:
                    fiscal_dates.add(stmt.fiscal_date_ending)

            rows = []
            for fiscal_date in sorted(fiscal_dates):
                income_stmt = next(
                    (s for s in income_stmts if s.fiscal_date_ending == fiscal_date),
                    None,
                )
                balance_stmt = next(
                    (s for s in balance_stmts if s.fiscal_date_ending == fiscal_date),
                    None,
                )

                if (
                    income_stmt is None
                    or balance_stmt is None
                    or income_stmt.filing_available_date is None
                    or balance_stmt.filing_available_date is None
                ):
                    continue
                available_date = max(
                    income_stmt.filing_available_date,
                    balance_stmt.filing_available_date,
                )
                if not (start_date <= date.fromisoformat(available_date) <= end_date):
                    continue
                ratios = compute_ratios(income_stmt, balance_stmt)
                required_ratios = (
                    (
                        ratios.gross_margin,
                        ratios.debt_to_equity,
                    )
                    if ratios is not None
                    else ()
                )
                if ratios is not None and all(
                    value is not None for value in required_ratios
                ):
                    provenance_stmt = max(
                        (income_stmt, balance_stmt),
                        key=lambda stmt: stmt.filing_available_date or "",
                    )
                    rows.append(
                        {
                            "date": pd.to_datetime(available_date),
                            "fiscal_period_end": fiscal_date,
                            "filing_available_date": available_date,
                            "filing_accession_number": (
                                provenance_stmt.filing_accession_number
                            ),
                            "filing_form": provenance_stmt.filing_form,
                            "filing_source": provenance_stmt.filing_source,
                            "gross_margin": ratios.gross_margin,
                            "debt_to_equity": ratios.debt_to_equity,
                        }
                    )

            if rows:
                df = pd.DataFrame(rows).set_index("date").sort_index()
                fundamentals[symbol] = df

        except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
            raise FundamentalsCacheError(
                f"Malformed fundamentals cache for {symbol}: {exc}"
            ) from exc

    return fundamentals


def load_point_in_time_fundamentals(
    symbols: list[str],
    as_of_date: date,
    base_path: Path | None = None,
) -> dict[str, PointInTimeFundamental]:
    """Load each symbol's latest complete filing known by ``as_of_date``."""
    frames = load_historical_fundamentals_from_cache(
        symbols=symbols,
        start_date=date.min,
        end_date=as_of_date,
        base_path=base_path,
    )
    result: dict[str, PointInTimeFundamental] = {}
    for symbol, frame in frames.items():
        if frame.empty:
            continue
        row = frame.iloc[-1]
        result[symbol] = PointInTimeFundamental(
            symbol=symbol,
            fiscal_period_end=str(row["fiscal_period_end"]),
            filing_available_date=str(row["filing_available_date"]),
            filing_accession_number=str(row["filing_accession_number"]),
            filing_form=str(row["filing_form"]),
            filing_source=str(row["filing_source"]),
            gross_margin=float(row["gross_margin"]),
            debt_to_equity=float(row["debt_to_equity"]),
        )
    return result
