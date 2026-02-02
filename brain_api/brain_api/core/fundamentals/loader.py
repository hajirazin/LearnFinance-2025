"""Shared loader for historical fundamentals from cache.

This module provides the ONE function for loading historical fundamentals
from the local cache. All consumers should use this function:
- POST /signals/fundamentals/historical endpoint
- PatchTST training
- PPO training/finetune
- SAC training/finetune
"""

from datetime import date
from pathlib import Path

import pandas as pd

from brain_api.core.fundamentals.parser import (
    compute_ratios,
    parse_quarterly_statements,
)
from brain_api.core.fundamentals.storage import load_raw_response


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
    - PPO training/finetune
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

            # Collect ratios for each fiscal date within the date range
            fiscal_dates: set[str] = set()

            for stmt in income_stmts:
                if (
                    start_date
                    <= date.fromisoformat(stmt.fiscal_date_ending)
                    <= end_date
                ):
                    fiscal_dates.add(stmt.fiscal_date_ending)
            for stmt in balance_stmts:
                if (
                    start_date
                    <= date.fromisoformat(stmt.fiscal_date_ending)
                    <= end_date
                ):
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

                ratios = compute_ratios(income_stmt, balance_stmt)
                if ratios:
                    rows.append(
                        {
                            "date": pd.to_datetime(fiscal_date),
                            "gross_margin": ratios.gross_margin,
                            "operating_margin": ratios.operating_margin,
                            "net_margin": ratios.net_margin,
                            "current_ratio": ratios.current_ratio,
                            "debt_to_equity": ratios.debt_to_equity,
                        }
                    )

            if rows:
                df = pd.DataFrame(rows).set_index("date").sort_index()
                fundamentals[symbol] = df

        except Exception as e:
            print(f"[Fundamentals] Error loading fundamentals for {symbol}: {e}")
            continue

    return fundamentals
