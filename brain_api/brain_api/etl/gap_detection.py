"""Gap detection for sentiment data.

Identifies missing date-symbol pairs by comparing expected combinations
against existing data in the output parquet file.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def generate_date_range(start_date: date, end_date: date) -> list[date]:
    """Generate all dates between start and end (inclusive).

    Args:
        start_date: Start date (earliest)
        end_date: End date (latest)

    Returns:
        List of dates from start to end
    """
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def read_existing_coverage(
    parquet_path: Path,
) -> set[tuple[date, str]]:
    """Read existing date-symbol pairs from parquet file.

    Args:
        parquet_path: Path to daily_sentiment.parquet

    Returns:
        Set of (date, symbol) tuples that have sentiment data
    """
    if not parquet_path.exists():
        return set()

    df = pd.read_parquet(parquet_path)
    if df.empty:
        return set()

    # Convert date column to date objects if needed
    if df["date"].dtype == "object":
        df["date"] = pd.to_datetime(df["date"]).dt.date
    elif hasattr(df["date"].dtype, "date"):
        df["date"] = df["date"].apply(lambda x: x if isinstance(x, date) else x.date())

    return set(zip(df["date"], df["symbol"]))


def find_gaps(
    symbols: list[str],
    start_date: date,
    end_date: date,
    parquet_path: Path,
) -> list[tuple[date, str]]:
    """Find missing date-symbol pairs in the parquet file.

    Args:
        symbols: List of symbols to check
        start_date: Earliest date to check
        end_date: Latest date to check
        parquet_path: Path to daily_sentiment.parquet

    Returns:
        List of (date, symbol) tuples that are missing, sorted in
        reverse chronological order (most recent first)
    """
    # Generate all expected (date, symbol) pairs
    all_dates = generate_date_range(start_date, end_date)
    all_pairs = {(d, symbol) for d in all_dates for symbol in symbols}

    # Read existing coverage from parquet file
    existing = read_existing_coverage(parquet_path)

    # Find gaps
    gaps = all_pairs - existing

    # Return sorted in reverse chronological order (today first)
    return sorted(gaps, key=lambda x: (x[0], x[1]), reverse=True)


def categorize_gaps(
    gaps: list[tuple[date, str]],
    api_earliest_date: date,
) -> tuple[list[tuple[date, str]], list[tuple[date, str]]]:
    """Categorize gaps into fillable and unfillable.

    Args:
        gaps: List of (date, symbol) gaps
        api_earliest_date: Earliest date the API has data for (e.g., 2015-01-01)

    Returns:
        Tuple of (fillable_gaps, unfillable_gaps)
        - fillable_gaps: Can be filled via API (date >= api_earliest_date)
        - unfillable_gaps: Cannot be filled (date < api_earliest_date)
    """
    fillable = []
    unfillable = []

    for gap in gaps:
        gap_date, _ = gap
        if gap_date >= api_earliest_date:
            fillable.append(gap)
        else:
            unfillable.append(gap)

    return fillable, unfillable


def get_gap_statistics(
    symbols: list[str],
    start_date: date,
    end_date: date,
    parquet_path: Path,
    api_earliest_date: date,
) -> dict:
    """Get comprehensive gap statistics.

    Args:
        symbols: List of symbols to check
        start_date: Earliest date to check
        end_date: Latest date to check
        parquet_path: Path to daily_sentiment.parquet
        api_earliest_date: Earliest date the API has data for

    Returns:
        Dictionary with gap statistics
    """
    all_dates = generate_date_range(start_date, end_date)
    total_pairs = len(all_dates) * len(symbols)

    existing = read_existing_coverage(parquet_path)
    existing_count = len(existing)

    gaps = find_gaps(symbols, start_date, end_date, parquet_path)
    fillable, unfillable = categorize_gaps(gaps, api_earliest_date)

    return {
        "total_date_symbol_pairs": total_pairs,
        "existing_in_parquet": existing_count,
        "gaps_found": len(gaps),
        "gaps_fillable": len(fillable),
        "gaps_pre_api_date": len(unfillable),
        "symbols_checked": len(symbols),
        "date_range_start": start_date.isoformat(),
        "date_range_end": end_date.isoformat(),
        "api_earliest_date": api_earliest_date.isoformat(),
    }

