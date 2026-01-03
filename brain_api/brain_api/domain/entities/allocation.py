"""Allocation-related domain entities."""

from dataclasses import dataclass


@dataclass
class HRPResult:
    """Result of HRP allocation computation."""

    # Percentage weights (sum to 100)
    percentage_weights: dict[str, float]

    # Symbols actually used in allocation
    symbols_used: list[str]

    # Symbols excluded (insufficient data)
    symbols_excluded: list[str]

    # Parameters used
    lookback_days: int
    as_of_date: str  # ISO format

