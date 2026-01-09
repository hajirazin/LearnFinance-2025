"""Shared utilities for training and inference routes."""

from datetime import date
from typing import TypeVar

T = TypeVar("T")


def get_as_of_date(as_of_str: str | None) -> date:
    """Parse as-of date string or return today.

    Args:
        as_of_str: Date string in YYYY-MM-DD format, or None

    Returns:
        Parsed date, or today's date if input is None
    """
    if as_of_str:
        return date.fromisoformat(as_of_str)
    return date.today()


def sort_predictions_by_return(
    predictions: list[T],
    key_attr: str = "predicted_weekly_return_pct",
) -> list[T]:
    """Sort predictions by predicted return descending, nulls last.

    Predictions with valid returns are sorted highest to lowest.
    Predictions with null returns (insufficient history) are placed at the end.

    Args:
        predictions: List of prediction objects
        key_attr: Attribute name for the return value (default: predicted_weekly_return_pct)

    Returns:
        Sorted list of predictions
    """
    valid = [p for p in predictions if getattr(p, key_attr) is not None]
    invalid = [p for p in predictions if getattr(p, key_attr) is None]

    valid_sorted = sorted(
        valid,
        key=lambda p: getattr(p, key_attr),
        reverse=True,
    )

    return valid_sorted + invalid
