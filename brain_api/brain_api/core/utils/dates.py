"""Date parsing utilities."""

from datetime import date


def parse_as_of_date(as_of_str: str | None) -> date:
    """Parse as-of date string or return today.

    Args:
        as_of_str: Date string in YYYY-MM-DD format, or None

    Returns:
        Parsed date, or today's date if input is None
    """
    if as_of_str:
        return date.fromisoformat(as_of_str)
    return date.today()


