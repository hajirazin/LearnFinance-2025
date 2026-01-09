"""Symbol/universe utilities."""

from typing import Any


def get_halal_symbols() -> list[str]:
    """Get symbols from halal universe.

    Returns:
        List of ticker symbols from the halal ETF universe
    """
    from brain_api.universe import get_halal_universe

    universe = get_halal_universe()
    return [stock["symbol"] for stock in universe["stocks"]]


def get_halal_universe_data() -> dict[str, Any]:
    """Get full halal universe data.

    Returns:
        Dict with 'stocks' list and metadata
    """
    from brain_api.universe import get_halal_universe

    return get_halal_universe()
