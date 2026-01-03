"""Halal stock universe from ETF holdings.

Re-exports from shared package for backwards compatibility.
"""

from shared.universe.halal import (
    HALAL_ETFS,
    get_halal_symbols,
    get_halal_universe,
)

__all__ = ["HALAL_ETFS", "get_halal_universe", "get_halal_symbols"]
