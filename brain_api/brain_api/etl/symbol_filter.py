"""Universe filtering for stock symbols.

Resolves universe slates via the ETL universe registry; per-universe
ETF metadata (used by ``etfs_used``) is kept inline because it is only
consumed by the news-sentiment ETL output stats.
"""

from datetime import UTC, datetime

from brain_api.etl.universe_registry import get_etl_symbols
from brain_api.universe import HALAL_ETFS
from brain_api.universe.halal_new import ALL_ETFS

# Universes whose ETF source list is the ``halal_new`` 5-ETF set
# (``halal_new`` itself plus any sticky-derived US slate built on top
# of it). All other registered universes fall back to the legacy
# 3-ETF ``HALAL_ETFS`` for stats purposes.
_HALAL_NEW_DERIVED_UNIVERSES = frozenset({"halal_new", "halal_filtered"})


class UniverseFilter:
    """Filter symbols to a specific universe.

    Supports:
    - Halal stocks (from ETF holdings via brain_api.universe)
    - Custom symbol list
    - No filtering (all symbols)
    """

    def __init__(
        self,
        symbols: set[str] | None = None,
        universe: str | None = None,
    ):
        """Initialize with a set of allowed symbols.

        Args:
            symbols: Set of uppercase symbols to allow. None = allow all.
            universe: Registered universe name this filter was built
                from (used by ``etfs_used``).
        """
        self._symbols = symbols
        self._universe = universe
        self._fetched_at: str | None = None

    @classmethod
    def from_universe(cls, universe: str) -> "UniverseFilter":
        """Create filter from a registered universe string.

        Args:
            universe: One of the registered universe strings (see
                :mod:`brain_api.etl.universe_registry`).

        Returns:
            UniverseFilter populated with the universe's symbols.
        """
        symbols = get_etl_symbols(universe)
        all_symbols = {s.upper() for s in symbols}
        instance = cls(all_symbols, universe=universe)
        instance._fetched_at = datetime.now(UTC).isoformat()
        return instance

    @classmethod
    def from_symbol_list(cls, symbols: list[str]) -> "UniverseFilter":
        """Create filter from a custom symbol list.

        Args:
            symbols: List of symbols to allow

        Returns:
            UniverseFilter with custom symbols
        """
        return cls({s.upper() for s in symbols})

    @classmethod
    def allow_all(cls) -> "UniverseFilter":
        """Create filter that allows all symbols.

        Returns:
            UniverseFilter that passes everything
        """
        return cls(None)

    def is_allowed(self, symbol: str) -> bool:
        """Check if a symbol is in the universe.

        Args:
            symbol: Stock symbol to check

        Returns:
            True if symbol is allowed
        """
        if self._symbols is None:
            return True
        return symbol.upper() in self._symbols

    def filter_symbols(self, symbols: list[str]) -> list[str]:
        """Filter a list of symbols to allowed ones.

        Args:
            symbols: List of symbols to filter

        Returns:
            List of allowed symbols
        """
        if self._symbols is None:
            return symbols
        return [s for s in symbols if s.upper() in self._symbols]

    @property
    def symbol_count(self) -> int | None:
        """Number of symbols in the universe, or None if unrestricted."""
        return len(self._symbols) if self._symbols else None

    @property
    def symbols(self) -> set[str] | None:
        """The set of allowed symbols, or None if unrestricted."""
        return self._symbols

    @property
    def fetched_at(self) -> str | None:
        """When the universe was fetched, if applicable."""
        return self._fetched_at

    @property
    def etfs_used(self) -> list[str]:
        """List of ETF tickers backing this universe.

        - ``halal_new`` / ``halal_filtered`` -> 5-ETF halal_new source.
        - everything else (including unrecognised universes) -> the
          legacy 3-ETF ``HALAL_ETFS`` list, preserving prior behaviour.
        """
        if self._universe in _HALAL_NEW_DERIVED_UNIVERSES:
            return [s.upper() for s in ALL_ETFS]
        return list(HALAL_ETFS)
