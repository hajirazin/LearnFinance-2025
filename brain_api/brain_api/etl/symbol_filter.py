"""Universe filtering for stock symbols.

Simplified version that uses brain_api.universe directly.
"""

from datetime import UTC, datetime

from brain_api.universe import (
    HALAL_ETFS,
    get_halal_symbols,
)


class UniverseFilter:
    """Filter symbols to a specific universe.

    Supports:
    - Halal stocks (from ETF holdings via brain_api.universe)
    - Custom symbol list
    - No filtering (all symbols)
    """

    def __init__(self, symbols: set[str] | None = None):
        """Initialize with a set of allowed symbols.

        Args:
            symbols: Set of uppercase symbols to allow. None = allow all.
        """
        self._symbols = symbols
        self._fetched_at: str | None = None

    @classmethod
    def from_halal_universe(cls) -> "UniverseFilter":
        """Create filter from halal ETF holdings.

        Uses brain_api.universe to fetch holdings from SPUS, HLAL, SPTE ETFs,
        filtered to US-listed stocks only.

        Returns:
            UniverseFilter with halal symbols
        """
        symbols = get_halal_symbols()
        all_symbols = {s.upper() for s in symbols}

        instance = cls(all_symbols)
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
        """List of ETF tickers used for halal universe."""
        return list(HALAL_ETFS)


