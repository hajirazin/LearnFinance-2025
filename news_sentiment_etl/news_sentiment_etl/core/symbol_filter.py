"""Universe filtering for stock symbols."""

from datetime import UTC, datetime

import yfinance as yf

# Halal ETFs to source holdings from (same as brain_api)
HALAL_ETFS = ["SPUS", "HLAL", "SPTE"]


def _fetch_etf_holdings(ticker: str) -> list[str]:
    """Fetch symbols from a single ETF's holdings.

    Args:
        ticker: ETF ticker symbol

    Returns:
        List of stock symbols in the ETF
    """
    try:
        etf = yf.Ticker(ticker)
        if not hasattr(etf, "funds_data") or etf.funds_data is None:
            return []

        top_holdings = etf.funds_data.top_holdings
        if top_holdings is None or top_holdings.empty:
            return []

        return [str(symbol).upper() for symbol in top_holdings.index]
    except Exception:
        return []


class UniverseFilter:
    """Filter symbols to a specific universe.

    Supports:
    - Halal stocks (from ETF holdings)
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

        Fetches current holdings from SPUS, HLAL, SPTE ETFs.

        Returns:
            UniverseFilter with halal symbols
        """
        all_symbols: set[str] = set()

        for etf_ticker in HALAL_ETFS:
            holdings = _fetch_etf_holdings(etf_ticker)
            all_symbols.update(holdings)

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

