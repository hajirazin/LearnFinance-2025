"""Historical ticker symbol aliases for normalization.

Moved from news_sentiment_etl/core/ticker_aliases.py.

Handles cases where companies changed their ticker symbols (e.g., FB → META).
This allows:
1. Query for both old and new tickers when filtering articles
2. Normalize extracted symbols to the current ticker
"""

# Historical ticker changes: old_ticker -> (current_ticker, effective_date)
# effective_date is when the new ticker started trading
TICKER_HISTORY: dict[str, tuple[str, str]] = {
    "FB": ("META", "2022-06-09"),  # Facebook → Meta Platforms rebrand
    # Add more as needed, e.g.:
    # "TWTR": ("X", "2023-07-24"),  # Twitter → X (if it were public)
}


def expand_with_aliases(symbols: set[str] | None) -> set[str] | None:
    """Expand a symbol set to include historical aliases.

    Given a set of current tickers, adds any historical predecessors
    so that SQL queries match both old and new ticker references.

    Example:
        expand_with_aliases({"META", "AAPL"}) → {"META", "FB", "AAPL"}

    Args:
        symbols: Set of current ticker symbols (uppercase), or None

    Returns:
        Expanded set including historical aliases, or None if input was None
    """
    if symbols is None:
        return None

    expanded = set(symbols)

    # Add predecessors for any current tickers in the set
    for old_ticker, (new_ticker, _effective_date) in TICKER_HISTORY.items():
        if new_ticker in symbols:
            expanded.add(old_ticker)

    return expanded


def normalize_symbol(symbol: str) -> str:
    """Map a historical ticker to its current ticker.

    Example:
        normalize_symbol("FB") → "META"
        normalize_symbol("AAPL") → "AAPL"

    Args:
        symbol: Ticker symbol (uppercase)

    Returns:
        Current ticker symbol
    """
    if symbol in TICKER_HISTORY:
        return TICKER_HISTORY[symbol][0]
    return symbol


def normalize_symbols(symbols: list[str]) -> list[str]:
    """Normalize a list of symbols to their current tickers.

    Maintains order and removes duplicates that arise from normalization.

    Example:
        normalize_symbols(["FB", "AAPL", "META"]) → ["META", "AAPL"]

    Args:
        symbols: List of ticker symbols (uppercase)

    Returns:
        List of current ticker symbols (deduplicated, order preserved)
    """
    normalized = []
    seen = set()

    for symbol in symbols:
        current = normalize_symbol(symbol)
        if current not in seen:
            seen.add(current)
            normalized.append(current)

    return normalized


