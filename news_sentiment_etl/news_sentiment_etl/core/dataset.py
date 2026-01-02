"""HuggingFace dataset streaming loader with symbol extraction."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

from datasets import load_dataset

from news_sentiment_etl.core.config import ETLConfig


@dataclass
class NewsArticle:
    """A single news article from the dataset.

    Attributes:
        date: Publication date (trading date from extra_fields if available)
        text: Article text (title + body)
        symbols: List of stock symbols mentioned
        dataset_source: Which subset this came from
        raw_date: Original date string from dataset
    """

    date: str  # YYYY-MM-DD format
    text: str
    symbols: list[str]
    dataset_source: str
    raw_date: str


def _parse_extra_fields(extra_fields_str: str | None) -> dict:
    """Parse the extra_fields JSON string.

    Args:
        extra_fields_str: JSON string from dataset

    Returns:
        Parsed dictionary or empty dict on error
    """
    if not extra_fields_str:
        return {}
    try:
        return json.loads(extra_fields_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def _extract_symbols(extra_fields: dict) -> list[str]:
    """Extract stock symbols from extra_fields.

    The dataset stores symbols in various fields:
    - stocks: list of symbols
    - tickers: alternative field name
    - symbol: single symbol

    Args:
        extra_fields: Parsed extra_fields dict

    Returns:
        List of uppercase stock symbols
    """
    symbols = []

    # Try different field names
    if "stocks" in extra_fields:
        stocks = extra_fields["stocks"]
        if isinstance(stocks, list):
            symbols.extend(stocks)
        elif isinstance(stocks, str):
            symbols.append(stocks)

    if "tickers" in extra_fields:
        tickers = extra_fields["tickers"]
        if isinstance(tickers, list):
            symbols.extend(tickers)
        elif isinstance(tickers, str):
            symbols.append(tickers)

    if "symbol" in extra_fields:
        symbol = extra_fields["symbol"]
        if isinstance(symbol, str):
            symbols.append(symbol)

    # Normalize: uppercase, remove duplicates, filter empty
    normalized = []
    seen = set()
    for s in symbols:
        if s and isinstance(s, str):
            upper = s.upper().strip()
            if upper and upper not in seen:
                seen.add(upper)
                normalized.append(upper)

    return normalized


def _extract_date(row: dict, extra_fields: dict) -> str:
    """Extract the trading date from a row.

    Prefers date_trading from extra_fields (backtest-safe),
    falls back to the top-level date field.

    Args:
        row: Dataset row
        extra_fields: Parsed extra_fields dict

    Returns:
        Date string in YYYY-MM-DD format
    """
    # Prefer trading date for backtest safety
    trading_date = extra_fields.get("date_trading")
    if trading_date:
        # Format: "2011-04-20T13:30:00Z" -> "2011-04-20"
        try:
            dt = datetime.fromisoformat(trading_date.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            pass

    # Fall back to top-level date
    date_str = row.get("date", "")
    if date_str:
        try:
            # Format: "2011-04-19T00:00:00Z" -> "2011-04-19"
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            pass

    return ""


def stream_articles(config: ETLConfig) -> Iterator[NewsArticle]:
    """Stream articles from the HuggingFace dataset.

    Filters to only articles that have stock symbols.

    Args:
        config: ETL configuration

    Yields:
        NewsArticle objects with symbols
    """
    # Load dataset in streaming mode
    dataset = load_dataset(
        config.dataset_name,
        data_files=config.data_files,
        split="train",
        streaming=True,
        token=config.hf_token,
    )

    count = 0
    for row in dataset:
        # Parse extra_fields
        extra_fields = _parse_extra_fields(row.get("extra_fields"))

        # Extract symbols - skip if none found
        symbols = _extract_symbols(extra_fields)
        if not symbols:
            continue

        # Extract date
        date = _extract_date(row, extra_fields)
        if not date:
            continue

        # Extract text
        text = row.get("text", "")
        if not text or len(text) < 10:  # Skip very short texts
            continue

        # Get dataset source
        dataset_source = extra_fields.get("dataset", "unknown")

        yield NewsArticle(
            date=date,
            text=text,
            symbols=symbols,
            dataset_source=dataset_source,
            raw_date=row.get("date", ""),
        )

        count += 1
        if config.max_articles and count >= config.max_articles:
            break


def batch_articles(
    articles: Iterator[NewsArticle], batch_size: int
) -> Iterator[list[NewsArticle]]:
    """Batch articles for efficient processing.

    Args:
        articles: Iterator of NewsArticle
        batch_size: Number of articles per batch

    Yields:
        Lists of NewsArticle
    """
    batch: list[NewsArticle] = []
    for article in articles:
        batch.append(article)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    # Yield remaining
    if batch:
        yield batch

