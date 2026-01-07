"""Helper functions for signal endpoints."""

from pathlib import Path

import pandas as pd

from brain_api.core.fundamentals import FundamentalRatios
from brain_api.core.news_sentiment import NewsSentimentResult, SymbolSentiment
from brain_api.routes.signals.models import (
    ArticleResponse,
    NewsSignalResponse,
    RatiosResponse,
    SentimentDataPoint,
    SymbolSentimentResponse,
)


def symbol_to_response(
    sentiment: SymbolSentiment,
    return_top_k: int,
) -> SymbolSentimentResponse:
    """Convert internal SymbolSentiment to API response format.

    Selects top K articles by score (most positive first).
    """
    # Sort articles by score descending
    sorted_articles = sorted(
        sentiment.scored_articles,
        key=lambda a: a.sentiment.score,
        reverse=True,
    )

    # Take top K
    top_articles = sorted_articles[:return_top_k]

    # Convert to response format
    article_responses = [
        ArticleResponse(
            title=sa.article.title,
            publisher=sa.article.publisher,
            link=sa.article.link,
            published=sa.article.published.isoformat() if sa.article.published else None,
            finbert_label=sa.sentiment.label,
            finbert_p_pos=sa.sentiment.p_pos,
            finbert_p_neg=sa.sentiment.p_neg,
            finbert_p_neu=sa.sentiment.p_neu,
            article_score=sa.sentiment.score,
        )
        for sa in top_articles
    ]

    return SymbolSentimentResponse(
        symbol=sentiment.symbol,
        article_count_fetched=sentiment.article_count_fetched,
        article_count_used=sentiment.article_count_used,
        sentiment_score=sentiment.sentiment_score,
        insufficient_news=sentiment.insufficient_news,
        top_k_articles=article_responses,
    )


def result_to_response(
    result: NewsSentimentResult,
    return_top_k: int,
) -> NewsSignalResponse:
    """Convert internal result to API response format."""
    return NewsSignalResponse(
        run_id=result.run_id,
        attempt=result.attempt,
        as_of_date=result.as_of_date,
        from_cache=result.from_cache,
        per_symbol=[
            symbol_to_response(s, return_top_k) for s in result.per_symbol
        ],
    )


def get_yfinance_ratios(symbol: str, as_of_date: str) -> RatiosResponse | None:
    """Fetch current fundamental ratios from yfinance.
    
    yfinance ticker.info contains pre-computed ratios from the latest filings.
    No rate limits, no caching needed.
    """
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        # yfinance provides these as decimals (e.g., 0.45 for 45%)
        gross_margin = info.get("grossMargins")
        operating_margin = info.get("operatingMargins")
        net_margin = info.get("profitMargins")
        current_ratio = info.get("currentRatio")
        debt_to_equity = info.get("debtToEquity")
        
        # debtToEquity from yfinance is as percentage (e.g., 150 for 1.5x)
        # Normalize to ratio
        if debt_to_equity is not None and debt_to_equity > 10:
            debt_to_equity = debt_to_equity / 100
        
        return RatiosResponse(
            symbol=symbol,
            as_of_date=as_of_date,
            gross_margin=round(gross_margin, 4) if gross_margin else None,
            operating_margin=round(operating_margin, 4) if operating_margin else None,
            net_margin=round(net_margin, 4) if net_margin else None,
            current_ratio=round(current_ratio, 4) if current_ratio else None,
            debt_to_equity=round(debt_to_equity, 4) if debt_to_equity else None,
        )
    except Exception:
        return None


def ratios_to_response(
    ratios: FundamentalRatios | None,
) -> RatiosResponse | None:
    """Convert internal FundamentalRatios to API response."""
    if ratios is None:
        return None
    return RatiosResponse(
        symbol=ratios.symbol,
        as_of_date=ratios.as_of_date,
        gross_margin=ratios.gross_margin,
        operating_margin=ratios.operating_margin,
        net_margin=ratios.net_margin,
        current_ratio=ratios.current_ratio,
        debt_to_equity=ratios.debt_to_equity,
    )


def load_historical_sentiment(
    parquet_path: Path,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> list[SentimentDataPoint]:
    """Load historical sentiment from parquet with neutral fallback for missing data.

    Args:
        parquet_path: Path to the daily_sentiment.parquet file
        symbols: List of symbols to fetch
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List of SentimentDataPoint for all (date, symbol) combinations in range.
        Missing combinations get neutral sentiment (score=0.0, other fields=None).
    """
    # Generate all date+symbol combinations requested
    all_dates = pd.date_range(start_date, end_date, freq="D")
    requested_df = pd.DataFrame(
        [(d.strftime("%Y-%m-%d"), s) for d in all_dates for s in symbols],
        columns=["date", "symbol"],
    )

    # Load parquet and filter
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        # Convert date to string for consistent comparison
        df["date"] = df["date"].astype(str)
        mask = (
            (df["date"] >= start_date)
            & (df["date"] <= end_date)
            & (df["symbol"].isin(symbols))
        )
        filtered_df = df[mask][
            ["date", "symbol", "sentiment_score", "article_count", "p_pos_avg", "p_neg_avg"]
        ].copy()
    else:
        filtered_df = pd.DataFrame(
            columns=["date", "symbol", "sentiment_score", "article_count", "p_pos_avg", "p_neg_avg"]
        )

    # Left join to include all requested combos
    result_df = requested_df.merge(filtered_df, how="left", on=["date", "symbol"])

    # Fill missing sentiment_score with neutral (0.0)
    if "sentiment_score" not in result_df.columns or result_df["sentiment_score"].isna().all():
        result_df["sentiment_score"] = 0.0
    else:
        result_df["sentiment_score"] = result_df["sentiment_score"].fillna(0.0).astype(float)

    # Convert to response objects
    data_points = []
    for _, row in result_df.iterrows():
        data_points.append(
            SentimentDataPoint(
                symbol=row["symbol"],
                date=row["date"],
                sentiment_score=float(row["sentiment_score"]),
                article_count=int(row["article_count"]) if pd.notna(row["article_count"]) else None,
                p_pos_avg=float(row["p_pos_avg"]) if pd.notna(row["p_pos_avg"]) else None,
                p_neg_avg=float(row["p_neg_avg"]) if pd.notna(row["p_neg_avg"]) else None,
            )
        )

    return data_points


