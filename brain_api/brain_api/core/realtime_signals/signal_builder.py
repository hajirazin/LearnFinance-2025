"""Real-time signal builder for RL inference."""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import ClassVar

import pandas as pd

logger = logging.getLogger(__name__)


class RealTimeSignalBuilder:
    """Builds real-time signals for RL inference.

    Fetches:
    - News sentiment: yfinance news + FinBERT scoring
    - Momentum: yfinance daily closes (same source as SAC training)
    """

    # Signal keys must exactly match SAC_SIGNAL_NAMES (order matters --
    # this is a literal list, not a re-export, so the plan's static
    # source-contract check can verify it independently of the
    # decision_context module).
    SIGNAL_KEYS: ClassVar[list[str]] = [
        "news_sentiment",
        "news_coverage",
        "momentum_1w",
        "momentum_4w",
        "momentum_12_1",
    ]

    def __init__(self, base_path: Path | None = None):
        """Initialize signal builder.

        Args:
            base_path: Base path for data storage (for caching). Defaults to brain_api/data.
        """
        self.base_path = base_path or (
            Path(__file__).parent.parent.parent.parent / "data"
        )

    def build(
        self,
        symbols: list[str],
        as_of_date: date,
    ) -> dict[str, dict[str, float]]:
        """Build current signals for all symbols.

        Args:
            symbols: List of stock ticker symbols
            as_of_date: Reference date for fetching data

        Returns:
            Dict mapping symbol -> dict of signal values
        """
        logger.info(
            f"[SignalBuilder] Fetching real-time signals for {len(symbols)} symbols"
        )

        # News retains its existing log-and-continue behavior. Momentum is
        # fail-loud (no zero-fill) so incomplete prices cannot reach the actor.
        signals = self._init_empty_signals(symbols)

        # Fetch news sentiment
        self._fetch_news_sentiment(symbols, as_of_date, signals)

        self._fetch_momentum(symbols, as_of_date, signals)

        return signals

    def _init_empty_signals(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """Initialize empty signal dict with zeros for all symbols."""
        return {symbol: dict.fromkeys(self.SIGNAL_KEYS, 0.0) for symbol in symbols}

    def _fetch_news_sentiment(
        self,
        symbols: list[str],
        as_of_date: date,
        signals: dict[str, dict[str, float]],
    ) -> None:
        """Fetch news sentiment using yfinance + FinBERT."""
        try:
            from brain_api.core.finbert import FinBERTScorer
            from brain_api.core.news_sentiment import (
                YFinanceNewsFetcher,
                process_news_sentiment,
            )

            fetcher = YFinanceNewsFetcher()
            scorer = FinBERTScorer()

            news_result = process_news_sentiment(
                symbols=symbols,
                fetcher=fetcher,
                scorer=scorer,
                as_of_date=as_of_date,
                max_articles_per_symbol=10,
                run_id=f"rl_inference:{as_of_date.isoformat()}",
                attempt=1,
                base_path=self.base_path,
            )

            # Extract sentiment scores
            for symbol_sentiment in news_result.per_symbol:
                if symbol_sentiment.symbol in signals:
                    signals[symbol_sentiment.symbol]["news_sentiment"] = (
                        symbol_sentiment.sentiment_score
                    )

            logger.info(
                f"[SignalBuilder] News sentiment fetched for {len(news_result.per_symbol)} symbols"
            )
        except Exception as e:
            logger.warning(f"[SignalBuilder] Failed to fetch news sentiment: {e}")

    def _fetch_momentum(
        self,
        symbols: list[str],
        as_of_date: date,
        signals: dict[str, dict[str, float]],
    ) -> None:
        """Compute momentum_1w/4w/12_1 (fail-loud).

        Momentum uses the same yfinance daily closes as SAC training
        (``brain_api.core.prices.load_prices_yfinance``). No zero-fill on
        failure: callers see the exact symbol/reason.
        """
        from brain_api.core.prices import load_prices_yfinance
        from brain_api.core.sac.momentum_signals import (
            MOM_12_1_LOOKBACK_BARS,
            compute_momentum_1w,
            compute_momentum_4w,
            compute_momentum_12_1,
        )

        # 252-bar lookback (~12 months) needs a wide enough calendar
        # window; ~2x trading days in calendar days covers weekends/holidays.
        lookback_start = as_of_date - timedelta(
            days=int(MOM_12_1_LOOKBACK_BARS * 365 / 252) + 30
        )
        prices = load_prices_yfinance(
            symbols, lookback_start, as_of_date, log_prefix="[SignalBuilder]"
        )
        for symbol in symbols:
            price_df = prices.get(symbol)
            if price_df is None or price_df.empty:
                raise ValueError(
                    f"[SignalBuilder] Missing price history for {symbol}; "
                    "cannot compute momentum"
                )
            closes = price_df["close"].to_numpy(dtype=float)
            as_of_index = len(closes) - 1

            signals[symbol]["momentum_1w"] = compute_momentum_1w(
                closes, as_of_index=as_of_index
            )
            signals[symbol]["momentum_4w"] = compute_momentum_4w(
                closes, as_of_index=as_of_index
            )
            signals[symbol]["momentum_12_1"] = compute_momentum_12_1(
                closes, as_of_index=as_of_index
            )
        logger.info(f"[SignalBuilder] Momentum computed for {len(symbols)} symbols")

    # =========================================================================
    # DataFrame methods for PatchTST inference (time series format)
    # =========================================================================

    def build_news_dataframes(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, pd.DataFrame]:
        """Build news sentiment DataFrames for PatchTST inference.

        Fetches recent news from yfinance, scores with FinBERT, and returns
        DataFrames indexed by date with 'sentiment_score' column.

        Args:
            symbols: List of stock ticker symbols
            start_date: Start of data window
            end_date: End of data window (typically day before target week)

        Returns:
            Dict mapping symbol -> DataFrame with 'sentiment_score' column
        """
        from brain_api.core.finbert import FinBERTScorer
        from brain_api.core.news_sentiment import (
            YFinanceNewsFetcher,
            process_news_sentiment,
        )

        logger.info(
            f"[SignalBuilder] Fetching news DataFrames for {len(symbols)} symbols"
        )

        result: dict[str, pd.DataFrame] = {}

        try:
            fetcher = YFinanceNewsFetcher()
            scorer = FinBERTScorer()

            news_result = process_news_sentiment(
                symbols=symbols,
                fetcher=fetcher,
                scorer=scorer,
                as_of_date=end_date,
                max_articles_per_symbol=20,  # More articles for time series
                run_id=f"patchtst_inference:{end_date.isoformat()}",
                attempt=1,
                base_path=self.base_path,
            )

            # Convert to DataFrames
            for symbol_sentiment in news_result.per_symbol:
                symbol = symbol_sentiment.symbol
                # Create a single-row DataFrame at the as-of date
                # (yfinance returns recent news, so we use end_date as the reference)
                df = pd.DataFrame(
                    {"sentiment_score": [symbol_sentiment.sentiment_score]},
                    index=pd.DatetimeIndex([pd.Timestamp(end_date)]),
                )
                result[symbol] = df

            logger.info(
                f"[SignalBuilder] News DataFrames built for {len(result)} symbols"
            )
        except Exception as e:
            logger.warning(f"[SignalBuilder] Failed to fetch news DataFrames: {e}")

        return result
