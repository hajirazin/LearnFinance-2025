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
    - Fundamentals: yfinance ticker.info (gross_margin, debt_to_equity)
    - Momentum: yfinance daily closes (same source as SAC training)
    - Earnings yield: SEC point-in-time diluted EPS (never yfinance-only
      EPS -- see ``brain_api.core.sac.momentum_signals``)
    """

    # Signal keys must exactly match SAC_SIGNAL_NAMES (order matters --
    # this is a literal list, not a re-export, so the plan's static
    # source-contract check can verify it independently of the
    # decision_context module).
    SIGNAL_KEYS: ClassVar[list[str]] = [
        "news_sentiment",
        "news_coverage",
        "gross_margin",
        "debt_to_equity",
        "fundamental_age",
        "momentum_1w",
        "momentum_4w",
        "momentum_12_1",
        "earnings_yield",
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

        # Initialize with zeros (fallback if fetching fails). This applies
        # only to the pre-existing news/fundamentals fetchers below, which
        # already log-and-continue on failure. Momentum + earnings_yield
        # are fail-loud (no zero-fill) -- see `_fetch_momentum_and_earnings_yield`.
        signals = self._init_empty_signals(symbols)

        # Fetch news sentiment
        self._fetch_news_sentiment(symbols, as_of_date, signals)

        # Fetch fundamentals
        self._fetch_fundamentals(symbols, as_of_date, signals)

        # Fetch momentum + earnings yield (fail-loud; raises on
        # insufficient bars or missing/non-finite EPS per AGENTS.md rule 1)
        self._fetch_momentum_and_earnings_yield(symbols, as_of_date, signals)

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

    def _fetch_fundamentals(
        self,
        symbols: list[str],
        as_of_date: date,
        signals: dict[str, dict[str, float]],
    ) -> None:
        """Fetch fundamentals from yfinance."""
        from brain_api.routes.signals.helpers import get_yfinance_ratios

        fundamentals_fetched = 0
        for symbol in symbols:
            try:
                ratios = get_yfinance_ratios(symbol, as_of_date.isoformat())
                if ratios:
                    signals[symbol]["gross_margin"] = ratios.gross_margin or 0.0
                    signals[symbol]["debt_to_equity"] = ratios.debt_to_equity or 0.0
                    # fundamental_age stays 0 for current data
                    fundamentals_fetched += 1
            except Exception as e:
                logger.debug(
                    f"[SignalBuilder] Failed to fetch fundamentals for {symbol}: {e}"
                )

        logger.info(
            f"[SignalBuilder] Fundamentals fetched for {fundamentals_fetched} symbols"
        )

    def _fetch_momentum_and_earnings_yield(
        self,
        symbols: list[str],
        as_of_date: date,
        signals: dict[str, dict[str, float]],
    ) -> None:
        """Compute momentum_1w/4w/12_1 + earnings_yield (fail-loud).

        Momentum uses the same yfinance daily closes as SAC training
        (``brain_api.core.prices.load_prices_yfinance``); earnings_yield
        uses SEC point-in-time diluted EPS
        (``load_point_in_time_fundamentals``) -- never yfinance-only EPS,
        per the plan's "do not invent AV/yfinance-only EPS" constraint.
        No zero-fill on failure: this raises so callers see the exact
        symbol/reason rather than a silently wrong actor input.
        """
        from brain_api.core.fundamentals import load_point_in_time_fundamentals
        from brain_api.core.prices import load_prices_yfinance
        from brain_api.core.sac.momentum_signals import (
            MOM_12_1_LOOKBACK_BARS,
            compute_earnings_yield,
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
        fundamentals = load_point_in_time_fundamentals(symbols, as_of_date=as_of_date)

        for symbol in symbols:
            price_df = prices.get(symbol)
            if price_df is None or price_df.empty:
                raise ValueError(
                    f"[SignalBuilder] Missing price history for {symbol}; "
                    "cannot compute momentum"
                )
            closes = price_df["close"].to_numpy(dtype=float)
            as_of_index = len(closes) - 1

            fundamental = fundamentals.get(symbol)
            if fundamental is None or fundamental.eps_diluted is None:
                raise ValueError(
                    f"[SignalBuilder] Missing SEC point-in-time eps_diluted "
                    f"for {symbol}; cannot compute earnings_yield"
                )

            signals[symbol]["momentum_1w"] = compute_momentum_1w(
                closes, as_of_index=as_of_index
            )
            signals[symbol]["momentum_4w"] = compute_momentum_4w(
                closes, as_of_index=as_of_index
            )
            signals[symbol]["momentum_12_1"] = compute_momentum_12_1(
                closes, as_of_index=as_of_index
            )
            signals[symbol]["earnings_yield"] = compute_earnings_yield(
                eps_diluted=fundamental.eps_diluted,
                as_of_close=float(closes[as_of_index]),
            )

        logger.info(
            f"[SignalBuilder] Momentum + earnings_yield computed for "
            f"{len(symbols)} symbols"
        )

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

    def build_fundamentals_dataframes(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, pd.DataFrame]:
        """Build fundamentals DataFrames for PatchTST inference.

        Fetches current fundamentals from yfinance and returns DataFrames
        indexed by date with fundamental ratio columns.

        Args:
            symbols: List of stock ticker symbols
            start_date: Start of data window (not used, but kept for API consistency)
            end_date: End of data window (used as the reference date)

        Returns:
            Dict mapping symbol -> DataFrame with fundamental ratio columns
        """
        from brain_api.routes.signals.helpers import get_yfinance_ratios

        logger.info(
            f"[SignalBuilder] Fetching fundamentals DataFrames for {len(symbols)} symbols"
        )

        result: dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            try:
                ratios = get_yfinance_ratios(symbol, end_date.isoformat())
                if ratios:
                    df = pd.DataFrame(
                        {
                            "gross_margin": [ratios.gross_margin or 0.0],
                            "debt_to_equity": [ratios.debt_to_equity or 0.0],
                        },
                        index=pd.DatetimeIndex([pd.Timestamp(end_date)]),
                    )
                    result[symbol] = df
            except Exception as e:
                logger.debug(
                    f"[SignalBuilder] Failed to fetch fundamentals for {symbol}: {e}"
                )

        logger.info(
            f"[SignalBuilder] Fundamentals DataFrames built for {len(result)} symbols"
        )
        return result
