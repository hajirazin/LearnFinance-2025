"""Tests for data_freshness module."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain_api.core.data_freshness import (
    ensure_fresh_training_data,
    get_symbols_not_fetched_today,
    refresh_stale_fundamentals,
)
from brain_api.core.fundamentals.models import FetchRecord
from brain_api.etl.gap_fill import GapFillProgress, GapFillResult


class TestGetSymbolsNotFetchedToday:
    """Tests for get_symbols_not_fetched_today function."""

    def test_none_fetched_returns_all_symbols(self, tmp_path: Path) -> None:
        """All symbols need fetching when DB is empty."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        symbols = ["AAPL", "MSFT", "GOOGL"]

        with patch(
            "brain_api.core.data_freshness.FundamentalsIndex"
        ) as mock_index_class:
            mock_index = MagicMock()
            mock_index.get_fetch_record.return_value = None
            mock_index_class.return_value = mock_index

            result = get_symbols_not_fetched_today(symbols, cache_dir)

            assert result == ["AAPL", "MSFT", "GOOGL"]
            assert mock_index.close.called

    def test_some_fetched_today_returns_stale_only(self, tmp_path: Path) -> None:
        """Only stale symbols returned when some fetched today."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        symbols = ["AAPL", "MSFT", "GOOGL"]
        today = date.today()
        yesterday = (
            date(today.year, today.month, today.day - 1)
            if today.day > 1
            else date(today.year, today.month - 1, 28)
        )

        def mock_get_record(symbol: str, endpoint: str) -> FetchRecord | None:
            if symbol == "AAPL":
                # Fetched today
                return FetchRecord(
                    symbol="AAPL",
                    endpoint="income_statement",
                    file_path="/path/to/aapl.json",
                    fetched_at=f"{today.isoformat()}T10:00:00+00:00",
                    latest_annual_date="2024-12-31",
                    latest_quarterly_date="2024-09-30",
                )
            elif symbol == "MSFT":
                # Fetched yesterday
                return FetchRecord(
                    symbol="MSFT",
                    endpoint="income_statement",
                    file_path="/path/to/msft.json",
                    fetched_at=f"{yesterday.isoformat()}T10:00:00+00:00",
                    latest_annual_date="2024-12-31",
                    latest_quarterly_date="2024-09-30",
                )
            else:
                # Never fetched
                return None

        with patch(
            "brain_api.core.data_freshness.FundamentalsIndex"
        ) as mock_index_class:
            mock_index = MagicMock()
            mock_index.get_fetch_record.side_effect = mock_get_record
            mock_index_class.return_value = mock_index

            result = get_symbols_not_fetched_today(symbols, cache_dir)

            # AAPL was fetched today, so only MSFT and GOOGL should be returned
            assert "AAPL" not in result
            assert "MSFT" in result
            assert "GOOGL" in result

    def test_all_fetched_today_returns_empty(self, tmp_path: Path) -> None:
        """Empty list when all fetched today."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        symbols = ["AAPL", "MSFT"]
        today = date.today()

        with patch(
            "brain_api.core.data_freshness.FundamentalsIndex"
        ) as mock_index_class:
            mock_index = MagicMock()
            mock_index.get_fetch_record.return_value = FetchRecord(
                symbol="ANY",
                endpoint="income_statement",
                file_path="/path/to/any.json",
                fetched_at=f"{today.isoformat()}T10:00:00+00:00",
                latest_annual_date="2024-12-31",
                latest_quarterly_date="2024-09-30",
            )
            mock_index_class.return_value = mock_index

            result = get_symbols_not_fetched_today(symbols, cache_dir)

            assert result == []


class TestEnsureFreshTrainingData:
    """Tests for ensure_fresh_training_data function."""

    def test_fills_sentiment_gaps(self, tmp_path: Path) -> None:
        """Calls gap_fill for sentiment."""
        parquet_path = tmp_path / "output" / "daily_sentiment.parquet"
        parquet_path.parent.mkdir(parents=True)
        parquet_path.touch()  # Create empty file

        symbols = ["AAPL", "MSFT"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        mock_gap_result = GapFillResult(
            success=True,
            progress=GapFillProgress(
                rows_added=100,
                gaps_pre_api_date=50,
            ),
        )

        with (
            patch(
                "brain_api.core.data_freshness.fill_sentiment_gaps",
                return_value=mock_gap_result,
            ) as mock_fill,
            patch(
                "brain_api.core.data_freshness.get_symbols_not_fetched_today",
                return_value=[],
            ),
            patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": ""}),
        ):
            result = ensure_fresh_training_data(
                symbols, start_date, end_date, parquet_path=parquet_path
            )

            mock_fill.assert_called_once_with(
                start_date=start_date,
                end_date=end_date,
                parquet_path=parquet_path,
                local_only=True,
            )
            assert result.sentiment_gaps_filled == 100
            assert result.sentiment_gaps_remaining == 50

    def test_refreshes_fundamentals_not_fetched_today(self, tmp_path: Path) -> None:
        """Fetches fundamentals not fetched today."""
        parquet_path = tmp_path / "output" / "daily_sentiment.parquet"
        base_path = tmp_path

        symbols = ["AAPL", "MSFT"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        with (
            patch(
                "brain_api.core.data_freshness.fill_sentiment_gaps",
            ),
            patch(
                "brain_api.core.data_freshness.get_symbols_not_fetched_today",
                return_value=["MSFT"],  # Only MSFT needs refreshing
            ),
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"}),
        ):
            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            # Parquet doesn't exist, so gap fill will be skipped
            result = ensure_fresh_training_data(
                symbols,
                start_date,
                end_date,
                parquet_path=parquet_path,
                fundamentals_base_path=base_path,
            )

            # Should have called fetch_symbol for MSFT
            mock_fetcher.fetch_symbol.assert_called_once_with("MSFT")
            assert "MSFT" in result.fundamentals_refreshed
            assert "AAPL" not in result.fundamentals_refreshed
            assert mock_fetcher.close.called

    def test_continues_on_fundamentals_failure(self, tmp_path: Path) -> None:
        """Logs warning and continues when fetch fails."""
        parquet_path = tmp_path / "output" / "daily_sentiment.parquet"
        base_path = tmp_path

        symbols = ["AAPL", "MSFT", "GOOGL"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        def mock_fetch(symbol: str) -> None:
            if symbol == "MSFT":
                raise Exception("API rate limit exceeded")
            # Other symbols succeed

        with (
            patch("brain_api.core.data_freshness.fill_sentiment_gaps"),
            patch(
                "brain_api.core.data_freshness.get_symbols_not_fetched_today",
                return_value=["AAPL", "MSFT", "GOOGL"],
            ),
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"}),
        ):
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_symbol.side_effect = mock_fetch
            mock_fetcher_class.return_value = mock_fetcher

            result = ensure_fresh_training_data(
                symbols,
                start_date,
                end_date,
                parquet_path=parquet_path,
                fundamentals_base_path=base_path,
            )

            # AAPL and GOOGL should succeed, MSFT should fail
            assert "AAPL" in result.fundamentals_refreshed
            assert "GOOGL" in result.fundamentals_refreshed
            assert "MSFT" in result.fundamentals_failed
            assert len(result.fundamentals_failed) == 1

    def test_skips_already_fresh_fundamentals(self, tmp_path: Path) -> None:
        """Skips symbols fetched today."""
        parquet_path = tmp_path / "output" / "daily_sentiment.parquet"
        base_path = tmp_path

        symbols = ["AAPL", "MSFT"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        with (
            patch("brain_api.core.data_freshness.fill_sentiment_gaps"),
            patch(
                "brain_api.core.data_freshness.get_symbols_not_fetched_today",
                return_value=[],  # All symbols already fetched today
            ),
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"}),
        ):
            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            result = ensure_fresh_training_data(
                symbols,
                start_date,
                end_date,
                parquet_path=parquet_path,
                fundamentals_base_path=base_path,
            )

            # fetch_symbol should not be called since all are fresh
            mock_fetcher.fetch_symbol.assert_not_called()
            assert result.fundamentals_refreshed == []
            assert result.fundamentals_skipped_today == ["AAPL", "MSFT"]

    def test_skips_fundamentals_when_no_api_key(self, tmp_path: Path) -> None:
        """Skips fundamentals refresh when API key is not set."""
        parquet_path = tmp_path / "output" / "daily_sentiment.parquet"
        base_path = tmp_path

        symbols = ["AAPL", "MSFT"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        with (
            patch("brain_api.core.data_freshness.fill_sentiment_gaps"),
            patch(
                "brain_api.core.data_freshness.get_symbols_not_fetched_today",
                return_value=["AAPL", "MSFT"],
            ),
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": ""}, clear=False),
        ):
            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            result = ensure_fresh_training_data(
                symbols,
                start_date,
                end_date,
                parquet_path=parquet_path,
                fundamentals_base_path=base_path,
            )

            # FundamentalsFetcher should not be instantiated
            mock_fetcher_class.assert_not_called()
            assert result.fundamentals_refreshed == []

    def test_returns_duration_seconds(self, tmp_path: Path) -> None:
        """Returns duration of the operation."""
        parquet_path = tmp_path / "output" / "daily_sentiment.parquet"
        base_path = tmp_path

        symbols = ["AAPL"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        with (
            patch("brain_api.core.data_freshness.fill_sentiment_gaps"),
            patch(
                "brain_api.core.data_freshness.get_symbols_not_fetched_today",
                return_value=[],
            ),
            patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": ""}),
        ):
            result = ensure_fresh_training_data(
                symbols,
                start_date,
                end_date,
                parquet_path=parquet_path,
                fundamentals_base_path=base_path,
            )

            assert result.duration_seconds >= 0


class TestRefreshStaleFundamentals:
    """Tests for refresh_stale_fundamentals function.

    This function is shared by:
    - PUT /signals/fundamentals/historical endpoint
    - ensure_fresh_training_data() before training
    """

    def test_skips_symbols_fetched_today(self, tmp_path: Path) -> None:
        """All symbols already fetched today are skipped."""
        base_path = tmp_path

        symbols = ["AAPL", "MSFT"]

        with (
            patch(
                "brain_api.core.data_freshness.get_symbols_not_fetched_today",
                return_value=[],  # All symbols already fetched today
            ),
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch(
                "brain_api.core.data_freshness.FundamentalsIndex"
            ) as mock_index_class,
            patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"}),
        ):
            mock_index = MagicMock()
            mock_index.get_api_calls_today.return_value = 5
            mock_index_class.return_value = mock_index

            result = refresh_stale_fundamentals(symbols, base_path=base_path)

            # FundamentalsFetcher should not be instantiated (nothing to fetch)
            mock_fetcher_class.assert_not_called()
            assert result.refreshed == []
            assert result.skipped == ["AAPL", "MSFT"]
            assert result.failed == []

    def test_refreshes_stale_symbols(self, tmp_path: Path) -> None:
        """Symbols not fetched today are refreshed via API."""
        base_path = tmp_path

        symbols = ["AAPL", "MSFT", "GOOGL"]

        with (
            patch(
                "brain_api.core.data_freshness.get_symbols_not_fetched_today",
                return_value=["MSFT", "GOOGL"],  # Only these need refreshing
            ),
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"}),
        ):
            mock_fetcher = MagicMock()
            mock_fetcher.get_api_status.return_value = {
                "calls_today": 7,
                "daily_limit": 25,
                "remaining": 18,
            }
            mock_fetcher_class.return_value = mock_fetcher

            result = refresh_stale_fundamentals(symbols, base_path=base_path)

            # Should have called fetch_symbol for MSFT and GOOGL
            assert mock_fetcher.fetch_symbol.call_count == 2
            mock_fetcher.fetch_symbol.assert_any_call("MSFT")
            mock_fetcher.fetch_symbol.assert_any_call("GOOGL")
            assert "MSFT" in result.refreshed
            assert "GOOGL" in result.refreshed
            assert "AAPL" in result.skipped
            assert mock_fetcher.close.called

    def test_returns_api_status(self, tmp_path: Path) -> None:
        """Returns API status in result."""
        base_path = tmp_path

        symbols = ["AAPL"]

        with (
            patch(
                "brain_api.core.data_freshness.get_symbols_not_fetched_today",
                return_value=["AAPL"],
            ),
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"}),
        ):
            mock_fetcher = MagicMock()
            mock_fetcher.get_api_status.return_value = {
                "calls_today": 10,
                "daily_limit": 25,
                "remaining": 15,
            }
            mock_fetcher_class.return_value = mock_fetcher

            result = refresh_stale_fundamentals(symbols, base_path=base_path)

            assert result.api_status["calls_today"] == 10
            assert result.api_status["daily_limit"] == 25
            assert result.api_status["remaining"] == 15

    def test_continues_on_failure(self, tmp_path: Path) -> None:
        """Continues fetching other symbols when one fails."""
        base_path = tmp_path

        symbols = ["AAPL", "MSFT", "GOOGL"]

        def mock_fetch(symbol: str) -> None:
            if symbol == "MSFT":
                raise Exception("API rate limit exceeded")
            # Other symbols succeed

        with (
            patch(
                "brain_api.core.data_freshness.get_symbols_not_fetched_today",
                return_value=["AAPL", "MSFT", "GOOGL"],
            ),
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"}),
        ):
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_symbol.side_effect = mock_fetch
            mock_fetcher.get_api_status.return_value = {
                "calls_today": 5,
                "daily_limit": 25,
                "remaining": 20,
            }
            mock_fetcher_class.return_value = mock_fetcher

            result = refresh_stale_fundamentals(symbols, base_path=base_path)

            # AAPL and GOOGL should succeed, MSFT should fail
            assert "AAPL" in result.refreshed
            assert "GOOGL" in result.refreshed
            assert "MSFT" in result.failed
            assert len(result.failed) == 1

    def test_skips_when_no_api_key(self, tmp_path: Path) -> None:
        """Fails all symbols when API key is not set."""
        base_path = tmp_path

        symbols = ["AAPL", "MSFT"]

        with (
            patch(
                "brain_api.core.data_freshness.get_symbols_not_fetched_today",
                return_value=["AAPL", "MSFT"],
            ),
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": ""}, clear=False),
        ):
            result = refresh_stale_fundamentals(symbols, base_path=base_path)

            # FundamentalsFetcher should not be instantiated
            mock_fetcher_class.assert_not_called()
            assert result.refreshed == []
            assert result.failed == ["AAPL", "MSFT"]
