"""Tests for data_freshness module (filing-head freshness, not fetch-today)."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain_api.core.data_freshness import (
    ensure_fresh_training_data,
    refresh_stale_fundamentals,
)
from brain_api.core.fundamentals.refresh_policy import RefreshAction
from brain_api.etl.gap_fill import GapFillProgress, GapFillResult


def _mock_fetcher(
    *,
    actions: dict[str, RefreshAction] | None = None,
    default_action: RefreshAction = RefreshAction.SKIP,
    fetch_side_effect=None,
    sec_eligible: set[str] | None = None,
    pending: set[str] | None = None,
) -> MagicMock:
    fetcher = MagicMock()
    actions = actions or {}
    sec_eligible = sec_eligible or set()
    pending = pending or set()

    def decide(symbol: str, *, force_refresh: bool = False) -> RefreshAction:
        if force_refresh:
            return RefreshAction.PULL
        return actions.get(symbol, default_action)

    def classify(symbol: str):
        result = MagicMock()
        result.sec_eligible = symbol in sec_eligible
        result.cik = "0000320193" if symbol in sec_eligible else None
        return result

    fetcher.decide_action_for_symbol.side_effect = decide
    fetcher.eligibility_client = MagicMock()
    fetcher.eligibility_client.classify.side_effect = classify
    fetcher._pending_new_filing = set(pending)
    if fetch_side_effect is not None:
        fetcher.fetch_symbol.side_effect = fetch_side_effect
    fetcher.get_api_status.return_value = {
        "calls_today": 0,
        "daily_limit": 25,
        "remaining": 25,
    }
    return fetcher


class TestEnsureFreshTrainingData:
    """Tests for ensure_fresh_training_data function."""

    def test_fills_sentiment_gaps(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "output" / "daily_sentiment.parquet"
        parquet_path.parent.mkdir(parents=True)
        parquet_path.touch()

        symbols = ["AAPL", "MSFT"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        mock_gap_result = GapFillResult(
            success=True,
            progress=GapFillProgress(rows_added=100, gaps_pre_api_date=50),
        )

        with (
            patch(
                "brain_api.core.data_freshness.fill_sentiment_gaps",
                return_value=mock_gap_result,
            ) as mock_fill,
            patch(
                "brain_api.core.data_freshness.refresh_stale_fundamentals",
            ) as mock_refresh,
            patch.dict(
                "os.environ",
                {
                    "ALPHA_VANTAGE_API_KEY": "test_key",
                    "SEC_USER_AGENT": "LearnFinance test@example.com",
                },
            ),
        ):
            mock_refresh.return_value = MagicMock(
                refreshed=[],
                skipped=["AAPL", "MSFT"],
                failed=[],
                errors={},
            )
            result = ensure_fresh_training_data(
                "halal_filtered",
                symbols,
                start_date,
                end_date,
                parquet_path=parquet_path,
            )

            mock_fill.assert_called_once_with(
                universe="halal_filtered",
                start_date=start_date,
                end_date=end_date,
                parquet_path=parquet_path,
                local_only=True,
            )
            assert result.sentiment_gaps_filled == 100
            assert result.sentiment_gaps_remaining == 50

    def test_refreshes_fundamentals_needing_pull(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "output" / "daily_sentiment.parquet"
        base_path = tmp_path
        symbols = ["AAPL", "MSFT"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        with (
            patch("brain_api.core.data_freshness.fill_sentiment_gaps"),
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict(
                "os.environ",
                {
                    "ALPHA_VANTAGE_API_KEY": "test_key",
                    "SEC_USER_AGENT": "LearnFinance test@example.com",
                },
            ),
        ):
            mock_fetcher = _mock_fetcher(
                actions={
                    "AAPL": RefreshAction.SKIP,
                    "MSFT": RefreshAction.PULL,
                },
            )
            mock_fetcher_class.return_value = mock_fetcher

            result = ensure_fresh_training_data(
                "halal_filtered",
                symbols,
                start_date,
                end_date,
                parquet_path=parquet_path,
                fundamentals_base_path=base_path,
            )

            mock_fetcher.fetch_symbol.assert_called_once_with(
                "MSFT", force_refresh=False
            )
            assert "MSFT" in result.fundamentals_refreshed
            assert "AAPL" not in result.fundamentals_refreshed
            assert mock_fetcher.close.called

    def test_continues_on_fundamentals_failure(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "output" / "daily_sentiment.parquet"
        base_path = tmp_path
        symbols = ["AAPL", "MSFT", "GOOGL"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        def mock_fetch(symbol: str, *, force_refresh: bool = False) -> None:
            if symbol == "MSFT":
                raise Exception("API rate limit exceeded")

        with (
            patch("brain_api.core.data_freshness.fill_sentiment_gaps"),
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict(
                "os.environ",
                {
                    "ALPHA_VANTAGE_API_KEY": "test_key",
                    "SEC_USER_AGENT": "LearnFinance test@example.com",
                },
            ),
        ):
            mock_fetcher = _mock_fetcher(
                default_action=RefreshAction.PULL,
                fetch_side_effect=mock_fetch,
            )
            mock_fetcher_class.return_value = mock_fetcher

            result = ensure_fresh_training_data(
                "halal_filtered",
                symbols,
                start_date,
                end_date,
                parquet_path=parquet_path,
                fundamentals_base_path=base_path,
            )

            assert "AAPL" in result.fundamentals_refreshed
            assert "GOOGL" in result.fundamentals_refreshed
            assert "MSFT" in result.fundamentals_failed
            assert len(result.fundamentals_failed) == 1

    def test_skips_fresh_fundamentals(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "output" / "daily_sentiment.parquet"
        base_path = tmp_path
        symbols = ["AAPL", "MSFT"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        with (
            patch("brain_api.core.data_freshness.fill_sentiment_gaps"),
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict(
                "os.environ",
                {
                    "ALPHA_VANTAGE_API_KEY": "test_key",
                    "SEC_USER_AGENT": "LearnFinance test@example.com",
                },
            ),
        ):
            mock_fetcher = _mock_fetcher(default_action=RefreshAction.SKIP)
            mock_fetcher_class.return_value = mock_fetcher

            result = ensure_fresh_training_data(
                "halal_filtered",
                symbols,
                start_date,
                end_date,
                parquet_path=parquet_path,
                fundamentals_base_path=base_path,
            )

            mock_fetcher.fetch_symbol.assert_not_called()
            assert result.fundamentals_refreshed == []
            assert result.fundamentals_skipped_today == ["AAPL", "MSFT"]

    def test_fails_without_sec_user_agent(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "output" / "daily_sentiment.parquet"
        base_path = tmp_path
        symbols = ["AAPL", "MSFT"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        with (
            patch("brain_api.core.data_freshness.fill_sentiment_gaps"),
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict("os.environ", {"SEC_USER_AGENT": ""}, clear=False),
        ):
            result = ensure_fresh_training_data(
                "halal_filtered",
                symbols,
                start_date,
                end_date,
                parquet_path=parquet_path,
                fundamentals_base_path=base_path,
            )

            mock_fetcher_class.assert_not_called()
            assert result.fundamentals_refreshed == []
            assert set(result.fundamentals_failed) == {"AAPL", "MSFT"}

    def test_returns_duration_seconds(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "output" / "daily_sentiment.parquet"
        base_path = tmp_path
        symbols = ["AAPL"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        with (
            patch("brain_api.core.data_freshness.fill_sentiment_gaps"),
            patch(
                "brain_api.core.data_freshness.refresh_stale_fundamentals",
            ) as mock_refresh,
            patch.dict(
                "os.environ",
                {
                    "ALPHA_VANTAGE_API_KEY": "k",
                    "SEC_USER_AGENT": "LearnFinance test@example.com",
                },
            ),
        ):
            mock_refresh.return_value = MagicMock(
                refreshed=[], skipped=["AAPL"], failed=[], errors={}
            )
            result = ensure_fresh_training_data(
                "halal_filtered",
                symbols,
                start_date,
                end_date,
                parquet_path=parquet_path,
                fundamentals_base_path=base_path,
            )
            assert result.duration_seconds >= 0


class TestRefreshStaleFundamentals:
    """Tests for refresh_stale_fundamentals (filing-head policy)."""

    def test_skips_fresh_symbols(self, tmp_path: Path) -> None:
        symbols = ["AAPL", "MSFT"]
        with (
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict(
                "os.environ",
                {
                    "ALPHA_VANTAGE_API_KEY": "test_key",
                    "SEC_USER_AGENT": "LearnFinance test@example.com",
                },
            ),
        ):
            mock_fetcher = _mock_fetcher(default_action=RefreshAction.SKIP)
            mock_fetcher_class.return_value = mock_fetcher

            result = refresh_stale_fundamentals(symbols, base_path=tmp_path)

            mock_fetcher.fetch_symbol.assert_not_called()
            assert result.refreshed == []
            assert result.skipped == ["AAPL", "MSFT"]
            assert result.failed == []

    def test_refreshes_pull_symbols(self, tmp_path: Path) -> None:
        symbols = ["AAPL", "MSFT", "GOOGL"]
        with (
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict(
                "os.environ",
                {
                    "ALPHA_VANTAGE_API_KEY": "test_key",
                    "SEC_USER_AGENT": "LearnFinance test@example.com",
                },
            ),
        ):
            mock_fetcher = _mock_fetcher(
                actions={
                    "AAPL": RefreshAction.SKIP,
                    "MSFT": RefreshAction.PULL,
                    "GOOGL": RefreshAction.PULL,
                },
            )
            mock_fetcher_class.return_value = mock_fetcher

            result = refresh_stale_fundamentals(symbols, base_path=tmp_path)

            assert mock_fetcher.fetch_symbol.call_count == 2
            mock_fetcher.fetch_symbol.assert_any_call("MSFT", force_refresh=False)
            mock_fetcher.fetch_symbol.assert_any_call("GOOGL", force_refresh=False)
            assert "MSFT" in result.refreshed
            assert "GOOGL" in result.refreshed
            assert "AAPL" in result.skipped
            assert mock_fetcher.close.called

    def test_returns_api_status(self, tmp_path: Path) -> None:
        with (
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict(
                "os.environ",
                {
                    "ALPHA_VANTAGE_API_KEY": "test_key",
                    "SEC_USER_AGENT": "LearnFinance test@example.com",
                },
            ),
        ):
            mock_fetcher = _mock_fetcher(
                actions={"AAPL": RefreshAction.PULL},
            )
            mock_fetcher.get_api_status.return_value = {
                "calls_today": 10,
                "daily_limit": 25,
                "remaining": 15,
            }
            mock_fetcher_class.return_value = mock_fetcher

            result = refresh_stale_fundamentals(["AAPL"], base_path=tmp_path)

            assert result.api_status["calls_today"] == 10
            assert result.api_status["daily_limit"] == 25
            assert result.api_status["remaining"] == 15

    def test_continues_on_failure(self, tmp_path: Path) -> None:
        def mock_fetch(symbol: str, *, force_refresh: bool = False) -> None:
            if symbol == "MSFT":
                raise Exception("API rate limit exceeded")

        with (
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict(
                "os.environ",
                {
                    "ALPHA_VANTAGE_API_KEY": "test_key",
                    "SEC_USER_AGENT": "LearnFinance test@example.com",
                },
            ),
        ):
            mock_fetcher = _mock_fetcher(
                default_action=RefreshAction.PULL,
                fetch_side_effect=mock_fetch,
            )
            mock_fetcher_class.return_value = mock_fetcher

            result = refresh_stale_fundamentals(
                ["AAPL", "MSFT", "GOOGL"], base_path=tmp_path
            )

            assert "AAPL" in result.refreshed
            assert "GOOGL" in result.refreshed
            assert "MSFT" in result.failed
            assert len(result.failed) == 1

    def test_fails_without_sec_user_agent(self, tmp_path: Path) -> None:
        with (
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict("os.environ", {"SEC_USER_AGENT": ""}, clear=False),
        ):
            result = refresh_stale_fundamentals(["AAPL", "MSFT"], base_path=tmp_path)

            mock_fetcher_class.assert_not_called()
            assert result.refreshed == []
            assert result.failed == ["AAPL", "MSFT"]

    def test_av_pull_requires_api_key(self, tmp_path: Path) -> None:
        with (
            patch(
                "brain_api.core.data_freshness.FundamentalsFetcher"
            ) as mock_fetcher_class,
            patch.dict(
                "os.environ",
                {
                    "ALPHA_VANTAGE_API_KEY": "",
                    "SEC_USER_AGENT": "LearnFinance test@example.com",
                },
                clear=False,
            ),
        ):
            mock_fetcher = _mock_fetcher(
                actions={"SAP": RefreshAction.PULL},
                sec_eligible=set(),
            )
            mock_fetcher_class.return_value = mock_fetcher

            result = refresh_stale_fundamentals(["SAP"], base_path=tmp_path)

            mock_fetcher.fetch_symbol.assert_not_called()
            assert result.failed == ["SAP"]
            assert "ALPHA_VANTAGE_API_KEY" in result.errors["SAP"]
