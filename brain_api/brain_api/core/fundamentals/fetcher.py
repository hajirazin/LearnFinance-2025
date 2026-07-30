"""Main FundamentalsFetcher orchestration class."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from brain_api.core.fundamentals.client import (
    AlphaVantageProviderError,
    RealAlphaVantageClient,
)
from brain_api.core.fundamentals.index import FundamentalsIndex
from brain_api.core.fundamentals.models import FundamentalRatios, FundamentalsResult
from brain_api.core.fundamentals.parser import (
    compute_ratios,
    get_statement_as_of,
    parse_quarterly_statements,
)
from brain_api.core.fundamentals.sec_filings import (
    FilingAvailabilityProvider,
    SECFilingAvailabilityClient,
    enrich_statement_periods_with_filing_availability,
)
from brain_api.core.fundamentals.storage import load_raw_response, save_raw_response


def _response_payload(wrapped: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the raw provider object from one cached wrapper."""
    if wrapped is None:
        return None
    response = wrapped.get("response")
    return response if isinstance(response, dict) else None


def _has_unresolved_quarterly_filings(payload: dict[str, Any]) -> bool:
    """Return whether any quarterly period lacks exact filing provenance."""
    reports = payload.get("quarterlyReports", [])
    return any(
        isinstance(report, dict)
        and not all(
            report.get(field)
            for field in (
                "filingDate",
                "accessionNumber",
                "filingForm",
                "filingSource",
            )
        )
        for report in reports
    )


def _has_quarterly_reports(payload: dict[str, Any] | None) -> bool:
    """Return whether a provider payload has at least one quarterly report."""
    return bool(
        payload is not None
        and isinstance(payload.get("quarterlyReports"), list)
        and payload["quarterlyReports"]
    )


def _has_resolved_quarterly_filing(payload: dict[str, Any]) -> bool:
    """Return whether at least one quarterly period has exact SEC provenance."""
    return any(
        isinstance(report, dict)
        and all(
            report.get(field)
            for field in (
                "filingDate",
                "accessionNumber",
                "filingForm",
                "filingSource",
            )
        )
        for report in payload.get("quarterlyReports", [])
    )


class FundamentalsConfigurationError(RuntimeError):
    """Raised when point-in-time fundamentals configuration is incomplete."""


class FundamentalsProviderError(RuntimeError):
    """Raised when AV or SEC cannot provide usable fundamentals evidence."""


def cached_fundamentals_require_sec_enrichment(
    base_path: Path,
    symbol: str,
) -> bool:
    """Return whether a cached statement has unresolved quarterly periods."""
    for endpoint in ("income_statement", "balance_sheet"):
        payload = _response_payload(load_raw_response(base_path, symbol, endpoint))
        if payload is not None and _has_unresolved_quarterly_filings(payload):
            return True
    return False


class FundamentalsFetcher:
    """Fetch and cache fundamental data from Alpha Vantage.

    Usage:
        fetcher = FundamentalsFetcher(api_key="...", base_path=Path("data"))
        result = fetcher.fetch_symbol("AAPL")
        ratios = fetcher.get_ratios("AAPL", as_of_date="2024-12-31")
    """

    def __init__(
        self,
        api_key: str,
        base_path: Path,
        cache_dir: Path | None = None,
        daily_limit: int = 25,
        filing_provider: FilingAvailabilityProvider | None = None,
    ):
        """Initialize the fetcher.

        Args:
            api_key: Alpha Vantage API key
            base_path: Base data directory for raw JSON files
            cache_dir: Directory for SQLite index (defaults to base_path/cache)
            daily_limit: Maximum API calls per day
        """
        self.base_path = base_path
        self.cache_dir = cache_dir or (base_path / "cache")
        sec_user_agent = os.environ.get("SEC_USER_AGENT", "")
        self.filing_provider = filing_provider
        if self.filing_provider is None and sec_user_agent:
            self.filing_provider = SECFilingAvailabilityClient(sec_user_agent)

        self.index = FundamentalsIndex(self.cache_dir)
        self.client = RealAlphaVantageClient(
            api_key=api_key,
            index=self.index,
            daily_limit=daily_limit,
        )

    def fetch_symbol(
        self,
        symbol: str,
        force_refresh: bool = False,
    ) -> FundamentalsResult:
        """Fetch fundamental data for a symbol.

        Uses cache if available, otherwise fetches from API.

        Args:
            symbol: Stock ticker
            force_refresh: If True, ignore cache and re-fetch

        Returns:
            FundamentalsResult with statements and cache status
        """
        api_calls_made = 0
        from_cache = True
        raw_income = None
        raw_balance = None

        # Try to load from cache
        income_data = None
        balance_data = None

        if not force_refresh:
            # Read the canonical/legacy cache directly. Requiring an index row
            # here would bypass the on-disk migration path and waste AV quota.
            income_data = load_raw_response(self.base_path, symbol, "income_statement")
            balance_data = load_raw_response(self.base_path, symbol, "balance_sheet")
            if not _has_quarterly_reports(_response_payload(income_data)):
                income_data = None
            if not _has_quarterly_reports(_response_payload(balance_data)):
                balance_data = None

        # Fetch missing data from API
        if income_data is None:
            try:
                raw_income = self.client.fetch_income_statement(symbol)
            except AlphaVantageProviderError as exc:
                raise FundamentalsProviderError(str(exc)) from exc
            if not _has_quarterly_reports(raw_income):
                raise FundamentalsProviderError(
                    "Alpha Vantage returned no usable quarterly income statement "
                    f"for {symbol}"
                )
            api_calls_made += 1
            from_cache = False
            income_data = {"response": raw_income}

        if balance_data is None:
            try:
                raw_balance = self.client.fetch_balance_sheet(symbol)
            except AlphaVantageProviderError as exc:
                raise FundamentalsProviderError(str(exc)) from exc
            if not _has_quarterly_reports(raw_balance):
                raise FundamentalsProviderError(
                    "Alpha Vantage returned no usable quarterly balance sheet "
                    f"for {symbol}"
                )
            api_calls_made += 1
            from_cache = False
            balance_data = {"response": raw_balance}

        payloads = {
            "income_statement": _response_payload(income_data),
            "balance_sheet": _response_payload(balance_data),
        }
        unresolved = {
            endpoint
            for endpoint, payload in payloads.items()
            if payload is not None and _has_unresolved_quarterly_filings(payload)
        }
        if unresolved:
            if self.filing_provider is None:
                raise FundamentalsConfigurationError(
                    "SEC filing availability is required for fundamentals; "
                    "set SEC_USER_AGENT or inject a filing_provider"
                )
            try:
                filings = self.filing_provider.fetch_symbol_filings(symbol)
            except Exception as exc:
                raise FundamentalsProviderError(
                    f"SEC filing availability request failed for {symbol}: {exc}"
                ) from exc
            for endpoint in unresolved:
                payload = payloads[endpoint]
                if payload is not None:
                    enrich_statement_periods_with_filing_availability(payload, filings)
                    if not _has_resolved_quarterly_filing(payload):
                        raise FundamentalsProviderError(
                            "SEC enrichment produced no exact quarterly filing "
                            f"matches for {symbol} {endpoint}"
                        )

        endpoints_to_save = set(unresolved)
        if raw_income is not None:
            endpoints_to_save.add("income_statement")
        if raw_balance is not None:
            endpoints_to_save.add("balance_sheet")
        for endpoint in endpoints_to_save:
            payload = payloads[endpoint]
            if payload is None:
                continue
            # Exact SEC report-date misses remain unresolved and are excluded
            # by the loader; persisting them records that enrichment was tried.
            file_path = save_raw_response(self.base_path, symbol, endpoint, payload)
            quarterly = payload.get("quarterlyReports", [])
            annual = payload.get("annualReports", [])
            latest_q = quarterly[0].get("fiscalDateEnding") if quarterly else None
            latest_a = annual[0].get("fiscalDateEnding") if annual else None
            self.index.record_fetch(
                symbol, endpoint, str(file_path), latest_a, latest_q
            )
            if endpoint == "income_statement":
                income_data = {"response": payload}
            else:
                balance_data = {"response": payload}

        # Parse statements
        income_statements = []
        balance_sheets = []

        if income_data:
            income_statements = parse_quarterly_statements(
                symbol, "income_statement", income_data
            )
        if balance_data:
            balance_sheets = parse_quarterly_statements(
                symbol, "balance_sheet", balance_data
            )

        calls_today = self.index.get_api_calls_today()

        return FundamentalsResult(
            symbol=symbol,
            income_statements=income_statements,
            balance_sheets=balance_sheets,
            from_cache=from_cache and api_calls_made == 0,
            api_calls_made=api_calls_made,
            api_calls_remaining=max(0, self.client.daily_limit - calls_today),
        )

    def get_ratios(
        self,
        symbol: str,
        as_of_date: str,
    ) -> FundamentalRatios | None:
        """Get financial ratios for a symbol as of a specific date.

        Uses cached data only - call fetch_symbol first to ensure data exists.

        Args:
            symbol: Stock ticker
            as_of_date: YYYY-MM-DD date for point-in-time lookup

        Returns:
            FundamentalRatios or None if no data available
        """
        # Load cached data
        income_data = load_raw_response(self.base_path, symbol, "income_statement")
        balance_data = load_raw_response(self.base_path, symbol, "balance_sheet")

        if income_data is None and balance_data is None:
            return None

        # Parse and get point-in-time statements
        income_stmt = None
        balance_stmt = None

        if income_data:
            income_stmts = parse_quarterly_statements(
                symbol, "income_statement", income_data
            )
            income_stmt = get_statement_as_of(income_stmts, as_of_date)

        if balance_data:
            balance_stmts = parse_quarterly_statements(
                symbol, "balance_sheet", balance_data
            )
            balance_stmt = get_statement_as_of(balance_stmts, as_of_date)

        return compute_ratios(income_stmt, balance_stmt)

    def get_api_status(self) -> dict[str, Any]:
        """Get current API usage status.

        Returns:
            Dict with calls_today, daily_limit, remaining
        """
        calls_today = self.index.get_api_calls_today()
        return {
            "calls_today": calls_today,
            "daily_limit": self.client.daily_limit,
            "remaining": max(0, self.client.daily_limit - calls_today),
        }

    def get_cached_symbols(self) -> list[str]:
        """Get list of symbols with cached data.

        Returns:
            List of symbol tickers
        """
        return self.index.get_all_fetched_symbols()

    def close(self) -> None:
        """Close database connections."""
        self.index.close()
