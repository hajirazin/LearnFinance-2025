"""Main FundamentalsFetcher orchestration class."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from brain_api.core.fundamentals.client import RealAlphaVantageClient
from brain_api.core.fundamentals.index import FundamentalsIndex
from brain_api.core.fundamentals.models import FundamentalRatios, FundamentalsResult
from brain_api.core.fundamentals.parser import (
    compute_ratios,
    get_statement_as_of,
    parse_quarterly_statements,
)
from brain_api.core.fundamentals.storage import load_raw_response, save_raw_response


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
        
        # Try to load from cache
        income_data = None
        balance_data = None
        
        if not force_refresh:
            income_record = self.index.get_fetch_record(symbol, "income_statement")
            balance_record = self.index.get_fetch_record(symbol, "balance_sheet")
            
            if income_record:
                income_data = load_raw_response(self.base_path, symbol, "income_statement")
            if balance_record:
                balance_data = load_raw_response(self.base_path, symbol, "balance_sheet")
        
        # Fetch missing data from API
        if income_data is None:
            raw_income = self.client.fetch_income_statement(symbol)
            if raw_income:
                file_path = save_raw_response(
                    self.base_path, symbol, "income_statement", raw_income
                )
                # Get latest dates for index
                quarterly = raw_income.get("quarterlyReports", [])
                annual = raw_income.get("annualReports", [])
                latest_q = quarterly[0].get("fiscalDateEnding") if quarterly else None
                latest_a = annual[0].get("fiscalDateEnding") if annual else None
                
                self.index.record_fetch(
                    symbol, "income_statement", str(file_path), latest_a, latest_q
                )
                income_data = {"response": raw_income}
                api_calls_made += 1
                from_cache = False
        
        if balance_data is None:
            raw_balance = self.client.fetch_balance_sheet(symbol)
            if raw_balance:
                file_path = save_raw_response(
                    self.base_path, symbol, "balance_sheet", raw_balance
                )
                quarterly = raw_balance.get("quarterlyReports", [])
                annual = raw_balance.get("annualReports", [])
                latest_q = quarterly[0].get("fiscalDateEnding") if quarterly else None
                latest_a = annual[0].get("fiscalDateEnding") if annual else None
                
                self.index.record_fetch(
                    symbol, "balance_sheet", str(file_path), latest_a, latest_q
                )
                balance_data = {"response": raw_balance}
                api_calls_made += 1
                from_cache = False
        
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
            income_stmts = parse_quarterly_statements(symbol, "income_statement", income_data)
            income_stmt = get_statement_as_of(income_stmts, as_of_date)
        
        if balance_data:
            balance_stmts = parse_quarterly_statements(symbol, "balance_sheet", balance_data)
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


