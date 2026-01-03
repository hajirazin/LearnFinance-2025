"""Alpha Vantage fundamentals fetcher with hybrid caching.

Storage strategy:
- Raw JSON files: Store complete API responses as source of truth
  Location: data/raw/fundamentals/{symbol}/income_statement.json
            data/raw/fundamentals/{symbol}/balance_sheet.json
- SQLite index: Track what's been fetched and when for quick lookups
  Location: data/cache/fundamentals.db

This approach ensures:
1. No data loss if parsing/schema changes
2. Efficient lookups without re-parsing all files
3. Rate limit awareness (25/day free tier)
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime, date
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol


# ============================================================================
# Data models
# ============================================================================


@dataclass
class QuarterlyStatement:
    """A single quarterly financial statement (income or balance sheet)."""

    symbol: str
    statement_type: str  # "income_statement" or "balance_sheet"
    fiscal_date_ending: str  # YYYY-MM-DD
    reported_currency: str
    raw_data: dict[str, Any]  # All fields from API

    def get_value(self, field: str) -> Decimal | None:
        """Get a numeric value from the statement, handling 'None' strings."""
        val = self.raw_data.get(field)
        if val is None or val == "None" or val == "":
            return None
        try:
            return Decimal(str(val))
        except Exception:
            return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "statement_type": self.statement_type,
            "fiscal_date_ending": self.fiscal_date_ending,
            "reported_currency": self.reported_currency,
            "raw_data": self.raw_data,
        }


@dataclass
class FundamentalRatios:
    """Computed financial ratios for a symbol at a point in time.
    
    5 core ratios for PPO:
    - Profitability: gross_margin, operating_margin, net_margin
    - Liquidity: current_ratio
    - Leverage: debt_to_equity
    """

    symbol: str
    as_of_date: str  # The fiscal_date_ending used
    
    # Profitability
    gross_margin: float | None  # grossProfit / totalRevenue
    operating_margin: float | None  # operatingIncome / totalRevenue
    net_margin: float | None  # netIncome / totalRevenue
    
    # Liquidity
    current_ratio: float | None  # totalCurrentAssets / totalCurrentLiabilities
    
    # Leverage
    debt_to_equity: float | None  # shortLongTermDebtTotal / totalShareholderEquity
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "as_of_date": self.as_of_date,
            "gross_margin": self.gross_margin,
            "operating_margin": self.operating_margin,
            "net_margin": self.net_margin,
            "current_ratio": self.current_ratio,
            "debt_to_equity": self.debt_to_equity,
        }


# ============================================================================
# SQLite Index for tracking fetched data
# ============================================================================


@dataclass
class FetchRecord:
    """Record of a fetched statement file."""
    
    symbol: str
    endpoint: str  # "income_statement" or "balance_sheet"
    file_path: str
    fetched_at: str  # ISO timestamp
    latest_annual_date: str | None
    latest_quarterly_date: str | None


class FundamentalsIndex:
    """SQLite index for tracking fetched fundamental data.
    
    This doesn't store the actual data - just metadata about what's been fetched.
    """

    def __init__(self, cache_dir: Path):
        """Initialize the index.
        
        Args:
            cache_dir: Directory to store the index database
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "fundamentals.db"
        self._conn: sqlite3.Connection | None = None

    def _ensure_connected(self) -> sqlite3.Connection:
        """Ensure database connection and schema exist."""
        if self._conn is not None:
            return self._conn

        self._conn = sqlite3.connect(str(self.db_path))
        
        # Track fetched files
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fetch_log (
                symbol TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                file_path TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                latest_annual_date TEXT,
                latest_quarterly_date TEXT,
                PRIMARY KEY (symbol, endpoint)
            )
        """)
        
        # Track API rate limit usage
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS api_calls (
                call_date TEXT NOT NULL,
                call_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (call_date)
            )
        """)
        
        self._conn.commit()
        return self._conn

    def get_fetch_record(self, symbol: str, endpoint: str) -> FetchRecord | None:
        """Get the fetch record for a symbol/endpoint pair.
        
        Args:
            symbol: Stock ticker
            endpoint: "income_statement" or "balance_sheet"
            
        Returns:
            FetchRecord if exists, None otherwise
        """
        conn = self._ensure_connected()
        cursor = conn.execute(
            """
            SELECT symbol, endpoint, file_path, fetched_at, 
                   latest_annual_date, latest_quarterly_date
            FROM fetch_log
            WHERE symbol = ? AND endpoint = ?
            """,
            (symbol, endpoint),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return FetchRecord(
            symbol=row[0],
            endpoint=row[1],
            file_path=row[2],
            fetched_at=row[3],
            latest_annual_date=row[4],
            latest_quarterly_date=row[5],
        )

    def record_fetch(
        self,
        symbol: str,
        endpoint: str,
        file_path: str,
        latest_annual_date: str | None,
        latest_quarterly_date: str | None,
    ) -> None:
        """Record that a file was fetched.
        
        Args:
            symbol: Stock ticker
            endpoint: "income_statement" or "balance_sheet"
            file_path: Path where JSON was saved
            latest_annual_date: Most recent annual report date
            latest_quarterly_date: Most recent quarterly report date
        """
        conn = self._ensure_connected()
        now = datetime.now(UTC).isoformat()
        conn.execute(
            """
            INSERT OR REPLACE INTO fetch_log
            (symbol, endpoint, file_path, fetched_at, latest_annual_date, latest_quarterly_date)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (symbol, endpoint, file_path, now, latest_annual_date, latest_quarterly_date),
        )
        conn.commit()

    def get_api_calls_today(self) -> int:
        """Get the number of API calls made today.
        
        Returns:
            Number of API calls made today
        """
        conn = self._ensure_connected()
        today = date.today().isoformat()
        cursor = conn.execute(
            "SELECT call_count FROM api_calls WHERE call_date = ?",
            (today,),
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def increment_api_calls(self, count: int = 1) -> int:
        """Increment the API call counter for today.
        
        Args:
            count: Number of calls to add
            
        Returns:
            New total for today
        """
        conn = self._ensure_connected()
        today = date.today().isoformat()
        
        # Use upsert
        conn.execute(
            """
            INSERT INTO api_calls (call_date, call_count)
            VALUES (?, ?)
            ON CONFLICT(call_date) DO UPDATE SET
            call_count = call_count + excluded.call_count
            """,
            (today, count),
        )
        conn.commit()
        
        return self.get_api_calls_today()

    def get_all_fetched_symbols(self) -> list[str]:
        """Get all symbols that have been fetched.
        
        Returns:
            List of unique symbols
        """
        conn = self._ensure_connected()
        cursor = conn.execute("SELECT DISTINCT symbol FROM fetch_log")
        return [row[0] for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ============================================================================
# Alpha Vantage API Client
# ============================================================================


class AlphaVantageClient(Protocol):
    """Protocol for Alpha Vantage API client."""

    def fetch_income_statement(self, symbol: str) -> dict[str, Any] | None:
        """Fetch income statement data for a symbol."""
        ...

    def fetch_balance_sheet(self, symbol: str) -> dict[str, Any] | None:
        """Fetch balance sheet data for a symbol."""
        ...


class RealAlphaVantageClient:
    """Real Alpha Vantage API client with rate limiting."""

    def __init__(
        self,
        api_key: str,
        index: FundamentalsIndex,
        daily_limit: int = 25,
        request_delay: float = 12.0,  # ~5 requests/minute for free tier
    ):
        """Initialize the client.
        
        Args:
            api_key: Alpha Vantage API key
            index: FundamentalsIndex for rate limit tracking
            daily_limit: Maximum API calls per day
            request_delay: Seconds to wait between requests
        """
        self.api_key = api_key
        self.index = index
        self.daily_limit = daily_limit
        self.request_delay = request_delay
        self._last_request_time: float = 0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()

    def _check_daily_limit(self) -> bool:
        """Check if daily API limit has been reached.
        
        Returns:
            True if we can make more calls, False if limit reached
        """
        calls_today = self.index.get_api_calls_today()
        return calls_today < self.daily_limit

    def _fetch_endpoint(self, function: str, symbol: str) -> dict[str, Any] | None:
        """Fetch data from Alpha Vantage API.
        
        Args:
            function: API function name (INCOME_STATEMENT, BALANCE_SHEET)
            symbol: Stock ticker
            
        Returns:
            API response as dict, or None if rate limited/error
        """
        import requests
        
        if not self._check_daily_limit():
            return None
        
        self._rate_limit()
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for error responses
            if "Error Message" in data or "Note" in data:
                # "Note" typically means rate limit hit
                return None
            
            self.index.increment_api_calls()
            return data
            
        except Exception:
            return None

    def fetch_income_statement(self, symbol: str) -> dict[str, Any] | None:
        """Fetch income statement data for a symbol."""
        return self._fetch_endpoint("INCOME_STATEMENT", symbol)

    def fetch_balance_sheet(self, symbol: str) -> dict[str, Any] | None:
        """Fetch balance sheet data for a symbol."""
        return self._fetch_endpoint("BALANCE_SHEET", symbol)


# ============================================================================
# File storage helpers
# ============================================================================


def get_fundamentals_dir(base_path: Path, symbol: str) -> Path:
    """Get directory for a symbol's fundamental data.
    
    Args:
        base_path: Base data directory
        symbol: Stock ticker
        
    Returns:
        Path to symbol's fundamentals directory
    """
    return base_path / "raw" / "fundamentals" / symbol


def save_raw_response(
    base_path: Path,
    symbol: str,
    endpoint: str,
    data: dict[str, Any],
) -> Path:
    """Save raw API response to JSON file.
    
    Args:
        base_path: Base data directory
        symbol: Stock ticker
        endpoint: "income_statement" or "balance_sheet"
        data: Raw API response
        
    Returns:
        Path where file was saved
    """
    dir_path = get_fundamentals_dir(base_path, symbol)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    file_path = dir_path / f"{endpoint}.json"
    
    # Add metadata to the saved file
    wrapped_data = {
        "symbol": symbol,
        "endpoint": endpoint,
        "fetched_at": datetime.now(UTC).isoformat(),
        "response": data,
    }
    
    with open(file_path, "w") as f:
        json.dump(wrapped_data, f, indent=2)
    
    return file_path


def load_raw_response(
    base_path: Path,
    symbol: str,
    endpoint: str,
) -> dict[str, Any] | None:
    """Load raw API response from JSON file.
    
    Args:
        base_path: Base data directory
        symbol: Stock ticker
        endpoint: "income_statement" or "balance_sheet"
        
    Returns:
        Wrapped data dict with "response" key, or None if not found
    """
    file_path = get_fundamentals_dir(base_path, symbol) / f"{endpoint}.json"
    
    if not file_path.exists():
        return None
    
    with open(file_path) as f:
        return json.load(f)


# ============================================================================
# Statement parsing
# ============================================================================


def parse_quarterly_statements(
    symbol: str,
    endpoint: str,
    raw_response: dict[str, Any],
) -> list[QuarterlyStatement]:
    """Parse quarterly statements from raw API response.
    
    Args:
        symbol: Stock ticker
        endpoint: "income_statement" or "balance_sheet"
        raw_response: Raw API response dict
        
    Returns:
        List of QuarterlyStatement objects sorted by date (newest first)
    """
    api_data = raw_response.get("response", raw_response)
    quarterly_reports = api_data.get("quarterlyReports", [])
    
    statements = []
    for report in quarterly_reports:
        fiscal_date = report.get("fiscalDateEnding", "")
        currency = report.get("reportedCurrency", "USD")
        
        if fiscal_date:
            statements.append(QuarterlyStatement(
                symbol=symbol,
                statement_type=endpoint,
                fiscal_date_ending=fiscal_date,
                reported_currency=currency,
                raw_data=report,
            ))
    
    # Sort by date descending (newest first)
    statements.sort(key=lambda s: s.fiscal_date_ending, reverse=True)
    return statements


def get_statement_as_of(
    statements: list[QuarterlyStatement],
    as_of_date: str,
) -> QuarterlyStatement | None:
    """Get the most recent statement as of a given date.
    
    This ensures point-in-time correctness - we only use data that
    would have been available on the as_of_date.
    
    Args:
        statements: List of statements sorted by date descending
        as_of_date: YYYY-MM-DD date string
        
    Returns:
        Most recent statement with fiscal_date_ending <= as_of_date
    """
    for stmt in statements:
        if stmt.fiscal_date_ending <= as_of_date:
            return stmt
    return None


# ============================================================================
# Ratio computation
# ============================================================================


def compute_ratios(
    income: QuarterlyStatement | None,
    balance: QuarterlyStatement | None,
) -> FundamentalRatios | None:
    """Compute financial ratios from income statement and balance sheet.
    
    Args:
        income: Quarterly income statement
        balance: Quarterly balance sheet (should be same fiscal period)
        
    Returns:
        FundamentalRatios or None if insufficient data
    """
    if income is None and balance is None:
        return None
    
    symbol = income.symbol if income else balance.symbol  # type: ignore
    as_of_date = income.fiscal_date_ending if income else balance.fiscal_date_ending  # type: ignore
    
    # Initialize all ratios as None
    gross_margin = None
    operating_margin = None
    net_margin = None
    current_ratio = None
    debt_to_equity = None
    
    # Compute profitability ratios from income statement
    if income:
        total_revenue = income.get_value("totalRevenue")
        gross_profit = income.get_value("grossProfit")
        operating_income = income.get_value("operatingIncome")
        net_income = income.get_value("netIncome")
        
        if total_revenue and total_revenue > 0:
            if gross_profit is not None:
                gross_margin = float(gross_profit / total_revenue)
            if operating_income is not None:
                operating_margin = float(operating_income / total_revenue)
            if net_income is not None:
                net_margin = float(net_income / total_revenue)
    
    # Compute liquidity and leverage from balance sheet
    if balance:
        total_current_assets = balance.get_value("totalCurrentAssets")
        total_current_liabilities = balance.get_value("totalCurrentLiabilities")
        total_debt = balance.get_value("shortLongTermDebtTotal")
        shareholder_equity = balance.get_value("totalShareholderEquity")
        
        if total_current_assets and total_current_liabilities and total_current_liabilities > 0:
            current_ratio = float(total_current_assets / total_current_liabilities)
        
        if total_debt is not None and shareholder_equity and shareholder_equity > 0:
            debt_to_equity = float(total_debt / shareholder_equity)
    
    return FundamentalRatios(
        symbol=symbol,
        as_of_date=as_of_date,
        gross_margin=round(gross_margin, 4) if gross_margin is not None else None,
        operating_margin=round(operating_margin, 4) if operating_margin is not None else None,
        net_margin=round(net_margin, 4) if net_margin is not None else None,
        current_ratio=round(current_ratio, 4) if current_ratio is not None else None,
        debt_to_equity=round(debt_to_equity, 4) if debt_to_equity is not None else None,
    )


# ============================================================================
# Main fetcher class
# ============================================================================


@dataclass
class FundamentalsResult:
    """Result of fundamentals fetch operation."""
    
    symbol: str
    income_statements: list[QuarterlyStatement]
    balance_sheets: list[QuarterlyStatement]
    from_cache: bool
    api_calls_made: int
    api_calls_remaining: int


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


