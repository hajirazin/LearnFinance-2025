"""Stock filtering and factor scoring for universe construction.

Provides pure functions for:
- Fetching fundamental metrics from yfinance (sequential with rate-limit safety)
- Junk filter: ROE > 0, Price > SMA200, Beta < 2
- Factor scoring: 0.4 * Momentum + 0.3 * Quality + 0.3 * Value

Sequential fetching with retry avoids Yahoo Finance 429 rate limits.
Monthly universe caching (see cache.py) means this only runs once per month.
"""

import logging
import threading
import time

import yfinance as yf

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
DELAY_BETWEEN_SYMBOLS_S = 1.0
API_FAILURE_THRESHOLD = 0.20


class YFinanceFetchError(Exception):
    """Raised when too many yfinance API calls fail, indicating rate limiting."""


def fetch_stock_metrics(
    symbols: list[str],
    shutdown_event: threading.Event | None = None,
) -> dict[str, dict]:
    """Fetch fundamental metrics for each symbol using yfinance.

    Uses a single batch yf.download() for 6-month price history (momentum),
    then sequential yf.Ticker().info calls with retry for fundamentals.

    Raises YFinanceFetchError if more than 20% of symbols fail due to API
    errors (as opposed to genuinely missing data).

    Args:
        symbols: List of stock ticker symbols.
        shutdown_event: If set, the loop stops early and returns partial results.

    Returns:
        Mapping of symbol -> metrics dict with keys:
        roe, price, sma200, beta, gross_margin, roic,
        earnings_yield, six_month_return.

    Raises:
        YFinanceFetchError: When API error rate exceeds threshold.
    """
    if not symbols:
        return {}

    six_month_returns = _fetch_six_month_returns(symbols)

    logger.info(f"Fetching .info for {len(symbols)} stocks (sequential)...")

    metrics: dict[str, dict] = {}
    api_error_count = 0
    total = len(symbols)

    for i, sym in enumerate(symbols, 1):
        if shutdown_event and shutdown_event.is_set():
            logger.warning(f"Shutdown requested after {i - 1}/{total} stocks, aborting")
            raise YFinanceFetchError(
                f"Shutdown requested after {i - 1}/{total} stocks fetched."
            )

        data, was_api_error = _fetch_one_with_retry(sym, six_month_returns)
        metrics[sym] = data
        if was_api_error:
            api_error_count += 1

        if i % 20 == 0:
            logger.info(f"  {i}/{total} done (api_errors={api_error_count})...")

        if i < total:
            time.sleep(DELAY_BETWEEN_SYMBOLS_S)

    pct = (api_error_count / total * 100) if total > 0 else 0
    logger.info(
        f"yfinance API errors: {api_error_count}/{total} stocks failed ({pct:.1f}%). "
        f"Threshold is {API_FAILURE_THRESHOLD * 100:.0f}%."
    )

    if total > 0 and (api_error_count / total) > API_FAILURE_THRESHOLD:
        raise YFinanceFetchError(
            f"{api_error_count}/{total} stocks failed due to yfinance API errors "
            f"({pct:.1f}%). Aborting to avoid garbage universe."
        )

    logger.info(f"Fetched metrics for {len(metrics)} stocks")
    return metrics


def _fetch_one_with_retry(
    sym: str,
    six_month_returns: dict[str, float | None],
) -> tuple[dict, bool]:
    """Fetch fundamental metrics for a single symbol with retry.

    Args:
        sym: Stock ticker symbol.
        six_month_returns: Pre-computed 6-month returns mapping.

    Returns:
        (metrics_dict, was_api_error) -- empty dict + True if all retries failed
        due to API errors; empty dict + False if the stock genuinely has no data.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            info = yf.Ticker(sym).info

            if not info or info.get("regularMarketPrice") is None:
                return {}, False

            return _extract_metrics(info, sym, six_month_returns), False

        except Exception as e:
            if attempt < MAX_RETRIES:
                delay = 2 ** (attempt + 1)
                logger.warning(
                    f"Retry {attempt + 1}/{MAX_RETRIES} for {sym} in {delay}s: {e}"
                )
                time.sleep(delay)
            else:
                logger.error(f"All {MAX_RETRIES} retries exhausted for {sym}: {e}")
                return {}, True


def _extract_metrics(
    info: dict,
    sym: str,
    six_month_returns: dict[str, float | None],
) -> dict:
    """Extract fundamental metrics from a yfinance .info dict.

    Args:
        info: Raw yfinance ticker info dict.
        sym: Symbol for six_month_returns lookup.
        six_month_returns: Pre-computed 6-month returns mapping.

    Returns:
        Metrics dict with roe, price, sma200, beta, etc.
    """
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    op_margins = info.get("operatingMargins")
    total_rev = info.get("totalRevenue")
    ev = info.get("enterpriseValue")
    book = info.get("bookValue")
    shares = info.get("sharesOutstanding")
    total_debt = info.get("totalDebt") or 0
    total_cash = info.get("totalCash") or 0

    # ROIC ≈ NOPAT / Invested Capital
    # NOPAT = Operating Income x 0.75 (assumed 25% tax)
    # Invested Capital ≈ Total Equity + Total Debt - Cash
    roic = None
    if op_margins and total_rev and book and shares:
        operating_income = op_margins * total_rev
        invested_capital = (book * shares) + total_debt - total_cash
        if invested_capital > 0:
            roic = (operating_income * 0.75) / invested_capital

    # Earnings Yield = EBIT / EV
    earnings_yield = None
    if op_margins and total_rev and ev and ev > 0:
        ebit = op_margins * total_rev
        earnings_yield = ebit / ev

    return {
        "roe": info.get("returnOnEquity"),
        "price": price,
        "sma200": info.get("twoHundredDayAverage"),
        "beta": info.get("beta"),
        "gross_margin": info.get("grossMargins"),
        "roic": roic,
        "earnings_yield": earnings_yield,
        "six_month_return": six_month_returns.get(sym),
    }


def _fetch_six_month_returns(symbols: list[str]) -> dict[str, float | None]:
    """Batch-download 6-month price history and compute returns.

    Args:
        symbols: List of stock ticker symbols.

    Returns:
        Mapping of symbol -> 6-month total return (or None).
    """
    logger.info("Downloading 6-month price history (batch, no threading)...")
    hist = yf.download(symbols, period="6mo", progress=False, threads=False)
    six_month_returns: dict[str, float | None] = {}

    if hist.empty:
        return dict.fromkeys(symbols)

    if "Close" in hist.columns:
        close = hist["Close"]
        for sym in symbols:
            try:
                series = close[sym].dropna()
                if len(series) >= 2:
                    ret = (series.iloc[-1] / series.iloc[0]) - 1
                    six_month_returns[sym] = float(ret)
                else:
                    six_month_returns[sym] = None
            except (KeyError, TypeError):
                six_month_returns[sym] = None

    for sym in symbols:
        if sym not in six_month_returns:
            six_month_returns[sym] = None

    return six_month_returns


def apply_junk_filter(
    holdings: list[dict], metrics: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    """Remove stocks that fail basic quality checks.

    Rules:
        1. ROE > 0  (company must be profitable)
        2. Price > 200-day SMA  (above long-term trend)
        3. Beta < 2.0  (exclude wildly volatile stocks)

    Missing metrics are treated as failures (conservative).

    Args:
        holdings: List of holdings dicts with at least 'symbol' key.
        metrics: Mapping of symbol -> metrics dict from fetch_stock_metrics.

    Returns:
        (passed, failed) - both lists of holdings dicts with metrics attached.
        Failed entries include 'fail_reasons' list.
    """
    passed: list[dict] = []
    failed: list[dict] = []

    for h in holdings:
        sym = h["symbol"]
        m = metrics.get(sym, {})
        roe = m.get("roe")
        price = m.get("price")
        sma200 = m.get("sma200")
        beta = m.get("beta")

        reasons: list[str] = []
        if roe is None or roe <= 0:
            reasons.append(f"ROE={roe}")
        if price is None or sma200 is None or price <= sma200:
            reasons.append(f"Price={price} <= SMA200={sma200}")
        if beta is None or beta >= 2.0:
            reasons.append(f"Beta={beta}")

        entry = {**h, "metrics": m}
        if reasons:
            entry["fail_reasons"] = reasons
            failed.append(entry)
        else:
            passed.append(entry)

    return passed, failed


def _percentile_rank(values: list[float | None]) -> list[float | None]:
    """Compute percentile rank (0-1) for each value. None stays None.

    Args:
        values: List of numeric values (or None).

    Returns:
        List of percentile ranks in [0, 1]. None inputs produce None outputs.
    """
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return [None] * len(values)
    sorted_vals = sorted(v for _, v in valid)
    n = len(sorted_vals)
    ranks: list[float | None] = [None] * len(values)
    for i, v in valid:
        pos = sorted_vals.index(v)
        ranks[i] = pos / max(n - 1, 1)
    return ranks


def compute_factor_scores(holdings: list[dict]) -> list[dict]:
    """Rank stocks by composite factor score.

    Score = 0.4 x Momentum + 0.3 x Quality + 0.3 x Value

    Momentum : 6-month return (percentile rank)
    Quality  : ROIC, falling back to Gross Margin (percentile rank)
    Value    : Earnings Yield = EBIT/EV (percentile rank)

    Stocks with any missing component get score=None and are ranked last.

    Args:
        holdings: List of holdings dicts, each with a 'metrics' sub-dict
                  (output of apply_junk_filter passed entries).

    Returns:
        Holdings sorted by score descending, with factor_score and
        factor_components attached.
    """
    momentum_raw = [h.get("metrics", {}).get("six_month_return") for h in holdings]
    quality_raw: list[float | None] = []
    for h in holdings:
        m = h.get("metrics", {})
        quality_raw.append(
            m.get("roic") if m.get("roic") is not None else m.get("gross_margin")
        )
    value_raw = [h.get("metrics", {}).get("earnings_yield") for h in holdings]

    momentum_pct = _percentile_rank(momentum_raw)
    quality_pct = _percentile_rank(quality_raw)
    value_pct = _percentile_rank(value_raw)

    scored: list[dict] = []
    for i, h in enumerate(holdings):
        mp = momentum_pct[i]
        qp = quality_pct[i]
        vp = value_pct[i]

        if mp is not None and qp is not None and vp is not None:
            score = round(0.4 * mp + 0.3 * qp + 0.3 * vp, 6)
        else:
            score = None

        scored.append(
            {
                **h,
                "factor_score": score,
                "factor_components": {
                    "momentum_6m_return": momentum_raw[i],
                    "momentum_pct": mp,
                    "quality_raw": quality_raw[i],
                    "quality_pct": qp,
                    "value_earnings_yield": value_raw[i],
                    "value_pct": vp,
                },
            }
        )

    scored.sort(
        key=lambda x: x.get("factor_score")
        if x.get("factor_score") is not None
        else -1,
        reverse=True,
    )
    return scored
