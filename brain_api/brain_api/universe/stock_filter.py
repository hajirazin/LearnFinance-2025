"""Stock filtering and factor scoring for universe construction.

Provides pure functions for:
- Fetching fundamental metrics from yfinance (batch + threaded)
- Junk filter: ROE > 0, Price > SMA200, Beta < 2
- Factor scoring: 0.4 * Momentum + 0.3 * Quality + 0.3 * Value

Ported from scripts/scrape_halal_holdings.py with proper logging,
type hints, and no global state.
"""

import concurrent.futures
import logging

import yfinance as yf

logger = logging.getLogger(__name__)

MAX_YFINANCE_WORKERS = 16


def fetch_stock_metrics(symbols: list[str]) -> dict[str, dict]:
    """Fetch fundamental metrics for each symbol using yfinance.

    Uses a single batch yf.download() for 6-month price history (momentum),
    then threaded yf.Ticker().info calls for fundamentals.

    Args:
        symbols: List of stock ticker symbols.

    Returns:
        Mapping of symbol -> metrics dict with keys:
        roe, price, sma200, beta, gross_margin, roic,
        earnings_yield, six_month_return.
    """
    if not symbols:
        return {}

    six_month_returns = _fetch_six_month_returns(symbols)

    logger.info(f"Fetching .info for {len(symbols)} stocks (threaded)...")

    def _fetch_one(sym: str) -> tuple[str, dict]:
        try:
            info = yf.Ticker(sym).info
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

            return sym, {
                "roe": info.get("returnOnEquity"),
                "price": price,
                "sma200": info.get("twoHundredDayAverage"),
                "beta": info.get("beta"),
                "gross_margin": info.get("grossMargins"),
                "roic": roic,
                "earnings_yield": earnings_yield,
                "six_month_return": six_month_returns.get(sym),
            }
        except Exception:
            logger.warning(f"Failed to fetch metrics for {sym}", exc_info=True)
            return sym, {}

    metrics: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_YFINANCE_WORKERS
    ) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in symbols}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            sym, data = future.result()
            metrics[sym] = data
            if done % 50 == 0:
                logger.info(f"  {done}/{len(symbols)} done...")

    logger.info(f"Fetched metrics for {len(metrics)} stocks")
    return metrics


def _fetch_six_month_returns(symbols: list[str]) -> dict[str, float | None]:
    """Batch-download 6-month price history and compute returns.

    Args:
        symbols: List of stock ticker symbols.

    Returns:
        Mapping of symbol -> 6-month total return (or None).
    """
    logger.info("Downloading 6-month price history (batch)...")
    hist = yf.download(symbols, period="6mo", progress=False, threads=True)
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
