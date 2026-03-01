"""Scrape Nifty 500 Shariah index constituents from NSE India JSON API.

Uses the same API that powers the nseindia.com website frontend:
    GET https://www.nseindia.com/api/equity-stockIndices?index=NIFTY500%20SHARIAH

Requires session management: NSE blocks direct API calls without cookies.
A browser-like session must first hit a page that sets the required cookies
(nse_ak, bm_sv, etc.), then the API endpoint returns JSON.

NSE is aggressive about bot detection. The session may need multiple
retries with fresh sessions if the first attempt gets a captcha/HTML page.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

NSE_BASE_URL = "https://www.nseindia.com"
NSE_INDEX_API = f"{NSE_BASE_URL}/api/equity-stockIndices"
NIFTY500_SHARIAH_INDEX = "NIFTY500 SHARIAH"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

SESSION_TIMEOUT = 15
API_TIMEOUT = 30
MAX_SESSION_RETRIES = 3
RETRY_DELAY_S = 2.0


class NseFetchError(Exception):
    """Raised when fetching data from NSE India API fails."""


def _create_nse_session() -> requests.Session:
    """Create a requests session with NSE-compatible headers and cookies."""
    session = requests.Session()
    session.headers.update(HEADERS)

    session.get(NSE_BASE_URL, timeout=SESSION_TIMEOUT)
    time.sleep(1)

    return session


def _fetch_index_data(session: requests.Session) -> dict:
    """Fetch index data from NSE API using an established session."""
    resp = session.get(
        NSE_INDEX_API,
        params={"index": NIFTY500_SHARIAH_INDEX},
        headers={
            "Referer": "https://www.nseindia.com/market-data/live-equity-market",
        },
        timeout=API_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def scrape_nifty500_shariah() -> list[dict]:
    """Fetch Nifty 500 Shariah index constituents from NSE India.

    Retries with fresh sessions up to MAX_SESSION_RETRIES times if NSE
    returns non-JSON (HTML captcha/block pages).

    Returns:
        List of dicts with keys: symbol, name, industry.
        Typically ~100-150 stocks.

    Raises:
        NseFetchError: On HTTP errors, empty response, or session failure.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_SESSION_RETRIES + 1):
        try:
            session = _create_nse_session()
        except requests.RequestException as e:
            last_error = e
            logger.warning(
                f"NSE session attempt {attempt}/{MAX_SESSION_RETRIES} failed: {e}"
            )
            if attempt < MAX_SESSION_RETRIES:
                time.sleep(RETRY_DELAY_S * attempt)
            continue

        try:
            payload = _fetch_index_data(session)
        except (requests.RequestException, ValueError) as e:
            last_error = e
            logger.warning(
                f"NSE API attempt {attempt}/{MAX_SESSION_RETRIES} failed: {e}"
            )
            if attempt < MAX_SESSION_RETRIES:
                time.sleep(RETRY_DELAY_S * attempt)
            continue

        raw_data = payload.get("data", [])
        if not raw_data:
            raise NseFetchError(
                f"NSE API returned empty data array for index '{NIFTY500_SHARIAH_INDEX}'"
            )

        constituents: list[dict] = []
        for entry in raw_data:
            symbol = entry.get("symbol", "")
            if not symbol or " " in symbol:
                continue

            constituents.append(
                {
                    "symbol": symbol,
                    "name": entry.get("meta", {}).get("companyName", "")
                    or entry.get("companyName", ""),
                    "industry": entry.get("meta", {}).get("industry", "")
                    or entry.get("industry", ""),
                }
            )

        if not constituents:
            raise NseFetchError(
                f"No valid constituents found after filtering for '{NIFTY500_SHARIAH_INDEX}'"
            )

        sym_list = [c["symbol"] for c in constituents]
        preview = (
            f"{sym_list[:20]}... (+{len(sym_list) - 20} more)"
            if len(sym_list) > 20
            else str(sym_list)
        )
        logger.info(
            f"Nifty 500 Shariah: fetched {len(constituents)} constituents "
            f"from NSE API (attempt {attempt}): {preview}"
        )
        return constituents

    raise NseFetchError(
        f"All {MAX_SESSION_RETRIES} NSE session attempts failed. Last error: {last_error}"
    )
