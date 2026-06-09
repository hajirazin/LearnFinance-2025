"""Scrape Nifty 500 Shariah index constituents from Finology.

NSE India's equity-stockIndices API is protected by Akamai Bot Manager
(requires JavaScript execution to validate the _abck cookie). As a result,
the direct NSE approach is no longer viable without a headless browser.

Finology (ticker.finology.in) exposes the same index constituent data via
an internal AJAX endpoint. The endpoint requires:
  1. An ASP.NET_SessionId cookie obtained by visiting the index page first.
  2. The X-Requested-With: XMLHttpRequest header on the API call.

curl-cffi with Chrome TLS impersonation is used so the session request
looks like a real browser to the server.
"""

import logging
import time

from curl_cffi.requests import Session

logger = logging.getLogger(__name__)

FINOLOGY_INDEX_PAGE = "https://ticker.finology.in/market/index/nse/shariah500"
FINOLOGY_API_URL = "https://ticker.finology.in/GetIndicesCompList.ashx"
SHARIAH500_INDEX_CODE = 161

SESSION_TIMEOUT = 15
API_TIMEOUT = 30
MAX_SESSION_RETRIES = 3
RETRY_DELAY_S = 2.0


class NseFetchError(Exception):
    """Raised when fetching Nifty 500 Shariah constituent data fails."""


def _create_finology_session() -> Session:
    """Create a curl-cffi session with Chrome impersonation and warm it up."""
    session = Session(impersonate="chrome")
    session.get(FINOLOGY_INDEX_PAGE, timeout=SESSION_TIMEOUT)
    time.sleep(1)
    return session


def _fetch_index_data(session: Session) -> list[dict]:
    """Fetch Nifty 500 Shariah constituents from the Finology API."""
    resp = session.get(
        FINOLOGY_API_URL,
        params={"indexcode": SHARIAH500_INDEX_CODE},
        headers={
            "Referer": FINOLOGY_INDEX_PAGE,
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/javascript, */*; q=0.01",
        },
        timeout=API_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def scrape_nifty500_shariah() -> list[dict]:
    """Fetch Nifty 500 Shariah index constituents from Finology.

    Retries with fresh sessions up to MAX_SESSION_RETRIES times on
    network errors or unexpected responses.

    Returns:
        List of dicts with keys: symbol, name, industry.
        Typically ~199 stocks. industry is always empty string
        (Finology does not expose industry classification).

    Raises:
        NseFetchError: On HTTP errors, empty response, or session failure.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_SESSION_RETRIES + 1):
        try:
            session = _create_finology_session()
        except Exception as e:
            last_error = e
            logger.warning(
                f"Finology session attempt {attempt}/{MAX_SESSION_RETRIES} failed: {e}"
            )
            if attempt < MAX_SESSION_RETRIES:
                time.sleep(RETRY_DELAY_S * attempt)
            continue

        try:
            raw_data = _fetch_index_data(session)
        except Exception as e:
            last_error = e
            logger.warning(
                f"Finology API attempt {attempt}/{MAX_SESSION_RETRIES} failed: {e}"
            )
            if attempt < MAX_SESSION_RETRIES:
                time.sleep(RETRY_DELAY_S * attempt)
            continue

        if not raw_data:
            raise NseFetchError(
                f"Finology API returned empty data for Nifty 500 Shariah "
                f"(indexcode={SHARIAH500_INDEX_CODE})"
            )

        constituents = [
            {
                "symbol": entry["symbol"],
                "name": entry.get("compname", ""),
                "industry": "",
            }
            for entry in raw_data
            if entry.get("symbol")
        ]

        if not constituents:
            raise NseFetchError(
                "No valid constituents found in Finology response for Nifty 500 Shariah"
            )

        sym_list = [c["symbol"] for c in constituents]
        preview = (
            f"{sym_list[:20]}... (+{len(sym_list) - 20} more)"
            if len(sym_list) > 20
            else str(sym_list)
        )
        logger.info(
            f"Nifty 500 Shariah: fetched {len(constituents)} constituents "
            f"from Finology (attempt {attempt}): {preview}"
        )
        return constituents

    raise NseFetchError(
        f"All {MAX_SESSION_RETRIES} Finology session attempts failed. "
        f"Last error: {last_error}"
    )
