"""Scrape holdings from sp-funds.com for SP Funds halal ETFs.

Supports SPUS, SPTE, and SPWO. Parses HTML tables from each ETF page
to extract ticker, name, and weighting for every holding.
"""

import logging

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SP_FUNDS_BASE_URL = "https://www.sp-funds.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def scrape_sp_funds(etf_slug: str) -> list[dict]:
    """Scrape full holdings from an SP Funds ETF page.

    Fetches the HTML page at sp-funds.com/{etf_slug}/ and parses the
    holdings table (identified by a 'StockTicker' column header).

    Args:
        etf_slug: Lowercase ETF identifier (e.g. "spus", "spte", "spwo").

    Returns:
        List of dicts with keys: symbol, name, weight.
        Empty list if the page cannot be fetched or parsed.
    """
    url = f"{SP_FUNDS_BASE_URL}/{etf_slug}/"
    logger.info(f"Scraping SP Funds holdings from {url}")

    resp = requests.get(url, timeout=30, headers=HEADERS)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    for table in soup.find_all("table"):
        header_cells = [th.get_text(strip=True) for th in table.find_all("th")]
        if "StockTicker" not in header_cells:
            continue

        ticker_idx = header_cells.index("StockTicker")
        name_idx = (
            header_cells.index("SecurityName")
            if "SecurityName" in header_cells
            else None
        )
        weight_idx = (
            header_cells.index("Weightings") if "Weightings" in header_cells else None
        )

        holdings: list[dict] = []
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) <= ticker_idx:
                continue

            ticker = cells[ticker_idx].get_text(strip=True)
            if not ticker or ticker == "Cash&Other":
                continue

            name = ""
            if name_idx is not None and len(cells) > name_idx:
                name = cells[name_idx].get_text(strip=True)

            weight = 0.0
            if weight_idx is not None and len(cells) > weight_idx:
                weight_text = cells[weight_idx].get_text(strip=True)
                try:
                    weight = float(weight_text)
                except ValueError:
                    weight = 0.0

            holdings.append({"symbol": ticker, "name": name, "weight": weight})

        logger.info(f"[{etf_slug.upper()}] Scraped {len(holdings)} holdings")
        return holdings

    logger.warning(f"[{etf_slug.upper()}] No holdings table found on page")
    return []
