"""SEC filing-availability enrichment for Alpha Vantage statement periods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import requests


@dataclass(frozen=True)
class SECFilingAvailability:
    """Public availability provenance for one SEC filing."""

    report_date: str
    filing_date: str
    accession_number: str
    form: str
    source: str


class FilingAvailabilityProvider(Protocol):
    """Provider used to resolve filing availability for a ticker."""

    def fetch_symbol_filings(self, symbol: str) -> list[SECFilingAvailability]:
        """Return filing availability records for ``symbol``."""
        ...


class SECFilingAvailabilityClient:
    """Small client for the SEC submissions data API."""

    _TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    _SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(self, user_agent: str, timeout_seconds: float = 30.0):
        if not user_agent.strip():
            raise ValueError("SEC user agent must identify the requesting application")
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self._cik_by_ticker: dict[str, str] | None = None

    def _get_json(self, url: str) -> dict[str, Any]:
        response = requests.get(
            url,
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"SEC returned a non-object payload for {url}")
        return payload

    def _load_cik_registry(self) -> dict[str, str]:
        if self._cik_by_ticker is None:
            payload = self._get_json(self._TICKERS_URL)
            registry: dict[str, str] = {}
            for company in payload.values():
                if not isinstance(company, dict):
                    continue
                ticker = str(company.get("ticker", "")).upper()
                cik = company.get("cik_str")
                if ticker and cik is not None:
                    registry[ticker] = f"{int(cik):010d}"
            self._cik_by_ticker = registry
        return self._cik_by_ticker

    def fetch_symbol_filings(self, symbol: str) -> list[SECFilingAvailability]:
        """Return recent 10-Q/10-K availability records for ``symbol``."""
        cik = self._load_cik_registry().get(symbol.upper())
        if cik is None:
            raise ValueError(f"No SEC CIK mapping found for {symbol}")

        payload = self._get_json(self._SUBMISSIONS_URL.format(cik=cik))
        recent = payload.get("filings", {}).get("recent", {})
        if not isinstance(recent, dict):
            raise ValueError(
                f"SEC submissions payload has no recent filings for {symbol}"
            )

        keys = ("reportDate", "filingDate", "accessionNumber", "form")
        columns = [recent.get(key, []) for key in keys]
        if not all(isinstance(column, list) for column in columns):
            raise ValueError(f"SEC submissions columns are malformed for {symbol}")

        filings = []
        for report_date, filing_date, accession, form in zip(*columns, strict=True):
            if form not in {"10-Q", "10-K", "20-F", "40-F"}:
                continue
            if not all((report_date, filing_date, accession)):
                continue
            filings.append(
                SECFilingAvailability(
                    report_date=str(report_date),
                    filing_date=str(filing_date),
                    accession_number=str(accession),
                    form=str(form),
                    source=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{str(accession).replace('-', '')}",
                )
            )
        return filings


def enrich_statement_periods_with_filing_availability(
    statement_payload: dict[str, Any],
    filings: list[SECFilingAvailability],
) -> dict[str, Any]:
    """Attach exact SEC availability provenance to matching AV periods.

    Periods without an exact SEC ``reportDate`` match remain unresolved and
    are excluded by the point-in-time loader.
    """
    by_report_date: dict[str, SECFilingAvailability] = {}
    for filing in sorted(
        filings,
        key=lambda item: (item.report_date, item.filing_date, item.accession_number),
    ):
        by_report_date.setdefault(filing.report_date, filing)

    for report in statement_payload.get("quarterlyReports", []):
        if not isinstance(report, dict):
            continue
        filing = by_report_date.get(str(report.get("fiscalDateEnding", "")))
        if filing is None:
            continue
        report["filingDate"] = filing.filing_date
        report["accessionNumber"] = filing.accession_number
        report["filingForm"] = filing.form
        report["filingSource"] = filing.source
    return statement_payload
