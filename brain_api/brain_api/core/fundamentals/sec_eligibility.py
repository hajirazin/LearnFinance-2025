"""SEC-eligible US classification (no CompanyFacts pre-download)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from brain_api.core.fundamentals.sec_filings import SECFilingAvailability
from brain_api.core.fundamentals.sec_rate_limit import wait_for_sec_slot

PERIODIC_US = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A"})
PERIODIC_FPI = frozenset({"20-F", "40-F", "20-F/A", "40-F/A"})
HEAD_FORMS_US = PERIODIC_US
HEAD_FORMS_FPI = PERIODIC_US | PERIODIC_FPI


@dataclass(frozen=True)
class FilingHead:
    """Newest applicable SEC periodic filing identity."""

    filing_date: str
    accession_number: str
    form: str
    report_date: str


@dataclass(frozen=True)
class EligibilityResult:
    """Classification outcome for one symbol."""

    symbol: str
    cik: str | None
    sec_eligible: bool
    recent_forms: tuple[str, ...]


class SECEligibilityClient:
    """Classify SEC-eligible US names from CIK + recent periodic forms."""

    _TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    _SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(
        self,
        user_agent: str,
        timeout_seconds: float = 30.0,
        majority_window: int = 8,
    ):
        if not user_agent.strip():
            raise ValueError("SEC user agent must identify the requesting application")
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.majority_window = majority_window
        self._cik_by_ticker: dict[str, str] | None = None

    def _get_json(self, url: str) -> dict[str, Any]:
        wait_for_sec_slot()
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

    def resolve_cik(self, symbol: str) -> str | None:
        """Return zero-padded CIK or None."""
        return self._load_cik_registry().get(symbol.upper())

    def _recent_forms_and_rows(
        self, cik: str
    ) -> tuple[list[str], list[tuple[str, str, str, str]]]:
        payload = self._get_json(self._SUBMISSIONS_URL.format(cik=cik))
        filings = payload.get("filings", {})
        if not isinstance(filings, dict):
            raise ValueError(f"SEC submissions payload has no filings for CIK {cik}")
        recent = filings.get("recent", {})
        if not isinstance(recent, dict):
            raise ValueError(
                f"SEC submissions payload has no recent filings for CIK {cik}"
            )
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        report_dates = recent.get("reportDate", [])
        if not all(
            isinstance(col, list)
            for col in (forms, filing_dates, accessions, report_dates)
        ):
            raise ValueError(f"SEC recent filings columns malformed for CIK {cik}")
        rows: list[tuple[str, str, str, str]] = []
        for form, filing_date, accession, report_date in zip(
            forms, filing_dates, accessions, report_dates, strict=False
        ):
            if not form:
                continue
            rows.append(
                (
                    str(form),
                    str(filing_date or ""),
                    str(accession or ""),
                    str(report_date or ""),
                )
            )
        return [r[0] for r in rows], rows

    @staticmethod
    def classify_sec_eligible(recent_forms: list[str], *, window: int = 8) -> bool:
        """Majority pin among last N periodic US vs FPI forms.

        Thin history uses whatever periodic forms exist (``periodic[:window]``).
        Empty periodic history → not eligible. A US/FPI tie (``us == fpi``) is
        SEC-eligible via ``us >= fpi`` (and ``us > 0``).
        """
        periodic = [f for f in recent_forms if f in PERIODIC_US or f in PERIODIC_FPI]
        windowed = periodic[:window]
        if not windowed:
            return False
        us = sum(1 for f in windowed if f in PERIODIC_US)
        fpi = sum(1 for f in windowed if f in PERIODIC_FPI)
        return us > 0 and us >= fpi

    def classify(self, symbol: str) -> EligibilityResult:
        """Classify symbol without downloading CompanyFacts."""
        cik = self.resolve_cik(symbol)
        if cik is None:
            return EligibilityResult(
                symbol=symbol.upper(),
                cik=None,
                sec_eligible=False,
                recent_forms=(),
            )
        forms, _rows = self._recent_forms_and_rows(cik)
        eligible = self.classify_sec_eligible(forms, window=self.majority_window)
        return EligibilityResult(
            symbol=symbol.upper(),
            cik=cik,
            sec_eligible=eligible,
            recent_forms=tuple(forms[: self.majority_window]),
        )

    def fetch_filing_head(self, symbol: str, *, sec_eligible: bool) -> FilingHead:
        """Return newest applicable periodic filing head for freshness checks."""
        cik = self.resolve_cik(symbol)
        if cik is None:
            raise ValueError(f"No SEC CIK mapping found for {symbol}")
        allowed = HEAD_FORMS_US if sec_eligible else HEAD_FORMS_FPI
        _forms, rows = self._recent_forms_and_rows(cik)
        candidates: list[FilingHead] = []
        for form, filing_date, accession, report_date in rows:
            if form not in allowed:
                continue
            if not (filing_date and accession):
                continue
            candidates.append(
                FilingHead(
                    filing_date=filing_date,
                    accession_number=accession,
                    form=form,
                    report_date=report_date,
                )
            )
        if not candidates:
            raise ValueError(f"No applicable periodic filings found for {symbol}")
        candidates.sort(key=lambda h: (h.filing_date, h.accession_number), reverse=True)
        return candidates[0]


def filings_to_availability(
    heads: list[FilingHead],
    *,
    cik: str,
) -> list[SECFilingAvailability]:
    """Adapt FilingHead rows to SECFilingAvailability for enrichment reuse."""
    out: list[SECFilingAvailability] = []
    for head in heads:
        if not head.report_date:
            continue
        out.append(
            SECFilingAvailability(
                report_date=head.report_date,
                filing_date=head.filing_date,
                accession_number=head.accession_number,
                form=head.form,
                source=(
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{head.accession_number.replace('-', '')}"
                ),
            )
        )
    return out
