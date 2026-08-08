"""Main FundamentalsFetcher orchestration (SEC-first router)."""

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
from brain_api.core.fundamentals.refresh_policy import (
    RefreshAction,
    decide_refresh_action,
)
from brain_api.core.fundamentals.sec_eligibility import (
    FilingHead,
    SECEligibilityClient,
)
from brain_api.core.fundamentals.sec_filings import (
    FilingAvailabilityProvider,
    SECFilingAvailabilityClient,
    enrich_statement_periods_with_filing_availability,
)
from brain_api.core.fundamentals.sec_statements import (
    CompanyFactsClient,
    SECStatementError,
    build_statement_payloads_from_companyfacts,
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


def _cache_provenance_heads(payload: dict[str, Any] | None) -> set[tuple[str, str]]:
    """Collect (filingDate, accessionNumber) pairs from cached quarterly reports."""
    heads: set[tuple[str, str]] = set()
    if payload is None:
        return heads
    for report in payload.get("quarterlyReports", []) or []:
        if not isinstance(report, dict):
            continue
        filing_date = report.get("filingDate")
        accession = report.get("accessionNumber")
        if filing_date and accession:
            heads.add((str(filing_date), str(accession)))
    return heads


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


def has_usable_cached_quarters(base_path: Path, symbol: str) -> bool:
    """Return whether both statement caches have at least one quarterly report."""
    income = _response_payload(load_raw_response(base_path, symbol, "income_statement"))
    balance = _response_payload(load_raw_response(base_path, symbol, "balance_sheet"))
    return _has_quarterly_reports(income) and _has_quarterly_reports(balance)


def cache_behind_filing_head(
    base_path: Path,
    symbol: str,
    head: FilingHead,
) -> bool:
    """True when cache lacks the head (filingDate, accession) identity."""
    income = _response_payload(load_raw_response(base_path, symbol, "income_statement"))
    balance = _response_payload(load_raw_response(base_path, symbol, "balance_sheet"))
    heads = _cache_provenance_heads(income) | _cache_provenance_heads(balance)
    return (head.filing_date, head.accession_number) not in heads


class FundamentalsConfigurationError(RuntimeError):
    """Raised when point-in-time fundamentals configuration is incomplete."""


class FundamentalsProviderError(RuntimeError):
    """Raised when a provider cannot provide usable fundamentals evidence."""


class FundamentalsFetcher:
    """Fetch and cache fundamental data via SEC-first router (AV for non-eligible)."""

    def __init__(
        self,
        api_key: str,
        base_path: Path,
        cache_dir: Path | None = None,
        daily_limit: int = 25,
        filing_provider: FilingAvailabilityProvider | None = None,
        eligibility_client: SECEligibilityClient | None = None,
        companyfacts_client: CompanyFactsClient | None = None,
    ):
        self.base_path = base_path
        self.cache_dir = cache_dir or (base_path / "cache")
        sec_user_agent = os.environ.get("SEC_USER_AGENT", "")
        self.filing_provider = filing_provider
        if self.filing_provider is None and sec_user_agent:
            self.filing_provider = SECFilingAvailabilityClient(sec_user_agent)
        self.eligibility_client = eligibility_client
        if self.eligibility_client is None and sec_user_agent:
            self.eligibility_client = SECEligibilityClient(sec_user_agent)
        self.companyfacts_client = companyfacts_client
        if self.companyfacts_client is None and sec_user_agent:
            self.companyfacts_client = CompanyFactsClient(sec_user_agent)

        self.index = FundamentalsIndex(self.cache_dir)
        self.client = RealAlphaVantageClient(
            api_key=api_key,
            index=self.index,
            daily_limit=daily_limit,
        )
        self._pending_new_filing: set[str] = set()

    def _mark_pending(self, symbol: str) -> None:
        self.index.mark_pending_new_filing(symbol)
        self._pending_new_filing.add(symbol.upper())

    def _clear_pending(self, symbol: str) -> None:
        self.index.clear_pending_new_filing(symbol)
        self._pending_new_filing.discard(symbol.upper())

    def decide_action_for_symbol(
        self,
        symbol: str,
        *,
        force_refresh: bool = False,
    ) -> RefreshAction:
        """Classify refresh action for one symbol (may call SEC head-check)."""
        if self.eligibility_client is None and not force_refresh:
            # Without SEC identity we can only pull when missing/force
            if force_refresh or not has_usable_cached_quarters(self.base_path, symbol):
                return RefreshAction.PULL
            if cached_fundamentals_require_sec_enrichment(self.base_path, symbol):
                raise FundamentalsConfigurationError(
                    "SEC filing availability is required for fundamentals; "
                    "set SEC_USER_AGENT or inject a filing_provider"
                )
            return RefreshAction.SKIP

        has_usable = has_usable_cached_quarters(self.base_path, symbol)
        unprovenanced = has_usable and cached_fundamentals_require_sec_enrichment(
            self.base_path, symbol
        )
        has_cik = False
        behind = False
        if self.eligibility_client is not None:
            try:
                eligibility = self.eligibility_client.classify(symbol)
            except Exception as exc:
                raise FundamentalsProviderError(
                    f"SEC eligibility/head-check failed for {symbol}: {exc}"
                ) from exc
            has_cik = eligibility.cik is not None
            if has_cik:
                try:
                    head = self.eligibility_client.fetch_filing_head(
                        symbol, sec_eligible=eligibility.sec_eligible
                    )
                    behind = cache_behind_filing_head(self.base_path, symbol, head)
                except Exception as exc:
                    raise FundamentalsProviderError(
                        f"SEC filing head-check failed for {symbol}: {exc}"
                    ) from exc

        return decide_refresh_action(
            force_refresh=force_refresh,
            has_usable_quarters=has_usable,
            has_cik=has_cik,
            behind_head=behind,
            unprovenanced=unprovenanced,
        )

    def _enrich_payloads(
        self,
        symbol: str,
        payloads: dict[str, dict[str, Any] | None],
    ) -> set[str]:
        unresolved = {
            endpoint
            for endpoint, payload in payloads.items()
            if payload is not None and _has_unresolved_quarterly_filings(payload)
        }
        if not unresolved:
            return set()
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
        return unresolved

    def _save_payloads(
        self,
        symbol: str,
        payloads: dict[str, dict[str, Any]],
        endpoints: set[str],
    ) -> None:
        for endpoint in endpoints:
            payload = payloads[endpoint]
            file_path = save_raw_response(self.base_path, symbol, endpoint, payload)
            quarterly = payload.get("quarterlyReports", [])
            annual = payload.get("annualReports", [])
            latest_q = quarterly[0].get("fiscalDateEnding") if quarterly else None
            latest_a = annual[0].get("fiscalDateEnding") if annual else None
            self.index.record_fetch(
                symbol, endpoint, str(file_path), latest_a, latest_q
            )

    def _pull_sec(
        self, symbol: str, *, expected_head: FilingHead | None
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        if self.eligibility_client is None or self.companyfacts_client is None:
            raise FundamentalsConfigurationError(
                "SEC_USER_AGENT is required for SEC CompanyFacts pulls"
            )
        eligibility = self.eligibility_client.classify(symbol)
        if not eligibility.sec_eligible or eligibility.cik is None:
            raise FundamentalsProviderError(
                f"{symbol} is not SEC-eligible for CompanyFacts pull"
            )
        facts = self.companyfacts_client.fetch_companyfacts(eligibility.cik)
        try:
            income, balance = build_statement_payloads_from_companyfacts(
                facts, symbol=symbol, cik=eligibility.cik
            )
        except SECStatementError as exc:
            raise FundamentalsProviderError(str(exc)) from exc

        pulled_newer = True
        if expected_head is not None:
            heads = _cache_provenance_heads(income) | _cache_provenance_heads(balance)
            pulled_newer = (
                expected_head.filing_date,
                expected_head.accession_number,
            ) in heads or any(d > expected_head.filing_date for d, _a in heads)
        return income, balance, pulled_newer

    def _pull_av_atomic(self, symbol: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fetch both AV statements; raise without writing if either fails."""
        try:
            raw_income = self.client.fetch_income_statement(symbol)
        except AlphaVantageProviderError as exc:
            raise FundamentalsProviderError(str(exc)) from exc
        if not _has_quarterly_reports(raw_income):
            raise FundamentalsProviderError(
                "Alpha Vantage returned no usable quarterly income statement "
                f"for {symbol}"
            )
        try:
            raw_balance = self.client.fetch_balance_sheet(symbol)
        except AlphaVantageProviderError as exc:
            raise FundamentalsProviderError(str(exc)) from exc
        if not _has_quarterly_reports(raw_balance):
            raise FundamentalsProviderError(
                f"Alpha Vantage returned no usable quarterly balance sheet for {symbol}"
            )
        raw_income = {**raw_income, "provider": "alpha_vantage"}
        raw_balance = {**raw_balance, "provider": "alpha_vantage"}
        return raw_income, raw_balance

    def fetch_symbol(
        self,
        symbol: str,
        force_refresh: bool = False,
    ) -> FundamentalsResult:
        """Fetch fundamental data for a symbol using the SEC-first router."""
        api_calls_made = 0
        action = self.decide_action_for_symbol(symbol, force_refresh=force_refresh)

        if action == RefreshAction.SKIP:
            income_data = load_raw_response(self.base_path, symbol, "income_statement")
            balance_data = load_raw_response(self.base_path, symbol, "balance_sheet")
            income_statements = (
                parse_quarterly_statements(symbol, "income_statement", income_data)
                if income_data
                else []
            )
            balance_sheets = (
                parse_quarterly_statements(symbol, "balance_sheet", balance_data)
                if balance_data
                else []
            )
            calls_today = self.index.get_api_calls_today()
            return FundamentalsResult(
                symbol=symbol,
                income_statements=income_statements,
                balance_sheets=balance_sheets,
                from_cache=True,
                api_calls_made=0,
                api_calls_remaining=max(0, self.client.daily_limit - calls_today),
            )

        expected_head: FilingHead | None = None
        sec_eligible = False
        if self.eligibility_client is not None:
            eligibility = self.eligibility_client.classify(symbol)
            sec_eligible = eligibility.sec_eligible
            if eligibility.cik is not None:
                expected_head = self.eligibility_client.fetch_filing_head(
                    symbol, sec_eligible=sec_eligible
                )

        if action == RefreshAction.ENRICH_ONLY:
            income_payload = _response_payload(
                load_raw_response(self.base_path, symbol, "income_statement")
            )
            balance_payload = _response_payload(
                load_raw_response(self.base_path, symbol, "balance_sheet")
            )
            payloads = {
                "income_statement": income_payload,
                "balance_sheet": balance_payload,
            }
            unresolved = self._enrich_payloads(symbol, payloads)
            to_save = {
                ep: payloads[ep] for ep in unresolved if payloads[ep] is not None
            }
            self._save_payloads(symbol, to_save, set(to_save))  # type: ignore[arg-type]
            income_data = {"response": payloads["income_statement"]}
            balance_data = {"response": payloads["balance_sheet"]}
        else:
            # PULL
            if sec_eligible:
                income, balance, pulled_newer = self._pull_sec(
                    symbol, expected_head=expected_head
                )
                if expected_head is not None and not pulled_newer:
                    self._mark_pending(symbol)
                    # Keep prior cache; surface via pending set for refresh result
                    income_data = load_raw_response(
                        self.base_path, symbol, "income_statement"
                    )
                    balance_data = load_raw_response(
                        self.base_path, symbol, "balance_sheet"
                    )
                    if income_data is None or balance_data is None:
                        raise FundamentalsProviderError(
                            f"SEC head newer for {symbol} but CompanyFacts "
                            "returned no matching new period and no prior cache"
                        )
                else:
                    self._clear_pending(symbol)
                    payloads = {
                        "income_statement": income,
                        "balance_sheet": balance,
                    }
                    self._save_payloads(
                        symbol, payloads, {"income_statement", "balance_sheet"}
                    )
                    income_data = {"response": income}
                    balance_data = {"response": balance}
            else:
                income, balance = self._pull_av_atomic(symbol)
                api_calls_made = 2
                payloads_opt: dict[str, dict[str, Any] | None] = {
                    "income_statement": income,
                    "balance_sheet": balance,
                }
                self._enrich_payloads(symbol, payloads_opt)
                assert payloads_opt["income_statement"] is not None
                assert payloads_opt["balance_sheet"] is not None
                to_save = {
                    "income_statement": payloads_opt["income_statement"],
                    "balance_sheet": payloads_opt["balance_sheet"],
                }
                if expected_head is not None:
                    heads = _cache_provenance_heads(
                        to_save["income_statement"]
                    ) | _cache_provenance_heads(to_save["balance_sheet"])
                    if (
                        expected_head.filing_date,
                        expected_head.accession_number,
                    ) not in heads and not any(
                        d > expected_head.filing_date for d, _a in heads
                    ):
                        self._mark_pending(symbol)
                    else:
                        self._clear_pending(symbol)
                self._save_payloads(
                    symbol, to_save, {"income_statement", "balance_sheet"}
                )
                income_data = {"response": to_save["income_statement"]}
                balance_data = {"response": to_save["balance_sheet"]}

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
            from_cache=api_calls_made == 0 and action == RefreshAction.ENRICH_ONLY,
            api_calls_made=api_calls_made,
            api_calls_remaining=max(0, self.client.daily_limit - calls_today),
        )

    def get_ratios(
        self,
        symbol: str,
        as_of_date: str,
    ) -> FundamentalRatios | None:
        """Get financial ratios for a symbol as of a specific date."""
        income_data = load_raw_response(self.base_path, symbol, "income_statement")
        balance_data = load_raw_response(self.base_path, symbol, "balance_sheet")

        if income_data is None and balance_data is None:
            return None

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
        """Get current API usage status (observability; not a local gate)."""
        calls_today = self.index.get_api_calls_today()
        return {
            "calls_today": calls_today,
            "daily_limit": self.client.daily_limit,
            "remaining": max(0, self.client.daily_limit - calls_today),
        }

    def get_cached_symbols(self) -> list[str]:
        """Get list of symbols with cached data."""
        return self.index.get_all_fetched_symbols()

    def close(self) -> None:
        """Close database connections."""
        self.index.close()
