"""Unit tests for SEC-first fundamentals router primitives."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from brain_api.core.fundamentals.client import (
    AlphaVantageProviderError,
    RealAlphaVantageClient,
)
from brain_api.core.fundamentals.fetcher import (
    FundamentalsFetcher,
    FundamentalsProviderError,
    cache_behind_filing_head,
)
from brain_api.core.fundamentals.refresh_policy import (
    RefreshAction,
    SymbolCacheState,
    decide_refresh_action,
    order_av_pull_queue,
)
from brain_api.core.fundamentals.sec_eligibility import (
    EligibilityResult,
    FilingHead,
    SECEligibilityClient,
)
from brain_api.core.fundamentals.sec_statements import (
    SECStatementError,
    build_statement_payloads_from_companyfacts,
)
from brain_api.core.fundamentals.storage import load_raw_response, save_raw_response


def test_decide_behind_beats_unprovenanced() -> None:
    assert (
        decide_refresh_action(
            force_refresh=False,
            has_usable_quarters=True,
            has_cik=True,
            behind_head=True,
            unprovenanced=True,
        )
        == RefreshAction.PULL
    )


def test_decide_enrich_only_when_unprovenanced_and_not_behind() -> None:
    assert (
        decide_refresh_action(
            force_refresh=False,
            has_usable_quarters=True,
            has_cik=True,
            behind_head=False,
            unprovenanced=True,
        )
        == RefreshAction.ENRICH_ONLY
    )


def test_av_queue_missing_before_stale() -> None:
    ordered = order_av_pull_queue(
        [
            ("STALE", SymbolCacheState.FILING_STALE),
            ("MISS", SymbolCacheState.MISSING),
            ("STALE2", SymbolCacheState.FILING_STALE),
        ]
    )
    assert ordered == ["MISS", "STALE", "STALE2"]


def test_eligibility_majority_10k_10q_is_eligible() -> None:
    client = SECEligibilityClient("LearnFinance test@example.com")
    forms = ["10-Q", "10-K", "10-Q", "10-Q", "10-K", "10-Q", "10-Q", "10-K"]
    rows = [(f, "2024-01-01", f"a{i}", "2023-12-31") for i, f in enumerate(forms)]
    with (
        patch.object(client, "resolve_cik", return_value="0000789019"),
        patch.object(client, "_recent_forms_and_rows", return_value=(forms, rows)),
    ):
        result = client.classify("MSFT")
    assert result.sec_eligible is True
    assert result.cik == "0000789019"


def test_eligibility_majority_20f_not_eligible() -> None:
    client = SECEligibilityClient("LearnFinance test@example.com")
    forms = ["20-F", "6-K", "20-F", "6-K", "20-F", "6-K", "20-F", "6-K"]
    rows = [(f, "2024-01-01", f"a{i}", "2023-12-31") for i, f in enumerate(forms)]
    with (
        patch.object(client, "resolve_cik", return_value="0001000180"),
        patch.object(client, "_recent_forms_and_rows", return_value=(forms, rows)),
    ):
        result = client.classify("SAP")
    assert result.sec_eligible is False


def test_amendment_forms_in_periodic_us() -> None:
    from brain_api.core.fundamentals.sec_eligibility import PERIODIC_US

    assert "10-K/A" in PERIODIC_US
    assert "10-Q/A" in PERIODIC_US


def test_ytd_to_quarter_differencing() -> None:
    facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "start": "2024-01-01",
                                "end": "2024-03-31",
                                "val": 100,
                                "filed": "2024-04-15",
                                "form": "10-Q",
                                "accn": "0001",
                                "fy": 2024,
                                "fp": "Q1",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-06-30",
                                "val": 250,
                                "filed": "2024-07-15",
                                "form": "10-Q",
                                "accn": "0002",
                                "fy": 2024,
                                "fp": "Q2",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-09-30",
                                "val": 400,
                                "filed": "2024-10-15",
                                "form": "10-Q",
                                "accn": "0003",
                                "fy": 2024,
                                "fp": "Q3",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "val": 600,
                                "filed": "2025-02-01",
                                "form": "10-K",
                                "accn": "0004",
                                "fy": 2024,
                                "fp": "FY",
                            },
                        ]
                    }
                },
                "GrossProfit": {
                    "units": {
                        "USD": [
                            {
                                "start": "2024-01-01",
                                "end": "2024-03-31",
                                "val": 40,
                                "filed": "2024-04-15",
                                "form": "10-Q",
                                "accn": "0001",
                                "fy": 2024,
                                "fp": "Q1",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-06-30",
                                "val": 100,
                                "filed": "2024-07-15",
                                "form": "10-Q",
                                "accn": "0002",
                                "fy": 2024,
                                "fp": "Q2",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-09-30",
                                "val": 155,
                                "filed": "2024-10-15",
                                "form": "10-Q",
                                "accn": "0003",
                                "fy": 2024,
                                "fp": "Q3",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "val": 240,
                                "filed": "2025-02-01",
                                "form": "10-K",
                                "accn": "0004",
                                "fy": 2024,
                                "fp": "FY",
                            },
                        ]
                    }
                },
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "start": "2024-01-01",
                                "end": "2024-03-31",
                                "val": 20,
                                "filed": "2024-04-15",
                                "form": "10-Q",
                                "accn": "0001",
                                "fy": 2024,
                                "fp": "Q1",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-06-30",
                                "val": 50,
                                "filed": "2024-07-15",
                                "form": "10-Q",
                                "accn": "0002",
                                "fy": 2024,
                                "fp": "Q2",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-09-30",
                                "val": 80,
                                "filed": "2024-10-15",
                                "form": "10-Q",
                                "accn": "0003",
                                "fy": 2024,
                                "fp": "Q3",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "val": 120,
                                "filed": "2025-02-01",
                                "form": "10-K",
                                "accn": "0004",
                                "fy": 2024,
                                "fp": "FY",
                            },
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "start": "2024-01-01",
                                "end": "2024-03-31",
                                "val": 10,
                                "filed": "2024-04-15",
                                "form": "10-Q",
                                "accn": "0001",
                                "fy": 2024,
                                "fp": "Q1",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-06-30",
                                "val": 25,
                                "filed": "2024-07-15",
                                "form": "10-Q",
                                "accn": "0002",
                                "fy": 2024,
                                "fp": "Q2",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-09-30",
                                "val": 40,
                                "filed": "2024-10-15",
                                "form": "10-Q",
                                "accn": "0003",
                                "fy": 2024,
                                "fp": "Q3",
                            },
                            {
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "val": 60,
                                "filed": "2025-02-01",
                                "form": "10-K",
                                "accn": "0004",
                                "fy": 2024,
                                "fp": "FY",
                            },
                        ]
                    }
                },
                "AssetsCurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-03-31",
                                "val": 1000,
                                "filed": "2024-04-15",
                                "form": "10-Q",
                                "accn": "0001",
                                "fy": 2024,
                                "fp": "Q1",
                            },
                            {
                                "end": "2024-06-30",
                                "val": 1100,
                                "filed": "2024-07-15",
                                "form": "10-Q",
                                "accn": "0002",
                                "fy": 2024,
                                "fp": "Q2",
                            },
                            {
                                "end": "2024-09-30",
                                "val": 1200,
                                "filed": "2024-10-15",
                                "form": "10-Q",
                                "accn": "0003",
                                "fy": 2024,
                                "fp": "Q3",
                            },
                            {
                                "end": "2024-12-31",
                                "val": 1300,
                                "filed": "2025-02-01",
                                "form": "10-K",
                                "accn": "0004",
                                "fy": 2024,
                                "fp": "FY",
                            },
                        ]
                    }
                },
                "LiabilitiesCurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": e,
                                "val": 400,
                                "filed": f,
                                "form": form,
                                "accn": a,
                                "fy": 2024,
                                "fp": fp,
                            }
                            for e, f, form, a, fp in [
                                ("2024-03-31", "2024-04-15", "10-Q", "0001", "Q1"),
                                ("2024-06-30", "2024-07-15", "10-Q", "0002", "Q2"),
                                ("2024-09-30", "2024-10-15", "10-Q", "0003", "Q3"),
                                ("2024-12-31", "2025-02-01", "10-K", "0004", "FY"),
                            ]
                        ]
                    }
                },
                "LongTermDebt": {
                    "units": {
                        "USD": [
                            {
                                "end": e,
                                "val": 50,
                                "filed": f,
                                "form": form,
                                "accn": a,
                                "fy": 2024,
                                "fp": fp,
                            }
                            for e, f, form, a, fp in [
                                ("2024-03-31", "2024-04-15", "10-Q", "0001", "Q1"),
                                ("2024-06-30", "2024-07-15", "10-Q", "0002", "Q2"),
                                ("2024-09-30", "2024-10-15", "10-Q", "0003", "Q3"),
                                ("2024-12-31", "2025-02-01", "10-K", "0004", "FY"),
                            ]
                        ]
                    }
                },
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            {
                                "end": e,
                                "val": 500,
                                "filed": f,
                                "form": form,
                                "accn": a,
                                "fy": 2024,
                                "fp": fp,
                            }
                            for e, f, form, a, fp in [
                                ("2024-03-31", "2024-04-15", "10-Q", "0001", "Q1"),
                                ("2024-06-30", "2024-07-15", "10-Q", "0002", "Q2"),
                                ("2024-09-30", "2024-10-15", "10-Q", "0003", "Q3"),
                                ("2024-12-31", "2025-02-01", "10-K", "0004", "FY"),
                            ]
                        ]
                    }
                },
            }
        }
    }
    income, balance = build_statement_payloads_from_companyfacts(
        facts, symbol="TEST", cik="0000320193"
    )
    by_end = {r["fiscalDateEnding"]: r for r in income["quarterlyReports"]}
    assert float(by_end["2024-03-31"]["totalRevenue"]) == 100.0
    assert float(by_end["2024-06-30"]["totalRevenue"]) == 150.0
    assert float(by_end["2024-09-30"]["totalRevenue"]) == 150.0
    assert float(by_end["2024-12-31"]["totalRevenue"]) == 200.0
    assert float(by_end["2024-06-30"]["grossProfit"]) == 60.0
    assert by_end["2024-06-30"]["filingDate"] == "2024-07-15"
    assert by_end["2024-06-30"]["accessionNumber"] == "0002"
    assert len(balance["quarterlyReports"]) == 4


def test_missing_tag_fails_loud() -> None:
    facts = {"facts": {"us-gaap": {}}}
    with pytest.raises(SECStatementError, match="No CompanyFacts USD points"):
        build_statement_payloads_from_companyfacts(facts, symbol="X", cik="0000000001")


def test_cache_behind_filing_head(tmp_path: Path) -> None:
    payload = {
        "quarterlyReports": [
            {
                "fiscalDateEnding": "2024-03-31",
                "filingDate": "2024-04-15",
                "accessionNumber": "0001",
            }
        ]
    }
    save_raw_response(tmp_path, "AAPL", "income_statement", payload)
    save_raw_response(tmp_path, "AAPL", "balance_sheet", payload)
    head = FilingHead(
        filing_date="2024-07-15",
        accession_number="0002",
        form="10-Q",
        report_date="2024-06-30",
    )
    assert cache_behind_filing_head(tmp_path, "AAPL", head) is True
    head_same = FilingHead(
        filing_date="2024-04-15",
        accession_number="0001",
        form="10-Q",
        report_date="2024-03-31",
    )
    assert cache_behind_filing_head(tmp_path, "AAPL", head_same) is False


def test_av_note_fails_without_local_quota_gate() -> None:
    index = MagicMock()
    index.get_api_calls_today.return_value = 25
    index.increment_api_calls = MagicMock()
    client = RealAlphaVantageClient(
        api_key="k", index=index, daily_limit=25, request_delay=0.0
    )
    with patch("requests.get") as mock_get:
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.return_value = {
            "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 25"
        }
        with pytest.raises(AlphaVantageProviderError, match="rejected"):
            client.fetch_income_statement("AAPL")
    # Still attempted the HTTP call despite calls_today == daily_limit
    mock_get.assert_called_once()
    index.increment_api_calls.assert_not_called()


def test_atomic_av_no_half_write(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "LearnFinance test@example.com")
    prior = {
        "quarterlyReports": [
            {
                "fiscalDateEnding": "2020-03-31",
                "totalRevenue": "1",
                "grossProfit": "1",
                "operatingIncome": "1",
                "netIncome": "1",
                "filingDate": "2020-04-01",
                "accessionNumber": "old",
                "filingForm": "10-Q",
                "filingSource": "https://example.com",
            }
        ]
    }
    save_raw_response(tmp_path, "SAP", "income_statement", prior)
    save_raw_response(
        tmp_path,
        "SAP",
        "balance_sheet",
        {
            "quarterlyReports": [
                {
                    "fiscalDateEnding": "2020-03-31",
                    "totalCurrentAssets": "1",
                    "totalCurrentLiabilities": "1",
                    "shortLongTermDebtTotal": "1",
                    "totalShareholderEquity": "1",
                    "filingDate": "2020-04-01",
                    "accessionNumber": "old",
                    "filingForm": "10-Q",
                    "filingSource": "https://example.com",
                }
            ]
        },
    )

    eligibility = EligibilityResult(
        symbol="SAP", cik="0001000180", sec_eligible=False, recent_forms=("20-F",)
    )
    elig_client = MagicMock()
    elig_client.classify.return_value = eligibility
    elig_client.fetch_filing_head.return_value = FilingHead(
        filing_date="2020-04-01",
        accession_number="old",
        form="20-F",
        report_date="2020-03-31",
    )

    fetcher = FundamentalsFetcher(
        api_key="k",
        base_path=tmp_path,
        eligibility_client=elig_client,
        companyfacts_client=MagicMock(),
        filing_provider=MagicMock(),
    )
    fetcher.client = MagicMock()
    fetcher.client.daily_limit = 25
    fetcher.client.fetch_income_statement.return_value = {
        "quarterlyReports": [{"fiscalDateEnding": "2024-06-30", "totalRevenue": "9"}]
    }
    fetcher.client.fetch_balance_sheet.side_effect = AlphaVantageProviderError(
        "Alpha Vantage rejected BALANCE_SHEET for SAP: Note"
    )

    with pytest.raises(FundamentalsProviderError):
        fetcher.fetch_symbol("SAP", force_refresh=True)

    income = load_raw_response(tmp_path, "SAP", "income_statement")
    assert income is not None
    assert income["response"]["quarterlyReports"][0]["fiscalDateEnding"] == "2020-03-31"
    fetcher.close()
