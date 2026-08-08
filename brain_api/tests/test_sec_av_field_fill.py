"""Tests for SEC→AV named demotion field/period merge."""

from __future__ import annotations

from brain_api.core.fundamentals.fetcher import (
    merge_sec_av_field_gaps,
    sec_payloads_need_av_fill,
)
from brain_api.core.fundamentals.sec_statements import (
    SAC_BALANCE_FIELDS,
    SAC_INCOME_FIELDS,
    build_statement_payloads_from_companyfacts,
)


def test_merge_skips_av_when_sec_complete() -> None:
    sec = {
        "provider": "sec_companyfacts",
        "quarterlyReports": [
            {
                "fiscalDateEnding": "2024-06-30",
                "totalRevenue": "100",
                "grossProfit": "40",
                "operatingIncome": "20",
                "netIncome": "10",
                "fieldSource": dict.fromkeys(SAC_INCOME_FIELDS, "sec_companyfacts"),
            }
        ],
    }
    av = {
        "provider": "alpha_vantage",
        "quarterlyReports": [
            {
                "fiscalDateEnding": "2024-06-30",
                "totalRevenue": "999",
                "grossProfit": "999",
                "operatingIncome": "999",
                "netIncome": "999",
            }
        ],
    }
    merged = merge_sec_av_field_gaps(sec, av, SAC_INCOME_FIELDS)
    assert merged["quarterlyReports"][0]["totalRevenue"] == "100"
    assert merged["provider"] == "sec_companyfacts"
    complete_income = {
        "quarterlyReports": [dict.fromkeys(SAC_INCOME_FIELDS, "1")],
    }
    complete_balance = {
        "quarterlyReports": [dict.fromkeys(SAC_BALANCE_FIELDS, "1")],
    }
    assert sec_payloads_need_av_fill(complete_income, complete_balance) is False


def test_merge_fills_missing_field_with_fieldsource() -> None:
    sec = {
        "provider": "sec_companyfacts",
        "quarterlyReports": [
            {
                "fiscalDateEnding": "2024-06-30",
                "totalRevenue": "100",
                "grossProfit": "40",
                "operatingIncome": "20",
                # netIncome missing
            }
        ],
    }
    av = {
        "quarterlyReports": [
            {
                "fiscalDateEnding": "2024-06-30",
                "totalRevenue": "100",
                "grossProfit": "40",
                "operatingIncome": "20",
                "netIncome": "7",
            }
        ],
    }
    merged = merge_sec_av_field_gaps(sec, av, SAC_INCOME_FIELDS)
    row = merged["quarterlyReports"][0]
    assert row["netIncome"] == "7"
    assert row["fieldSource"]["netIncome"] == "alpha_vantage"
    assert row["fieldSource"]["totalRevenue"] == "sec_companyfacts"
    assert merged["provider"] == "sec_companyfacts+alpha_vantage"


def test_merge_av_only_when_sec_empty() -> None:
    av = {
        "quarterlyReports": [
            {
                "fiscalDateEnding": "2024-06-30",
                **dict.fromkeys(SAC_BALANCE_FIELDS, "1"),
            }
        ],
    }
    merged = merge_sec_av_field_gaps(None, av, SAC_BALANCE_FIELDS)
    assert merged["provider"] == "sec_companyfacts+alpha_vantage"
    assert merged["quarterlyReports"][0]["fieldSource"]["shortLongTermDebtTotal"] == (
        "alpha_vantage"
    )


def test_sec_complete_does_not_need_av() -> None:
    income = {
        "quarterlyReports": [dict.fromkeys(SAC_INCOME_FIELDS, "1")],
    }
    balance = {
        "quarterlyReports": [dict.fromkeys(SAC_BALANCE_FIELDS, "1")],
    }
    assert sec_payloads_need_av_fill(income, balance) is False
    assert sec_payloads_need_av_fill(None, balance) is True


def test_gross_profit_derive_from_cost_tag() -> None:
    def _flow(end: str, start: str, val: float) -> dict:
        return {
            "units": {
                "USD": [
                    {
                        "end": end,
                        "start": start,
                        "val": val,
                        "filed": "2024-07-15",
                        "form": "10-Q",
                        "accn": "0001",
                        "fy": 2024,
                        "fp": "Q2",
                    }
                ]
            }
        }

    def _instant(val: float) -> dict:
        return {
            "units": {
                "USD": [
                    {
                        "end": "2024-06-30",
                        "val": val,
                        "filed": "2024-07-15",
                        "form": "10-Q",
                        "accn": "0001",
                        "fy": 2024,
                        "fp": "Q2",
                    }
                ]
            }
        }

    facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": _flow(
                    "2024-06-30", "2024-04-01", 100.0
                ),
                "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization": _flow(
                    "2024-06-30", "2024-04-01", 40.0
                ),
                "OperatingIncomeLoss": _flow("2024-06-30", "2024-04-01", 30.0),
                "NetIncomeLoss": _flow("2024-06-30", "2024-04-01", 20.0),
                "AssetsCurrent": _instant(200),
                "LiabilitiesCurrent": _instant(50),
                "LongTermDebtNoncurrent": _instant(10),
                "StockholdersEquity": _instant(100),
            }
        }
    }
    income, balance = build_statement_payloads_from_companyfacts(
        facts, symbol="CTSH", cik="0001058290"
    )
    row = income["quarterlyReports"][0]
    assert float(row["grossProfit"]) == 60.0
    assert row["fieldSource"]["grossProfit"] == "sec_derived"
    assert float(balance["quarterlyReports"][0]["shortLongTermDebtTotal"]) == 10.0
