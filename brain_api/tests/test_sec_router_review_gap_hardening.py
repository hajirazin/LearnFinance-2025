"""Tests for review-gap hardening: debt first-match, shared throttle, pending age."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain_api.core.fundamentals.fetcher import cache_behind_filing_head
from brain_api.core.fundamentals.index import FundamentalsIndex
from brain_api.core.fundamentals.sec_eligibility import (
    FilingHead,
    SECEligibilityClient,
)
from brain_api.core.fundamentals.sec_rate_limit import (
    reset_sec_rate_limit_for_tests,
    wait_for_sec_slot,
)
from brain_api.core.fundamentals.sec_statements import (
    SECStatementError,
    resolve_debt_points,
)
from brain_api.core.fundamentals.storage import save_raw_response


def test_eligibility_thin_history_and_tie() -> None:
    classify = SECEligibilityClient.classify_sec_eligible
    assert classify(["10-Q", "10-K"]) is True
    assert classify(["8-K", "6-K"]) is False
    tie = ["10-Q", "20-F", "10-K", "40-F", "10-Q", "20-F", "10-K", "40-F"]
    assert classify(tie) is True
    fpi_heavy = ["20-F", "20-F", "20-F", "20-F", "20-F", "10-Q", "10-Q", "10-Q"]
    assert classify(fpi_heavy) is False


def _facts_for_tags(tag_vals: dict[str, float], *, end: str = "2024-06-30") -> dict:
    us_gaap = {}
    for tag, val in tag_vals.items():
        us_gaap[tag] = {
            "units": {
                "USD": [
                    {
                        "end": end,
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
    return {"facts": {"us-gaap": us_gaap}}


def test_debt_combined_beats_long_term_debt() -> None:
    facts = _facts_for_tags(
        {
            "LongTermDebt": 10.0,
            "DebtLongtermAndShorttermCombinedAmount": 50.0,
        }
    )
    resolved = resolve_debt_points(facts)
    assert resolved["2024-06-30"].value == 50.0


def test_debt_composed_sum_when_combined_absent() -> None:
    facts = _facts_for_tags(
        {
            "DebtCurrent": 5.0,
            "LongTermDebtNoncurrent": 20.0,
            "LongTermDebt": 99.0,
        }
    )
    resolved = resolve_debt_points(facts)
    assert resolved["2024-06-30"].value == 25.0


def test_debt_fail_loud_when_current_without_noncurrent() -> None:
    facts = _facts_for_tags(
        {
            "ShortTermBorrowings": 3.0,
            "LongTermDebtCurrent": 1.0,
            "LongTermDebt": 40.0,
        }
    )
    with pytest.raises(SECStatementError, match="LongTermDebtNoncurrent"):
        resolve_debt_points(facts)


def test_debt_fail_loud_mixed_period_does_not_suppress() -> None:
    """One incomplete composed period must fail the whole debt resolve."""
    facts = {
        "facts": {
            "us-gaap": {
                "DebtCurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-03-31",
                                "val": 5.0,
                                "filed": "2024-04-15",
                                "form": "10-Q",
                                "accn": "0001",
                                "fy": 2024,
                                "fp": "Q1",
                            },
                            {
                                "end": "2024-06-30",
                                "val": 6.0,
                                "filed": "2024-07-15",
                                "form": "10-Q",
                                "accn": "0002",
                                "fy": 2024,
                                "fp": "Q2",
                            },
                        ]
                    }
                },
                "LongTermDebtNoncurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-03-31",
                                "val": 20.0,
                                "filed": "2024-04-15",
                                "form": "10-Q",
                                "accn": "0001",
                                "fy": 2024,
                                "fp": "Q1",
                            }
                            # Q2 missing noncurrent → fail loud, do not keep Q1 only
                        ]
                    }
                },
            }
        }
    }
    with pytest.raises(SECStatementError, match="LongTermDebtNoncurrent"):
        resolve_debt_points(facts)


def test_shared_throttle_blocks_second_client(monkeypatch) -> None:
    reset_sec_rate_limit_for_tests()
    times = [0.0]

    def fake_monotonic() -> float:
        return times[0]

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        times[0] += seconds

    monkeypatch.setattr(
        "brain_api.core.fundamentals.sec_rate_limit.time.monotonic", fake_monotonic
    )
    monkeypatch.setattr(
        "brain_api.core.fundamentals.sec_rate_limit.time.sleep", fake_sleep
    )

    wait_for_sec_slot(min_interval_seconds=0.125)
    times[0] = 0.05
    wait_for_sec_slot(min_interval_seconds=0.125)
    assert sleeps and sleeps[0] == pytest.approx(0.075)
    reset_sec_rate_limit_for_tests()


def test_pending_age_persists_across_index_reload(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    idx1 = FundamentalsIndex(cache)
    first = idx1.mark_pending_new_filing(
        "AAPL", first_pending_at="2026-07-01T00:00:00+00:00"
    )
    idx1.close()
    idx2 = FundamentalsIndex(cache)
    assert idx2.get_pending_new_filing("AAPL") == first
    idx2.clear_pending_new_filing("AAPL")
    assert idx2.get_pending_new_filing("AAPL") is None
    idx2.close()


def test_amendment_newest_head_is_behind(tmp_path: Path) -> None:
    payload = {
        "quarterlyReports": [
            {
                "fiscalDateEnding": "2024-03-31",
                "filingDate": "2024-07-15",
                "accessionNumber": "0001",
            }
        ]
    }
    save_raw_response(tmp_path, "AAPL", "income_statement", payload)
    save_raw_response(tmp_path, "AAPL", "balance_sheet", payload)
    head = FilingHead(
        filing_date="2024-08-01",
        accession_number="0002-A",
        form="10-Q/A",
        report_date="2024-03-31",
    )
    assert cache_behind_filing_head(tmp_path, "AAPL", head) is True
