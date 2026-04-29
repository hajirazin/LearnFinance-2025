"""Summary generation and email reporting activities."""

import logging

from temporalio import activity

from activities.client import get_client
from models import (
    FundamentalsResponse,
    HRPAllocationResponse,
    LSTMInferenceResponse,
    NewsSignalResponse,
    PatchTSTInferenceResponse,
    SACInferenceResponse,
    SkippedAllocation,
    SkippedSubmitResponse,
    SubmitOrdersResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)

logger = logging.getLogger(__name__)


def _alloc_to_dict(alloc, is_hrp: bool = False, as_of_date: str = "") -> dict:
    """Convert allocation to dict, handling skipped allocations."""
    if isinstance(alloc, SkippedAllocation) or getattr(alloc, "skipped", False):
        if is_hrp:
            return {
                "percentage_weights": {},
                "symbols_used": 0,
                "symbols_excluded": [],
                "lookback_days": 0,
                "as_of_date": as_of_date,
            }
        return {
            "target_weights": {},
            "turnover": 0,
            "model_version": "skipped",
            "target_week_start": "",
            "target_week_end": "",
            "weight_changes": [],
        }
    return alloc.model_dump()


def _submit_to_dict(submit) -> dict:
    """Convert submit response to dict."""
    if isinstance(submit, SkippedSubmitResponse) or getattr(submit, "skipped", False):
        return {"orders_submitted": 0, "orders_failed": 0, "skipped": True}
    return {
        "orders_submitted": submit.orders_submitted,
        "orders_failed": submit.orders_failed,
        "skipped": False,
    }


@activity.defn
def generate_summary(
    lstm: LSTMInferenceResponse,
    patchtst: PatchTSTInferenceResponse,
    news: NewsSignalResponse,
    fundamentals: FundamentalsResponse,
    hrp: HRPAllocationResponse | SkippedAllocation,
    sac: SACInferenceResponse | SkippedAllocation,
) -> WeeklySummaryResponse:
    """Generate LLM summary of weekly results."""
    logger.info("Generating LLM summary...")
    with get_client() as client:
        response = client.post(
            "/llm/weekly-summary",
            json={
                "lstm": lstm.model_dump(),
                "patchtst": patchtst.model_dump(),
                "news": news.model_dump(),
                "fundamentals": fundamentals.model_dump(),
                "hrp": _alloc_to_dict(hrp, is_hrp=True),
                "sac": _alloc_to_dict(sac),
            },
        )
        response.raise_for_status()
    result = WeeklySummaryResponse(**response.json())
    logger.info(f"Generated summary via {result.provider} ({result.model_used})")
    return result


@activity.defn
def send_weekly_email(
    summary: WeeklySummaryResponse,
    lstm: LSTMInferenceResponse,
    patchtst: PatchTSTInferenceResponse,
    hrp: HRPAllocationResponse | SkippedAllocation,
    sac: SACInferenceResponse | SkippedAllocation,
    sac_submit: SubmitOrdersResponse | SkippedSubmitResponse,
    hrp_submit: SubmitOrdersResponse | SkippedSubmitResponse,
    target_week_start: str,
    target_week_end: str,
    as_of_date: str,
    skipped_algorithms: list[str],
) -> WeeklyReportEmailResponse:
    """Send weekly report email."""
    logger.info("Sending weekly report email...")
    with get_client() as client:
        response = client.post(
            "/email/weekly-report",
            json={
                "summary": summary.summary,
                "order_results": {
                    "sac": _submit_to_dict(sac_submit),
                    "hrp": _submit_to_dict(hrp_submit),
                },
                "skipped_algorithms": skipped_algorithms,
                "target_week_start": target_week_start,
                "target_week_end": target_week_end,
                "as_of_date": as_of_date,
                "sac": _alloc_to_dict(sac, as_of_date=as_of_date),
                "hrp": _alloc_to_dict(hrp, is_hrp=True, as_of_date=as_of_date),
                "lstm": lstm.model_dump(),
                "patchtst": patchtst.model_dump(),
            },
        )
        response.raise_for_status()
    result = WeeklyReportEmailResponse(**response.json())
    logger.info(f"Email sent: {result.subject}")
    return result


@activity.defn
def generate_india_summary(
    hrp: HRPAllocationResponse,
    universe: str,
) -> WeeklySummaryResponse:
    """Generate LLM summary of India HRP allocation."""
    logger.info("Generating India LLM summary...")
    with get_client() as client:
        response = client.post(
            "/llm/india-weekly-summary",
            json={"hrp": hrp.model_dump(), "universe": universe},
        )
        response.raise_for_status()
    result = WeeklySummaryResponse(**response.json())
    logger.info(f"Generated India summary via {result.provider} ({result.model_used})")
    return result


@activity.defn
def send_india_weekly_email(
    summary: WeeklySummaryResponse,
    hrp: HRPAllocationResponse,
    universe: str,
    target_week_start: str,
    target_week_end: str,
    as_of_date: str,
) -> WeeklyReportEmailResponse:
    """Send India weekly report email (HRP + AI summary)."""
    logger.info("Sending India weekly report email...")
    with get_client() as client:
        response = client.post(
            "/email/india-weekly-report",
            json={
                "summary": summary.summary,
                "hrp": hrp.model_dump(),
                "universe": universe,
                "target_week_start": target_week_start,
                "target_week_end": target_week_end,
                "as_of_date": as_of_date,
            },
        )
        response.raise_for_status()
    result = WeeklyReportEmailResponse(**response.json())
    logger.info(f"India email sent: {result.subject}")
    return result


@activity.defn
def generate_double_hrp_summary(
    stage1: HRPAllocationResponse,
    stage2: HRPAllocationResponse,
    universe: str,
    top_n: int,
) -> WeeklySummaryResponse:
    """Generate LLM summary of Double HRP two-stage allocation."""
    logger.info("Generating Double HRP LLM summary...")
    with get_client() as client:
        response = client.post(
            "/llm/india-double-hrp-summary",
            json={
                "stage1": stage1.model_dump(),
                "stage2": stage2.model_dump(),
                "universe": universe,
                "top_n": top_n,
            },
        )
        response.raise_for_status()
    result = WeeklySummaryResponse(**response.json())
    logger.info(
        f"Generated Double HRP summary via {result.provider} ({result.model_used})"
    )
    return result


@activity.defn
def send_double_hrp_email(
    summary: WeeklySummaryResponse,
    stage1: HRPAllocationResponse,
    stage2: HRPAllocationResponse,
    universe: str,
    top_n: int,
    target_week_start: str,
    target_week_end: str,
    as_of_date: str,
) -> WeeklyReportEmailResponse:
    """Send Double HRP report email (both stages + AI summary)."""
    logger.info("Sending Double HRP report email...")
    with get_client() as client:
        response = client.post(
            "/email/india-double-hrp-report",
            json={
                "summary": summary.summary,
                "stage1": stage1.model_dump(),
                "stage2": stage2.model_dump(),
                "universe": universe,
                "top_n": top_n,
                "target_week_start": target_week_start,
                "target_week_end": target_week_end,
                "as_of_date": as_of_date,
            },
        )
        response.raise_for_status()
    result = WeeklyReportEmailResponse(**response.json())
    logger.info(f"Double HRP email sent: {result.subject}")
    return result


@activity.defn
def generate_us_double_hrp_summary(
    stage1: HRPAllocationResponse,
    stage2: HRPAllocationResponse,
    universe: str,
    top_n: int,
) -> WeeklySummaryResponse:
    """Generate LLM summary of US Double HRP two-stage allocation."""
    logger.info("Generating US Double HRP LLM summary...")
    with get_client() as client:
        response = client.post(
            "/llm/us-double-hrp-summary",
            json={
                "stage1": stage1.model_dump(),
                "stage2": stage2.model_dump(),
                "universe": universe,
                "top_n": top_n,
            },
        )
        response.raise_for_status()
    result = WeeklySummaryResponse(**response.json())
    logger.info(
        f"Generated US Double HRP summary via {result.provider} ({result.model_used})"
    )
    return result


@activity.defn
def send_us_double_hrp_email(
    summary: WeeklySummaryResponse,
    stage1: HRPAllocationResponse,
    stage2: HRPAllocationResponse,
    universe: str,
    top_n: int,
    target_week_start: str,
    target_week_end: str,
    as_of_date: str,
    sticky_kept_count: int = 0,
    sticky_fillers_count: int = 0,
    previous_year_week_used: str | None = None,
    order_results: SubmitOrdersResponse | SkippedSubmitResponse | None = None,
    skipped: bool = False,
) -> WeeklyReportEmailResponse:
    """Send US Double HRP report email.

    On the skip path (``skipped=True`` or ``order_results`` is a
    :class:`SkippedSubmitResponse`), the email body suppresses the
    allocation/orders tables and shows a short banner about why the
    week was skipped (last week's orders still open).
    """
    logger.info("Sending US Double HRP report email...")
    payload: dict = {
        "summary": summary.summary,
        "stage1": stage1.model_dump(),
        "stage2": stage2.model_dump(),
        "universe": universe,
        "top_n": top_n,
        "target_week_start": target_week_start,
        "target_week_end": target_week_end,
        "as_of_date": as_of_date,
        "sticky_kept_count": sticky_kept_count,
        "sticky_fillers_count": sticky_fillers_count,
        "previous_year_week_used": previous_year_week_used,
        "skipped": skipped,
    }
    if order_results is not None:
        payload["order_results"] = _submit_to_dict(order_results)
    with get_client() as client:
        response = client.post(
            "/email/us-double-hrp-report",
            json=payload,
        )
        response.raise_for_status()
    result = WeeklyReportEmailResponse(**response.json())
    logger.info(f"US Double HRP email sent: {result.subject}")
    return result
