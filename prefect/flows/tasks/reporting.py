"""Summary generation and email reporting tasks."""

from prefect import task
from prefect.logging import get_run_logger

from flows.models import (
    FundamentalsResponse,
    HRPAllocationResponse,
    LSTMInferenceResponse,
    NewsSignalResponse,
    PatchTSTInferenceResponse,
    PPOInferenceResponse,
    SACInferenceResponse,
    SkippedAllocation,
    SkippedSubmitResponse,
    SubmitOrdersResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)
from flows.tasks.client import get_client


def _alloc_to_dict(alloc, is_hrp: bool = False, as_of_date: str = "") -> dict:
    """Convert allocation to dict, handling skipped allocations."""
    if isinstance(alloc, SkippedAllocation) or getattr(alloc, "skipped", False):
        if is_hrp:
            return {
                "percentage_weights": {},
                "symbols_used": 0,
                "symbols_excluded": [],
                "as_of_date": as_of_date,
            }
        return {
            "target_weights": {},
            "turnover": 0,
            "model_version": "skipped",
        }
    return alloc.model_dump()


def _submit_to_dict(submit) -> dict:
    """Convert submit response to dict."""
    if isinstance(submit, SkippedSubmitResponse) or submit.skipped:
        return {"orders_submitted": 0, "orders_failed": 0, "skipped": True}
    return {
        "orders_submitted": submit.orders_submitted,
        "orders_failed": submit.orders_failed,
        "skipped": False,
    }


@task(name="Generate Summary", retries=1, retry_delay_seconds=30)
def generate_summary(
    lstm: LSTMInferenceResponse,
    patchtst: PatchTSTInferenceResponse,
    news: NewsSignalResponse,
    fundamentals: FundamentalsResponse,
    hrp: HRPAllocationResponse | SkippedAllocation,
    sac: SACInferenceResponse | SkippedAllocation,
    ppo: PPOInferenceResponse | SkippedAllocation,
) -> WeeklySummaryResponse:
    """Generate LLM summary of weekly results."""
    logger = get_run_logger()
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
                "ppo": _alloc_to_dict(ppo),
            },
        )
        response.raise_for_status()
        data = response.json()

    result = WeeklySummaryResponse(**data)
    logger.info(f"Generated summary via {result.provider} ({result.model_used})")
    return result


@task(name="Send Weekly Email", retries=1, retry_delay_seconds=30)
def send_weekly_email(
    summary: WeeklySummaryResponse,
    lstm: LSTMInferenceResponse,
    patchtst: PatchTSTInferenceResponse,
    hrp: HRPAllocationResponse | SkippedAllocation,
    sac: SACInferenceResponse | SkippedAllocation,
    ppo: PPOInferenceResponse | SkippedAllocation,
    ppo_submit: SubmitOrdersResponse | SkippedSubmitResponse,
    sac_submit: SubmitOrdersResponse | SkippedSubmitResponse,
    hrp_submit: SubmitOrdersResponse | SkippedSubmitResponse,
    target_week_start: str,
    target_week_end: str,
    as_of_date: str,
    skipped_algorithms: list[str],
) -> WeeklyReportEmailResponse:
    """Send weekly report email."""
    logger = get_run_logger()
    logger.info("Sending weekly report email...")

    with get_client() as client:
        response = client.post(
            "/email/weekly-report",
            json={
                "summary": summary.summary,
                "order_results": {
                    "ppo": _submit_to_dict(ppo_submit),
                    "sac": _submit_to_dict(sac_submit),
                    "hrp": _submit_to_dict(hrp_submit),
                },
                "skipped_algorithms": skipped_algorithms,
                "target_week_start": target_week_start,
                "target_week_end": target_week_end,
                "as_of_date": as_of_date,
                "sac": _alloc_to_dict(sac, as_of_date=as_of_date),
                "ppo": _alloc_to_dict(ppo, as_of_date=as_of_date),
                "hrp": _alloc_to_dict(hrp, is_hrp=True, as_of_date=as_of_date),
                "lstm": lstm.model_dump(),
                "patchtst": patchtst.model_dump(),
            },
        )
        response.raise_for_status()
        data = response.json()

    result = WeeklyReportEmailResponse(**data)
    logger.info(f"Email sent: {result.subject}")
    return result
