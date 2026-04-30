"""Weekly report email endpoint."""

import logging
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from .gmail import GmailConfigError, send_html_email
from .models import (
    AlphaHRPEmailRequest,
    DoubleHRPEmailRequest,
    IndiaAlphaHRPEmailRequest,
    SACWeeklyReportEmailRequest,
    USAlphaHRPEmailRequest,
    USDoubleHRPEmailRequest,
    WeeklyReportEmailResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Template directory (relative to brain_api package)
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"


def get_jinja_env() -> Environment:
    """Get Jinja2 environment for loading templates."""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,  # We're generating HTML, enable autoescape
    )


def _render_and_send_email(
    *,
    template_name: str,
    context: dict,
    subject: str,
    log_label: str,
) -> WeeklyReportEmailResponse:
    """Load a Jinja template, render with context, send via Gmail.

    Shared scaffolding for the four near-identical "render template,
    pick subject, send via Gmail" blocks across the weekly-report
    endpoints. Variants only differ in the template + subject + log
    string; that's exactly what this helper parameterises.

    Math/policy stays in the caller (which Jinja context to build,
    which subject to render); this helper only owns I/O.
    """
    try:
        env = get_jinja_env()
        template = env.get_template(template_name)
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Template not found: {template_name}",
        ) from e

    html_body = template.render(**context)
    logger.debug(f"Generated {log_label} HTML body length: {len(html_body)} chars")

    try:
        send_html_email(subject=subject, html_body=html_body)
    except GmailConfigError as e:
        logger.error(f"Gmail configuration error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Gmail configuration error: {e}",
        ) from e
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to send email: {e}",
        ) from e

    logger.info(f"{log_label} email sent successfully")

    return WeeklyReportEmailResponse(
        is_success=True,
        subject=subject,
        body=html_body,
    )


def _build_subject(
    *,
    target_week_start: str,
    target_week_end: str,
    base: str,
    skipped: bool = False,
    skipped_suffix: str | None = None,
    skipped_base: str | None = None,
) -> str:
    """Compose a "Title (start -> end)" subject with optional skip variant.

    The skip-suffix arm exists so the email subject distinguishes a
    weekly run from its skipped-because-open-orders cousin without
    duplicating the subject-building string in two callers.
    """
    if skipped:
        title = skipped_base or f"{base} Skipped"
        if skipped_suffix:
            title = f"{title} {skipped_suffix}"
    else:
        title = base
    return f"{title} ({target_week_start} -> {target_week_end})"


# Type alias retained for callers that want a typed ``subject_fn`` style.
SubjectFn = Callable[[bool], str]


def _alpha_hrp_email_context(request: AlphaHRPEmailRequest) -> dict:
    """Build the Jinja context dict for an Alpha-HRP email render.

    Centralises the shape contract of the alpha-HRP email template
    context across US and India. The base ``AlphaHRPEmailRequest``
    carries every common field; the US subclass adds ``order_results``
    and ``skipped`` (India does not trade so its base instance leaves
    them at their defaults of ``None`` / ``False``).

    The skipped/order-execution blocks in the template render based on
    these context keys -- always present, always honest about whether
    they apply to this market.
    """
    order_results = getattr(request, "order_results", None)
    if order_results is not None:
        order_results = order_results.model_dump()
    return {
        "summary": request.summary,
        "stage1_top_scores": [item.model_dump() for item in request.stage1_top_scores],
        "model_version": request.model_version,
        "predicted_count": request.predicted_count,
        "requested_count": request.requested_count,
        "selected_symbols": request.selected_symbols,
        "kept_count": request.kept_count,
        "fillers_count": request.fillers_count,
        "evicted_from_previous": request.evicted_from_previous,
        "previous_year_week_used": request.previous_year_week_used,
        "stage2": request.stage2.model_dump(),
        "universe": request.universe,
        "top_n": request.top_n,
        "hold_threshold": request.hold_threshold,
        "target_week_start": request.target_week_start,
        "target_week_end": request.target_week_end,
        "as_of_date": request.as_of_date,
        "order_results": order_results,
        "skipped": getattr(request, "skipped", False),
    }


@router.post("/sac-weekly-report", response_model=WeeklyReportEmailResponse)
def send_sac_weekly_report_email(
    request: SACWeeklyReportEmailRequest,
) -> WeeklyReportEmailResponse:
    """Send a SAC-only weekly portfolio analysis email.

    Takes the AI-generated summary, the SAC Alpaca order execution
    results, the SAC allocation, and forecast predictions (LSTM,
    PatchTST), renders an HTML email using Jinja2, and sends via Gmail
    SMTP. HRP weekly reporting runs through ``/email/us-alpha-hrp-report``.

    Email configuration comes from environment variables:
    - GMAIL_USER: sender address
    - GMAIL_APP_PASSWORD: Gmail app password
    - TRAINING_EMAIL_TO: recipient address
    - TRAINING_EMAIL_CC: CC recipients (optional)
    """
    logger.info("Generating SAC weekly report email")

    return _render_and_send_email(
        template_name="sac_weekly_report_email.html.j2",
        context={
            "summary": request.summary,
            "order_results": request.order_results.model_dump(),
            "skipped_algorithms": request.skipped_algorithms,
            "target_week_start": request.target_week_start,
            "target_week_end": request.target_week_end,
            "as_of_date": request.as_of_date,
            "sac": request.sac.model_dump(),
            "lstm": request.lstm.model_dump(),
            "patchtst": request.patchtst.model_dump(),
        },
        subject=_build_subject(
            target_week_start=request.target_week_start,
            target_week_end=request.target_week_end,
            base="SAC Weekly Portfolio Analysis",
        ),
        log_label="SAC weekly report",
    )


@router.post("/india-alpha-hrp-report", response_model=WeeklyReportEmailResponse)
def send_india_alpha_hrp_report_email(
    request: IndiaAlphaHRPEmailRequest,
) -> WeeklyReportEmailResponse:
    """Send an India Alpha-HRP portfolio analysis email.

    India weekly allocation is structurally "PatchTST alpha screen on
    Nifty Shariah 500 -> rank-band sticky -> HRP". The email mirrors the
    US Alpha-HRP report (Stage 1 top-25, sticky kept/fillers/evicted,
    Stage 2 weights) minus the Alpaca order-execution / skipped blocks,
    since India does not trade through a paper account.
    """
    logger.info("Generating India Alpha-HRP report email")

    return _render_and_send_email(
        template_name="india_alpha_hrp_report_email.html.j2",
        context=_alpha_hrp_email_context(request),
        subject=_build_subject(
            target_week_start=request.target_week_start,
            target_week_end=request.target_week_end,
            base="India Alpha-HRP Portfolio Analysis",
        ),
        log_label="India Alpha-HRP report",
    )


@router.post("/india-double-hrp-report", response_model=WeeklyReportEmailResponse)
def send_india_double_hrp_report_email(
    request: DoubleHRPEmailRequest,
) -> WeeklyReportEmailResponse:
    """Send a Double HRP portfolio analysis email.

    Two-stage HRP: Stage 1 screens the full universe, Stage 2
    re-allocates the top-N selected stocks. The email shows both
    stages alongside the AI summary.
    """
    logger.info("Generating Double HRP report email")

    return _render_and_send_email(
        template_name="india_double_hrp_report_email.html.j2",
        context={
            "summary": request.summary,
            "stage1": request.stage1.model_dump(),
            "stage2": request.stage2.model_dump(),
            "universe": request.universe,
            "top_n": request.top_n,
            "target_week_start": request.target_week_start,
            "target_week_end": request.target_week_end,
            "as_of_date": request.as_of_date,
        },
        subject=_build_subject(
            target_week_start=request.target_week_start,
            target_week_end=request.target_week_end,
            base="Double HRP Portfolio Analysis",
        ),
        log_label="Double HRP report",
    )


@router.post("/us-double-hrp-report", response_model=WeeklyReportEmailResponse)
def send_us_double_hrp_report_email(
    request: USDoubleHRPEmailRequest,
) -> WeeklyReportEmailResponse:
    """Send a US Double HRP portfolio analysis email.

    Mirrors the India Double HRP email but adds an Alpaca order execution
    section (because US trades through a paper account) and a skip block
    when last week's orders were still open at run time.
    """
    logger.info("Generating US Double HRP report email")

    order_results = (
        request.order_results.model_dump() if request.order_results else None
    )

    return _render_and_send_email(
        template_name="us_double_hrp_report_email.html.j2",
        context={
            "summary": request.summary,
            "stage1": request.stage1.model_dump(),
            "stage2": request.stage2.model_dump(),
            "universe": request.universe,
            "top_n": request.top_n,
            "target_week_start": request.target_week_start,
            "target_week_end": request.target_week_end,
            "as_of_date": request.as_of_date,
            "order_results": order_results,
            "skipped": request.skipped,
            "sticky_kept_count": request.sticky_kept_count,
            "sticky_fillers_count": request.sticky_fillers_count,
            "previous_year_week_used": request.previous_year_week_used,
        },
        subject=_build_subject(
            target_week_start=request.target_week_start,
            target_week_end=request.target_week_end,
            base="US Double HRP Portfolio Analysis",
            skipped=request.skipped,
            skipped_base="US Double HRP Skipped",
        ),
        log_label="US Double HRP report",
    )


@router.post("/us-alpha-hrp-report", response_model=WeeklyReportEmailResponse)
def send_us_alpha_hrp_report_email(
    request: USAlphaHRPEmailRequest,
) -> WeeklyReportEmailResponse:
    """Send a US Alpha-HRP portfolio analysis email.

    Stage 1 = PatchTST alpha screen on halal_new (~410 stocks);
    rank-band sticky selection picks 15 (K_in=15, K_hold=30); Stage 2 =
    HRP on the chosen 15. Trades through the ``hrp`` Alpaca paper
    account (algorithm tag ``alpha_hrp`` for forward-going audit).

    On the skip path (``skipped=True``), the template hides the
    allocation/orders tables and shows a banner about the open-orders
    gate that prevented this week's submission.
    """
    logger.info("Generating US Alpha-HRP report email")

    return _render_and_send_email(
        template_name="us_alpha_hrp_report_email.html.j2",
        context=_alpha_hrp_email_context(request),
        subject=_build_subject(
            target_week_start=request.target_week_start,
            target_week_end=request.target_week_end,
            base="US Alpha-HRP Portfolio Analysis",
            skipped=request.skipped,
            skipped_base="US Alpha-HRP Skipped",
        ),
        log_label="US Alpha-HRP report",
    )
