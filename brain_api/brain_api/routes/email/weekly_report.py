"""Weekly report email endpoint."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from .gmail import GmailConfigError, send_html_email
from .models import (
    DoubleHRPEmailRequest,
    IndiaWeeklyReportEmailRequest,
    USDoubleHRPEmailRequest,
    WeeklyReportEmailRequest,
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


@router.post("/weekly-report", response_model=WeeklyReportEmailResponse)
def send_weekly_report_email(
    request: WeeklyReportEmailRequest,
) -> WeeklyReportEmailResponse:
    """Send a weekly portfolio analysis email.

    Takes the AI-generated summary, Alpaca order execution results,
    allocation data (SAC, HRP), and forecast predictions (LSTM, PatchTST),
    renders an HTML email using Jinja2, and sends via Gmail SMTP.

    Email configuration comes from environment variables:
    - GMAIL_USER: sender address
    - GMAIL_APP_PASSWORD: Gmail app password
    - TRAINING_EMAIL_TO: recipient address
    - TRAINING_EMAIL_CC: CC recipients (optional)

    Args:
        request: Weekly report data including summary, order results,
                 allocations, and forecasts.

    Returns:
        Response with success status, subject, and HTML body.

    Raises:
        HTTPException: If template loading or email sending fails.
    """
    logger.info("Generating weekly report email")

    # Load and render the Jinja2 template
    try:
        env = get_jinja_env()
        template = env.get_template("weekly_report_email.html.j2")
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="Template not found: weekly_report_email.html.j2",
        ) from e

    # Render HTML body with all data
    html_body = template.render(
        summary=request.summary,
        order_results=request.order_results.model_dump(),
        skipped_algorithms=request.skipped_algorithms,
        target_week_start=request.target_week_start,
        target_week_end=request.target_week_end,
        as_of_date=request.as_of_date,
        sac=request.sac.model_dump(),
        hrp=request.hrp.model_dump(),
        lstm=request.lstm.model_dump(),
        patchtst=request.patchtst.model_dump(),
    )

    logger.debug(f"Generated HTML body length: {len(html_body)} chars")

    # Build subject line
    subject = (
        f"Weekly Portfolio Analysis ({request.target_week_start} "
        f"-> {request.target_week_end})"
    )

    # Send email
    try:
        send_html_email(subject=subject, html_body=html_body)
        is_success = True
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

    logger.info("Weekly report email sent successfully")

    return WeeklyReportEmailResponse(
        is_success=is_success,
        subject=subject,
        body=html_body,
    )


@router.post("/india-weekly-report", response_model=WeeklyReportEmailResponse)
def send_india_weekly_report_email(
    request: IndiaWeeklyReportEmailRequest,
) -> WeeklyReportEmailResponse:
    """Send an India weekly portfolio analysis email.

    India pipeline is HRP-only (no SAC/news/fundamentals/orders).
    Renders an HTML email with the AI summary and HRP allocation table,
    and sends via Gmail SMTP.

    Args:
        request: India weekly report data (AI summary + HRP allocation).

    Returns:
        Response with success status, subject, and HTML body.

    Raises:
        HTTPException: If template loading or email sending fails.
    """
    logger.info("Generating India weekly report email")

    try:
        env = get_jinja_env()
        template = env.get_template("india_weekly_report_email.html.j2")
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="Template not found: india_weekly_report_email.html.j2",
        ) from e

    html_body = template.render(
        summary=request.summary,
        hrp=request.hrp.model_dump(),
        universe=request.universe,
        target_week_start=request.target_week_start,
        target_week_end=request.target_week_end,
        as_of_date=request.as_of_date,
    )

    logger.debug(f"Generated India HTML body length: {len(html_body)} chars")

    subject = (
        f"India Weekly Portfolio Analysis ({request.target_week_start} "
        f"-> {request.target_week_end})"
    )

    try:
        send_html_email(subject=subject, html_body=html_body)
        is_success = True
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

    logger.info("India weekly report email sent successfully")

    return WeeklyReportEmailResponse(
        is_success=is_success,
        subject=subject,
        body=html_body,
    )


@router.post("/india-double-hrp-report", response_model=WeeklyReportEmailResponse)
def send_india_double_hrp_report_email(
    request: DoubleHRPEmailRequest,
) -> WeeklyReportEmailResponse:
    """Send a Double HRP portfolio analysis email.

    Two-stage HRP: Stage 1 screens the full universe, Stage 2
    re-allocates the top-N selected stocks. The email shows both
    stages alongside the AI summary.

    Args:
        request: Double HRP report data (both stages + AI summary).

    Returns:
        Response with success status, subject, and HTML body.

    Raises:
        HTTPException: If template loading or email sending fails.
    """
    logger.info("Generating Double HRP report email")

    try:
        env = get_jinja_env()
        template = env.get_template("india_double_hrp_report_email.html.j2")
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="Template not found: india_double_hrp_report_email.html.j2",
        ) from e

    html_body = template.render(
        summary=request.summary,
        stage1=request.stage1.model_dump(),
        stage2=request.stage2.model_dump(),
        universe=request.universe,
        top_n=request.top_n,
        target_week_start=request.target_week_start,
        target_week_end=request.target_week_end,
        as_of_date=request.as_of_date,
    )

    logger.debug(f"Generated Double HRP HTML body length: {len(html_body)} chars")

    subject = (
        f"Double HRP Portfolio Analysis ({request.target_week_start} "
        f"-> {request.target_week_end})"
    )

    try:
        send_html_email(subject=subject, html_body=html_body)
        is_success = True
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

    logger.info("Double HRP report email sent successfully")

    return WeeklyReportEmailResponse(
        is_success=is_success,
        subject=subject,
        body=html_body,
    )


@router.post("/us-double-hrp-report", response_model=WeeklyReportEmailResponse)
def send_us_double_hrp_report_email(
    request: USDoubleHRPEmailRequest,
) -> WeeklyReportEmailResponse:
    """Send a US Double HRP portfolio analysis email.

    Mirrors the India Double HRP email but adds an Alpaca order execution
    section (because US trades through a paper account) and a skip block
    when last week's orders were still open at run time.

    Args:
        request: US Double HRP report data (stages + summary + orders).

    Returns:
        Response with success status, subject, and HTML body.

    Raises:
        HTTPException: If template loading or email sending fails.
    """
    logger.info("Generating US Double HRP report email")

    try:
        env = get_jinja_env()
        template = env.get_template("us_double_hrp_report_email.html.j2")
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="Template not found: us_double_hrp_report_email.html.j2",
        ) from e

    order_results = (
        request.order_results.model_dump() if request.order_results else None
    )

    html_body = template.render(
        summary=request.summary,
        stage1=request.stage1.model_dump(),
        stage2=request.stage2.model_dump(),
        universe=request.universe,
        top_n=request.top_n,
        target_week_start=request.target_week_start,
        target_week_end=request.target_week_end,
        as_of_date=request.as_of_date,
        order_results=order_results,
        skipped=request.skipped,
        sticky_kept_count=request.sticky_kept_count,
        sticky_fillers_count=request.sticky_fillers_count,
        previous_year_week_used=request.previous_year_week_used,
    )

    logger.debug(f"Generated US Double HRP HTML body length: {len(html_body)} chars")

    if request.skipped:
        subject = (
            f"US Double HRP Skipped ({request.target_week_start} "
            f"-> {request.target_week_end})"
        )
    else:
        subject = (
            f"US Double HRP Portfolio Analysis ({request.target_week_start} "
            f"-> {request.target_week_end})"
        )

    try:
        send_html_email(subject=subject, html_body=html_body)
        is_success = True
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

    logger.info("US Double HRP report email sent successfully")

    return WeeklyReportEmailResponse(
        is_success=is_success,
        subject=subject,
        body=html_body,
    )
