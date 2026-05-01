"""Training summary email endpoints.

Three endpoints share the same render -> SMTP-send pipeline (see
:func:`_send_training_email`); each only owns its template name, request
DTO, and subject prefix:

* ``/email/forecasters-training-summary`` -- US LSTM + PatchTST
  (called by the US Forecasters Temporal workflow).
* ``/email/sac-training-summary`` -- US SAC (called by the US SAC
  Temporal workflow which runs 12+ hours later).
* ``/email/india-training-summary`` -- India PatchTST.
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from .gmail import GmailConfigError, send_html_email
from .models import (
    ForecastersTrainingSummaryEmailRequest,
    IndiaTrainingSummaryEmailRequest,
    IndiaTrainingSummaryEmailResponse,
    SACTrainingSummaryEmailRequest,
    TrainingSummaryEmailResponse,
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


def _send_training_email(
    *,
    template_name: str,
    template_context: dict[str, Any],
    subject: str,
    log_label: str,
) -> tuple[bool, str]:
    """Render the Jinja HTML template and send via Gmail SMTP.

    Shared by every training-summary email endpoint in this module so
    that template loading, Gmail-config errors, and SMTP errors map to
    the same HTTP responses across the three variants.

    Args:
        template_name: Jinja template filename in
            ``brain_api/templates``.
        template_context: Variables to render into the HTML body.
        subject: Email subject line.
        log_label: Short tag used in log lines (``forecasters``,
            ``sac``, ``india``).

    Returns:
        Tuple of ``(is_success, html_body)``.

    Raises:
        HTTPException: 500 if the template is missing or Gmail is
            misconfigured, 503 if SMTP send fails.
    """
    logger.info(f"Generating {log_label} training summary email")

    try:
        env = get_jinja_env()
        template = env.get_template(template_name)
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Template not found: {template_name}",
        ) from e

    html_body = template.render(**template_context)
    logger.debug(f"Generated HTML body length: {len(html_body)} chars")

    try:
        send_html_email(subject=subject, html_body=html_body)
    except GmailConfigError as e:
        logger.error(f"Gmail configuration error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Gmail configuration error: {e}",
        ) from e
    except Exception as e:
        logger.error(f"Failed to send {log_label} training email: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to send email: {e}",
        ) from e

    logger.info(f"{log_label} training summary email sent successfully")
    return True, html_body


@router.post(
    "/forecasters-training-summary", response_model=TrainingSummaryEmailResponse
)
def send_forecasters_training_summary_email(
    request: ForecastersTrainingSummaryEmailRequest,
) -> TrainingSummaryEmailResponse:
    """Send the US Forecasters (LSTM + PatchTST) training summary email."""
    subject = (
        f"US Forecasters Training: {request.lstm.data_window_start} "
        f"to {request.lstm.data_window_end}"
    )
    is_success, html_body = _send_training_email(
        template_name="forecasters_training_summary_email.html.j2",
        template_context={
            "lstm": request.lstm.model_dump(),
            "patchtst": request.patchtst.model_dump(),
            "summary": request.summary,
        },
        subject=subject,
        log_label="forecasters",
    )
    return TrainingSummaryEmailResponse(
        is_success=is_success,
        subject=subject,
        body=html_body,
    )


@router.post("/sac-training-summary", response_model=TrainingSummaryEmailResponse)
def send_sac_training_summary_email(
    request: SACTrainingSummaryEmailRequest,
) -> TrainingSummaryEmailResponse:
    """Send the US SAC training summary email."""
    subject = (
        f"US SAC Training: {request.sac.data_window_start} "
        f"to {request.sac.data_window_end}"
    )
    is_success, html_body = _send_training_email(
        template_name="sac_training_summary_email.html.j2",
        template_context={
            "sac": request.sac.model_dump(),
            "summary": request.summary,
        },
        subject=subject,
        log_label="sac",
    )
    return TrainingSummaryEmailResponse(
        is_success=is_success,
        subject=subject,
        body=html_body,
    )


@router.post(
    "/india-training-summary", response_model=IndiaTrainingSummaryEmailResponse
)
def send_india_training_summary_email(
    request: IndiaTrainingSummaryEmailRequest,
) -> IndiaTrainingSummaryEmailResponse:
    """Send the India PatchTST training summary email."""
    subject = (
        f"India Training Summary: {request.patchtst.data_window_start} "
        f"to {request.patchtst.data_window_end}"
    )
    is_success, html_body = _send_training_email(
        template_name="india_training_summary_email.html.j2",
        template_context={
            "patchtst": request.patchtst.model_dump(),
            "summary": request.summary,
        },
        subject=subject,
        log_label="india",
    )
    return IndiaTrainingSummaryEmailResponse(
        is_success=is_success,
        subject=subject,
        body=html_body,
    )
