"""Training summary email endpoint."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from .gmail import GmailConfigError, send_html_email
from .models import TrainingSummaryEmailRequest, TrainingSummaryEmailResponse

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


@router.post("/training-summary", response_model=TrainingSummaryEmailResponse)
def send_training_summary_email(
    request: TrainingSummaryEmailRequest,
) -> TrainingSummaryEmailResponse:
    """Send a training summary email.

    Takes all 4 training results (LSTM, PatchTST, PPO, SAC) and the LLM-generated
    summary, renders an HTML email using Jinja2, and sends via Gmail SMTP.

    Email configuration comes from environment variables:
    - GMAIL_USER: sender address
    - GMAIL_APP_PASSWORD: Gmail app password
    - TRAINING_EMAIL_TO: recipient address
    - TRAINING_EMAIL_CC: CC recipients (optional)

    Args:
        request: Training results and LLM summary.

    Returns:
        Response with success status, subject, and HTML body.

    Raises:
        HTTPException: If template loading or email sending fails.
    """
    logger.info("Generating training summary email")

    # Load and render the Jinja2 template
    try:
        env = get_jinja_env()
        template = env.get_template("training_summary_email.html.j2")
    except TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="Template not found: training_summary_email.html.j2",
        ) from e

    # Render HTML body with training data
    html_body = template.render(
        lstm=request.lstm.model_dump(),
        patchtst=request.patchtst.model_dump(),
        ppo=request.ppo.model_dump(),
        sac=request.sac.model_dump(),
        summary=request.summary,
    )

    logger.debug(f"Generated HTML body length: {len(html_body)} chars")

    # Build subject line
    subject = (
        f"Training Summary: {request.lstm.data_window_start} "
        f"to {request.lstm.data_window_end}"
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

    logger.info("Training summary email sent successfully")

    return TrainingSummaryEmailResponse(
        is_success=is_success,
        subject=subject,
        body=html_body,
    )
