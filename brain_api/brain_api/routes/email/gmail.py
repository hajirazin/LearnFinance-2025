"""Gmail SMTP helper for sending HTML emails."""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


class GmailConfigError(Exception):
    """Raised when Gmail configuration is missing or invalid."""


def get_gmail_config() -> dict[str, str | list[str]]:
    """Get Gmail configuration from environment variables.

    Returns:
        dict with keys: user, password, to, cc (optional)

    Raises:
        GmailConfigError: If required environment variables are missing.
    """
    user = os.environ.get("GMAIL_USER")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    to = os.environ.get("TRAINING_EMAIL_TO")
    cc_str = os.environ.get("TRAINING_EMAIL_CC", "")

    if not user:
        raise GmailConfigError("GMAIL_USER environment variable is required")
    if not password:
        raise GmailConfigError("GMAIL_APP_PASSWORD environment variable is required")
    if not to:
        raise GmailConfigError("TRAINING_EMAIL_TO environment variable is required")

    # Parse CC as comma-separated list
    cc = [addr.strip() for addr in cc_str.split(",") if addr.strip()]

    return {
        "user": user,
        "password": password,
        "to": to,
        "cc": cc,
    }


def send_html_email(subject: str, html_body: str) -> bool:
    """Send an HTML email via Gmail SMTP.

    All configuration comes from environment variables:
    - GMAIL_USER: sender address (from)
    - GMAIL_APP_PASSWORD: Gmail app password (not regular password)
    - TRAINING_EMAIL_TO: recipient address
    - TRAINING_EMAIL_CC: CC recipients (optional, comma-separated)

    Uses SMTP_SSL on port 465 to smtp.gmail.com.

    Args:
        subject: Email subject line.
        html_body: HTML content of the email.

    Returns:
        True if email was sent successfully.

    Raises:
        GmailConfigError: If required configuration is missing.
        smtplib.SMTPException: If SMTP operation fails.
    """
    config = get_gmail_config()

    # Create message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config["user"]
    msg["To"] = config["to"]

    # Add CC if provided
    cc_list: list[str] = config["cc"]  # type: ignore[assignment]
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)

    # Attach HTML body
    html_part = MIMEText(html_body, "html")
    msg.attach(html_part)

    # Build recipient list (To + CC)
    recipients = [config["to"]]
    recipients.extend(cc_list)

    logger.info(f"Sending email to {config['to']} (CC: {cc_list})")

    # Send via Gmail SMTP
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(config["user"], config["password"])  # type: ignore[arg-type]
        server.sendmail(config["user"], recipients, msg.as_string())  # type: ignore[arg-type]

    logger.info("Email sent successfully")
    return True
