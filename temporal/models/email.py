"""Models for email endpoints."""

from pydantic import BaseModel


class TrainingSummaryEmailResponse(BaseModel):
    """Response model for POST /email/training-summary.

    Works with the brain_api /email/training-summary endpoint.
    """

    is_success: bool
    subject: str
    body: str  # Full HTML body (for debugging/logging)
