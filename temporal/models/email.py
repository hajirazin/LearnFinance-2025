"""Models for email endpoints."""

from pydantic import BaseModel


class TrainingSummaryEmailResponse(BaseModel):
    """Response model for the split training-summary email endpoints.

    Works with both
    ``/email/forecasters-training-summary`` and
    ``/email/sac-training-summary`` (and is shape-compatible with the
    India variant). Returned by the matching activities in
    ``temporal/activities/training.py``.
    """

    is_success: bool
    subject: str
    body: str  # Full HTML body (for debugging/logging)
