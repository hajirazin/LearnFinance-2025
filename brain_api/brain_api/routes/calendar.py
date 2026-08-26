"""Monday 09:00 New York decision window — RL calendar, not the news store."""

from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from brain_api.core.weekly_decision import (
    MondayCutoffError,
    monday_decision_cutoff,
    monday_window_bounds,
    require_monday_decision_cutoff,
)

router = APIRouter()


class MondayDecisionWindowRequest(BaseModel):
    as_of: datetime | None = None
    run_date: date | None = None


class MondayDecisionWindowResponse(BaseModel):
    cutoff: datetime
    start_exclusive: datetime
    end_inclusive: datetime


@router.post("/monday-decision-window", response_model=MondayDecisionWindowResponse)
def monday_decision_window(
    request: MondayDecisionWindowRequest,
) -> MondayDecisionWindowResponse:
    if request.as_of is not None and request.run_date is not None:
        raise HTTPException(
            status_code=422, detail="provide as_of or run_date, not both"
        )
    try:
        if request.as_of is not None:
            cutoff = require_monday_decision_cutoff(request.as_of)
        elif request.run_date is not None:
            cutoff = monday_decision_cutoff(request.run_date)
        else:
            raise HTTPException(status_code=422, detail="as_of or run_date is required")
    except MondayCutoffError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    start_exclusive, end_inclusive = monday_window_bounds(cutoff.date())
    return MondayDecisionWindowResponse(
        cutoff=cutoff,
        start_exclusive=start_exclusive,
        end_inclusive=end_inclusive,
    )
