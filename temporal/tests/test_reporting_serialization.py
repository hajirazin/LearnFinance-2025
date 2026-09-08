"""Regression tests for reporting activity JSON payloads."""

import json
from datetime import UTC, datetime, timedelta

from activities.reporting import _alloc_to_dict
from models import SACInferenceResponse
from models.news import SACNewsAudit


def test_sac_allocation_dump_is_json_serializable() -> None:
    """Nested news audit datetimes must be encoded before httpx sees them."""
    as_of = datetime(2026, 8, 31, 13, 0, tzinfo=UTC)
    allocation = SACInferenceResponse(
        target_weights={"AAPL": 0.8, "CASH": 0.2},
        turnover=0.1,
        model_version="test-version",
        asset_eligibility={"AAPL": True},
        regime_posterior=[0.7, 0.2, 0.1],
        sac_schema_version=3,
        architecture="masked_attention",
        news_audit=SACNewsAudit(
            as_of=as_of,
            start_exclusive=as_of - timedelta(days=7),
            end_inclusive=as_of,
            per_symbol=[],
        ),
    )

    payload = _alloc_to_dict(allocation)

    assert payload["news_audit"]["as_of"] == "2026-08-31T13:00:00Z"
    json.dumps(payload)
