"""API tests for /allocation/sticky-top-n and /allocation/record-final-weights.

These tests inject an ephemeral StickyHistoryRepository pointed at a temp
directory so we never write to the project's data/ folder during tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain_api.main import app
from brain_api.storage.sticky_history import (
    StickyHistoryRepository,
    get_sticky_history_repo,
)


@pytest.fixture()
def repo(tmp_path: Path) -> StickyHistoryRepository:
    """Repository with a per-test SQLite file."""
    return StickyHistoryRepository(db_path=tmp_path / "sticky.db")


@pytest.fixture()
def client(repo: StickyHistoryRepository) -> TestClient:
    """TestClient with sticky-history dependency overridden to a temp repo."""
    app.dependency_overrides[get_sticky_history_repo] = lambda: repo
    yield TestClient(app)
    app.dependency_overrides.clear()


def _stage1_payload(weights: dict[str, float]) -> dict:
    """Build an HRPAllocationResponse-shaped dict for the stage1 field."""
    return {
        "percentage_weights": weights,
        "symbols_used": len(weights),
        "symbols_excluded": [],
        "lookback_days": 756,
        "as_of_date": "2026-02-23",
    }


def _finalize_week(
    client: TestClient, *, universe: str, year_week: str, selected: list[str]
) -> None:
    """Mimic Stage 2 by writing equal final weights for ``selected`` stocks.

    Post-M1 the sticky carry-set reflects Stage 2 reality
    (``final_allocation_pct IS NOT NULL``), so any test that primes a
    "previous week" must explicitly finalize it.
    """
    weight = round(100.0 / len(selected), 4)
    client.post(
        "/allocation/record-final-weights",
        json={
            "universe": universe,
            "year_week": year_week,
            "final_weights_pct": dict.fromkeys(selected, weight),
        },
    )


# ----------------------------------------------------------------------------
# /allocation/sticky-top-n
# ----------------------------------------------------------------------------


class TestStickyTopNEndpoint:
    def test_cold_start_returns_top_rank_only(
        self, client: TestClient, repo: StickyHistoryRepository
    ):
        weights = {f"S{i:03d}": 100.0 - i for i in range(20)}
        response = client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": _stage1_payload(weights),
                "universe": "halal_new",
                "year_week": "202609",
                "as_of_date": "2026-02-23",
                "run_id": "paper:2026-02-23",
                "top_n": 5,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["selected"] == [f"S{i:03d}" for i in range(5)]
        assert body["kept_count"] == 0
        assert body["fillers_count"] == 5
        assert body["previous_year_week_used"] is None
        assert all(r == "top_rank" for r in body["reasons"].values())

        # Persisted: all 20 stocks recorded
        rows = repo.read_week("halal_new", "202609")
        assert len(rows) == 20
        # Top 5 marked as selected_in_final
        selected_rows = [r for r in rows if r.selected_in_final]
        assert {r.stock for r in selected_rows} == set(body["selected"])

    def test_sticky_retention_after_prior_week(
        self, client: TestClient, repo: StickyHistoryRepository
    ):
        # Prime last week's selection.
        client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": _stage1_payload(
                    {"AAPL": 5.0, "MSFT": 4.0, "GOOG": 3.0, "NVDA": 2.0, "AMZN": 1.0}
                ),
                "universe": "halal_new",
                "year_week": "202608",
                "as_of_date": "2026-02-16",
                "run_id": "paper:2026-02-16",
                "top_n": 3,
            },
        )
        _finalize_week(
            client,
            universe="halal_new",
            year_week="202608",
            selected=["AAPL", "MSFT", "GOOG"],
        )

        # This week: GOOG slipped slightly (3.0 -> 2.7), AAPL stable (5.0 -> 5.2).
        # MSFT dropped a lot (4.0 -> 1.0). New stock TSLA appears at 4.5.
        response = client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": _stage1_payload(
                    {
                        "AAPL": 5.2,
                        "TSLA": 4.5,
                        "GOOG": 2.7,
                        "NVDA": 2.0,
                        "MSFT": 1.0,
                    }
                ),
                "universe": "halal_new",
                "year_week": "202609",
                "as_of_date": "2026-02-23",
                "run_id": "paper:2026-02-23",
                "top_n": 3,
                "stickiness_threshold_pp": 1.0,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["previous_year_week_used"] == "202608"
        # AAPL kept (diff 0.2), GOOG kept (diff 0.3); MSFT evicted (diff 3.0).
        assert set(body["selected"]) == {"AAPL", "GOOG", "TSLA"}
        assert body["reasons"]["AAPL"] == "sticky"
        assert body["reasons"]["GOOG"] == "sticky"
        assert body["reasons"]["TSLA"] == "top_rank"
        assert body["kept_count"] == 2
        assert body["fillers_count"] == 1
        assert body["evicted_from_previous"] == {"MSFT": "weight_diff"}

    def test_dropped_from_universe_evicted(
        self, client: TestClient, repo: StickyHistoryRepository
    ):
        client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": _stage1_payload({"AAPL": 5.0, "MSFT": 4.0, "GOOG": 3.0}),
                "universe": "halal_new",
                "year_week": "202608",
                "as_of_date": "2026-02-16",
                "run_id": "paper:2026-02-16",
                "top_n": 2,
            },
        )
        _finalize_week(
            client,
            universe="halal_new",
            year_week="202608",
            selected=["AAPL", "MSFT"],
        )

        # MSFT no longer in the universe this week.
        response = client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": _stage1_payload({"AAPL": 5.5, "GOOG": 3.0, "TSLA": 2.0}),
                "universe": "halal_new",
                "year_week": "202609",
                "as_of_date": "2026-02-23",
                "run_id": "paper:2026-02-23",
                "top_n": 2,
            },
        )
        body = response.json()
        assert "MSFT" in body["evicted_from_previous"]
        assert body["evicted_from_previous"]["MSFT"] == "dropped_from_universe"

    def test_threshold_parameterized(self, client: TestClient):
        # Loose threshold (3pp) keeps both stocks despite material moves.
        client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": _stage1_payload({"A": 5.0, "B": 4.0, "C": 1.0}),
                "universe": "halal_new",
                "year_week": "202608",
                "as_of_date": "2026-02-16",
                "run_id": "paper:2026-02-16",
                "top_n": 2,
            },
        )
        _finalize_week(
            client,
            universe="halal_new",
            year_week="202608",
            selected=["A", "B"],
        )
        response = client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": _stage1_payload({"A": 7.0, "B": 6.5, "C": 1.0}),
                "universe": "halal_new",
                "year_week": "202609",
                "as_of_date": "2026-02-23",
                "run_id": "paper:2026-02-23",
                "top_n": 2,
                "stickiness_threshold_pp": 3.0,
            },
        )
        body = response.json()
        assert body["kept_count"] == 2
        assert body["evicted_from_previous"] == {}

    def test_missing_required_field_returns_422(self, client: TestClient):
        # Missing year_week
        response = client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": _stage1_payload({"A": 1.0}),
                "universe": "halal_new",
                "as_of_date": "2026-02-23",
                "run_id": "paper:2026-02-23",
            },
        )
        assert response.status_code == 422

    def test_year_week_must_be_six_chars(self, client: TestClient):
        response = client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": _stage1_payload({"A": 1.0}),
                "universe": "halal_new",
                "year_week": "2026-09",  # 7 chars
                "as_of_date": "2026-02-23",
                "run_id": "paper:2026-02-23",
            },
        )
        assert response.status_code == 422

    def test_top_n_out_of_range_returns_422(self, client: TestClient):
        response = client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": _stage1_payload({"A": 1.0}),
                "universe": "halal_new",
                "year_week": "202609",
                "as_of_date": "2026-02-23",
                "run_id": "paper:2026-02-23",
                "top_n": 0,
            },
        )
        assert response.status_code == 422

    def test_negative_threshold_returns_422(self, client: TestClient):
        response = client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": _stage1_payload({"A": 1.0}),
                "universe": "halal_new",
                "year_week": "202609",
                "as_of_date": "2026-02-23",
                "run_id": "paper:2026-02-23",
                "stickiness_threshold_pp": -1.0,
            },
        )
        assert response.status_code == 422

    def test_empty_weights_returns_400(self, client: TestClient):
        # Pydantic accepts empty dict; we 400 from the endpoint.
        response = client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": _stage1_payload({}),
                "universe": "halal_new",
                "year_week": "202609",
                "as_of_date": "2026-02-23",
                "run_id": "paper:2026-02-23",
            },
        )
        assert response.status_code == 400

    def test_rerun_overwrites_persisted_rows(
        self, client: TestClient, repo: StickyHistoryRepository
    ):
        body1 = {
            "stage1": _stage1_payload({"A": 5.0, "B": 4.0}),
            "universe": "halal_new",
            "year_week": "202609",
            "as_of_date": "2026-02-23",
            "run_id": "paper:2026-02-23",
            "top_n": 1,
        }
        client.post("/allocation/sticky-top-n", json=body1)
        body2 = {
            **body1,
            "stage1": _stage1_payload({"A": 5.5, "C": 4.0}),
            "run_id": "paper:2026-02-23-redo",
        }
        client.post("/allocation/sticky-top-n", json=body2)

        rows = repo.read_week("halal_new", "202609")
        assert {r.stock for r in rows} == {"A", "C"}
        assert all(r.run_id == "paper:2026-02-23-redo" for r in rows)


# ----------------------------------------------------------------------------
# /allocation/record-final-weights
# ----------------------------------------------------------------------------


class TestRecordFinalWeightsEndpoint:
    def test_records_final_weights_after_sticky(
        self, client: TestClient, repo: StickyHistoryRepository
    ):
        client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": _stage1_payload({"A": 5.0, "B": 4.0, "C": 3.0}),
                "universe": "halal_new",
                "year_week": "202609",
                "as_of_date": "2026-02-23",
                "run_id": "paper:2026-02-23",
                "top_n": 2,
            },
        )

        response = client.post(
            "/allocation/record-final-weights",
            json={
                "universe": "halal_new",
                "year_week": "202609",
                "final_weights_pct": {"A": 55.0, "B": 45.0},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["rows_updated"] == 2
        assert body["universe"] == "halal_new"
        assert body["year_week"] == "202609"

        rows = {r.stock: r for r in repo.read_week("halal_new", "202609")}
        assert rows["A"].final_allocation_pct == 55.0
        assert rows["A"].selected_in_final is True
        assert rows["B"].final_allocation_pct == 45.0
        assert rows["C"].final_allocation_pct is None  # not in final_weights

    def test_no_prior_row_returns_zero_gracefully(
        self, client: TestClient, repo: StickyHistoryRepository
    ):
        # No sticky-top-n called first; record-final-weights silently
        # updates 0 rows.
        response = client.post(
            "/allocation/record-final-weights",
            json={
                "universe": "halal_new",
                "year_week": "202609",
                "final_weights_pct": {"A": 100.0},
            },
        )
        assert response.status_code == 200
        assert response.json()["rows_updated"] == 0

    def test_empty_final_weights_returns_zero(self, client: TestClient):
        response = client.post(
            "/allocation/record-final-weights",
            json={
                "universe": "halal_new",
                "year_week": "202609",
                "final_weights_pct": {},
            },
        )
        assert response.status_code == 200
        assert response.json()["rows_updated"] == 0

    def test_missing_field_returns_422(self, client: TestClient):
        response = client.post(
            "/allocation/record-final-weights",
            json={"universe": "halal_new", "year_week": "202609"},
        )
        assert response.status_code == 422

    def test_year_week_validation(self, client: TestClient):
        response = client.post(
            "/allocation/record-final-weights",
            json={
                "universe": "halal_new",
                "year_week": "2026",
                "final_weights_pct": {"A": 100.0},
            },
        )
        assert response.status_code == 422
