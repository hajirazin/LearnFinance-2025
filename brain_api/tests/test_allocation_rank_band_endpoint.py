"""API tests for /allocation/rank-band-top-n.

These tests inject an ephemeral StickyHistoryRepository pointed at a temp
directory so we never write to the project's ``data/`` folder during
tests.

Per the repo's testing rule, schemas are exercised through API calls
(no schema-only tests).
"""

from __future__ import annotations

import json
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


def _request(
    *,
    scores: dict[str, float],
    universe: str = "halal_new_alpha",
    year_week: str = "202609",
    as_of_date: str = "2026-02-23",
    run_id: str = "paper:2026-02-23",
    top_n: int = 5,
    hold_threshold: int = 10,
) -> dict:
    return {
        "current_scores": scores,
        "universe": universe,
        "year_week": year_week,
        "as_of_date": as_of_date,
        "run_id": run_id,
        "top_n": top_n,
        "hold_threshold": hold_threshold,
    }


def _record_final_weights(
    client: TestClient,
    *,
    selected: list[str],
    universe: str = "halal_new_alpha",
    year_week: str = "202608",
) -> None:
    """Simulate Stage 2 HRP recording to populate the sticky carry-set.

    The sticky carry-set (next week's ``previous_selected_set``) is
    sourced from rows where ``final_allocation_pct IS NOT NULL`` --
    i.e. names that actually received a Stage 2 weight. Tests that want
    eviction / sticky behaviour to fire must therefore record fake
    Stage 2 weights for the prior week before reading sticky.
    """
    weight = round(100.0 / len(selected), 4)
    final_weights = dict.fromkeys(selected, weight)
    response = client.post(
        "/allocation/record-final-weights",
        json={
            "universe": universe,
            "year_week": year_week,
            "final_weights_pct": final_weights,
        },
    )
    assert response.status_code == 200, response.text


# ----------------------------------------------------------------------------
# /allocation/rank-band-top-n
# ----------------------------------------------------------------------------


class TestRankBandTopNEndpoint:
    def test_cold_start_returns_top_rank_only(
        self, client: TestClient, repo: StickyHistoryRepository
    ):
        scores = {f"S{i:03d}": 100.0 - i for i in range(20)}
        response = client.post(
            "/allocation/rank-band-top-n",
            json=_request(scores=scores, top_n=5, hold_threshold=10),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["selected"] == [f"S{i:03d}" for i in range(5)]
        assert body["kept_count"] == 0
        assert body["fillers_count"] == 5
        assert body["previous_year_week_used"] is None
        assert body["top_n"] == 5
        assert body["hold_threshold"] == 10
        assert all(r == "top_rank" for r in body["reasons"].values())

        rows = repo.read_week("halal_new_alpha", "202609")
        assert len(rows) == 20
        selected_rows = [r for r in rows if r.selected_in_final]
        assert {r.stock for r in selected_rows} == set(body["selected"])
        # Rank-band stores the raw signal in ``signal_score``; the
        # ``initial_allocation_pct`` column (units = "% of weight") stays
        # NULL because the score is not a portfolio weight.
        s000 = next(r for r in rows if r.stock == "S000")
        assert s000.signal_score == 100.0
        assert s000.initial_allocation_pct is None

    def test_churnless_second_week_keeps_all(
        self, client: TestClient, repo: StickyHistoryRepository
    ):
        scores_w1 = {"AAA": 5.0, "BBB": 4.0, "CCC": 3.0, "DDD": 2.0, "EEE": 1.0}
        client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores=scores_w1,
                year_week="202608",
                as_of_date="2026-02-16",
                run_id="paper:2026-02-16",
                top_n=3,
                hold_threshold=4,
            ),
        )
        # Stage 2 finalisation: the carry set is the realized Stage 2
        # portfolio (final_allocation_pct IS NOT NULL), not the Stage 1
        # selection. Simulate Stage 2 here so week 2 sees the right
        # previous_selected_set.
        _record_final_weights(client, selected=["AAA", "BBB", "CCC"])
        # Same ranks => all 3 kept.
        response = client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores=scores_w1,
                year_week="202609",
                top_n=3,
                hold_threshold=4,
            ),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["previous_year_week_used"] == "202608"
        assert set(body["selected"]) == {"AAA", "BBB", "CCC"}
        assert body["kept_count"] == 3
        assert body["fillers_count"] == 0
        assert body["evicted_from_previous"] == {}

    def test_rank_out_of_hold_eviction(
        self, client: TestClient, repo: StickyHistoryRepository
    ):
        # Week 1: top_n=3 selects all three, so CCC is in previous_final_set.
        client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 5.0, "BBB": 4.0, "CCC": 3.0},
                year_week="202608",
                as_of_date="2026-02-16",
                run_id="paper:2026-02-16",
                top_n=3,
                hold_threshold=3,
            ),
        )
        _record_final_weights(client, selected=["AAA", "BBB", "CCC"])
        # Week 2: CCC slips to rank 4 (> hold_threshold=3) -> evicted.
        response = client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 5.5, "BBB": 4.5, "DDD": 3.5, "CCC": 1.0},
                year_week="202609",
                top_n=3,
                hold_threshold=3,
            ),
        )
        body = response.json()
        # AAA (rank 1) and BBB (rank 2) still inside hold -> kept.
        # CCC (rank 4) outside hold -> evicted; one filler (DDD, rank 3).
        assert body["evicted_from_previous"] == {"CCC": "rank_out_of_hold"}
        assert set(body["selected"]) == {"AAA", "BBB", "DDD"}
        assert body["kept_count"] == 2
        assert body["fillers_count"] == 1
        assert body["reasons"]["AAA"] == "sticky"
        assert body["reasons"]["BBB"] == "sticky"
        assert body["reasons"]["DDD"] == "top_rank"

    def test_dropped_from_universe_eviction(self, client: TestClient):
        client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 5.0, "BBB": 4.0, "CCC": 3.0},
                year_week="202608",
                as_of_date="2026-02-16",
                run_id="paper:2026-02-16",
                top_n=2,
                hold_threshold=3,
            ),
        )
        _record_final_weights(client, selected=["AAA", "BBB"])
        # BBB no longer in this week's universe.
        response = client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 5.5, "CCC": 3.0, "DDD": 2.0},
                year_week="202609",
                top_n=2,
                hold_threshold=3,
            ),
        )
        body = response.json()
        assert "BBB" in body["evicted_from_previous"]
        assert body["evicted_from_previous"]["BBB"] == "dropped_from_universe"

    def test_universe_scoping_isolates_halal_new_from_halal_new_alpha(
        self, client: TestClient, repo: StickyHistoryRepository
    ):
        # Write under halal_new (e.g. simulated USDoubleHRPWorkflow row)
        # and then write under halal_new_alpha; reads must not cross.
        # ``repo`` is injected to ensure the sticky_history DB exists.
        del repo
        client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 9.0, "BBB": 8.0, "CCC": 7.0},
                universe="halal_new",
                year_week="202608",
                as_of_date="2026-02-16",
                run_id="paper:2026-02-16",
                top_n=2,
                hold_threshold=2,
            ),
        )
        response = client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"ZZZ": 5.0, "YYY": 4.0, "XXX": 3.0},
                universe="halal_new_alpha",
                year_week="202608",
                as_of_date="2026-02-16",
                run_id="paper:2026-02-16",
                top_n=2,
                hold_threshold=2,
            ),
        )
        _record_final_weights(
            client,
            selected=["ZZZ", "YYY"],
            universe="halal_new_alpha",
            year_week="202608",
        )
        # halal_new_alpha cold start should not see halal_new's prior set:
        assert response.json()["previous_year_week_used"] is None
        # And the next week under halal_new_alpha should *only* see
        # halal_new_alpha's prior set, not halal_new's.
        response2 = client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"ZZZ": 5.5, "YYY": 4.5, "XXX": 3.5},
                universe="halal_new_alpha",
                year_week="202609",
                as_of_date="2026-02-23",
                run_id="paper:2026-02-23",
                top_n=2,
                hold_threshold=2,
            ),
        )
        body = response2.json()
        assert body["previous_year_week_used"] == "202608"
        # Sticky kept stocks must come from halal_new_alpha's prior set
        # (ZZZ, YYY), not halal_new's (AAA, BBB).
        assert set(body["selected"]) == {"ZZZ", "YYY"}

    def test_empty_scores_returns_400(self, client: TestClient):
        response = client.post(
            "/allocation/rank-band-top-n",
            json=_request(scores={}),
        )
        assert response.status_code == 400

    def test_hold_threshold_below_top_n_returns_422(self, client: TestClient):
        response = client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 1.0, "BBB": 0.5},
                top_n=5,
                hold_threshold=3,
            ),
        )
        assert response.status_code == 422

    def test_top_n_out_of_range_returns_422(self, client: TestClient):
        response = client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 1.0},
                top_n=0,
                hold_threshold=1,
            ),
        )
        assert response.status_code == 422

    def test_year_week_validation(self, client: TestClient):
        response = client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 1.0},
                year_week="2026-09",
            ),
        )
        assert response.status_code == 422

    def test_record_final_weights_pairs_with_rank_band(
        self, client: TestClient, repo: StickyHistoryRepository
    ):
        # Confirm the existing /allocation/record-final-weights endpoint
        # works against rank-band-persisted rows (same schema).
        client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 5.0, "BBB": 4.0, "CCC": 3.0},
                top_n=2,
                hold_threshold=3,
            ),
        )
        response = client.post(
            "/allocation/record-final-weights",
            json={
                "universe": "halal_new_alpha",
                "year_week": "202609",
                "final_weights_pct": {"AAA": 60.0, "BBB": 40.0},
            },
        )
        assert response.status_code == 200
        assert response.json()["rows_updated"] == 2
        rows = {r.stock: r for r in repo.read_week("halal_new_alpha", "202609")}
        assert rows["AAA"].final_allocation_pct == 60.0
        assert rows["BBB"].final_allocation_pct == 40.0
        assert rows["CCC"].final_allocation_pct is None

    def test_carry_set_excludes_stage1_picks_dropped_by_hrp(
        self, client: TestClient, repo: StickyHistoryRepository
    ):
        """M1 regression: only realized Stage 2 holdings carry forward.

        Stage 1 picks {AAA, BBB, CCC}; Stage 2 (simulated by
        ``record-final-weights``) only weights AAA + BBB because HRP
        dropped CCC for missing covariance data. Next week CCC must NOT
        appear in ``previous_selected_set`` -- it never traded.
        """
        client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 5.0, "BBB": 4.0, "CCC": 3.0, "DDD": 1.0},
                year_week="202608",
                as_of_date="2026-02-16",
                run_id="paper:2026-02-16",
                top_n=3,
                hold_threshold=4,
            ),
        )
        # Stage 2 only weights AAA+BBB. CCC was selected at Stage 1 but
        # never traded.
        client.post(
            "/allocation/record-final-weights",
            json={
                "universe": "halal_new_alpha",
                "year_week": "202608",
                "final_weights_pct": {"AAA": 60.0, "BBB": 40.0},
            },
        )
        # Week 2: rank CCC last so it would otherwise be evicted with
        # ``rank_out_of_hold`` if it were in the carry set. With the M1
        # fix CCC never enters the carry set so no eviction is recorded.
        response = client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 5.5, "BBB": 4.5, "DDD": 3.5, "CCC": 0.1},
                year_week="202609",
                top_n=2,
                hold_threshold=2,
            ),
        )
        body = response.json()
        # Only AAA + BBB are in the carry set. CCC must NOT show up as
        # an evicted previous holding.
        assert "CCC" not in body["evicted_from_previous"]
        assert body["evicted_from_previous"] == {}
        assert body["kept_count"] == 2
        assert set(body["selected"]) == {"AAA", "BBB"}

    def test_nan_score_returns_422(self, client: TestClient):
        """M3: NaN scores break the strict-weak ordering -> reject loudly."""
        payload = _request(
            scores={"AAA": 1.0, "BBB": float("nan"), "CCC": 0.5},
            top_n=2,
            hold_threshold=3,
        )
        # ``json.dumps`` rejects NaN/inf, so build the raw body manually
        # to drive the request through the API's own validation path.
        raw = json.dumps(payload, allow_nan=True)
        response = client.post(
            "/allocation/rank-band-top-n",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert "finite" in response.text.lower() or "nan" in response.text.lower()

    def test_inf_score_returns_422(self, client: TestClient):
        """M3: +inf would dominate ranking without economic meaning."""
        payload = _request(
            scores={"AAA": 1.0, "BBB": float("inf"), "CCC": 0.5},
            top_n=2,
            hold_threshold=3,
        )
        raw = json.dumps(payload, allow_nan=True)
        response = client.post(
            "/allocation/rank-band-top-n",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_single_symbol_universe_cold_start(self, client: TestClient):
        """M6: single-symbol universe with top_n=1 still works."""
        response = client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 1.0},
                top_n=1,
                hold_threshold=1,
            ),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["selected"] == ["AAA"]
        assert body["kept_count"] == 0
        assert body["fillers_count"] == 1

    def test_hold_equals_top_n_degenerate_band(
        self, client: TestClient, repo: StickyHistoryRepository
    ):
        """M6: hold_threshold == top_n collapses the asymmetric band.

        When K_hold == K_in, eviction-by-rank fires the moment a
        previously-held stock falls below rank top_n (no slack). We
        assert the boundary case behaves consistently with the rank
        eviction predicate (rank > hold_threshold).
        """
        client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 5.0, "BBB": 4.0, "CCC": 3.0},
                year_week="202608",
                as_of_date="2026-02-16",
                run_id="paper:2026-02-16",
                top_n=2,
                hold_threshold=2,
            ),
        )
        _record_final_weights(client, selected=["AAA", "BBB"])
        # Week 2: BBB slips to rank 3, exactly one past hold_threshold=2 -> evicted.
        response = client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 6.0, "DDD": 5.0, "BBB": 4.0},
                year_week="202609",
                top_n=2,
                hold_threshold=2,
            ),
        )
        body = response.json()
        assert body["evicted_from_previous"] == {"BBB": "rank_out_of_hold"}
        assert set(body["selected"]) == {"AAA", "DDD"}

    def test_hold_threshold_larger_than_universe_is_permissive(
        self, client: TestClient
    ):
        """M6: hold_threshold > N never fires rank eviction.

        With only 3 symbols, no rank exceeds 3, so a hold_threshold of
        100 means every previously-held name still in the universe is
        retained. Pydantic's ``le=500`` validator caps the user-facing
        max so we do not hit any silent bound issue.
        """
        client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"AAA": 5.0, "BBB": 4.0, "CCC": 3.0},
                year_week="202608",
                as_of_date="2026-02-16",
                run_id="paper:2026-02-16",
                top_n=2,
                hold_threshold=100,
            ),
        )
        _record_final_weights(client, selected=["AAA", "BBB"])
        response = client.post(
            "/allocation/rank-band-top-n",
            json=_request(
                scores={"BBB": 5.0, "AAA": 4.0, "CCC": 3.0},
                year_week="202609",
                top_n=2,
                hold_threshold=100,
            ),
        )
        body = response.json()
        # No rank eviction, both kept.
        assert body["evicted_from_previous"] == {}
        assert body["kept_count"] == 2
        assert set(body["selected"]) == {"AAA", "BBB"}
