"""SQLite-backed history of Stage 1 HRP weights and final selections.

This is the persistence layer behind the sticky-selection (rebalance-band)
primitive. Stage 1 of a Double HRP workflow runs across the full universe
(~410 symbols for halal_new) and produces a per-stock weight. Those rows
are persisted here keyed by (universe, year_week, stock). The next week,
the workflow reads back last week's selected set + initial weights, applies
a no-trade band against this week's stage 1 weights, and chooses the final
top-N. Stage 2 weights are recorded back into the same rows after they are
computed.

Rerun semantics
---------------
``persist_stage1`` performs a full **delete-then-insert** for every
``(universe, year_week)`` pair present in the batch. A rerun for the
same week therefore produces an authoritative replacement of the prior
week's rows — including dropping stocks that were in the first attempt
but not in the rerun. This matches the project's rerun rule that a
rerun produces a fresh result for the week rather than merging history.
If you ever need an immutable audit log, add a separate append-only
table; do not change this rule without coordinating with the workflow's
rerun logic.

Year-week ordering
------------------
ISO year-week is stored as a 6-character string ``YYYYWW`` (e.g.
``"202608"``). Lexicographic comparison gives correct chronological
ordering across year boundaries because the year prefix dominates
(``"202552" < "202601"``).

Schema
------
``stage1_weight_history`` — one row per (universe, year_week, stock):

    universe              TEXT     NOT NULL
    year_week             TEXT     NOT NULL    -- "YYYYWW"
    as_of_date            TEXT     NOT NULL    -- "YYYY-MM-DD"
    stock                 TEXT     NOT NULL
    stage1_rank           INTEGER  NOT NULL    -- 1 = highest signal that week
    initial_allocation_pct REAL                -- stage 1 HRP weight in %
                                                -- (weight-band selection only;
                                                -- NULL for rank-band selection)
    signal_score          REAL                 -- raw stage 1 signal score
                                                -- (rank-band selection only;
                                                -- NULL for weight-band selection)
    final_allocation_pct  REAL                 -- stage 2 HRP weight in %, NULL
                                                -- if not selected or HRP dropped
    selected_in_final     INTEGER  NOT NULL DEFAULT 0
    selection_reason      TEXT                 -- "sticky" | "top_rank" | NULL
    run_id                TEXT     NOT NULL
    created_at            TEXT     NOT NULL DEFAULT (datetime('now'))
    PRIMARY KEY (universe, year_week, stock)

The two stage-1 columns (``initial_allocation_pct`` and ``signal_score``)
are mutually exclusive per row -- which one is populated depends on the
selection policy that wrote the row. Mixing them in a single column
would corrupt downstream consumers that assume the column units
(percent allocation Σ ≈ 100%) -- see ``select_with_rank_band`` in
``brain_api/core/sticky_selection.py`` for the rank-band signal contract.

Multi-universe
--------------
The same DB file holds rows for every universe (``halal_new``,
``nifty_shariah_500``, ...). The ``universe`` column scopes every read.
``read_previous_final_set`` is universe-aware: it never returns rows from
a different universe even if their year_week is closer.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from brain_api.storage.base import DEFAULT_DATA_PATH

DEFAULT_DB_PATH = DEFAULT_DATA_PATH / "allocation" / "sticky_history.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stage1_weight_history (
  universe              TEXT     NOT NULL,
  year_week             TEXT     NOT NULL,
  as_of_date            TEXT     NOT NULL,
  stock                 TEXT     NOT NULL,
  stage1_rank           INTEGER  NOT NULL,
  initial_allocation_pct REAL,
  signal_score          REAL,
  final_allocation_pct  REAL,
  selected_in_final     INTEGER  NOT NULL DEFAULT 0,
  selection_reason      TEXT,
  run_id                TEXT     NOT NULL,
  created_at            TEXT     NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (universe, year_week, stock)
);
CREATE INDEX IF NOT EXISTS idx_universe_yearweek
  ON stage1_weight_history(universe, year_week);
"""

# Recreate-table migration for legacy DBs whose
# ``initial_allocation_pct`` column was created with NOT NULL (before the
# rank-band signal_score split). SQLite has no ALTER COLUMN so we copy
# rows into a new table with the relaxed schema.
_LEGACY_RECREATE_SQL = """
BEGIN;
CREATE TABLE _stage1_weight_history_new (
  universe              TEXT     NOT NULL,
  year_week             TEXT     NOT NULL,
  as_of_date            TEXT     NOT NULL,
  stock                 TEXT     NOT NULL,
  stage1_rank           INTEGER  NOT NULL,
  initial_allocation_pct REAL,
  signal_score          REAL,
  final_allocation_pct  REAL,
  selected_in_final     INTEGER  NOT NULL DEFAULT 0,
  selection_reason      TEXT,
  run_id                TEXT     NOT NULL,
  created_at            TEXT     NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (universe, year_week, stock)
);
INSERT INTO _stage1_weight_history_new (
  universe, year_week, as_of_date, stock, stage1_rank,
  initial_allocation_pct, signal_score, final_allocation_pct,
  selected_in_final, selection_reason, run_id, created_at
)
SELECT
  universe, year_week, as_of_date, stock, stage1_rank,
  initial_allocation_pct, NULL, final_allocation_pct,
  selected_in_final, selection_reason, run_id, created_at
FROM stage1_weight_history;
DROP TABLE stage1_weight_history;
ALTER TABLE _stage1_weight_history_new RENAME TO stage1_weight_history;
CREATE INDEX IF NOT EXISTS idx_universe_yearweek
  ON stage1_weight_history(universe, year_week);
COMMIT;
"""


@dataclass(frozen=True)
class WeightRow:
    """One row of stage 1 weight history.

    ``selected_in_final`` and ``selection_reason`` reflect the sticky
    selection outcome. ``final_allocation_pct`` is filled in later by
    :meth:`StickyHistoryRepository.update_final_weights` once stage 2 has
    run. Exactly one of ``initial_allocation_pct`` (weight-band
    selection) or ``signal_score`` (rank-band selection) is populated;
    the other stays ``None``. Mixing the two would let a rank-band
    score (e.g. PatchTST predicted return) leak into a column whose
    semantics is "Stage 1 HRP weight in %" and break consumers that
    assume Σ ≈ 100% across the universe.
    """

    universe: str
    year_week: str
    as_of_date: str
    stock: str
    stage1_rank: int
    initial_allocation_pct: float | None
    final_allocation_pct: float | None
    selected_in_final: bool
    selection_reason: str | None
    run_id: str
    signal_score: float | None = None


@dataclass(frozen=True)
class PreviousWeekSnapshot:
    """View of the most recent prior week for a given universe.

    Returned by :meth:`StickyHistoryRepository.read_previous_final_set` to
    drive sticky-selection logic without exposing repository internals.

    ``initial_allocation_by_stock`` is keyed only by stocks that wrote a
    Stage 1 HRP weight (weight-band selection); rank-band rows are
    omitted because their Stage 1 column carries scores, not weights.
    Rank-band stickiness only needs ``final_set``, so omitting them
    here also avoids leaking score values into a "weight" map.
    """

    year_week: str
    initial_allocation_by_stock: dict[str, float]
    final_set: set[str]


class StickyHistoryRepository:
    """SQLite repository for stage 1 weight history.

    Stateless except for the configured DB path; safe to instantiate once
    per request via FastAPI dependency injection.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._migrate_schema(conn)
            conn.commit()

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        """Apply idempotent schema migrations for legacy DBs.

        Adds the ``signal_score`` column when missing and, if
        ``initial_allocation_pct`` was created with NOT NULL (pre
        rank-band schema), copies the table into a new shape that
        relaxes that constraint. SQLite has no ``ALTER COLUMN`` so the
        latter requires a temporary table.
        """
        info = conn.execute("PRAGMA table_info(stage1_weight_history)").fetchall()
        cols_by_name = {row["name"]: row for row in info}
        if "signal_score" not in cols_by_name:
            conn.execute(
                "ALTER TABLE stage1_weight_history ADD COLUMN signal_score REAL"
            )
            info = conn.execute("PRAGMA table_info(stage1_weight_history)").fetchall()
            cols_by_name = {row["name"]: row for row in info}
        initial_col = cols_by_name.get("initial_allocation_pct")
        if initial_col is not None and initial_col["notnull"] == 1:
            conn.executescript(_LEGACY_RECREATE_SQL)

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived connection. Caller is responsible for closing.

        Used inside ``with`` blocks so commits + closes are deterministic.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def persist_stage1(self, rows: Iterable[WeightRow]) -> int:
        """Replace stage 1 weight rows for every (universe, year_week) batch.

        Implementation note: this is a delete-then-insert, scoped to the
        ``(universe, year_week)`` pairs present in ``rows``. A rerun
        therefore drops stocks that were in the first attempt but not in
        the rerun, matching the project's rerun-as-replacement rule.
        Other weeks and universes are untouched.

        Returns the number of rows persisted (after replacement).
        """
        rows = list(rows)
        if not rows:
            return 0

        affected_weeks: set[tuple[str, str]] = {(r.universe, r.year_week) for r in rows}

        records = [
            (
                r.universe,
                r.year_week,
                r.as_of_date,
                r.stock,
                r.stage1_rank,
                r.initial_allocation_pct,
                r.signal_score,
                r.final_allocation_pct,
                int(r.selected_in_final),
                r.selection_reason,
                r.run_id,
            )
            for r in rows
        ]

        with self._connect() as conn:
            for universe, year_week in affected_weeks:
                conn.execute(
                    """
                    DELETE FROM stage1_weight_history
                    WHERE universe = ? AND year_week = ?
                    """,
                    (universe, year_week),
                )
            conn.executemany(
                """
                INSERT INTO stage1_weight_history (
                    universe, year_week, as_of_date, stock, stage1_rank,
                    initial_allocation_pct, signal_score, final_allocation_pct,
                    selected_in_final, selection_reason, run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()
        return len(records)

    def update_final_weights(
        self,
        universe: str,
        year_week: str,
        final: dict[str, float],
    ) -> int:
        """Fill in stage 2 final weights for the (universe, year_week).

        Only rows whose stock appears in ``final`` are touched; their
        ``final_allocation_pct`` is set to the provided weight and
        ``selected_in_final`` is set to 1. Rows for stocks that were
        considered but not selected are left untouched.

        Returns the number of rows updated. Missing stocks (e.g. selected
        symbols that were not part of the persisted stage 1 set) are
        ignored silently — the caller has already committed to those
        symbols by passing them in stage 2.
        """
        if not final:
            return 0

        params = [(pct, universe, year_week, stock) for stock, pct in final.items()]
        with self._connect() as conn:
            cursor = conn.executemany(
                """
                UPDATE stage1_weight_history
                SET final_allocation_pct = ?, selected_in_final = 1
                WHERE universe = ? AND year_week = ? AND stock = ?
                """,
                params,
            )
            conn.commit()
            return cursor.rowcount

    def read_week(self, universe: str, year_week: str) -> list[WeightRow]:
        """Return all rows for one (universe, year_week), ordered by rank."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT universe, year_week, as_of_date, stock, stage1_rank,
                       initial_allocation_pct, signal_score, final_allocation_pct,
                       selected_in_final, selection_reason, run_id
                FROM stage1_weight_history
                WHERE universe = ? AND year_week = ?
                ORDER BY stage1_rank ASC
                """,
                (universe, year_week),
            ).fetchall()
        return [
            WeightRow(
                universe=r["universe"],
                year_week=r["year_week"],
                as_of_date=r["as_of_date"],
                stock=r["stock"],
                stage1_rank=r["stage1_rank"],
                initial_allocation_pct=r["initial_allocation_pct"],
                signal_score=r["signal_score"],
                final_allocation_pct=r["final_allocation_pct"],
                selected_in_final=bool(r["selected_in_final"]),
                selection_reason=r["selection_reason"],
                run_id=r["run_id"],
            )
            for r in rows
        ]

    def read_previous_final_set(
        self,
        universe: str,
        current_year_week: str,
    ) -> PreviousWeekSnapshot | None:
        """Return the most recent prior week's snapshot for sticky logic.

        ``previous`` means the largest ``year_week`` strictly less than
        ``current_year_week`` for the given universe. Returns ``None`` if
        no such week exists (cold start).

        Math invariant -- ``final_set`` membership:

        ``final_set`` contains exactly the symbols that received a
        non-null Stage 2 weight (``final_allocation_pct IS NOT NULL``).
        We do NOT use ``selected_in_final`` because Stage 1 may flag a
        symbol as "selected" before HRP runs, and HRP can drop symbols
        for missing covariance / data alignment. The asymmetric K_in /
        K_hold rebalance band is Markov in the *realized* Stage 2
        portfolio, so feeding it the Stage-1-selected superset would
        bias retention vs eviction. If Stage 2 never ran (workflow
        crashed before ``record_final_weights``), the carry set is
        correctly empty -- the strategy effectively starts from a
        cold-start the next week, which matches reality on the
        Alpaca account.
        """
        with self._connect() as conn:
            prev_yw = conn.execute(
                """
                SELECT MAX(year_week) AS yw
                FROM stage1_weight_history
                WHERE universe = ? AND year_week < ?
                """,
                (universe, current_year_week),
            ).fetchone()["yw"]

            if prev_yw is None:
                return None

            rows = conn.execute(
                """
                SELECT stock, initial_allocation_pct, signal_score,
                       final_allocation_pct
                FROM stage1_weight_history
                WHERE universe = ? AND year_week = ?
                """,
                (universe, prev_yw),
            ).fetchall()

        initial_by_stock: dict[str, float] = {}
        final_set: set[str] = set()
        for r in rows:
            initial = r["initial_allocation_pct"]
            if initial is not None:
                initial_by_stock[r["stock"]] = initial
            if r["final_allocation_pct"] is not None:
                final_set.add(r["stock"])

        return PreviousWeekSnapshot(
            year_week=prev_yw,
            initial_allocation_by_stock=initial_by_stock,
            final_set=final_set,
        )


def get_sticky_history_repo() -> StickyHistoryRepository:
    """FastAPI dependency factory.

    Tests override this via ``app.dependency_overrides[get_sticky_history_repo]``
    to inject a repository pointed at a temp path.
    """
    return StickyHistoryRepository()
