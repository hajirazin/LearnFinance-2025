"""SQLite-backed history of single-stage screening rounds.

This module is the persistence layer for **single-stage** rank-band
screening strategies whose selection IS the final outcome -- there is no
Stage 2 HRP shrinkage to wait on. The canonical user is the
``halal_filtered`` universe builder (PatchTST predicted weekly return ->
rank-band sticky -> top 15) running on a **monthly cache cadence**.

Why a separate table from ``stage1_weight_history``
---------------------------------------------------
The two-stage table (``stage1_weight_history``, in
``brain_api.storage.sticky_history``) carries columns whose math
invariants only make sense in a Stage 1 + Stage 2 pipeline:

- ``initial_allocation_pct`` -- Stage 1 HRP weight in % (Σ ≈ 100%).
- ``final_allocation_pct``   -- Stage 2 HRP weight in %, NULL if HRP
  dropped the symbol; this is the column ``read_previous_final_set``
  uses to derive the carry-set for the next round.
- ``selected_in_final``      -- set 1 by Stage 1 BUT never re-set to 0
  even if HRP later drops the symbol; therefore it does NOT represent
  the realized carry-set on its own.

For a single-stage strategy:

- No HRP weight is ever computed -> both pct columns would be NULL
  forever, polluting the table.
- The carry-set is exactly "symbols selected this round" -- i.e. the
  ``selected_in_final = 1`` semantics rather than
  ``final_allocation_pct IS NOT NULL``.
- Mixing the two semantics in a single table would force every reader
  to branch on partition or on a kwarg, which is a math-invariant
  violation by code-reuse: the same column would mean different things
  to different callers.

A separate sibling table makes each table's invariants self-evident from
its schema and removes all branching at read time. Both tables live in
the SAME SQLite file (``data/allocation/sticky_history.db``) for ops
simplicity (one backup, one connection path).

Rerun semantics
---------------
``persist_screening_round`` performs a full **delete-then-insert** for
every ``(partition, period_key)`` pair present in the batch, mirroring
the rerun-as-replacement rule of the two-stage table. A rerun in the
same monthly bucket therefore drops symbols that were in the first
attempt but not in the rerun.

Period-key convention
---------------------
``period_key`` is a 6-character ISO ``YYYYWW`` string. **Monthly**
single-stage strategies anchor every write inside a calendar month to
the YYYYWW of that month's first Monday (see
``iso_year_week_of_month_anchor`` in
``brain_api.core.sticky_selection``). This keeps the column type stable
(always a real ISO week), so lex-comparison gives correct chronological
ordering across year boundaries.

Schema
------
``screening_history`` -- one row per ``(partition, period_key, stock)``:

    partition         TEXT     NOT NULL    -- e.g. "halal_filtered_alpha"
    period_key        TEXT     NOT NULL    -- "YYYYWW"
    as_of_date        TEXT     NOT NULL    -- "YYYY-MM-DD"
    stock             TEXT     NOT NULL
    rank              INTEGER  NOT NULL    -- 1 = highest signal that round
    signal_score      REAL     NOT NULL    -- raw numeric signal (e.g. PatchTST %)
    selected          INTEGER  NOT NULL DEFAULT 0
    selection_reason  TEXT                 -- "sticky" | "top_rank" | NULL
    run_id            TEXT     NOT NULL
    created_at        TEXT     NOT NULL DEFAULT (datetime('now'))
    PRIMARY KEY (partition, period_key, stock)

Cross-strategy isolation
------------------------
Partition strings (see ``brain_api.core.strategy_partitions``) MUST be
unique across the union of THIS table and ``stage1_weight_history``.
``ScreeningHistoryRepository`` only reads/writes ``screening_history``;
it never touches the two-stage table. Symmetrically,
``StickyHistoryRepository`` only reads/writes ``stage1_weight_history``.
Cross-table reads are forbidden by construction.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from brain_api.storage.base import DEFAULT_DATA_PATH

DEFAULT_DB_PATH = DEFAULT_DATA_PATH / "allocation" / "sticky_history.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS screening_history (
  partition         TEXT     NOT NULL,
  period_key        TEXT     NOT NULL,
  as_of_date        TEXT     NOT NULL,
  stock             TEXT     NOT NULL,
  rank              INTEGER  NOT NULL,
  signal_score      REAL     NOT NULL,
  selected          INTEGER  NOT NULL DEFAULT 0,
  selection_reason  TEXT,
  run_id            TEXT     NOT NULL,
  created_at        TEXT     NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (partition, period_key, stock)
);
CREATE INDEX IF NOT EXISTS idx_screening_partition_period
  ON screening_history(partition, period_key);
"""


@dataclass(frozen=True)
class ScreeningRow:
    """One row of single-stage screening history.

    ``signal_score`` is REQUIRED (not optional) because rank-band
    selection cannot run without a numeric score per candidate. Persisting
    a NULL would let downstream readers silently see a "missing score"
    that the selector never actually saw.
    """

    partition: str
    period_key: str
    as_of_date: str
    stock: str
    rank: int
    signal_score: float
    selected: bool
    selection_reason: str | None
    run_id: str


@dataclass(frozen=True)
class PreviousScreeningRound:
    """View of the most recent prior screening round for a partition.

    Returned by :meth:`ScreeningHistoryRepository.read_previous_selected_set`
    to drive single-stage rank-band stickiness without exposing repo
    internals.

    ``selected_set`` contains exactly the symbols flagged
    ``selected = 1`` last round -- the carry-set for the rebalance band.
    For single-stage strategies this is the realized "held last round"
    set (no Stage 2 shrinkage exists), so the rank-band band reads it
    directly.
    """

    period_key: str
    selected_set: set[str]


class ScreeningHistoryRepository:
    """SQLite repository for single-stage screening history.

    Stateless except for the configured DB path. Targets the
    ``screening_history`` table only; never queries
    ``stage1_weight_history``.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def persist_screening_round(self, rows: Iterable[ScreeningRow]) -> int:
        """Replace screening rows for every (partition, period_key) batch.

        Implementation note: delete-then-insert scoped to the
        ``(partition, period_key)`` pairs present in ``rows``. A rerun
        of the same monthly bucket therefore produces an authoritative
        replacement -- including dropping stocks that were in the first
        attempt but not in the rerun.

        Returns the number of rows persisted (after replacement).
        """
        rows = list(rows)
        if not rows:
            return 0

        affected_buckets: set[tuple[str, str]] = {
            (r.partition, r.period_key) for r in rows
        }

        records = [
            (
                r.partition,
                r.period_key,
                r.as_of_date,
                r.stock,
                r.rank,
                r.signal_score,
                int(r.selected),
                r.selection_reason,
                r.run_id,
            )
            for r in rows
        ]

        with self._connect() as conn:
            for partition, period_key in affected_buckets:
                conn.execute(
                    """
                    DELETE FROM screening_history
                    WHERE partition = ? AND period_key = ?
                    """,
                    (partition, period_key),
                )
            conn.executemany(
                """
                INSERT INTO screening_history (
                    partition, period_key, as_of_date, stock, rank,
                    signal_score, selected, selection_reason, run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()
        return len(records)

    def read_round(self, partition: str, period_key: str) -> list[ScreeningRow]:
        """Return all rows for one (partition, period_key), ordered by rank."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT partition, period_key, as_of_date, stock, rank,
                       signal_score, selected, selection_reason, run_id
                FROM screening_history
                WHERE partition = ? AND period_key = ?
                ORDER BY rank ASC
                """,
                (partition, period_key),
            ).fetchall()
        return [
            ScreeningRow(
                partition=r["partition"],
                period_key=r["period_key"],
                as_of_date=r["as_of_date"],
                stock=r["stock"],
                rank=r["rank"],
                signal_score=r["signal_score"],
                selected=bool(r["selected"]),
                selection_reason=r["selection_reason"],
                run_id=r["run_id"],
            )
            for r in rows
        ]

    def read_previous_selected_set(
        self,
        partition: str,
        current_period_key: str,
    ) -> PreviousScreeningRound | None:
        """Return the most recent prior round's snapshot for sticky logic.

        ``previous`` means the largest ``period_key`` strictly less than
        ``current_period_key`` for the given partition. Returns ``None``
        if no such round exists (cold start).

        Math invariant -- ``selected_set`` membership:

        ``selected_set`` contains exactly the symbols flagged
        ``selected = 1`` in last round's rows. For single-stage
        strategies the selection IS the final outcome -- there is no
        Stage 2 shrinkage to wait on -- so the realized carry-set is
        precisely the set marked ``selected``. This semantics is
        physically distinct from the two-stage table's
        ``final_allocation_pct IS NOT NULL`` rule and the two MUST NOT
        be conflated.
        """
        with self._connect() as conn:
            prev_pk_row = conn.execute(
                """
                SELECT MAX(period_key) AS pk
                FROM screening_history
                WHERE partition = ? AND period_key < ?
                """,
                (partition, current_period_key),
            ).fetchone()

            prev_pk = prev_pk_row["pk"] if prev_pk_row is not None else None

            if prev_pk is None:
                return None

            rows = conn.execute(
                """
                SELECT stock
                FROM screening_history
                WHERE partition = ? AND period_key = ? AND selected = 1
                """,
                (partition, prev_pk),
            ).fetchall()

        return PreviousScreeningRound(
            period_key=prev_pk,
            selected_set={r["stock"] for r in rows},
        )


def get_screening_history_repo() -> ScreeningHistoryRepository:
    """FastAPI dependency factory.

    Tests override this via
    ``app.dependency_overrides[get_screening_history_repo]`` to inject a
    repository pointed at a temp DB path.
    """
    return ScreeningHistoryRepository()
