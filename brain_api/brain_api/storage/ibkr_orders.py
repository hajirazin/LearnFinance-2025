"""SQLite-backed local order ledger for IBKR submissions.

This is the persistence layer for IBKR orders submitted via
``POST /ibkr/submit-orders``. Two reasons brain_api owns this ledger
instead of relying on the IB Gateway alone:

1. **Pre-submit dedup gate.** IBKR will NOT auto-reject duplicate
   ``Order.orderRef`` values (unlike Alpaca, which dedupes on
   ``client_order_id`` server-side). The gateway's open-orders book
   only sees in-flight trades; a previously *terminal* order with the
   same ``orderRef`` would not block a re-submit. The local ledger
   provides the cross-attempt history the dedup gate needs to refuse
   re-submits within the same ``(run_id, attempt)``.

2. **Week-long order history.** ``ib.reqCompletedOrders()`` only
   returns the current Gateway session, which resets daily on the IBC
   soft-restart cadence. The Temporal ``resolve_next_attempt`` /
   ``check_order_statuses`` flows need order visibility for at least
   the past trading week, so we mirror every submission into the
   ledger and serve ``GET /ibkr/order-history`` from it.

Schema kept deliberately broker-agnostic in field names so any future
broker that needs the same ledger shape can reuse the table or a
sibling.

Rerun semantics
---------------
``record_submission`` is an upsert keyed on ``order_ref`` (which is
the deterministic ``client_order_id``). A re-submit attempt would
either be blocked by the dedup gate before reaching this layer, or --
in the case of an updated lifecycle status -- update the existing row
rather than insert a duplicate. The PK is ``order_ref`` (not
``(run_id, attempt, symbol, side)``) because that's what the IBKR
side anchors on too.

Schema
------
``ibkr_submitted_orders`` -- one row per ``order_ref``::

    account            TEXT     NOT NULL    -- "sac_halal"
    run_id             TEXT     NOT NULL
    attempt            INTEGER  NOT NULL
    symbol             TEXT     NOT NULL
    side               TEXT     NOT NULL    -- "buy" | "sell"
    qty                REAL     NOT NULL
    limit_price        REAL                 -- NULL for market orders
    order_ref          TEXT     PRIMARY KEY -- == client_order_id
    ibkr_perm_id       INTEGER              -- IBKR's broker-side permId
    status             TEXT     NOT NULL    -- IBKR lifecycle status
    filled_qty         REAL                 -- mirrored from gateway events
    filled_avg_price   REAL                 -- mirrored from gateway events
    submitted_at       TEXT     NOT NULL DEFAULT (datetime('now'))
    last_updated_at    TEXT     NOT NULL DEFAULT (datetime('now'))
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from brain_api.storage.base import DEFAULT_DATA_PATH

DEFAULT_DB_PATH = DEFAULT_DATA_PATH / "ibkr" / "submitted_orders.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ibkr_submitted_orders (
  account            TEXT     NOT NULL,
  run_id             TEXT     NOT NULL,
  attempt            INTEGER  NOT NULL,
  symbol             TEXT     NOT NULL,
  side               TEXT     NOT NULL,
  qty                REAL     NOT NULL,
  limit_price        REAL,
  order_ref          TEXT     PRIMARY KEY,
  ibkr_perm_id       INTEGER,
  status             TEXT     NOT NULL,
  filled_qty         REAL,
  filled_avg_price   REAL,
  submitted_at       TEXT     NOT NULL DEFAULT (datetime('now')),
  last_updated_at    TEXT     NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ibkr_submitted_account_submitted
  ON ibkr_submitted_orders(account, submitted_at);
"""


@dataclass(frozen=True)
class SubmittedOrderRow:
    """One row of the local IBKR order ledger."""

    account: str
    run_id: str
    attempt: int
    symbol: str
    side: str
    qty: float
    limit_price: float | None
    order_ref: str
    ibkr_perm_id: int | None
    status: str
    filled_qty: float | None
    filled_avg_price: float | None


class IBKROrderLedger:
    """SQLite repository for the local IBKR order ledger.

    Stateless except for the configured DB path. ``record_submission``
    upserts on ``order_ref`` so a status update from the gateway just
    refreshes the existing row.
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

    def record_submission(self, row: SubmittedOrderRow) -> None:
        """Upsert a single submitted order into the ledger.

        ``order_ref`` is the PK; on conflict we refresh the IBKR perm
        id, status, fill data, and ``last_updated_at`` while preserving
        ``submitted_at`` (the original submission timestamp). This lets
        the same row carry the full lifecycle from "Submitted" through
        "Filled" / "Cancelled".
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ibkr_submitted_orders (
                    account, run_id, attempt, symbol, side, qty,
                    limit_price, order_ref, ibkr_perm_id, status,
                    filled_qty, filled_avg_price
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_ref) DO UPDATE SET
                    ibkr_perm_id     = COALESCE(excluded.ibkr_perm_id, ibkr_submitted_orders.ibkr_perm_id),
                    status           = excluded.status,
                    filled_qty       = COALESCE(excluded.filled_qty, ibkr_submitted_orders.filled_qty),
                    filled_avg_price = COALESCE(excluded.filled_avg_price, ibkr_submitted_orders.filled_avg_price),
                    last_updated_at  = datetime('now')
                """,
                (
                    row.account,
                    row.run_id,
                    row.attempt,
                    row.symbol,
                    row.side,
                    row.qty,
                    row.limit_price,
                    row.order_ref,
                    row.ibkr_perm_id,
                    row.status,
                    row.filled_qty,
                    row.filled_avg_price,
                ),
            )
            conn.commit()

    def has_order_ref(self, order_ref: str) -> bool:
        """Return True iff a row with this ``order_ref`` already exists.

        Powers the pre-submit dedup gate. Combined with
        :func:`brain_api.core.ibkr_client.list_open_order_refs` (which
        scans the gateway's live open-trades book), the dedup gate
        catches both:

        * a previously-submitted order that has since reached terminal
          state (visible in the ledger but not on the gateway), AND
        * an open order from a previous workflow attempt within the
          same week (visible on the gateway and in the ledger).
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM ibkr_submitted_orders WHERE order_ref = ? LIMIT 1",
                (order_ref,),
            )
            return cursor.fetchone() is not None

    def list_after(self, account: str, after_iso_date: str) -> list[SubmittedOrderRow]:
        """Return ledger rows for ``account`` submitted on/after ``after_iso_date``.

        ``after_iso_date`` is a calendar date (``YYYY-MM-DD``) so the
        comparison string-prefix-matches against ``submitted_at``
        (which is ``YYYY-MM-DD HH:MM:SS`` from ``datetime('now')``).
        Used by ``GET /ibkr/order-history`` to mirror Alpaca's
        ``after`` semantics.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT account, run_id, attempt, symbol, side, qty,
                       limit_price, order_ref, ibkr_perm_id, status,
                       filled_qty, filled_avg_price
                FROM ibkr_submitted_orders
                WHERE account = ? AND submitted_at >= ?
                ORDER BY submitted_at ASC
                """,
                (account, after_iso_date),
            ).fetchall()
        return [
            SubmittedOrderRow(
                account=r["account"],
                run_id=r["run_id"],
                attempt=r["attempt"],
                symbol=r["symbol"],
                side=r["side"],
                qty=r["qty"],
                limit_price=r["limit_price"],
                order_ref=r["order_ref"],
                ibkr_perm_id=r["ibkr_perm_id"],
                status=r["status"],
                filled_qty=r["filled_qty"],
                filled_avg_price=r["filled_avg_price"],
            )
            for r in rows
        ]

    def update_status_batch(
        self, updates: Iterable[tuple[str, str, float | None, float | None]]
    ) -> None:
        """Bulk-update ``(order_ref, status, filled_qty, filled_avg_price)`` rows.

        Used during ``sell_wait_buy`` polling so the ledger eventually
        converges with the gateway's view even after the gateway
        soft-restarts and forgets the daily session. Rows that don't
        already exist are silently ignored -- this method only
        refreshes existing submissions, never inserts new ones.
        """
        with self._connect() as conn:
            for order_ref, status, filled_qty, filled_avg_price in updates:
                conn.execute(
                    """
                    UPDATE ibkr_submitted_orders SET
                      status           = ?,
                      filled_qty       = COALESCE(?, filled_qty),
                      filled_avg_price = COALESCE(?, filled_avg_price),
                      last_updated_at  = datetime('now')
                    WHERE order_ref = ?
                    """,
                    (status, filled_qty, filled_avg_price, order_ref),
                )
            conn.commit()


def get_ibkr_order_ledger() -> IBKROrderLedger:
    """FastAPI dependency factory.

    Tests override this via
    ``app.dependency_overrides[get_ibkr_order_ledger]`` to inject a
    repository pointed at a temp DB path.
    """
    return IBKROrderLedger()
