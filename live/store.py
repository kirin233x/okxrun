"""SQLite state and the append-only event log.

The exchange is the source of truth for positions; this database is the source
of truth for *why* the executor did something, plus the small amount of state
the exchange cannot tell us (a position's entry anchor and its high-water mark
since the day opened).

Everything the executor decides lands in ``events``, which is only ever
appended to. That is what the portal reads and what makes a bad day auditable
months later.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_ts ON events (ts);
CREATE INDEX IF NOT EXISTS events_kind_ts ON events (kind, ts);

CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    inst_id TEXT NOT NULL,
    side TEXT NOT NULL,
    contracts TEXT NOT NULL,
    notional_usdt REAL NOT NULL,
    reduce_only INTEGER NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    response TEXT
);
CREATE INDEX IF NOT EXISTS orders_ts ON orders (ts);

CREATE TABLE IF NOT EXISTS position_state (
    inst_id TEXT PRIMARY KEY,
    trading_day TEXT NOT NULL,
    direction INTEGER NOT NULL,
    entry REAL NOT NULL,
    stop_fraction REAL NOT NULL,
    peak REAL NOT NULL,
    trough REAL NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_marks (
    ts TEXT PRIMARY KEY,
    trading_day TEXT NOT NULL,
    equity_usdt REAL NOT NULL,
    gross_notional_usdt REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS equity_marks_day ON equity_marks (trading_day);

CREATE TABLE IF NOT EXISTS trading_days (
    trading_day TEXT PRIMARY KEY,
    opening_equity_usdt REAL NOT NULL,
    killed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""


@dataclass
class PositionState:
    inst_id: str
    trading_day: str
    direction: int
    entry: float
    stop_fraction: float
    peak: float
    trough: float


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        with closing(self._connection.cursor()) as cursor:
            cursor.executescript(SCHEMA)

    def close(self) -> None:
        self._connection.close()

    # ----------------------------------------------------------------- events

    def log(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        self._connection.execute(
            "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
            (_now(), kind, json.dumps(payload or {}, ensure_ascii=False, default=str)),
        )

    def recent_events(self, limit: int = 200, kind: str | None = None) -> list[dict[str, Any]]:
        if kind:
            rows = self._connection.execute(
                "SELECT ts, kind, payload FROM events WHERE kind = ? ORDER BY id DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT ts, kind, payload FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"ts": row["ts"], "kind": row["kind"], "payload": json.loads(row["payload"])} for row in rows]

    # ----------------------------------------------------------------- orders

    def record_order(
        self,
        client_order_id: str,
        inst_id: str,
        side: str,
        contracts: Decimal,
        notional_usdt: float,
        reduce_only: bool,
        reason: str,
        status: str = "PENDING",
    ) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO orders "
            "(client_order_id, ts, inst_id, side, contracts, notional_usdt, reduce_only, reason, status, response) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT response FROM orders WHERE client_order_id = ?), NULL))",
            (
                client_order_id,
                _now(),
                inst_id,
                side,
                str(contracts),
                notional_usdt,
                int(reduce_only),
                reason,
                status,
                client_order_id,
            ),
        )

    def finish_order(self, client_order_id: str, status: str, response: dict[str, Any] | None) -> None:
        self._connection.execute(
            "UPDATE orders SET status = ?, response = ? WHERE client_order_id = ?",
            (status, json.dumps(response or {}, ensure_ascii=False, default=str), client_order_id),
        )

    def has_order(self, client_order_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return row is not None

    def orders_in_last_hour(self, now: datetime | None = None) -> int:
        cutoff = ((now or datetime.now(timezone.utc)) - timedelta(hours=1)).isoformat()
        row = self._connection.execute(
            "SELECT COUNT(*) AS total FROM orders WHERE ts >= ? AND status != 'REJECTED'", (cutoff,)
        ).fetchone()
        return int(row["total"])

    def recent_orders(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM orders ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    # --------------------------------------------------------------- position

    def position_state(self, inst_id: str) -> PositionState | None:
        row = self._connection.execute(
            "SELECT * FROM position_state WHERE inst_id = ?", (inst_id,)
        ).fetchone()
        if row is None:
            return None
        return PositionState(
            inst_id=row["inst_id"],
            trading_day=row["trading_day"],
            direction=int(row["direction"]),
            entry=float(row["entry"]),
            stop_fraction=float(row["stop_fraction"]),
            peak=float(row["peak"]),
            trough=float(row["trough"]),
        )

    def all_position_state(self) -> dict[str, PositionState]:
        rows = self._connection.execute("SELECT inst_id FROM position_state").fetchall()
        states = {}
        for row in rows:
            state = self.position_state(row["inst_id"])
            if state is not None:
                states[state.inst_id] = state
        return states

    def save_position_state(self, state: PositionState) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO position_state "
            "(inst_id, trading_day, direction, entry, stop_fraction, peak, trough, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                state.inst_id,
                state.trading_day,
                state.direction,
                state.entry,
                state.stop_fraction,
                state.peak,
                state.trough,
                _now(),
            ),
        )

    def drop_position_state(self, inst_id: str) -> None:
        self._connection.execute("DELETE FROM position_state WHERE inst_id = ?", (inst_id,))

    # ----------------------------------------------------------------- equity

    def mark_equity(self, trading_day: str, equity_usdt: float, gross_notional_usdt: float) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO equity_marks (ts, trading_day, equity_usdt, gross_notional_usdt) "
            "VALUES (?, ?, ?, ?)",
            (_now(), trading_day, equity_usdt, gross_notional_usdt),
        )

    def equity_history(self, limit: int = 2880) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT ts, trading_day, equity_usdt, gross_notional_usdt FROM equity_marks "
            "ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    # ----------------------------------------------------------- trading days

    def open_trading_day(self, trading_day: str, opening_equity_usdt: float) -> float:
        """Record the day's opening equity, or return the one already recorded.

        The 3% kill switch measures against this, so it must survive a restart
        mid-day: re-anchoring to the current equity after a drawdown would
        quietly reset the day's loss budget.
        """
        row = self._connection.execute(
            "SELECT opening_equity_usdt FROM trading_days WHERE trading_day = ?", (trading_day,)
        ).fetchone()
        if row is not None:
            return float(row["opening_equity_usdt"])
        self._connection.execute(
            "INSERT INTO trading_days (trading_day, opening_equity_usdt, killed, created_at) VALUES (?, ?, 0, ?)",
            (trading_day, opening_equity_usdt, _now()),
        )
        return opening_equity_usdt

    def mark_day_killed(self, trading_day: str) -> None:
        self._connection.execute(
            "UPDATE trading_days SET killed = 1 WHERE trading_day = ?", (trading_day,)
        )

    def day_killed(self, trading_day: str) -> bool:
        row = self._connection.execute(
            "SELECT killed FROM trading_days WHERE trading_day = ?", (trading_day,)
        ).fetchone()
        return bool(row and row["killed"])
