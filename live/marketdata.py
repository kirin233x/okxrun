"""Live market data, shaped exactly like the backtest's inputs.

The executor scores the same way the backtest does, which means it must build
its daily bars from confirmed hourly candles through the same code path. Only
confirmed candles are kept: acting on the in-progress bar would score a day
that has not finished.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from strategy import build_daily_inputs

from .okx import OKXClient


# 30-day formation plus a 20-day volatility window, with room for the shift and
# for days dropped by the completeness rule.
REQUIRED_DAYS = 90
PAGE = 100
CONFIRMED = "1"


def fetch_hourly(client: OKXClient, inst_id: str, days: int = REQUIRED_DAYS) -> pd.DataFrame:
    """Confirmed hourly candles for the last ``days``, oldest first."""
    wanted = days * 24
    rows: dict[int, list[str]] = {}
    after: int | None = None
    while len(rows) < wanted:
        batch = client.candles(inst_id, "1H", PAGE, after)
        if not batch:
            break
        for row in batch:
            # row: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
            if len(row) >= 9 and row[8] == CONFIRMED:
                rows[int(row[0])] = row
        after = min(int(row[0]) for row in batch)
        if len(batch) < PAGE:
            break
        time.sleep(0.05)

    if not rows:
        raise RuntimeError(f"{inst_id}: no confirmed hourly candles returned")

    frame = pd.DataFrame(
        [
            {
                "ts": ts,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volumeQuote": float(row[7]),
            }
            for ts, row in rows.items()
        ]
    ).sort_values("ts")
    frame.index = pd.to_datetime(frame.pop("ts"), unit="ms", utc=True)
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def load_inputs(
    client: OKXClient,
    universe: list[str],
    days: int = REQUIRED_DAYS,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Fetch the universe and build the daily frames ``compute_signals`` wants."""
    hourly = {inst_id: fetch_hourly(client, inst_id, days) for inst_id in universe}
    daily, closes = build_daily_inputs(hourly)
    return hourly, daily, closes


def append_pending_day(
    daily: dict[str, pd.DataFrame],
    trading_day: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Add an empty row for the day being traded.

    The backtest scores day D from a frame that already contains D, relying on
    every signal's ``.shift(1)`` to read only D-1 and earlier. Live, D has not
    happened yet and its incomplete bar was dropped, so the frame ends at D-1
    and scoring it would use D-2 — one day stale. Appending an empty D restores
    the backtest's alignment exactly.
    """
    extended: dict[str, pd.DataFrame] = {}
    for inst_id, frame in daily.items():
        if trading_day in frame.index:
            extended[inst_id] = frame
            continue
        pending = pd.DataFrame(
            {column: [float("nan")] for column in frame.columns},
            index=pd.DatetimeIndex([trading_day], name=frame.index.name),
        )
        extended[inst_id] = pd.concat([frame, pending]).sort_index()
    closes = pd.concat(
        {inst_id: frame["close"] for inst_id, frame in extended.items()}, axis=1
    ).sort_index()
    return extended, closes


def mark_prices(client: OKXClient, universe: set[str]) -> dict[str, float]:
    """Last traded price per instrument, used for sizing and stop checks."""
    prices: dict[str, float] = {}
    for ticker in client.tickers("SWAP"):
        inst_id = str(ticker.get("instId"))
        if inst_id not in universe:
            continue
        last = ticker.get("last") or ticker.get("markPx")
        if last:
            prices[inst_id] = float(last)
    return prices


def instrument_specs(client: OKXClient, universe: set[str]) -> dict[str, dict[str, Any]]:
    return {
        str(row["instId"]): row
        for row in client.instruments("SWAP")
        if str(row.get("instId")) in universe
    }
