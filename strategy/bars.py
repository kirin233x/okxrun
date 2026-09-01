"""Hourly-to-daily bar construction.

Shared because it changes the signals: the completeness rule below decides
which days exist at all, and a live executor that built its daily bars even
slightly differently would score a different universe than the backtest did.
"""

from __future__ import annotations

import pandas as pd


MIN_HOURS_PER_DAY = 23


def daily_from_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate confirmed hourly bars into daily bars.

    Days with fewer than 23 hourly bars are dropped: a partial day would
    produce a formation return and an ATR computed over a stub, and the live
    executor must not act on the day that is still in progress.
    """
    count = frame["close"].resample("1D").count()
    daily = frame.resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volumeQuote=("volumeQuote", "sum"),
    )
    return daily[count >= MIN_HOURS_PER_DAY].dropna(subset=["open", "high", "low", "close"])


def build_daily_inputs(hourly: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    daily = {inst_id: daily_from_hourly(frame) for inst_id, frame in hourly.items()}
    closes = pd.concat({inst_id: frame["close"] for inst_id, frame in daily.items()}, axis=1).sort_index()
    return daily, closes
