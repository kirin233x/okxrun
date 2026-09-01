"""Short, memory-bounded R9.2 backtest for the portal.

Public OKX candles only — no trading keys. Cache lives under ``state/backtest``
and is capped at one year of hourly bars. Each click has a hard time budget so
the portal cannot grow a long-running job on a 1 GB box.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy import (
    DAILY_KILL_LOSS,
    R92,
    build_daily_inputs,
    compute_signals,
    stop_fraction_from_atr,
    stop_price,
    target_for_day,
)

from .settings import ROOT


CANDLE_API = "https://www.okx.com/api/v5/market/history-candles"
FUNDING_API = "https://www.okx.com/api/v5/public/funding-rate-history"
UNIVERSE_FILE = ROOT / "research" / "universe.json"
MAX_CACHE_DAYS = 365
MAX_HOURS = MAX_CACHE_DAYS * 24
WARMUP_DAYS = 40
PAGE = 100
PAGES_PER_COIN = 8
PAGES_FUNDING = 4
BUDGET_SECONDS = 40.0
WEEK_HOURS = (WARMUP_DAYS + 14) * 24
BASE_COST = 0.0006
DEFAULT_PRINCIPAL = 10_000.0
ONE_DAY = timedelta(days=1)
ONE_HOUR_MS = 3_600_000
MIN_LEVERAGE = 0.25
MAX_LEVERAGE = 20.0
MIN_PRINCIPAL = 10.0
MAX_PRINCIPAL = 1_000_000.0
WINDOWS = {"week": 7, "year": 365}

_LOCK = threading.Lock()


def _state_dir() -> Path:
    return Path(os.environ.get("OKXRUN_STATE_DIR") or (ROOT / "state"))


def cache_dir() -> Path:
    path = _state_dir() / "backtest" / "candles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def last_path() -> Path:
    path = _state_dir() / "backtest"
    path.mkdir(parents=True, exist_ok=True)
    return path / "last.json"


def load_universe() -> list[str]:
    payload = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    return list(payload["universe"])


def candle_path(inst_id: str) -> Path:
    return cache_dir() / f"{inst_id.lower()}-1h.csv"


def funding_dir() -> Path:
    path = _state_dir() / "backtest" / "funding"
    path.mkdir(parents=True, exist_ok=True)
    return path


def funding_path(inst_id: str) -> Path:
    return funding_dir() / f"{inst_id.lower()}-funding.csv"


def _now_ms() -> int:
    return int(time.time() * 1000)


def clamp_leverage(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("杠杆必须是正数")
    return min(MAX_LEVERAGE, max(MIN_LEVERAGE, float(value)))


def clamp_principal(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("本金必须是正数")
    return min(MAX_PRINCIPAL, max(MIN_PRINCIPAL, float(value)))


def _request_json(url: str, params: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": "okxrun-backtest/1.0", "Connection": "close"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    if payload.get("code") != "0":
        raise RuntimeError(f"OKX {payload.get('code')}: {payload.get('msg')}")
    return payload


def _request_candles(inst_id: str, after: int | None = None, before: int | None = None) -> list[list[str]]:
    params = {"instId": inst_id, "bar": "1H", "limit": str(PAGE)}
    if after is not None:
        params["after"] = str(after)
    if before is not None:
        params["before"] = str(before)
    return _request_json(CANDLE_API, params).get("data") or []


def _rows_to_frame(rows: dict[int, list[str]]) -> pd.DataFrame:
    records = [
        {
            "ts": ts,
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volumeQuote": float(row[7]) if len(row) > 7 else 0.0,
        }
        for ts, row in rows.items()
        if len(row) >= 9 and row[8] == "1"
    ]
    if not records:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volumeQuote"])
    return pd.DataFrame(records)


def read_cached(inst_id: str) -> pd.DataFrame:
    path = candle_path(inst_id)
    if not path.exists():
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volumeQuote"])
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    return frame.astype({"ts": "int64"}, errors="ignore")


def write_cached(inst_id: str, frame: pd.DataFrame) -> None:
    trimmed = trim_hours(frame)
    trimmed.to_csv(candle_path(inst_id), index=False)


def read_funding(inst_id: str) -> pd.DataFrame:
    path = funding_path(inst_id)
    if not path.exists():
        return pd.DataFrame(columns=["ts", "fundingRate"])
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["ts", "fundingRate"])
    return frame.astype({"ts": "int64"}, errors="ignore")


def write_funding(inst_id: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    out = frame.drop_duplicates("ts", keep="last").sort_values("ts")
    floor = _now_ms() - MAX_HOURS * ONE_HOUR_MS
    out = out[out["ts"] >= floor]
    out.to_csv(funding_path(inst_id), index=False)


def fetch_funding_incremental(inst_id: str, cached: pd.DataFrame, deadline: float) -> pd.DataFrame:
    collected: dict[int, float] = {}
    newest_cached = int(cached["ts"].max()) if not cached.empty else None
    oldest_cached = int(cached["ts"].min()) if not cached.empty else None
    floor = _now_ms() - MAX_HOURS * ONE_HOUR_MS
    after: int | None = None
    for _ in range(PAGES_FUNDING):
        if time.monotonic() >= deadline:
            break
        params = {"instId": inst_id, "limit": str(PAGE)}
        if after is not None:
            params["after"] = str(after)
        try:
            batch = _request_json(FUNDING_API, params).get("data") or []
        except (urllib.error.URLError, TimeoutError, RuntimeError):
            break
        if not batch:
            break
        for row in batch:
            ts = int(row["fundingTime"])
            rate = float(row.get("realizedRate") or row.get("fundingRate") or 0)
            collected[ts] = rate
        oldest = min(int(row["fundingTime"]) for row in batch)
        if newest_cached is not None and oldest <= newest_cached and after is None:
            break
        if oldest <= floor or len(batch) < PAGE:
            break
        after = oldest
        time.sleep(0.03)
    if oldest_cached is not None and oldest_cached > floor + ONE_HOUR_MS and time.monotonic() < deadline:
        after = oldest_cached
        for _ in range(PAGES_FUNDING):
            if time.monotonic() >= deadline:
                break
            params = {"instId": inst_id, "limit": str(PAGE), "after": str(after)}
            try:
                batch = _request_json(FUNDING_API, params).get("data") or []
            except (urllib.error.URLError, TimeoutError, RuntimeError):
                break
            if not batch:
                break
            for row in batch:
                collected[int(row["fundingTime"])] = float(
                    row.get("realizedRate") or row.get("fundingRate") or 0
                )
            oldest = min(int(row["fundingTime"]) for row in batch)
            if oldest <= floor or len(batch) < PAGE:
                break
            after = oldest
            time.sleep(0.03)
    if not collected:
        return cached
    fresh = pd.DataFrame([{"ts": ts, "fundingRate": rate} for ts, rate in collected.items()])
    if cached.empty:
        return fresh
    return pd.concat([cached, fresh], ignore_index=True)


def funding_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    index = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    result = pd.Series(frame["fundingRate"].to_numpy(dtype=float), index=index)
    return result.groupby(level=0).sum().sort_index()


def load_funding_from_cache(universe: list[str]) -> dict[str, pd.Series]:
    return {inst_id: funding_series(read_funding(inst_id)) for inst_id in universe}


def trim_hours(frame: pd.DataFrame, max_hours: int = MAX_HOURS) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = (
        frame.drop_duplicates("ts", keep="last")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    cutoff = _now_ms() - max_hours * ONE_HOUR_MS
    out = out[out["ts"] >= cutoff]
    if len(out) > max_hours:
        out = out.tail(max_hours)
    return out.reset_index(drop=True)


def merge_hours(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old.empty:
        return trim_hours(new)
    if new.empty:
        return trim_hours(old)
    return trim_hours(pd.concat([old, new], ignore_index=True))


def _collect_batch(batch: list[list[str]]) -> dict[int, list[str]]:
    rows: dict[int, list[str]] = {}
    for row in batch:
        if len(row) >= 9 and row[8] == "1":
            rows[int(row[0])] = row
    return rows


def _pull_pages(
    inst_id: str,
    after: int | None,
    max_pages: int,
    deadline: float,
    stop_at: int | None = None,
    floor: int | None = None,
) -> dict[int, list[str]]:
    collected: dict[int, list[str]] = {}
    cursor = after
    for _ in range(max_pages):
        if time.monotonic() >= deadline:
            break
        try:
            batch = _request_candles(inst_id, after=cursor)
        except (urllib.error.URLError, TimeoutError, RuntimeError):
            break
        if not batch:
            break
        collected.update(_collect_batch(batch))
        oldest = min(int(row[0]) for row in batch)
        if stop_at is not None and oldest <= stop_at:
            break
        if floor is not None and oldest <= floor:
            break
        if len(batch) < PAGE:
            break
        cursor = oldest
        time.sleep(0.03)
    return collected


def fetch_incremental(inst_id: str, cached: pd.DataFrame, deadline: float, target_hours: int) -> pd.DataFrame:
    """Pull a bounded number of new pages so every coin gets a turn each click."""
    collected: dict[int, list[str]] = {}
    newest_cached = int(cached["ts"].max()) if not cached.empty else None
    oldest_cached = int(cached["ts"].min()) if not cached.empty else None
    floor = _now_ms() - MAX_HOURS * ONE_HOUR_MS

    collected.update(_pull_pages(inst_id, None, 3, deadline, stop_at=newest_cached))

    have = 0 if cached.empty else int(len(cached))
    if have < target_hours or (oldest_cached is not None and oldest_cached > floor + ONE_HOUR_MS):
        cursor = oldest_cached if oldest_cached is not None else (min(collected) if collected else None)
        collected.update(_pull_pages(inst_id, cursor, PAGES_PER_COIN, deadline, floor=floor))

    if not collected:
        return cached
    return merge_hours(cached, _rows_to_frame(collected))


def sync_cache(universe: list[str], deadline: float) -> dict[str, Any]:
    remain = max(0.0, deadline - time.monotonic())
    candle_deadline = time.monotonic() + remain * 0.65
    # First pass: get every coin to ~two months so a week window can run.
    for inst_id in universe:
        if time.monotonic() >= candle_deadline:
            break
        cached = read_cached(inst_id)
        merged = fetch_incremental(inst_id, cached, candle_deadline, WEEK_HOURS)
        if not merged.empty:
            write_cached(inst_id, merged)
    # Second pass: if time remains, deepen toward the one-year cap.
    for inst_id in universe:
        if time.monotonic() >= candle_deadline:
            break
        cached = read_cached(inst_id)
        if cached.empty or len(cached) >= MAX_HOURS - 24:
            continue
        merged = fetch_incremental(inst_id, cached, candle_deadline, MAX_HOURS)
        if not merged.empty:
            write_cached(inst_id, merged)
    for inst_id in universe:
        if time.monotonic() >= deadline:
            break
        cached = read_funding(inst_id)
        merged = fetch_funding_incremental(inst_id, cached, deadline)
        if not merged.empty:
            write_funding(inst_id, merged)
    return cache_status(universe)


def cache_status(universe: list[str] | None = None) -> dict[str, Any]:
    universe = universe or load_universe()
    hours = 0
    bytes_used = 0
    mins: list[int] = []
    maxs: list[int] = []
    coins = 0
    for inst_id in universe:
        path = candle_path(inst_id)
        if not path.exists():
            continue
        bytes_used += path.stat().st_size
        fund = funding_path(inst_id)
        if fund.exists():
            bytes_used += fund.stat().st_size
        frame = read_cached(inst_id)
        if frame.empty:
            continue
        coins += 1
        hours += int(len(frame))
        mins.append(int(frame["ts"].min()))
        maxs.append(int(frame["ts"].max()))
    span_days = 0.0
    if mins and maxs:
        span_days = (max(maxs) - min(mins)) / (ONE_HOUR_MS * 24)
    return {
        "coins": coins,
        "universe": len(universe),
        "hours": hours,
        "spanDays": round(span_days, 1),
        "bytes": bytes_used,
        "fullYear": span_days >= MAX_CACHE_DAYS - 2,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }


def indexed_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.to_datetime(result.pop("ts"), unit="ms", utc=True)
    numeric = result[["open", "high", "low", "close", "volumeQuote"]].apply(pd.to_numeric, errors="coerce")
    clean = numeric.dropna(subset=["open", "high", "low", "close"])
    return clean[~clean.index.duplicated(keep="last")].sort_index()


def load_hourly_from_cache(universe: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for inst_id in universe:
        frame = read_cached(inst_id)
        if frame.empty:
            continue
        indexed = indexed_hourly(frame)
        frames[inst_id] = indexed[~indexed.index.duplicated(keep="last")].sort_index()
    return frames


@dataclass
class Position:
    inst_id: str
    direction: int
    weight: float
    entry: float
    quantity: float
    stop_fraction: float
    peak: float
    trough: float
    entry_cost: float = 0.0
    realized_price_pnl: float = 0.0
    funding_pnl: float = 0.0
    exit_cost: float = 0.0
    exit_price: float | None = None
    exit_reason: str | None = None


def _stop_fill(position: Position, bar: pd.Series) -> float | None:
    stop = stop_price(position.entry, position.direction, position.stop_fraction, position.peak, position.trough)
    if position.direction > 0:
        if float(bar["open"]) <= stop:
            return float(bar["open"])
        if float(bar["low"]) <= stop:
            return stop
    else:
        if float(bar["open"]) >= stop:
            return float(bar["open"])
        if float(bar["high"]) >= stop:
            return stop
    return None


def _close(position: Position, price: float, cost_rate: float, reason: str) -> None:
    position.exit_price = price
    position.exit_reason = reason
    position.realized_price_pnl = position.direction * position.quantity * (price - position.entry)
    position.exit_cost = abs(position.quantity * price) * cost_rate


def simulate_day(
    date: pd.Timestamp,
    starting_equity: float,
    target: dict[str, float],
    previous_weights: dict[str, float],
    hourly: dict[str, pd.DataFrame],
    atr: pd.DataFrame,
    funding: dict[str, pd.Series],
    cost_rate: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    date = pd.Timestamp(date)
    day_end = date + ONE_DAY
    positions: dict[str, Position] = {}
    entry_cost_by_asset = {
        inst_id: starting_equity * abs(target.get(inst_id, 0.0) - previous_weights.get(inst_id, 0.0)) * cost_rate
        for inst_id in sorted(set(target) | set(previous_weights))
    }
    entry_cost = float(sum(entry_cost_by_asset.values()))
    for inst_id, weight in target.items():
        bars = hourly[inst_id].loc[(hourly[inst_id].index >= date) & (hourly[inst_id].index < day_end)]
        if bars.empty:
            continue
        entry = float(bars.iloc[0]["open"])
        notional = starting_equity * abs(weight)
        if notional <= 0 or entry <= 0 or date not in atr.index or inst_id not in atr.columns:
            continue
        stop_fraction = stop_fraction_from_atr(float(atr.loc[date, inst_id]))
        if not math.isfinite(stop_fraction):
            continue
        positions[inst_id] = Position(
            inst_id=inst_id,
            direction=1 if weight > 0 else -1,
            weight=weight,
            entry=entry,
            quantity=notional / entry,
            stop_fraction=stop_fraction,
            peak=entry,
            trough=entry,
            entry_cost=entry_cost_by_asset.get(inst_id, 0.0),
        )

    active = set(positions)
    timestamps: list[pd.Timestamp] = []
    if active:
        stamps: set[pd.Timestamp] = set()
        for inst_id in active:
            stamps.update(
                hourly[inst_id].loc[(hourly[inst_id].index >= date) & (hourly[inst_id].index < day_end)].index
            )
        timestamps = sorted(stamps)
    kill_triggered = False
    stop_count = 0
    first_timestamp = timestamps[0] if timestamps else None
    last_marks: dict[str, float] = {inst_id: position.entry for inst_id, position in positions.items()}

    for timestamp in timestamps:
        for inst_id in list(active):
            frame = hourly[inst_id]
            if timestamp not in frame.index:
                continue
            bar = frame.loc[timestamp]
            fill = _stop_fill(positions[inst_id], bar)
            if fill is not None:
                _close(positions[inst_id], fill, cost_rate, "POSITION_STOP")
                last_marks[inst_id] = fill
                active.remove(inst_id)
                stop_count += 1
                continue
            positions[inst_id].peak = max(positions[inst_id].peak, float(bar["high"]))
            positions[inst_id].trough = min(positions[inst_id].trough, float(bar["low"]))
            last_marks[inst_id] = float(bar["close"])

        if first_timestamp is not None and timestamp > first_timestamp:
            for inst_id in list(active):
                series = funding.get(inst_id)
                rate = float(series.get(timestamp, 0.0)) if series is not None and not series.empty else 0.0
                if rate:
                    notional = abs(positions[inst_id].quantity * last_marks[inst_id])
                    positions[inst_id].funding_pnl += -positions[inst_id].direction * notional * rate

        realized = sum(position.realized_price_pnl for position in positions.values())
        paid_cost = entry_cost + sum(position.exit_cost for position in positions.values())
        funding_pnl = sum(position.funding_pnl for position in positions.values())
        unrealized = sum(
            positions[inst_id].direction * positions[inst_id].quantity * (last_marks[inst_id] - positions[inst_id].entry)
            for inst_id in active
        )
        marked_equity = starting_equity + realized + unrealized + funding_pnl - paid_cost
        if active and marked_equity <= starting_equity * (1.0 - DAILY_KILL_LOSS):
            for inst_id in list(active):
                _close(positions[inst_id], last_marks[inst_id], cost_rate, "DAILY_KILL")
                active.remove(inst_id)
            kill_triggered = True
            break

    carried = sorted(active)
    for inst_id in carried:
        position = positions[inst_id]
        position.exit_price = last_marks[inst_id]
        position.exit_reason = "CARRY"
        position.realized_price_pnl = position.direction * position.quantity * (last_marks[inst_id] - position.entry)

    price_pnl = sum(position.realized_price_pnl for position in positions.values())
    funding_pnl = sum(position.funding_pnl for position in positions.values())
    cost = entry_cost + sum(position.exit_cost for position in positions.values())
    ending_equity = starting_equity + price_pnl + funding_pnl - cost
    ending_weights = {
        inst_id: positions[inst_id].direction * abs(positions[inst_id].quantity * last_marks[inst_id]) / ending_equity
        for inst_id in carried
        if ending_equity > 0
    }
    rebalance_turnover = sum(
        abs(target.get(inst_id, 0.0) - previous_weights.get(inst_id, 0.0))
        for inst_id in set(target) | set(previous_weights)
    )
    return {
        "net": ending_equity / starting_equity - 1.0,
        "price": price_pnl / starting_equity,
        "funding": funding_pnl / starting_equity,
        "cost": cost / starting_equity,
        "endGross": float(sum(abs(weight) for weight in ending_weights.values())),
    }, {
        "endingEquity": ending_equity,
        "endingWeights": ending_weights,
        "stops": stop_count,
        "killTriggered": kill_triggered,
        "longs": [inst_id for inst_id, weight in target.items() if weight > 0],
        "shorts": [inst_id for inst_id, weight in target.items() if weight < 0],
    }


def simulate(
    hourly: dict[str, pd.DataFrame],
    daily: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    leverage: float,
    principal: float,
    start_at: pd.Timestamp,
    funding: dict[str, pd.Series] | None = None,
    cost_rate: float = BASE_COST,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    funding = funding or {}
    signals = compute_signals(closes, daily)
    score = signals["score"]
    dates = [date for date in closes.index if date in score.index]  # type: ignore[operator]
    equity = principal
    previous_weights: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    days: list[dict[str, Any]] = []
    atr = signals["atr"]
    now = pd.Timestamp.now(tz="UTC")
    for date in dates:
        date = pd.Timestamp(date)
        if date.tzinfo is None:
            date = date.tz_localize("UTC")
        # Daily bar D covers [D, D+1). Keep days that overlap [start_at, now].
        if date + ONE_DAY <= start_at or date >= now:
            continue
        target, signal_detail = target_for_day(date, signals, previous_weights, R92, leverage)
        if not signal_detail.get("tradable", False):
            continue
        validation_assets = set(target) | {"BTC-USDT-SWAP"}
        if any(
            inst_id not in hourly
            or len(hourly[inst_id].loc[(hourly[inst_id].index >= date) & (hourly[inst_id].index < date + ONE_DAY)]) < 23
            for inst_id in validation_assets
        ):
            continue
        result, detail = simulate_day(
            date, equity, target, previous_weights, hourly, atr, funding, cost_rate
        )
        equity = detail["endingEquity"]
        previous_weights = detail["endingWeights"]
        rows.append({"date": date, **result})
        days.append(
            {
                "date": date.isoformat(),
                "regime": signal_detail.get("regime"),
                "netReturn": result["net"],
                "equity": equity,
                "longs": [inst_id.replace("-USDT-SWAP", "") for inst_id in detail["longs"]],
                "shorts": [inst_id.replace("-USDT-SWAP", "") for inst_id in detail["shorts"]],
                "stops": detail["stops"],
                "killTriggered": detail["killTriggered"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=["net", "price", "funding", "cost", "endGross"]), []
    frame = pd.DataFrame(rows).set_index("date").sort_index()
    return frame, days


def window_metrics(frame: pd.DataFrame, principal: float, cost_rate: float = BASE_COST) -> dict[str, Any] | None:
    if frame.empty:
        return None
    sample = frame.copy()
    liquidation_cost = float(sample.iloc[-1]["endGross"] * cost_rate)
    sample.loc[sample.index[-1], "net"] -= liquidation_cost
    sample.loc[sample.index[-1], "cost"] += liquidation_cost
    growth = (1.0 + sample["net"].fillna(0.0)).cumprod()
    equity = principal * growth
    drawdown = growth / growth.cummax().clip(lower=1.0) - 1.0
    funding_col = sample["funding"] if "funding" in sample.columns else 0.0
    return {
        "start": sample.index.min().isoformat(),
        "end": sample.index.max().isoformat(),
        "observations": int(len(sample)),
        "totalReturn": float(equity.iloc[-1] / principal - 1.0),
        "maxDrawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "positiveDayShare": float((sample["net"] > 0).mean()),
        "endingEquity": float(equity.iloc[-1]),
        "fundingReturn": float(funding_col.sum()) if not isinstance(funding_col, float) else 0.0,
        "costReturn": float(sample["cost"].sum()),
    }


def evaluate(
    hourly: dict[str, pd.DataFrame],
    leverage: float,
    window_days: int,
    principal: float = DEFAULT_PRINCIPAL,
    funding: dict[str, pd.Series] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    clock = pd.Timestamp(now or datetime.now(timezone.utc))
    if clock.tzinfo is None:
        clock = clock.tz_localize("UTC")
    else:
        clock = clock.tz_convert("UTC")
    start_at = clock - timedelta(days=window_days)
    daily, closes = build_daily_inputs(hourly)
    frame, days = simulate(hourly, daily, closes, leverage, principal, start_at, funding)
    metrics = window_metrics(frame, principal)
    if metrics is None:
        raise RuntimeError("这个时间窗口里还没有完整交易日，请再点一次补数据")
    return {
        "metrics": metrics,
        "recentDays": days[-14:] if window_days > 14 else days,
        "tradedDays": int(len(frame)),
        "startAt": start_at.isoformat(),
    }


def run_backtest(
    leverage: float,
    window: str,
    principal: float = DEFAULT_PRINCIPAL,
    budget_seconds: float = BUDGET_SECONDS,
) -> dict[str, Any]:
    if window not in WINDOWS:
        raise ValueError("window 只能是 week 或 year")
    leverage = clamp_leverage(leverage)
    principal = clamp_principal(principal)
    window_days = WINDOWS[window]
    if not _LOCK.acquire(blocking=False):
        return {"ok": False, "busy": True, "error": "已有一次回测在跑，请等它结束再点"}
    started = time.monotonic()
    try:
        universe = load_universe()
        fetch_deadline = started + max(8.0, budget_seconds * 0.55)
        cache = sync_cache(universe, fetch_deadline)
        hourly = load_hourly_from_cache(universe)
        funding = load_funding_from_cache(universe)
        funding_rows = sum(0 if series.empty else int(series.notna().sum()) for series in funding.values())
        if len(hourly) < 6:
            return {
                "ok": False,
                "busy": False,
                "error": "缓存里可交易标的不足 6 个，再点一次继续增量拉取",
                "cache": cache,
                "elapsedSec": round(time.monotonic() - started, 2),
            }
        span = max(frame.index.max() for frame in hourly.values()) - min(frame.index.min() for frame in hourly.values())
        span_days = span.total_seconds() / 86400
        if span_days < WARMUP_DAYS:
            return {
                "ok": False,
                "busy": False,
                "error": f"缓存只有 {span_days:.0f} 天，信号至少需要约 {WARMUP_DAYS} 天热身。再点一次继续补。",
                "cache": cache,
                "elapsedSec": round(time.monotonic() - started, 2),
            }
        result = evaluate(hourly, leverage, window_days, principal, funding)
        payload = {
            "ok": True,
            "busy": False,
            "leverage": leverage,
            "principal": principal,
            "window": window,
            "windowDays": window_days,
            "fundingIncluded": funding_rows > 0,
            "cache": cache,
            "elapsedSec": round(time.monotonic() - started, 2),
            "note": (
                "从一周前此刻空仓起步到现在。"
                if window == "week"
                else "从一年前此刻空仓起步到现在（K线最多缓存一年，热身期会吃掉前面约40天）。"
            )
            + "含开平仓手续费和资金费。",
            **result,
        }
        last_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    except Exception as error:  # noqa: BLE001 - surface to the portal
        return {
            "ok": False,
            "busy": False,
            "error": str(error),
            "elapsedSec": round(time.monotonic() - started, 2),
            "cache": cache_status(),
        }
    finally:
        _LOCK.release()


def last_result() -> dict[str, Any]:
    path = last_path()
    cache = cache_status()
    if not path.exists():
        return {"ok": True, "hasResult": False, "cache": cache}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ok": True, "hasResult": False, "cache": cache}
    payload["hasResult"] = bool(payload.get("ok"))
    payload["cache"] = cache
    return payload
