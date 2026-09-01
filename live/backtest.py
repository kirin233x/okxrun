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
UNIVERSE_FILE = ROOT / "research" / "universe.json"
MAX_CACHE_DAYS = 365
MAX_HOURS = MAX_CACHE_DAYS * 24
WARMUP_DAYS = 40
PAGE = 100
PAGES_PER_COIN = 8
BUDGET_SECONDS = 40.0
WEEK_HOURS = (WARMUP_DAYS + 14) * 24
BASE_COST = 0.0006
STARTING_EQUITY = 10_000.0
ONE_DAY = timedelta(days=1)
ONE_HOUR_MS = 3_600_000
MIN_LEVERAGE = 0.25
MAX_LEVERAGE = 5.0
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


def _now_ms() -> int:
    return int(time.time() * 1000)


def clamp_leverage(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("杠杆必须是正数")
    return min(MAX_LEVERAGE, max(MIN_LEVERAGE, float(value)))


def _request_candles(inst_id: str, after: int | None = None, before: int | None = None) -> list[list[str]]:
    params = {"instId": inst_id, "bar": "1H", "limit": str(PAGE)}
    if after is not None:
        params["after"] = str(after)
    if before is not None:
        params["before"] = str(before)
    request = urllib.request.Request(
        f"{CANDLE_API}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": "okxrun-backtest/1.0", "Connection": "close"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    if payload.get("code") != "0":
        raise RuntimeError(f"OKX {payload.get('code')}: {payload.get('msg')}")
    return payload.get("data") or []


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
    # First pass: get every coin to ~two months so a week window can run.
    for inst_id in universe:
        if time.monotonic() >= deadline:
            break
        cached = read_cached(inst_id)
        merged = fetch_incremental(inst_id, cached, deadline, WEEK_HOURS)
        if not merged.empty:
            write_cached(inst_id, merged)
    # Second pass: if time remains, deepen toward the one-year cap.
    for inst_id in universe:
        if time.monotonic() >= deadline:
            break
        cached = read_cached(inst_id)
        if cached.empty or len(cached) >= MAX_HOURS - 24:
            continue
        merged = fetch_incremental(inst_id, cached, deadline, MAX_HOURS)
        if not merged.empty:
            write_cached(inst_id, merged)
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

        realized = sum(position.realized_price_pnl for position in positions.values())
        paid_cost = entry_cost + sum(position.exit_cost for position in positions.values())
        unrealized = sum(
            positions[inst_id].direction * positions[inst_id].quantity * (last_marks[inst_id] - positions[inst_id].entry)
            for inst_id in active
        )
        marked_equity = starting_equity + realized + unrealized - paid_cost
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
    cost = entry_cost + sum(position.exit_cost for position in positions.values())
    ending_equity = starting_equity + price_pnl - cost
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
    cost_rate: float = BASE_COST,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    signals = compute_signals(closes, daily)
    score = signals["score"]
    dates = [date for date in closes.index if date in score.index]  # type: ignore[operator]
    equity = STARTING_EQUITY
    previous_weights: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    days: list[dict[str, Any]] = []
    atr = signals["atr"]
    for date in dates:
        target, signal_detail = target_for_day(date, signals, previous_weights, R92, leverage)
        if not signal_detail.get("tradable", False):
            continue
        date = pd.Timestamp(date)
        validation_assets = set(target) | {"BTC-USDT-SWAP"}
        if any(
            inst_id not in hourly
            or len(hourly[inst_id].loc[(hourly[inst_id].index >= date) & (hourly[inst_id].index < date + ONE_DAY)]) < 23
            for inst_id in validation_assets
        ):
            continue
        result, detail = simulate_day(date, equity, target, previous_weights, hourly, atr, cost_rate)
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
        return pd.DataFrame(columns=["net", "price", "cost", "endGross"]), []
    frame = pd.DataFrame(rows).set_index("date").sort_index()
    return frame, days


def window_metrics(frame: pd.DataFrame, days: int, cost_rate: float = BASE_COST) -> dict[str, Any] | None:
    if frame.empty:
        return None
    sample = frame[frame.index > frame.index.max() - timedelta(days=days)].copy()
    if sample.empty:
        return None
    liquidation_cost = float(sample.iloc[-1]["endGross"] * cost_rate)
    sample.loc[sample.index[-1], "net"] -= liquidation_cost
    sample.loc[sample.index[-1], "cost"] += liquidation_cost
    growth = (1.0 + sample["net"].fillna(0.0)).cumprod()
    equity = STARTING_EQUITY * growth
    drawdown = growth / growth.cummax().clip(lower=1.0) - 1.0
    return {
        "start": sample.index.min().isoformat(),
        "end": sample.index.max().isoformat(),
        "observations": int(len(sample)),
        "totalReturn": float(equity.iloc[-1] / STARTING_EQUITY - 1.0),
        "maxDrawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "positiveDayShare": float((sample["net"] > 0).mean()),
        "endingEquity": float(equity.iloc[-1]),
    }


def evaluate(hourly: dict[str, pd.DataFrame], leverage: float, window_days: int) -> dict[str, Any]:
    daily, closes = build_daily_inputs(hourly)
    frame, days = simulate(hourly, daily, closes, leverage)
    metrics = window_metrics(frame, window_days)
    if metrics is None:
        raise RuntimeError("缓存的K线还不够形成一个完整交易日，请再点一次补数据")
    return {
        "metrics": metrics,
        "recentDays": days[-14:],
        "tradedDays": int(len(frame)),
    }


def run_backtest(leverage: float, window: str, budget_seconds: float = BUDGET_SECONDS) -> dict[str, Any]:
    if window not in WINDOWS:
        raise ValueError("window 只能是 week 或 year")
    leverage = clamp_leverage(leverage)
    window_days = WINDOWS[window]
    if not _LOCK.acquire(blocking=False):
        return {"ok": False, "busy": True, "error": "已有一次回测在跑，请等它结束再点"}
    started = time.monotonic()
    try:
        universe = load_universe()
        # Leave roughly half the budget for the actual simulation.
        fetch_deadline = started + max(8.0, budget_seconds * 0.55)
        cache = sync_cache(universe, fetch_deadline)
        hourly = load_hourly_from_cache(universe)
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
        result = evaluate(hourly, leverage, window_days)
        payload = {
            "ok": True,
            "busy": False,
            "leverage": leverage,
            "window": window,
            "windowDays": window_days,
            "fundingIncluded": False,
            "cache": cache,
            "elapsedSec": round(time.monotonic() - started, 2),
            "note": "K线增量缓存最多一年；未计入资金费。一周回测仍用更长历史做排名热身。",
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
