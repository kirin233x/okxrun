from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# See the equivalent note in r9_momentum_backtest.py: direct invocation puts
# research/ on sys.path, so the repo root has to be restored for `strategy`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from strategy import (  # noqa: E402
    ASSETS_PER_SIDE,
    BTC_SHOCK_Z,
    DAILY_KILL_LOSS,
    MAX_STOP,
    MIN_STOP,
    RANK_EXIT_BUFFER,
    STOP_ATR_MULTIPLIER,
    TRAIL_DISTANCE,
    TRAIL_TRIGGER,
    buffered_selection,
    compute_signals,
)

try:
    from .high_return_candidates import (
        BASE_COST,
        DATA_DIR,
        STARTING_EQUITY,
        STRESS_COST,
        fetch_recent_funding,
        funding_series,
    )
    from .r9_momentum_backtest import (
        ONE_DAY,
        Position,
        build_daily_inputs,
        close_position,
        load_hourly,
        load_universe,
        metrics,
        robustness_metrics,
        stop_fill,
    )
except ImportError:
    from high_return_candidates import (
        BASE_COST,
        DATA_DIR,
        STARTING_EQUITY,
        STRESS_COST,
        fetch_recent_funding,
        funding_series,
    )
    from r9_momentum_backtest import (
        ONE_DAY,
        Position,
        build_daily_inputs,
        close_position,
        load_hourly,
        load_universe,
        metrics,
        robustness_metrics,
        stop_fill,
    )


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research" / "artifacts" / "r10-dual-speed-report.json"
BTC = "BTC-USDT-SWAP"
FAST_RETURN_HOURS = 12
FAST_EMA_HOURS = 20
SHOCK_COOLDOWN_HOURS = 4
SHOCK_RISK_MULTIPLIER = 0.50
TARGET_DAILY_VOLATILITY = 0.035
MIN_DYNAMIC_LEVERAGE = 1.5
MAX_DYNAMIC_LEVERAGE = 5.0
BASE_SIDE_GROSS = 0.75
MAX_BASE_ASSET_WEIGHT = 0.25


def dynamic_leverage(btc_daily_volatility: float) -> float:
    """Map lagged BTC daily volatility to a half-step leverage between 1.5x and 5x."""
    if not math.isfinite(btc_daily_volatility) or btc_daily_volatility <= 0:
        return MIN_DYNAMIC_LEVERAGE
    raw = TARGET_DAILY_VOLATILITY / btc_daily_volatility
    clipped = float(np.clip(raw, MIN_DYNAMIC_LEVERAGE, MAX_DYNAMIC_LEVERAGE))
    return round(clipped * 2.0) / 2.0


def fast_snapshot(btc_hourly: pd.DataFrame, decision_time: pd.Timestamp) -> dict[str, float | bool]:
    """Use only hourly bars completed before decision_time."""
    history = btc_hourly.loc[btc_hourly.index < decision_time, "close"].dropna()
    if len(history) < FAST_EMA_HOURS + 1:
        return {
            "tradable": False,
            "price": math.nan,
            "return12h": math.nan,
            "ema20h": math.nan,
            "up": False,
            "down": False,
        }
    price = float(history.iloc[-1])
    return_12h = float(price / history.iloc[-(FAST_RETURN_HOURS + 1)] - 1.0)
    ema_20h = float(history.ewm(span=FAST_EMA_HOURS, adjust=False).mean().iloc[-1])
    return {
        "tradable": True,
        "price": price,
        "return12h": return_12h,
        "ema20h": ema_20h,
        "up": return_12h > 0 and price > ema_20h,
        "down": return_12h < 0 and price < ema_20h,
    }


def capped_side_weights(volatility: pd.Series, gross: float = BASE_SIDE_GROSS) -> dict[str, float]:
    valid = volatility.replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[valid > 0]
    if valid.empty or gross <= 0:
        return {}
    raw = (1.0 / valid) / (1.0 / valid).sum() * gross
    capped = raw.clip(upper=MAX_BASE_ASSET_WEIGHT)
    return {str(inst_id): float(weight) for inst_id, weight in capped.items() if weight > 0}


def target_for_day(
    date: pd.Timestamp,
    signals: dict[str, pd.DataFrame | pd.Series],
    hourly: dict[str, pd.DataFrame],
    daily: dict[str, pd.DataFrame],
    previous_weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    previous_weights = previous_weights or {}
    date = pd.Timestamp(date)
    score = signals["score"].loc[date].dropna()  # type: ignore[union-attr]
    volatility = signals["volatility"].loc[date].reindex(score.index).dropna()  # type: ignore[union-attr]
    ranked = score.reindex(volatility.index).dropna().sort_values()
    if len(ranked) < ASSETS_PER_SIDE * 2:
        return {}, {"tradable": False, "regime": "INSUFFICIENT_DATA", "entryHour": 0}

    longs, shorts = buffered_selection(ranked, previous_weights)
    btc_prior = float(signals["btcPrior"].loc[date])  # type: ignore[union-attr]
    btc_ma = float(signals["btcMa"].loc[date])  # type: ignore[union-attr]
    btc_return_1 = float(signals["btcReturn1"].loc[date])  # type: ignore[union-attr]
    btc_return_7 = float(signals["btcReturn7"].loc[date])  # type: ignore[union-attr]
    btc_shock_z = float(signals["btcShockZ"].loc[date])  # type: ignore[union-attr]
    btc_volatility = float(signals["volatility"].loc[date, BTC])  # type: ignore[union-attr]
    values = (btc_prior, btc_ma, btc_return_1, btc_return_7, btc_shock_z, btc_volatility)
    if not all(math.isfinite(value) for value in values):
        return {}, {"tradable": False, "regime": "INSUFFICIENT_BTC_DATA", "entryHour": 0}

    slow_up = btc_prior > btc_ma and btc_return_7 > 0
    slow_down = btc_prior < btc_ma and btc_return_7 < 0
    shock_direction = 1 if btc_shock_z > BTC_SHOCK_Z else -1 if btc_shock_z < -BTC_SHOCK_Z else 0
    entry_hour = SHOCK_COOLDOWN_HOURS if shock_direction else 0
    decision_time = date + timedelta(hours=entry_hour)
    fast = fast_snapshot(hourly[BTC], decision_time)
    leverage = dynamic_leverage(btc_volatility)
    risk_multiplier = SHOCK_RISK_MULTIPLIER if shock_direction else 1.0

    previous_day = date - ONE_DAY
    btc_previous_bar = daily[BTC].loc[previous_day] if previous_day in daily[BTC].index else None
    shock_midpoint = (
        float((btc_previous_bar["high"] + btc_previous_bar["low"]) / 2.0)
        if btc_previous_bar is not None
        else math.nan
    )
    shock_continues_up = (
        shock_direction > 0
        and bool(fast["up"])
        and math.isfinite(shock_midpoint)
        and float(fast["price"]) >= shock_midpoint
    )
    shock_continues_down = (
        shock_direction < 0
        and bool(fast["down"])
        and math.isfinite(shock_midpoint)
        and float(fast["price"]) <= shock_midpoint
    )

    direction = 0
    if shock_direction > 0:
        direction = 1 if slow_up and shock_continues_up else 0
        regime = "UP_SHOCK_CONTINUE" if direction else "UP_SHOCK_FAILED"
    elif shock_direction < 0:
        direction = -1 if slow_down and shock_continues_down else 0
        regime = "DOWN_SHOCK_CONTINUE" if direction else "DOWN_SHOCK_FAILED"
    elif slow_up and bool(fast["up"]):
        direction = 1
        regime = "BULL_CONFIRMED"
    elif slow_down and bool(fast["down"]):
        direction = -1
        regime = "BEAR_CONFIRMED"
    elif slow_up and bool(fast["down"]):
        regime = "CORRECTION_FLAT"
    elif slow_down and bool(fast["up"]):
        regime = "REBOUND_FLAT"
    else:
        regime = "UNCONFIRMED_FLAT"

    selected = longs if direction > 0 else shorts if direction < 0 else []
    base_weights = capped_side_weights(volatility.reindex(selected))
    target = {
        inst_id: direction * weight * leverage * risk_multiplier
        for inst_id, weight in base_weights.items()
    }
    gross_target = float(sum(abs(weight) for weight in target.values()))
    detail = {
        "tradable": True,
        "regime": regime,
        "entryHour": entry_hour,
        "direction": "LONG" if direction > 0 else "SHORT" if direction < 0 else "FLAT",
        "slowUp": slow_up,
        "slowDown": slow_down,
        "fastUp": bool(fast["up"]),
        "fastDown": bool(fast["down"]),
        "fastReturn12h": fast["return12h"],
        "fastEma20h": fast["ema20h"],
        "fastPrice": fast["price"],
        "btcReturn1": btc_return_1,
        "btcReturn7": btc_return_7,
        "btcShockZ": btc_shock_z,
        "shockMidpoint": shock_midpoint,
        "dynamicLeverage": leverage,
        "riskMultiplier": risk_multiplier,
        "grossTarget": gross_target,
        "grossAfterCap": gross_target,
        "longs": longs,
        "shorts": shorts,
        "selected": selected,
        "rankBuffer": RANK_EXIT_BUFFER,
    }
    return target, detail


def simulate_day(
    date: pd.Timestamp,
    starting_equity: float,
    target: dict[str, float],
    previous_weights: dict[str, float],
    hourly: dict[str, pd.DataFrame],
    atr: pd.DataFrame,
    funding: dict[str, pd.Series],
    cost_rate: float,
    entry_hour: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    date = pd.Timestamp(date)
    entry_time = date + timedelta(hours=entry_hour)
    day_end = date + ONE_DAY
    positions: dict[str, Position] = {}
    entry_cost_by_asset = {
        inst_id: starting_equity * abs(target.get(inst_id, 0.0) - previous_weights.get(inst_id, 0.0)) * cost_rate
        for inst_id in set(target) | set(previous_weights)
    }
    entry_cost = float(sum(entry_cost_by_asset.values()))

    for inst_id, weight in target.items():
        bars = hourly[inst_id].loc[
            (hourly[inst_id].index >= entry_time) & (hourly[inst_id].index < day_end)
        ]
        if bars.empty:
            continue
        entry = float(bars.iloc[0]["open"])
        notional = starting_equity * abs(weight)
        stop_fraction = float(np.clip(STOP_ATR_MULTIPLIER * float(atr.loc[date, inst_id]), MIN_STOP, MAX_STOP))
        if notional <= 0 or entry <= 0 or not math.isfinite(stop_fraction):
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
    timestamps = (
        sorted(
            set().union(
                *[
                    set(
                        hourly[inst_id].loc[
                            (hourly[inst_id].index >= entry_time) & (hourly[inst_id].index < day_end)
                        ].index
                    )
                    for inst_id in active
                ]
            )
        )
        if active
        else []
    )
    first_timestamp = timestamps[0] if timestamps else None
    last_marks = {inst_id: position.entry for inst_id, position in positions.items()}
    kill_triggered = False
    stop_count = 0

    for timestamp in timestamps:
        for inst_id in list(active):
            frame = hourly[inst_id]
            if timestamp not in frame.index:
                continue
            bar = frame.loc[timestamp]
            fill = stop_fill(positions[inst_id], bar)
            if fill is not None:
                close_position(positions[inst_id], fill, cost_rate, "POSITION_STOP")
                last_marks[inst_id] = fill
                active.remove(inst_id)
                stop_count += 1
                continue
            positions[inst_id].peak = max(positions[inst_id].peak, float(bar["high"]))
            positions[inst_id].trough = min(positions[inst_id].trough, float(bar["low"]))
            last_marks[inst_id] = float(bar["close"])

        if first_timestamp is not None and timestamp > first_timestamp:
            for inst_id in list(active):
                rate = float(funding[inst_id].get(timestamp, 0.0))
                if rate:
                    notional = abs(positions[inst_id].quantity * last_marks[inst_id])
                    positions[inst_id].funding_pnl += -positions[inst_id].direction * notional * rate

        realized = sum(position.realized_price_pnl for position in positions.values())
        paid_cost = entry_cost + sum(position.exit_cost for position in positions.values())
        funding_pnl = sum(position.funding_pnl for position in positions.values())
        unrealized = sum(
            positions[inst_id].direction
            * positions[inst_id].quantity
            * (last_marks[inst_id] - positions[inst_id].entry)
            for inst_id in active
        )
        marked_equity = starting_equity + realized + unrealized + funding_pnl - paid_cost
        if active and marked_equity <= starting_equity * (1.0 - DAILY_KILL_LOSS):
            for inst_id in list(active):
                close_position(positions[inst_id], last_marks[inst_id], cost_rate, "DAILY_KILL")
                active.remove(inst_id)
            kill_triggered = True
            break

    carried = list(active)
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
    stopped_turnover = (
        sum(
            abs(position.quantity * float(position.exit_price or position.entry)) / starting_equity
            for position in positions.values()
            if position.exit_reason in {"POSITION_STOP", "DAILY_KILL"}
        )
        if starting_equity
        else 0.0
    )
    turnover = rebalance_turnover + stopped_turnover
    return {
        "net": ending_equity / starting_equity - 1.0,
        "gross": (price_pnl + funding_pnl) / starting_equity,
        "price": price_pnl / starting_equity,
        "funding": funding_pnl / starting_equity,
        "cost": cost / starting_equity,
        "turnover": turnover,
        "endGross": float(sum(abs(weight) for weight in ending_weights.values())),
    }, {
        "trades": sum(
            abs(target.get(inst_id, 0.0) - previous_weights.get(inst_id, 0.0)) > 1e-8
            for inst_id in set(target) | set(previous_weights)
        )
        + stop_count,
        "stops": stop_count,
        "killTriggered": kill_triggered,
        "endingEquity": ending_equity,
        "endingWeights": ending_weights,
        "positions": [
            {
                "instId": position.inst_id,
                "side": "LONG" if position.direction > 0 else "SHORT",
                "weight": position.weight,
                "entry": position.entry,
                "exit": position.exit_price,
                "stopFraction": position.stop_fraction,
                "exitReason": position.exit_reason,
                "pricePnl": position.realized_price_pnl,
                "fundingPnl": position.funding_pnl,
                "cost": position.entry_cost + position.exit_cost,
            }
            for position in positions.values()
        ],
    }


def simulate(
    hourly: dict[str, pd.DataFrame],
    daily: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    funding: dict[str, pd.Series],
    cost_rate: float,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    signals = compute_signals(closes, daily)
    dates = [date for date in closes.index if date in signals["score"].index]  # type: ignore[operator]
    equity = STARTING_EQUITY
    previous_weights: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    days: list[dict[str, Any]] = []
    regime_counts: dict[str, int] = {}
    leverage_values: list[float] = []
    total_stops = 0
    total_kills = 0
    total_trades = 0

    for date in dates:
        target, signal_detail = target_for_day(date, signals, hourly, daily, previous_weights)
        if not signal_detail.get("tradable", False):
            continue
        date = pd.Timestamp(date)
        validation_assets = set(target) | {BTC}
        if any(
            len(hourly[inst_id].loc[(hourly[inst_id].index >= date) & (hourly[inst_id].index < date + ONE_DAY)])
            < 23
            for inst_id in validation_assets
        ):
            continue
        result, detail = simulate_day(
            date,
            equity,
            target,
            previous_weights,
            hourly,
            signals["atr"],  # type: ignore[arg-type]
            funding,
            cost_rate,
            int(signal_detail["entryHour"]),
        )
        equity = detail["endingEquity"]
        previous_weights = detail["endingWeights"]
        rows.append({"date": date, **result})
        regime = str(signal_detail["regime"])
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        leverage_values.append(float(signal_detail["dynamicLeverage"]))
        total_stops += detail["stops"]
        total_kills += int(detail["killTriggered"])
        total_trades += detail["trades"]
        days.append(
            {
                "date": date.isoformat(),
                "signal": signal_detail,
                "netReturn": result["net"],
                "equity": equity,
                "stops": detail["stops"],
                "killTriggered": detail["killTriggered"],
                "positions": detail["positions"],
            }
        )

    frame = pd.DataFrame(rows).set_index("date").sort_index()
    diagnostics = {
        "days": int(len(frame)),
        "trades": total_trades,
        "stops": total_stops,
        "dailyKillDays": total_kills,
        "regimeCounts": regime_counts,
        "averageDynamicLeverage": float(np.mean(leverage_values)) if leverage_values else 0.0,
        "maxDynamicLeverage": float(max(leverage_values)) if leverage_values else 0.0,
    }
    return frame, diagnostics, days


def load_funding(universe: list[str], refresh_recent: bool) -> dict[str, pd.Series]:
    def load_one(inst_id: str) -> tuple[str, pd.Series]:
        path = DATA_DIR / f"{inst_id.lower()}-funding.csv"
        cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["ts", "fundingRate"])
        if refresh_recent:
            recent = fetch_recent_funding(inst_id, 14)
            cached = pd.concat([cached, recent], ignore_index=True).drop_duplicates("ts", keep="last")
            cached = cached.sort_values("ts")
            path.parent.mkdir(parents=True, exist_ok=True)
            cached.to_csv(path, index=False)
            print(f"funding {inst_id}: {len(recent)} recent rows", flush=True)
        return inst_id, funding_series(cached, "hourly").sort_index()

    if not refresh_recent:
        return dict(load_one(inst_id) for inst_id in universe)
    with ThreadPoolExecutor(max_workers=4) as executor:
        return dict(executor.map(load_one, universe))


def evaluate(
    hourly: dict[str, pd.DataFrame],
    daily: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    funding: dict[str, pd.Series],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base, diagnostics, day_details = simulate(hourly, daily, closes, funding, BASE_COST)
    stress, stress_diagnostics, _ = simulate(hourly, daily, closes, funding, STRESS_COST)
    windows = (("oneYear", 365), ("oneMonth", 30), ("oneWeek", 7), ("oneDay", 1))
    result = {
        "windows": {label: metrics(base, days, BASE_COST) for label, days in windows},
        "stressWindows": {label: metrics(stress, days, STRESS_COST) for label, days in windows},
        "diagnostics": diagnostics,
        "stressDiagnostics": stress_diagnostics,
        "robustness": robustness_metrics(base, day_details),
    }
    return result, day_details


def acceptance_gates(result: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "baseOneYearPositive": result["windows"]["oneYear"]["totalReturn"] > 0,
        "stressOneYearPositive": result["stressWindows"]["oneYear"]["totalReturn"] > 0,
        "maxDrawdownBelow30Pct": result["windows"]["oneYear"]["maxDrawdown"] > -0.30,
        "rolling30PositiveAbove60Pct": result["robustness"]["rolling30PositiveShare"] > 0.60,
        "recentMonthPositive": result["windows"]["oneMonth"]["totalReturn"] > 0,
        "recentWeekPositive": result["windows"]["oneWeek"]["totalReturn"] > 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-market", action="store_true")
    parser.add_argument("--refresh-funding", action="store_true")
    args = parser.parse_args()

    universe = load_universe()
    hourly = load_hourly(universe, args.refresh_market)
    daily, closes = build_daily_inputs(hourly)
    funding = load_funding(universe, args.refresh_funding)
    result, day_details = evaluate(hourly, daily, closes, funding)
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "mode": "OFFLINE_BACKTEST_ONLY",
        "strategy": {
            "id": "r10-dual-speed-shock-continuation",
            "name": "R10 双速趋势延续 + 冲击后重入 + 动态杠杆",
            "slowSignal": "前一完整UTC日BTC位于20日均线同侧，且7日收益同方向",
            "fastSignal": "决策前已完成小时K线的12小时收益与20小时EMA同方向",
            "shock": "|BTC日收益/20日波动|>1.5时冷静4小时；趋势仍成立且价格未穿越冲击K线中点才半风险重入",
            "selection": "上涨只做多动量前3，下跌只做空后3；30日/7日/20日高点综合排名；排名缓冲至前后6",
            "sizing": "20日逆波动率；单币基础权重上限25%；BTC波动率映射1.5x-5x动态杠杆；冲击重入减半",
            "risk": "1.5 ATR止损（3%-6%）；盈利4%后2.5%移动止盈；日内亏损3%组合熔断",
        },
        "data": {
            "source": "OKX public market API",
            "universe": universe,
            "universeSize": len(universe),
            "hourlyStart": min(frame.index.min() for frame in hourly.values()).isoformat(),
            "hourlyEnd": max(frame.index.max() for frame in hourly.values()).isoformat(),
            "backtestEnd": result["windows"]["oneDay"]["end"],
        },
        "costs": {
            "baseOneWay": BASE_COST,
            "stressOneWay": STRESS_COST,
            "fundingIncluded": True,
        },
        "candidate": result,
        "acceptanceGates": acceptance_gates(result),
        "recentDays": day_details[-35:],
        "limitations": [
            "固定使用当前20币流动性池回看历史，仍存在幸存者偏差。",
            "慢信号和排名只使用前一完整UTC日；快信号只使用决策时点前已完成的小时K线。",
            "小时K线无法知道同小时高低点顺序；移动止盈只使用前一小时已确认峰值。",
            "组合熔断按小时盯市，快速行情可能越过3%阈值后才成交。",
            "未模拟交易所真实强平、维持保证金阶梯和自动减仓。",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "data": report["data"],
                "candidate": result,
                "acceptanceGates": report["acceptanceGates"],
                "recentDays": day_details[-7:],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
