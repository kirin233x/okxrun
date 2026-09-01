from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Running this file directly puts research/ on sys.path, not the repo root, so
# the shared strategy package needs the root added back before it can import.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from strategy import (  # noqa: E402
    DAILY_KILL_LOSS,
    R9,
    R91,
    R92,
    StrategyVariant,
    compute_signals,
    stop_fraction_from_atr,
    stop_price,
    target_for_day,
)

try:
    from .high_return_candidates import (
        BASE_COST,
        DATA_DIR,
        STARTING_EQUITY,
        STRESS_COST,
        funding_series,
        indexed_frame,
        load_candles,
    )
except ImportError:
    from high_return_candidates import (
        BASE_COST,
        DATA_DIR,
        STARTING_EQUITY,
        STRESS_COST,
        funding_series,
        indexed_frame,
        load_candles,
    )


ROOT = Path(__file__).resolve().parents[1]
OVERLAY_REPORT = ROOT / "research" / "artifacts" / "reversal-overlay-report.json"
UNIVERSE_FILE = ROOT / "research" / "universe.json"
OUTPUT = ROOT / "research" / "artifacts" / "r9-momentum-report.json"
R91_OUTPUT = ROOT / "research" / "artifacts" / "r9.1-momentum-report.json"
R92_OUTPUT = ROOT / "research" / "artifacts" / "r9.2-momentum-report.json"
HOURLY_DAYS = 420
ONE_DAY = timedelta(days=1)


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


def load_universe() -> list[str]:
    source = OVERLAY_REPORT if OVERLAY_REPORT.exists() else UNIVERSE_FILE
    payload = json.loads(source.read_text(encoding="utf-8"))
    return list(payload["universe"])


def load_hourly_asset(inst_id: str, refresh: bool) -> tuple[str, pd.DataFrame]:
    frame = indexed_frame(load_candles(inst_id, "1H", HOURLY_DAYS, refresh))
    numeric = frame[["open", "high", "low", "close", "volumeQuote"]].apply(pd.to_numeric, errors="coerce")
    return inst_id, numeric.dropna(subset=["open", "high", "low", "close"])


def load_hourly(universe: list[str], refresh: bool) -> dict[str, pd.DataFrame]:
    if not refresh:
        return {inst_id: load_hourly_asset(inst_id, False)[1] for inst_id in universe}
    frames: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(load_hourly_asset, inst_id, True): inst_id for inst_id in universe}
        for future in as_completed(futures):
            inst_id, frame = future.result()
            frames[inst_id] = frame
            print(f"downloaded {inst_id}: {len(frame)} hours", flush=True)
    return frames


def daily_from_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    count = frame["close"].resample("1D").count()
    daily = frame.resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volumeQuote=("volumeQuote", "sum"),
    )
    return daily[count >= 23].dropna(subset=["open", "high", "low", "close"])


def build_daily_inputs(hourly: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    daily = {inst_id: daily_from_hourly(frame) for inst_id, frame in hourly.items()}
    closes = pd.concat({inst_id: frame["close"] for inst_id, frame in daily.items()}, axis=1).sort_index()
    return daily, closes


def load_funding(universe: list[str]) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    for inst_id in universe:
        path = DATA_DIR / f"{inst_id.lower()}-funding.csv"
        frame = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["ts", "fundingRate"])
        result[inst_id] = funding_series(frame, "hourly").sort_index()
    return result


def position_stop_price(position: Position) -> float:
    return stop_price(
        position.entry,
        position.direction,
        position.stop_fraction,
        position.peak,
        position.trough,
    )


def stop_fill(position: Position, bar: pd.Series) -> float | None:
    stop = position_stop_price(position)
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


def close_position(position: Position, price: float, cost_rate: float, reason: str) -> None:
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
        for inst_id in set(target) | set(previous_weights)
    }
    entry_cost = float(sum(entry_cost_by_asset.values()))
    for inst_id, weight in target.items():
        bars = hourly[inst_id].loc[(hourly[inst_id].index >= date) & (hourly[inst_id].index < day_end)]
        if bars.empty:
            continue
        entry = float(bars.iloc[0]["open"])
        notional = starting_equity * abs(weight)
        if notional <= 0 or entry <= 0:
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
    timestamps = sorted(
        set().union(
            *[
                set(hourly[inst_id].loc[(hourly[inst_id].index >= date) & (hourly[inst_id].index < day_end)].index)
                for inst_id in active
            ]
        )
    ) if active else []
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
    rebalance_turnover = sum(abs(target.get(inst_id, 0.0) - previous_weights.get(inst_id, 0.0)) for inst_id in set(target) | set(previous_weights))
    stopped_turnover = sum(
        abs(position.quantity * float(position.exit_price or position.entry)) / starting_equity
        for position in positions.values()
        if position.exit_reason in {"POSITION_STOP", "DAILY_KILL"}
    ) if starting_equity else 0.0
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
        ) + stop_count,
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
    variant: StrategyVariant = R9,
    leverage: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    signals = compute_signals(closes, daily)
    score = signals["score"]
    dates = [date for date in closes.index if date in score.index]  # type: ignore[operator]
    equity = STARTING_EQUITY
    previous_weights: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    days: list[dict[str, Any]] = []
    regime_counts: dict[str, int] = {}
    total_stops = 0
    total_kills = 0
    total_trades = 0
    gross_exposures: list[float] = []
    for date in dates:
        target, signal_detail = target_for_day(date, signals, previous_weights, variant, leverage)
        if not signal_detail.get("tradable", False):
            continue
        # Flat signals are real trading days too: they close prior holdings and pay
        # the corresponding cost. BTC is used to reject an incomplete current day.
        date = pd.Timestamp(date)
        validation_assets = set(target) | {"BTC-USDT-SWAP"}
        if any(
            len(hourly[inst_id].loc[(hourly[inst_id].index >= date) & (hourly[inst_id].index < date + ONE_DAY)]) < 23
            for inst_id in validation_assets
        ):
            continue
        result, detail = simulate_day(
            date, equity, target, previous_weights, hourly, signals["atr"], funding, cost_rate  # type: ignore[arg-type]
        )
        equity = detail["endingEquity"]
        previous_weights = detail["endingWeights"]
        rows.append({"date": date, **result})
        regime = signal_detail["regime"]
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        total_stops += detail["stops"]
        total_kills += int(detail["killTriggered"])
        total_trades += detail["trades"]
        gross_exposures.append(signal_detail["grossAfterCap"])
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
        "averageGrossExposure": float(np.mean(gross_exposures)) if gross_exposures else 0.0,
        "variant": variant.id,
        "leverage": leverage,
    }
    return frame, diagnostics, days


def metrics(frame: pd.DataFrame, days: int, cost_rate: float) -> dict[str, Any]:
    sample = frame[frame.index > frame.index.max() - timedelta(days=days)].copy()
    liquidation_cost = float(sample.iloc[-1]["endGross"] * cost_rate)
    sample.loc[sample.index[-1], "net"] -= liquidation_cost
    sample.loc[sample.index[-1], "cost"] += liquidation_cost
    sample.loc[sample.index[-1], "turnover"] += float(sample.iloc[-1]["endGross"])
    growth = (1.0 + sample["net"].fillna(0.0)).cumprod()
    equity = STARTING_EQUITY * growth
    drawdown = growth / growth.cummax().clip(lower=1.0) - 1.0
    return {
        "start": sample.index.min().isoformat(),
        "end": sample.index.max().isoformat(),
        "observations": int(len(sample)),
        "endingEquity": float(equity.iloc[-1]),
        "totalReturn": float(equity.iloc[-1] / STARTING_EQUITY - 1.0),
        "priceReturnApprox": float(sample["price"].sum()),
        "fundingReturn": float(sample["funding"].sum()),
        "costReturn": float(sample["cost"].sum()),
        "turnover": float(sample["turnover"].sum()),
        "maxDrawdown": float(drawdown.min()),
        "annualizedVolatility": float(sample["net"].std(ddof=0) * math.sqrt(365)),
        "positiveDayShare": float((sample["net"] > 0).mean()),
    }


def robustness_metrics(frame: pd.DataFrame, day_details: list[dict[str, Any]]) -> dict[str, Any]:
    sample = frame[frame.index > frame.index.max() - timedelta(days=365)]
    monthly = (1.0 + sample["net"]).resample("MS").prod() - 1.0
    rolling_30 = (1.0 + sample["net"]).rolling(30).apply(np.prod, raw=True) - 1.0
    positive = sample.loc[sample["net"] > 0, "net"]
    negative = sample.loc[sample["net"] < 0, "net"]
    regime_metrics: dict[str, Any] = {}
    for regime in sorted({day["signal"]["regime"] for day in day_details}):
        values = np.array(
            [day["netReturn"] for day in day_details if day["signal"]["regime"] == regime], dtype=float
        )
        regime_metrics[regime] = {
            "days": int(len(values)),
            "compoundedReturn": float(np.prod(1.0 + values) - 1.0),
            "averageDailyReturn": float(values.mean()),
            "positiveDayShare": float((values > 0).mean()),
        }
    return {
        "accountingMaxError": float((frame["net"] - (frame["gross"] - frame["cost"])).abs().max()),
        "positiveMonths": int((monthly > 0).sum()),
        "observedMonths": int(len(monthly)),
        "positiveMonthShare": float((monthly > 0).mean()),
        "bestMonth": float(monthly.max()),
        "worstMonth": float(monthly.min()),
        "medianMonth": float(monthly.median()),
        "rolling30PositiveShare": float((rolling_30.dropna() > 0).mean()),
        "rolling30Worst": float(rolling_30.min()),
        "rolling30Median": float(rolling_30.median()),
        "approxProfitFactor": float(positive.sum() / -negative.sum()),
        "topFiveWinningDaysShare": float(positive.nlargest(5).sum() / positive.sum()),
        "bestDay": float(sample["net"].max()),
        "worstDay": float(sample["net"].min()),
        "maxAssetWeight": float(
            max(abs(position["weight"]) for day in day_details for position in day["positions"])
        ),
        "maxGrossTarget": float(max(day["signal"]["grossAfterCap"] for day in day_details)),
        "byRegime": regime_metrics,
        "monthlyReturns": {month.strftime("%Y-%m"): float(value) for month, value in monthly.items()},
    }


def evaluate_variant(
    variant: StrategyVariant,
    hourly: dict[str, pd.DataFrame],
    daily: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    funding: dict[str, pd.Series],
    leverage: float = 1.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base, diagnostics, day_details = simulate(hourly, daily, closes, funding, BASE_COST, variant, leverage)
    stress, stress_diagnostics, _ = simulate(hourly, daily, closes, funding, STRESS_COST, variant, leverage)
    windows = (("oneYear", 365), ("oneMonth", 30), ("oneWeek", 7), ("oneDay", 1))
    result = {
        "strategy": {"id": variant.id, "name": variant.name, "leverage": leverage},
        "windows": {label: metrics(base, days, BASE_COST) for label, days in windows},
        "stressWindows": {label: metrics(stress, days, STRESS_COST) for label, days in windows},
        "diagnostics": diagnostics,
        "stressDiagnostics": stress_diagnostics,
        "robustness": robustness_metrics(base, day_details),
    }
    return result, day_details


def acceptance_gates(result: dict[str, Any]) -> dict[str, Any]:
    values = {
        "baseOneYearAbove20Pct": result["windows"]["oneYear"]["totalReturn"] > 0.20,
        "stressOneYearAbove10Pct": result["stressWindows"]["oneYear"]["totalReturn"] > 0.10,
        "maxDrawdownBelow25Pct": result["windows"]["oneYear"]["maxDrawdown"] > -0.25,
        "rolling30PositiveAbove60Pct": result["robustness"]["rolling30PositiveShare"] > 0.60,
        "recentMonthPositive": result["windows"]["oneMonth"]["totalReturn"] > 0.0,
        "recentWeekPositive": result["windows"]["oneWeek"]["totalReturn"] > 0.0,
    }
    return {"passed": all(values.values()), "checks": values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--leverage", type=float, default=1.0)
    args = parser.parse_args()
    if not math.isfinite(args.leverage) or args.leverage <= 0:
        parser.error("--leverage must be a positive finite number")
    universe = load_universe()
    hourly = load_hourly(universe, args.refresh)
    daily, closes = build_daily_inputs(hourly)
    funding = load_funding(universe)
    baseline, _ = evaluate_variant(R91, hourly, daily, closes, funding, args.leverage)
    candidate, day_details = evaluate_variant(R92, hourly, daily, closes, funding, args.leverage)
    gates = acceptance_gates(candidate)
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "mode": "OFFLINE_BACKTEST_ONLY",
        "leverage": args.leverage,
        "strategy": {
            "id": "r9.2-liquid-momentum-turnover-light",
            "name": "R9.2 流动币动量 + BTC 状态倾斜（低换手版）",
            "formation": "60% 30日排名 + 25% 7日排名 + 15% 接近20日高点排名",
            "selection": f"首次进入多前3/空后3；已有仓位保留到跌出对应前/后8；20日逆波动率权重；单币上限{20 * args.leverage:g}%权益名义本金",
            "regime": "BTC趋势时75/25倾斜；BTC单日上涨超过20日波动1.5倍时次日空仓；下跌冲击日只做空",
            "risk": "1.5倍ATR止损（3%–6%）；盈利4%后2.5%移动止盈；日内亏损3%组合熔断",
            "rebalance": "每日00:00 UTC；归一化同方向目标权重变化不足10%不调仓；持有至止损、熔断或退出排名缓冲区",
        },
        "data": {
            "source": "OKX public market API",
            "universe": universe,
            "universeSize": len(universe),
            "hourlyStart": min(frame.index.min() for frame in hourly.values()).isoformat(),
            "hourlyEnd": max(frame.index.max() for frame in hourly.values()).isoformat(),
            "backtestEnd": candidate["windows"]["oneDay"]["end"],
        },
        "costs": {"baseOneWay": BASE_COST, "stressOneWay": STRESS_COST, "fundingIncluded": True},
        "baseline": baseline,
        "candidate": candidate,
        "acceptanceGates": gates,
        "recentDays": day_details[-35:],
        "limitations": [
            "使用当前存续且当前流动的20币固定池回看历史，仍有幸存者偏差，不等同于历史动态币池。",
            "小时K线内无法知道最高价与最低价的先后顺序；移动止盈只使用前一小时已确认的峰值，避免同小时内的乐观排序。",
            "按小时收盘判断3%组合熔断，真实15分钟执行可能更早或发生滑点。",
            f"没有单独模拟交易所强平、维持保证金阶梯或自动减仓；总敞口不超过{args.leverage:g}倍且单币不超过{0.2 * args.leverage:g}倍权益名义本金。",
        ],
    }
    leverage_label = f"{args.leverage:g}".replace(".", "-")
    output = R92_OUTPUT if math.isclose(args.leverage, 1.0) else R92_OUTPUT.with_name(
        f"r9.2-momentum-{leverage_label}x-report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "data": report["data"],
                "baseline": baseline,
                "candidate": candidate,
                "acceptanceGates": gates,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
