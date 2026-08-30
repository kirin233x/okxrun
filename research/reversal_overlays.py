from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .high_return_candidates import (
        BASE_COST,
        CANDLE_API,
        DATA_DIR,
        STARTING_EQUITY,
        cache_path,
        fetch_recent_funding,
        funding_series,
        indexed_frame,
        request_json,
        window_metrics,
    )
except ImportError:
    from high_return_candidates import (
        BASE_COST,
        CANDLE_API,
        DATA_DIR,
        STARTING_EQUITY,
        cache_path,
        fetch_recent_funding,
        funding_series,
        indexed_frame,
        request_json,
        window_metrics,
    )


ROOT = Path(__file__).resolve().parents[1]
BASE_REPORT = ROOT / "research" / "artifacts" / "high-return-report.json"
OUTPUT = ROOT / "research" / "artifacts" / "reversal-overlay-report.json"
FORMATION_DAYS = 56
VOLATILITY_DAYS = 30
BETA_DAYS = 60
BTC_TREND_DAYS = 20
BTC_STRONG_TREND = 0.10
TARGET_VOLATILITY = 0.35
MIN_GROSS_SCALE = 0.35
STOP_LOSS = 0.12


def recent_daily(inst_id: str) -> pd.DataFrame:
    rows = request_json(CANDLE_API, {"instId": inst_id, "bar": "1Dutc", "limit": "100"}).get("data", [])
    records = [
        {
            "ts": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "volumeCcy": float(row[6]),
            "volumeQuote": float(row[7]),
        }
        for row in rows
        if len(row) >= 9 and row[8] == "1"
    ]
    return pd.DataFrame(records)


def load_daily(inst_id: str, refresh: bool) -> pd.DataFrame:
    path = cache_path(inst_id, "1Dutc")
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if refresh or cached.empty:
        latest = recent_daily(inst_id)
        combined = pd.concat([cached, latest], ignore_index=True)
        combined = combined.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path, index=False)
        return combined
    return cached


def load_funding(inst_id: str, refresh: bool) -> pd.DataFrame:
    path = DATA_DIR / f"{inst_id.lower()}-funding.csv"
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["ts", "fundingRate"])
    if refresh:
        recent = fetch_recent_funding(inst_id, days=14)
        combined = pd.concat([cached, recent], ignore_index=True)
        combined["ts"] = pd.to_numeric(combined["ts"], errors="coerce")
        combined["fundingRate"] = pd.to_numeric(combined["fundingRate"], errors="coerce")
        combined = combined.dropna().drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
        combined.to_csv(path, index=False)
        return combined
    return cached


def compute_inputs(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    universe = base["data"]["reversalUniverse"]
    prices: dict[str, pd.Series] = {}
    funding: dict[str, pd.Series] = {}
    for inst_id in universe:
        prices[inst_id] = indexed_frame(load_daily(inst_id, refresh))["close"]
        funding[inst_id] = funding_series(load_funding(inst_id, refresh), "daily")
        if refresh:
            time.sleep(0.04)
    price_panel = pd.concat(prices, axis=1).sort_index().ffill(limit=2)
    eligible = [column for column in price_panel if price_panel[column].notna().sum() >= 365]
    price_panel = price_panel[eligible]
    funding_panel = pd.concat({key: funding[key] for key in eligible}, axis=1).reindex(price_panel.index).fillna(0.0)
    return price_panel, funding_panel


def build_weights(price_panel: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    returns = price_panel.pct_change(fill_method=None)
    formation = price_panel.pct_change(FORMATION_DAYS, fill_method=None).shift(1)
    volatility = returns.rolling(VOLATILITY_DAYS).std(ddof=0).shift(1)
    btc_return = returns["BTC-USDT-SWAP"]
    btc_trend = price_panel["BTC-USDT-SWAP"].pct_change(BTC_TREND_DAYS, fill_method=None).shift(1)
    beta_covariance = returns.rolling(BETA_DAYS).cov(btc_return).shift(1)
    beta_variance = btc_return.rolling(BETA_DAYS).var(ddof=0).shift(1)
    betas = beta_covariance.div(beta_variance, axis=0).clip(lower=0.1, upper=3.0)
    rebalance_dates = set(price_panel.resample("W-MON").first().index)
    weights = pd.DataFrame(0.0, index=price_panel.index, columns=price_panel.columns)
    previous = pd.Series(0.0, index=price_panel.columns)
    entry_prices: dict[str, float] = {}
    stops = 0
    regime_pauses = 0
    rebalances = 0
    gross_scales: list[float] = []

    use_regime = variant in {"btc-regime", "combined"}
    use_beta = variant in {"beta-neutral", "combined"}
    use_risk = variant in {"risk-overlay", "combined"}

    for date in price_panel.index:
        target = previous.copy()
        if date in rebalance_dates:
            signal = formation.loc[date].dropna()
            vol = volatility.loc[date].reindex(signal.index).dropna()
            ranked = signal.reindex(vol[vol >= vol.median()].index).dropna().sort_values()
            if len(ranked) >= 8:
                rebalances += 1
                basket = max(2, len(ranked) // 4)
                longs = list(ranked.index[:basket])
                shorts = list(ranked.index[-basket:])
                target = pd.Series(0.0, index=price_panel.columns)
                if use_regime and abs(float(btc_trend.loc[date])) > BTC_STRONG_TREND:
                    regime_pauses += 1
                    gross_scales.append(0.0)
                    entry_prices = {}
                else:
                    long_gross = 0.5
                    short_gross = 0.5
                    if use_beta:
                        long_beta = float(betas.loc[date, longs].mean())
                        short_beta = float(betas.loc[date, shorts].mean())
                        total_beta = long_beta + short_beta
                        if math.isfinite(total_beta) and total_beta > 0:
                            long_gross = float(np.clip(short_beta / total_beta, 0.35, 0.65))
                            short_gross = 1.0 - long_gross
                    target.loc[longs] = long_gross / basket
                    target.loc[shorts] = -short_gross / basket
                    scale = 1.0
                    if use_risk:
                        history = returns.loc[returns.index < date, target.index[target != 0]].tail(30)
                        covariance = history.cov().fillna(0.0)
                        vector = target.reindex(covariance.index).to_numpy(dtype=float)
                        expected_variance = float(vector @ covariance.to_numpy(dtype=float) @ vector)
                        expected_volatility = math.sqrt(max(expected_variance, 0.0) * 365)
                        if expected_volatility > 0:
                            scale = float(np.clip(TARGET_VOLATILITY / expected_volatility, MIN_GROSS_SCALE, 1.0))
                            target *= scale
                    gross_scales.append(scale)
                    entry_prices = {
                        inst_id: float(price_panel.loc[date, inst_id]) for inst_id in target.index[target != 0]
                    }
        elif use_risk and entry_prices:
            for inst_id in list(entry_prices):
                weight = float(target[inst_id])
                if weight == 0 or pd.isna(price_panel.loc[date, inst_id]):
                    continue
                direction = 1.0 if weight > 0 else -1.0
                position_return = direction * (float(price_panel.loc[date, inst_id]) / entry_prices[inst_id] - 1)
                if position_return <= -STOP_LOSS:
                    target[inst_id] = 0.0
                    stops += 1
                    del entry_prices[inst_id]
        weights.loc[date] = target
        previous = target

    applied = weights.shift(1).fillna(0.0)
    beta_exposure = (applied * betas.reindex_like(applied).fillna(0.0)).sum(axis=1)
    diagnostics = {
        "variant": variant,
        "rebalances": rebalances,
        "regimePausedRebalances": regime_pauses,
        "stops": stops,
        "averageGrossExposure": float(applied.abs().sum(axis=1).mean()),
        "averageAbsoluteBetaExposure": float(beta_exposure.abs().mean()),
        "averageRebalanceScale": float(np.mean(gross_scales)) if gross_scales else 0.0,
    }
    return weights, diagnostics


def simulate(price_panel: pd.DataFrame, funding_panel: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    returns = price_panel.pct_change(fill_method=None)
    weights, diagnostics = build_weights(price_panel, variant)
    applied = weights.shift(1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    price_return = (applied * returns).sum(axis=1)
    funding_return = -(applied * funding_panel.reindex_like(weights).fillna(0.0)).sum(axis=1)
    gross = price_return + funding_return
    cost = turnover * BASE_COST
    result = pd.DataFrame(
        {"net": gross - cost, "gross": gross, "funding": funding_return, "cost": cost, "turnover": turnover}
    ).iloc[60:]
    diagnostics["turnover"] = float(turnover.sum())
    return result, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    prices, funding = compute_inputs(args.refresh)
    variants = ["baseline", "btc-regime", "beta-neutral", "risk-overlay", "combined"]
    results = []
    for variant in variants:
        returns, diagnostics = simulate(prices, funding, variant)
        results.append(
            {
                "id": variant,
                "windows": {
                    label: window_metrics(returns, days)
                    for label, days in (("oneYear", 365), ("oneMonth", 30), ("oneWeek", 7))
                },
                "diagnostics": diagnostics,
            }
        )
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "mode": "OFFLINE_EXPERIMENT_ONLY",
        "runningPaperStrategyChanged": False,
        "dataEnd": prices.index.max().isoformat(),
        "universe": list(prices.columns),
        "cost": BASE_COST,
        "rules": {
            "baseline": "原始高波动 8 周反转，周调仓，金额中性。",
            "btc-regime": "使用前一日可见数据；BTC 过去 20 日绝对涨跌超过 10% 时，本周暂停开仓。",
            "beta-neutral": "使用前 60 日 Beta，使多空两侧 BTC Beta 尽量相抵；单侧名义限制在 35%–65%。",
            "risk-overlay": "组合目标年化波动 35%，总敞口只降不升且不低于 35%；单仓日收盘亏损 12% 后退出到下次调仓。",
            "combined": "同时应用 BTC 强趋势暂停、Beta 中性和风险约束。",
        },
        "limitations": [
            "仍使用当前存续与流动性资产池，存在幸存者偏差。",
            "一周只有 7 个日线观察，不足以选择参数；必须同时参考一个月和一年。",
            "止损按日收盘执行，未模拟盘中跳空和盘口冲击。",
        ],
        "variants": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
