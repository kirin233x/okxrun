from __future__ import annotations

import argparse
import http.client
import io
import json
import math
import time
import urllib.parse
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "research" / "data" / "high-return"
REPORT_PATH = ROOT / "research" / "artifacts" / "high-return-report.json"
CANDLE_API = "https://www.okx.com/api/v5/market/history-candles"
INSTRUMENT_API = "https://www.okx.com/api/v5/public/instruments"
TICKERS_API = "https://www.okx.com/api/v5/market/tickers"
FUNDING_API = "https://www.okx.com/api/v5/public/funding-rate-history"
FUNDING_ARCHIVE = (
    "https://static.okx.com/cdn/okex/traderecords/swaprates/monthly/"
    "{yyyymm}/{instrument}-fundingrates-{year}-{month:02d}.zip?v=999"
)
STARTING_EQUITY = 10_000.0
BASE_COST = 0.0006
STRESS_COST = 0.0012
MODEL_ASSETS = ("BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP")
MAX_REVERSAL_ASSETS = 30
MIN_DAILY_NOTIONAL = 25_000_000.0
MIN_LISTING_DAYS = 180
ONE_HOUR_MS = 3_600_000
ONE_DAY_MS = 86_400_000
LOOKBACK_DAYS = 520
XGB_TEST_DAYS = 365
XGB_MIN_TRAIN_HOURS = 3_000
EXCLUDED_UNDERLYINGS = {"XAU"}


def request_json(url: str, params: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(6):
        request = urllib.request.Request(
            f"{url}?{urllib.parse.urlencode(params)}",
            headers={"User-Agent": "PulseBacktest/3.0", "Connection": "close"},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.load(response)
            break
        except (urllib.error.URLError, http.client.RemoteDisconnected, TimeoutError) as error:
            last_error = error
            time.sleep(min(1.5 * (attempt + 1), 6))
    if payload is None:
        raise RuntimeError(f"OKX 请求重试失败: {last_error}")
    if payload.get("code") != "0":
        raise RuntimeError(f"OKX {payload.get('code')}: {payload.get('msg')}")
    return payload


def cache_path(inst_id: str, bar: str) -> Path:
    return DATA_DIR / f"{inst_id.lower()}-{bar.lower()}.csv"


def fetch_candles(inst_id: str, bar: str, days: int) -> pd.DataFrame:
    interval_ms = ONE_HOUR_MS if bar == "1H" else ONE_DAY_MS
    requested_bars = math.ceil(days * ONE_DAY_MS / interval_ms) + 5
    cutoff = int(time.time() * 1000) - requested_bars * interval_ms
    after: int | None = None
    rows: dict[int, list[str]] = {}
    page = 0
    while True:
        params = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after is not None:
            params["after"] = str(after)
        batch = request_json(CANDLE_API, params).get("data", [])
        if not batch:
            break
        for row in batch:
            if len(row) >= 9 and row[8] == "1":
                rows[int(row[0])] = row
        oldest = min(int(row[0]) for row in batch)
        if oldest <= cutoff or len(batch) < 100:
            break
        after = oldest
        page += 1
        if page > math.ceil(requested_bars / 100) + 5:
            break
        time.sleep(0.04)

    records = [
        {
            "ts": ts,
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "volumeCcy": float(row[6]),
            "volumeQuote": float(row[7]),
        }
        for ts, row in rows.items()
        if ts >= cutoff
    ]
    frame = pd.DataFrame(records).sort_values("ts").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"{inst_id} {bar} 没有可用数据。")
    path = cache_path(inst_id, bar)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_candles(inst_id: str, bar: str, days: int, refresh: bool) -> pd.DataFrame:
    path = cache_path(inst_id, bar)
    if refresh or not path.exists():
        return fetch_candles(inst_id, bar, days)
    return pd.read_csv(path)


def indexed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.to_datetime(result.pop("ts"), unit="ms", utc=True)
    return result[~result.index.duplicated(keep="last")].sort_index()


def fetch_recent_funding(inst_id: str, days: int = 120) -> pd.DataFrame:
    cutoff = int(time.time() * 1000) - days * ONE_DAY_MS
    after: int | None = None
    rows: dict[int, float] = {}
    while True:
        params = {"instId": inst_id, "limit": "100"}
        if after is not None:
            params["after"] = str(after)
        batch = request_json(FUNDING_API, params).get("data", [])
        if not batch:
            break
        for row in batch:
            timestamp = int(row["fundingTime"])
            if timestamp >= cutoff:
                rows[timestamp] = float(row.get("realizedRate") or row.get("fundingRate") or 0)
        oldest = min(int(row["fundingTime"]) for row in batch)
        if oldest <= cutoff or len(batch) < 100:
            break
        after = oldest
        time.sleep(0.12)
    return pd.DataFrame([{"ts": timestamp, "fundingRate": rate} for timestamp, rate in rows.items()])


def fetch_funding(inst_id: str, refresh: bool) -> pd.DataFrame:
    path = DATA_DIR / f"{inst_id.lower()}-funding.csv"
    if not refresh and path.exists():
        return pd.read_csv(path)
    now = pd.Timestamp.now(tz="UTC")
    first_month = (now - pd.Timedelta(395, unit="D")).normalize().replace(day=1)
    last_complete_month = now.normalize().replace(day=1) - pd.offsets.MonthBegin(1)
    frames: list[pd.DataFrame] = []
    for month in pd.date_range(first_month, last_complete_month, freq="MS", tz="UTC"):
        url = FUNDING_ARCHIVE.format(
            yyyymm=month.strftime("%Y%m"), instrument=inst_id, year=month.year, month=month.month
        )
        request = urllib.request.Request(url, headers={"User-Agent": "PulseBacktest/3.0", "Connection": "close"})
        archive: bytes | None = None
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    archive = response.read()
                break
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    break
                last_error = error
            except (urllib.error.URLError, http.client.RemoteDisconnected, TimeoutError) as error:
                last_error = error
            time.sleep(min(1.5 * (attempt + 1), 6))
        if archive is None:
            if isinstance(last_error, urllib.error.HTTPError) and last_error.code == 404:
                continue
            raise RuntimeError(f"{inst_id} {month:%Y-%m} 资金费率归档重试失败: {last_error}")
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            csv_names = [name for name in zipped.namelist() if name.endswith(".csv")]
            if len(csv_names) != 1:
                continue
            frame = pd.read_csv(zipped.open(csv_names[0])).rename(
                columns={"funding_time": "ts", "funding_rate": "fundingRate"}
            )
            frames.append(frame[["ts", "fundingRate"]])
        time.sleep(0.025)
    recent = fetch_recent_funding(inst_id)
    if not recent.empty:
        frames.append(recent)
    if not frames:
        return pd.DataFrame(columns=["ts", "fundingRate"])
    combined = pd.concat(frames, ignore_index=True)
    combined["ts"] = pd.to_numeric(combined["ts"], errors="coerce")
    combined["fundingRate"] = pd.to_numeric(combined["fundingRate"], errors="coerce")
    combined = combined.dropna().drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def funding_series(frame: pd.DataFrame, frequency: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    index = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    result = pd.Series(frame["fundingRate"].to_numpy(dtype=float), index=index)
    if frequency == "daily":
        result.index = result.index.normalize()
        return result.groupby(level=0).sum()
    return result.groupby(level=0).sum()


def discover_reversal_universe(refresh: bool) -> list[dict[str, Any]]:
    universe_path = DATA_DIR / "reversal-universe.json"
    if not refresh and universe_path.exists():
        return json.loads(universe_path.read_text(encoding="utf-8"))

    instruments = request_json(INSTRUMENT_API, {"instType": "SWAP"}).get("data", [])
    tickers = request_json(TICKERS_API, {"instType": "SWAP"}).get("data", [])
    ticker_map = {ticker["instId"]: ticker for ticker in tickers}
    now_ms = int(time.time() * 1000)
    candidates: list[dict[str, Any]] = []
    for instrument in instruments:
        inst_id = instrument.get("instId", "")
        underlying = inst_id.split("-", 1)[0]
        ticker = ticker_map.get(inst_id)
        if (
            instrument.get("settleCcy") != "USDT"
            or not inst_id.endswith("-USDT-SWAP")
            or instrument.get("state") != "live"
            or ticker is None
            or underlying in EXCLUDED_UNDERLYINGS
        ):
            continue
        listing_time = int(instrument.get("listTime") or 0)
        listing_days = (now_ms - listing_time) / ONE_DAY_MS if listing_time else 0
        quote_volume = float(ticker.get("volCcy24h") or 0)
        last_price = float(ticker.get("last") or 0)
        notional = quote_volume * last_price
        if listing_days < MIN_LISTING_DAYS or notional < MIN_DAILY_NOTIONAL:
            continue
        candidates.append(
            {"instId": inst_id, "listingDays": round(listing_days, 1), "dailyNotional": notional}
        )
    candidates.sort(key=lambda item: item["dailyNotional"], reverse=True)
    selected = candidates[:MAX_REVERSAL_ASSETS]
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    universe_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected


def hourly_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    close = frame["close"]
    returns = close.pct_change()
    log_returns = np.log(close).diff()
    high_low = np.log(frame["high"] / frame["low"]).replace([np.inf, -np.inf], np.nan)
    quote_volume = frame["volumeQuote"].replace(0, np.nan)
    features = pd.DataFrame(index=frame.index)
    for lag in (1, 2, 3, 6, 12, 24, 48, 72, 168):
        features[f"return_{lag}h"] = close.pct_change(lag)
    for window in (6, 12, 24, 48, 72, 168, 336):
        features[f"volatility_{window}h"] = returns.rolling(window).std(ddof=0)
        features[f"range_{window}h"] = high_low.rolling(window).mean()
        logged_volume = np.log1p(quote_volume)
        features[f"volume_z_{window}h"] = (
            logged_volume - logged_volume.rolling(window).mean()
        ) / logged_volume.rolling(window).std(ddof=0)
        features[f"distance_sma_{window}h"] = close / close.rolling(window).mean() - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    features["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=48, adjust=False).mean()
    features["macd_norm"] = (ema_fast - ema_slow) / close
    features["hour_sin"] = np.sin(2 * np.pi * frame.index.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * frame.index.hour / 24)
    target = log_returns.shift(-1)
    next_return = close.pct_change().shift(-1)
    return features.replace([np.inf, -np.inf], np.nan), target, next_return


def walk_forward_predictions(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    features, target, next_return = hourly_features(frame)
    valid = features.notna().all(axis=1) & target.notna() & next_return.notna()
    features, target, next_return = features.loc[valid], target.loc[valid], next_return.loc[valid]
    test_start = features.index.max() - pd.Timedelta(XGB_TEST_DAYS, unit="D")
    test_months = pd.period_range(
        test_start.tz_localize(None).to_period("M"),
        features.index.max().tz_localize(None).to_period("M"),
        freq="M",
    )
    prediction_parts: list[pd.Series] = []
    return_parts: list[pd.Series] = []
    params = {
        "n_estimators": 350,
        "max_depth": 3,
        "learning_rate": 0.035,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 8,
        "reg_alpha": 0.15,
        "reg_lambda": 2.0,
        "objective": "reg:squarederror",
        "n_jobs": 4,
        "random_state": 42,
    }
    for month in test_months:
        month_start = pd.Timestamp(month.start_time, tz="UTC")
        month_end = pd.Timestamp(month.end_time, tz="UTC")
        test_mask = (features.index >= max(month_start, test_start)) & (features.index <= month_end)
        train_mask = features.index < month_start
        if not test_mask.any() or train_mask.sum() < XGB_MIN_TRAIN_HOURS:
            continue
        model = XGBRegressor(**params)
        model.fit(features.loc[train_mask], target.loc[train_mask], verbose=False)
        prediction_parts.append(pd.Series(model.predict(features.loc[test_mask]), index=features.loc[test_mask].index))
        return_parts.append(next_return.loc[test_mask])
    if not prediction_parts:
        raise RuntimeError("XGBoost 没有产生样本外预测。")
    return pd.concat(prediction_parts).sort_index(), pd.concat(return_parts).sort_index()


def positions_from_forecast(prediction: pd.Series, cost_rate: float, lambda_cost: float = 2.0) -> pd.Series:
    positions: list[float] = []
    previous = 0.0
    for forecast in prediction:
        desired = 1.0 if forecast > 0 else 0.0
        threshold = lambda_cost * cost_rate * abs(desired - previous)
        if abs(forecast) > threshold:
            previous = desired
        positions.append(previous)
    return pd.Series(positions, index=prediction.index)


def xgb_asset_returns(frame: pd.DataFrame, funding: pd.Series, cost_rate: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction, forward_return = walk_forward_predictions(frame)
    position = positions_from_forecast(prediction, cost_rate)
    turnover = position.diff().abs().fillna(position.abs())
    price_return = position * forward_return
    next_funding = funding.reindex(position.index, fill_value=0.0).shift(-1).fillna(0.0)
    funding_return = -position * next_funding
    gross = price_return + funding_return
    cost = turnover * cost_rate
    diagnostics = {
        "predictions": int(len(prediction)),
        "orders": int((turnover > 0).sum()),
        "turnover": float(turnover.sum()),
        "activeShare": float((position != 0).mean()),
        "longShare": float((position > 0).mean()),
        "flatShare": float((position == 0).mean()),
        "directionalHitRate": float((np.sign(prediction) == np.sign(forward_return)).mean()),
        "averageAbsoluteForecast": float(prediction.abs().mean()),
    }
    return pd.DataFrame({"net": gross - cost, "gross": gross, "funding": funding_return, "cost": cost, "turnover": turnover}), diagnostics


def combine_xgb(frames: dict[str, pd.DataFrame], funding: dict[str, pd.Series], cost_rate: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    asset_returns: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}
    for inst_id, frame in frames.items():
        result, detail = xgb_asset_returns(frame, funding[inst_id], cost_rate)
        asset_returns.append(result.rename(columns={column: f"{inst_id}:{column}" for column in result.columns}))
        diagnostics[inst_id] = detail
    panel = pd.concat(asset_returns, axis=1).sort_index()
    combined = pd.DataFrame(index=panel.index)
    for metric in ("net", "gross", "funding", "cost", "turnover"):
        combined[metric] = panel[[column for column in panel if column.endswith(f":{metric}")]].mean(axis=1, skipna=True)
    return combined.dropna(subset=["net"]), diagnostics


def reversal_returns(price_panel: pd.DataFrame, funding_panel: pd.DataFrame, cost_rate: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    returns = price_panel.pct_change()
    formation = price_panel.pct_change(56).shift(1)
    volatility = returns.rolling(30).std(ddof=0).shift(1)
    weights = pd.DataFrame(np.nan, index=price_panel.index, columns=price_panel.columns)
    rebalance_dates = set(price_panel.resample("W-MON").first().index)
    previous = pd.Series(0.0, index=price_panel.columns)
    for date in price_panel.index:
        if date in rebalance_dates:
            signal = formation.loc[date].dropna()
            vol = volatility.loc[date].reindex(signal.index).dropna()
            ranked = signal.reindex(vol[vol >= vol.median()].index).dropna().sort_values()
            if len(ranked) >= 8:
                basket = max(2, len(ranked) // 4)
                target = pd.Series(0.0, index=price_panel.columns)
                target.loc[ranked.index[:basket]] = 0.5 / basket
                target.loc[ranked.index[-basket:]] = -0.5 / basket
                previous = target
        weights.loc[date] = previous

    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    gross = (weights.shift(1) * returns).sum(axis=1)
    funding_return = -(weights.shift(1) * funding_panel.reindex_like(weights).fillna(0.0)).sum(axis=1)
    gross = gross + funding_return
    cost = turnover * cost_rate
    result = pd.DataFrame({"net": gross - cost, "gross": gross, "funding": funding_return, "cost": cost, "turnover": turnover}).iloc[60:]
    diagnostics = {
        "universeAssets": int(price_panel.shape[1]),
        "averageActivePositions": float((weights != 0).sum(axis=1).mean()),
        "rebalanceCount": int((turnover > 0).sum()),
        "turnover": float(turnover.sum()),
        "formationDays": 56,
        "rebalance": "weekly",
    }
    return result, diagnostics


def window_metrics(frame: pd.DataFrame, days: int) -> dict[str, Any]:
    sample = frame[frame.index > frame.index.max() - pd.Timedelta(days, unit="D")]
    net = sample["net"].fillna(0)
    equity = STARTING_EQUITY * (1 + net).cumprod()
    periods_per_year = 24 * 365 if len(sample) > days * 2 else 365
    drawdown = equity / equity.cummax() - 1
    return {
        "start": sample.index.min().isoformat(),
        "end": sample.index.max().isoformat(),
        "observations": int(len(sample)),
        "startingEquity": STARTING_EQUITY,
        "endingEquity": float(equity.iloc[-1]),
        "totalReturn": float(equity.iloc[-1] / STARTING_EQUITY - 1),
        "priceReturnApprox": float((sample["gross"] - sample["funding"]).sum()),
        "grossReturnApprox": float(sample["gross"].sum()),
        "costReturn": float(sample["cost"].sum()),
        "fundingReturn": float(sample["funding"].sum()),
        "turnover": float(sample["turnover"].sum()),
        "maxDrawdown": float(drawdown.min()),
        "annualizedVolatility": float(net.std(ddof=0) * math.sqrt(periods_per_year)),
        "positivePeriodShare": float((net > 0).mean()),
    }


def strategy_report(strategy_id: str, name: str, rule: str, source: str, source_url: str, base: pd.DataFrame, stress: pd.DataFrame, diagnostics: dict[str, Any]) -> dict[str, Any]:
    windows = {label: window_metrics(base, days) for label, days in (("oneYear", 365), ("oneMonth", 30), ("oneWeek", 7))}
    stress_windows = {label: window_metrics(stress, days) for label, days in (("oneYear", 365), ("oneMonth", 30), ("oneWeek", 7))}
    passed = windows["oneYear"]["totalReturn"] > 0 and stress_windows["oneYear"]["totalReturn"] > 0
    return {"id": strategy_id, "name": name, "rule": rule, "source": source, "sourceUrl": source_url, "windows": windows, "stressWindows": stress_windows, "diagnostics": diagnostics, "passedOneYearGate": bool(passed)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    hourly_frames = {inst_id: indexed_frame(load_candles(inst_id, "1H", LOOKBACK_DAYS, args.refresh)) for inst_id in MODEL_ASSETS}
    universe = discover_reversal_universe(args.refresh)
    daily_frames: dict[str, pd.DataFrame] = {}
    for item in universe:
        try:
            daily_frames[item["instId"]] = indexed_frame(load_candles(item["instId"], "1Dutc", LOOKBACK_DAYS, args.refresh))
        except Exception as error:
            item["excludedReason"] = str(error)
    price_panel = pd.concat({inst_id: frame["close"] for inst_id, frame in daily_frames.items()}, axis=1).sort_index()
    eligible = [column for column in price_panel if price_panel[column].notna().sum() >= 365]
    price_panel = price_panel[eligible].ffill(limit=2)

    funding_frames = {inst_id: fetch_funding(inst_id, args.refresh) for inst_id in sorted(set(eligible) | set(MODEL_ASSETS))}
    hourly_funding = {inst_id: funding_series(funding_frames[inst_id], "hourly") for inst_id in MODEL_ASSETS}
    daily_funding = pd.concat(
        {inst_id: funding_series(funding_frames[inst_id], "daily") for inst_id in eligible}, axis=1
    ).reindex(price_panel.index).fillna(0.0)

    xgb_base, xgb_diagnostics = combine_xgb(hourly_frames, hourly_funding, BASE_COST)
    xgb_stress, _ = combine_xgb(hourly_frames, hourly_funding, STRESS_COST)
    reversal_base, reversal_diagnostics = reversal_returns(price_panel, daily_funding, BASE_COST)
    reversal_stress, _ = reversal_returns(price_panel, daily_funding, STRESS_COST)
    strategies = [
        strategy_report("hourly-xgb-cost-aware", "小时 XGBoost 成本过滤", "BTC/ETH/SOL 下一小时收益滚动预测；每月只用当时可见历史重训。预测幅度未超过仓位变化成本的 2 倍时不换仓，三币等权。", "Bysik & Ślepaczuk (2026)", "https://arxiv.org/abs/2606.00060", xgb_base, xgb_stress, xgb_diagnostics),
        strategy_report("liquid-altcoin-reversal", "高波动流动币横截面反转", "OKX USDT 永续中按实时流动性与上市时间筛选；在高波动半组中做多过去 8 周跌幅最大四分位、做空涨幅最大四分位，每周调仓，净敞口为零。", "Kiefer & Nowotny (2026)", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6703978", reversal_base, reversal_stress, reversal_diagnostics),
    ]
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "mode": "BACKTEST_ONLY",
        "executionEnabled": False,
        "objective": "验证两个公开研究启发的较高收益候选在 OKX 数据与真实成本后的最近一年、一个月和一周表现。",
        "data": {"source": "OKX public market API", "hourlyAssets": list(MODEL_ASSETS), "reversalUniverse": eligible, "reversalUniverseSize": len(eligible), "latestTimestamp": max(xgb_base.index.max(), reversal_base.index.max()).isoformat()},
        "costs": {"baselinePerUnitTurnover": BASE_COST, "baselineBreakdown": "0.05% taker fee + 0.01% slippage", "stressPerUnitTurnover": STRESS_COST, "fundingIncluded": True, "fundingDisclosure": "使用 OKX 官方月度资金费率归档与最近公开 API；按实际多空方向计入。"},
        "validation": {"xgb": "月度 walk-forward；测试月不参与训练；所有特征只使用信号时点及以前数据。", "reversal": "使用 shift(1) 的 8 周形成期与前一日波动率；周调仓；动态流动性筛选只代表当前可交易池，存在历史存续偏差。", "gate": "最近一年在基础成本和双倍成本下均为正，仅作为进入下一轮严谨验证的最低条件。", "limitations": ["反转池按当前 OKX 上市与流动性筛选，仍有存续偏差。", "XGBoost 采用论文思想的可复现缩小版，并非作者完整 27 折超参数搜索。", "资金费率已计入，但借币、盘口冲击与极端行情成交偏差仍未建模；结果不得直接用于实盘。"]},
        "strategies": strategies,
        "passedStrategyIds": [item["id"] for item in strategies if item["passedOneYearGate"]],
    }
    report["success"] = len(report["passedStrategyIds"]) == 2
    report["decision"] = "TWO_CANDIDATES_PASS" if report["success"] else "CANDIDATE_RESEARCH_INCOMPLETE"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
