from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from live.backtest import (
    MAX_HOURS,
    clamp_leverage,
    evaluate,
    merge_hours,
    trim_hours,
    window_metrics,
)


def _hours(start: datetime, count: int, price: float) -> pd.DataFrame:
    rows = []
    for i in range(count):
        ts = int((start + timedelta(hours=i)).timestamp() * 1000)
        px = price * (1.0 + 0.0001 * np.sin(i / 7.0))
        rows.append(
            {
                "ts": ts,
                "open": px,
                "high": px * 1.002,
                "low": px * 0.998,
                "close": px,
                "volumeQuote": 1_000_000.0,
            }
        )
    return pd.DataFrame(rows)


class CacheTests(unittest.TestCase):
    def test_trim_keeps_only_one_year(self) -> None:
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        frame = _hours(now - timedelta(hours=MAX_HOURS + 200), MAX_HOURS + 200, 100.0)
        trimmed = trim_hours(frame)
        self.assertLessEqual(len(trimmed), MAX_HOURS)
        self.assertGreater(trimmed["ts"].min(), int((now - timedelta(days=370)).timestamp() * 1000))

    def test_merge_is_incremental_and_dedupes(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        old = _hours(start, 10, 10.0)
        extra = _hours(start + timedelta(hours=8), 6, 10.0)
        merged = merge_hours(old, extra)
        self.assertEqual(len(merged), 14)
        self.assertEqual(merged["ts"].nunique(), 14)


class MetricsTests(unittest.TestCase):
    def test_principal_scales_ending_equity(self) -> None:
        idx = pd.date_range("2026-01-01", periods=7, freq="D", tz="UTC")
        net = pd.Series([0.01] * 7, index=idx)
        frame = pd.DataFrame({"net": net, "price": net, "funding": 0.0, "cost": 0.0, "endGross": 1.0})
        a = window_metrics(frame, 1_000.0)
        b = window_metrics(frame, 2_000.0)
        assert a is not None and b is not None
        self.assertAlmostEqual(a["totalReturn"], b["totalReturn"], places=6)
        self.assertAlmostEqual(b["endingEquity"] / a["endingEquity"], 2.0, places=4)

    def test_leverage_caps_at_20(self) -> None:
        self.assertEqual(clamp_leverage(20), 20)
        self.assertEqual(clamp_leverage(50), 20)


class EvaluateTests(unittest.TestCase):
    def test_synthetic_universe_produces_a_week_window(self) -> None:
        coins = [
            "BTC-USDT-SWAP",
            "ETH-USDT-SWAP",
            "SOL-USDT-SWAP",
            "DOGE-USDT-SWAP",
            "XRP-USDT-SWAP",
            "ADA-USDT-SWAP",
            "LINK-USDT-SWAP",
            "BNB-USDT-SWAP",
        ]
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        hourly = {}
        for i, inst_id in enumerate(coins):
            raw = _hours(start, 70 * 24, 100.0 + i * 3)
            raw["close"] = raw["close"] * (1.0 + 0.0003 * i * np.sin(np.arange(len(raw)) / 11.0))
            raw["high"] = raw[["open", "close"]].max(axis=1) * 1.003
            raw["low"] = raw[["open", "close"]].min(axis=1) * 0.997
            indexed = raw.copy()
            indexed.index = pd.to_datetime(indexed.pop("ts"), unit="ms", utc=True)
            hourly[inst_id] = indexed
        end = start + timedelta(days=70)
        result = evaluate(hourly, leverage=1.0, window_days=7, principal=1_000.0, now=end)
        self.assertGreaterEqual(result["tradedDays"], 5)
        self.assertLessEqual(result["tradedDays"], 8)
        self.assertIsNotNone(result["metrics"])
        doubled = evaluate(hourly, leverage=1.0, window_days=7, principal=2_000.0, now=end)
        self.assertAlmostEqual(
            doubled["metrics"]["endingEquity"] / result["metrics"]["endingEquity"],
            2.0,
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
