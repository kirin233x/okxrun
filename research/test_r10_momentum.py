from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.r10_momentum_backtest import (
    MAX_DYNAMIC_LEVERAGE,
    MIN_DYNAMIC_LEVERAGE,
    dynamic_leverage,
    fast_snapshot,
    target_for_day,
)


def synthetic_inputs(shock_z: float = 2.0, rising: bool = True):
    date = pd.Timestamp("2026-02-02", tz="UTC")
    assets = ["BTC-USDT-SWAP"] + [f"A{index}-USDT-SWAP" for index in range(7)]
    score = pd.DataFrame(
        [[float(index) for index in range(len(assets))]],
        index=[date],
        columns=assets,
    )
    volatility = pd.DataFrame(0.02, index=[date], columns=assets)
    signals = {
        "score": score,
        "volatility": volatility,
        "btcPrior": pd.Series([110.0], index=[date]),
        "btcMa": pd.Series([100.0], index=[date]),
        "btcReturn1": pd.Series([0.08], index=[date]),
        "btcReturn7": pd.Series([0.12], index=[date]),
        "btcShockZ": pd.Series([shock_z], index=[date]),
    }
    index = pd.date_range("2026-01-31 00:00", "2026-02-02 03:00", freq="h", tz="UTC")
    close = np.linspace(90.0, 115.0, len(index)) if rising else np.linspace(115.0, 90.0, len(index))
    btc_hourly = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volumeQuote": 1_000.0,
        },
        index=index,
    )
    hourly = {"BTC-USDT-SWAP": btc_hourly}
    daily = {
        "BTC-USDT-SWAP": pd.DataFrame(
            {"open": [96.0], "high": [110.0], "low": [90.0], "close": [108.0]},
            index=[date - pd.offsets.Day(1)],
        )
    }
    return date, signals, hourly, daily


class R10MomentumTests(unittest.TestCase):
    def test_dynamic_leverage_falls_as_volatility_rises(self) -> None:
        low_volatility = dynamic_leverage(0.005)
        high_volatility = dynamic_leverage(0.05)
        self.assertEqual(low_volatility, MAX_DYNAMIC_LEVERAGE)
        self.assertEqual(high_volatility, MIN_DYNAMIC_LEVERAGE)
        self.assertGreater(low_volatility, high_volatility)

    def test_fast_snapshot_does_not_use_decision_bar(self) -> None:
        index = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
        frame = pd.DataFrame({"close": np.linspace(100.0, 110.0, len(index))}, index=index)
        decision_time = index[-1]
        before = fast_snapshot(frame, decision_time)
        frame.loc[decision_time, "close"] = 1_000.0
        after = fast_snapshot(frame, decision_time)
        self.assertEqual(before, after)

    def test_positive_shock_waits_four_hours_then_reenters_long_at_half_risk(self) -> None:
        date, signals, hourly, daily = synthetic_inputs(shock_z=2.0, rising=True)
        target, detail = target_for_day(date, signals, hourly, daily)
        self.assertEqual(detail["entryHour"], 4)
        self.assertEqual(detail["regime"], "UP_SHOCK_CONTINUE")
        self.assertEqual(detail["direction"], "LONG")
        self.assertEqual(detail["riskMultiplier"], 0.5)
        self.assertTrue(target)
        self.assertTrue(all(weight > 0 for weight in target.values()))
        self.assertLessEqual(len(target), 3)
        self.assertEqual(detail["grossTarget"], detail["grossAfterCap"])

    def test_failed_positive_shock_stays_flat(self) -> None:
        date, signals, hourly, daily = synthetic_inputs(shock_z=2.0, rising=False)
        target, detail = target_for_day(date, signals, hourly, daily)
        self.assertEqual(detail["entryHour"], 4)
        self.assertEqual(detail["regime"], "UP_SHOCK_FAILED")
        self.assertEqual(detail["direction"], "FLAT")
        self.assertEqual(target, {})

    def test_confirmed_bull_never_opens_shorts(self) -> None:
        date, signals, hourly, daily = synthetic_inputs(shock_z=0.2, rising=True)
        target, detail = target_for_day(date, signals, hourly, daily)
        self.assertEqual(detail["entryHour"], 0)
        self.assertEqual(detail["regime"], "BULL_CONFIRMED")
        self.assertTrue(target)
        self.assertTrue(all(weight > 0 for weight in target.values()))


if __name__ == "__main__":
    unittest.main()
