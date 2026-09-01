from __future__ import annotations

import unittest

import pandas as pd

from research.r9_momentum_backtest import (
    BASE_COST,
    Position,
    load_universe,
    simulate_day,
    stop_fill,
)
from strategy import (
    MAX_ASSET_WEIGHT,
    R91,
    R92,
    apply_rebalance_threshold,
    buffered_selection,
    capped_inverse_vol_weights,
    scale_target_weights,
)


class R9MomentumTests(unittest.TestCase):
    def test_r92_uses_the_backtested_low_turnover_settings(self) -> None:
        self.assertEqual(R91.rank_exit_buffer, 6)
        self.assertAlmostEqual(R91.min_rebalance_delta, 0.05)
        self.assertEqual(R92.rank_exit_buffer, 8)
        self.assertAlmostEqual(R92.min_rebalance_delta, 0.10)

    def test_public_universe_is_available_without_generated_reports(self) -> None:
        universe = load_universe()
        self.assertIn("BTC-USDT-SWAP", universe)
        self.assertEqual(len(universe), len(set(universe)))

    def test_inverse_vol_weights_respect_single_asset_cap(self) -> None:
        volatility = pd.Series({"A": 0.01, "B": 0.02, "C": 0.03})
        weights = capped_inverse_vol_weights(volatility, 0.75)
        self.assertLessEqual(max(weights.values()), 0.20)
        self.assertLessEqual(sum(weights.values()), 0.75)

    def test_long_fixed_stop_uses_gap_price_when_worse(self) -> None:
        position = Position("A", 1, 0.2, 100.0, 1.0, 0.04, 100.0, 100.0)
        bar = pd.Series({"open": 95.0, "high": 97.0, "low": 94.0, "close": 96.0})
        self.assertEqual(stop_fill(position, bar), 95.0)

    def test_short_trailing_stop_uses_confirmed_prior_trough(self) -> None:
        position = Position("A", -1, -0.2, 100.0, 1.0, 0.06, 100.0, 94.0)
        bar = pd.Series({"open": 96.0, "high": 97.0, "low": 93.0, "close": 95.0})
        self.assertAlmostEqual(stop_fill(position, bar) or 0.0, 96.35)

    def test_rank_buffer_retains_positions_until_they_leave_top_or_bottom_six(self) -> None:
        ranked = pd.Series({f"A{rank}": float(rank) for rank in range(1, 11)}).sort_values()
        previous = {"A6": -0.1, "A5": -0.1, "A4": -0.1, "A7": 0.1, "A8": 0.1, "A9": 0.1}
        longs, shorts = buffered_selection(ranked, previous)
        self.assertEqual(longs, ["A9", "A8", "A7"])
        self.assertEqual(shorts, ["A4", "A5", "A6"])

    def test_rank_buffer_replaces_a_position_outside_buffer(self) -> None:
        ranked = pd.Series({f"A{rank}": float(rank) for rank in range(1, 11)}).sort_values()
        previous = {"A1": 0.1, "A8": 0.1, "A9": 0.1}
        longs, _ = buffered_selection(ranked, previous)
        self.assertEqual(longs, ["A9", "A8", "A10"])

    def test_r92_rank_buffer_retains_positions_through_rank_eight(self) -> None:
        ranked = pd.Series({f"A{rank}": float(rank) for rank in range(1, 13)}).sort_values()
        previous = {"A5": 0.1, "A8": -0.1}
        longs, shorts = buffered_selection(ranked, previous, exit_buffer=8)
        self.assertIn("A5", longs)
        self.assertIn("A8", shorts)

    def test_small_same_side_rebalance_is_ignored_but_exits_are_not(self) -> None:
        target = {"A": 0.18, "C": -0.20}
        previous = {"A": 0.20, "B": 0.20, "C": 0.20}
        adjusted = apply_rebalance_threshold(target, previous, 0.05)
        self.assertEqual(adjusted["A"], MAX_ASSET_WEIGHT)
        self.assertEqual(adjusted["C"], -0.20)
        self.assertNotIn("B", adjusted)

    def test_flat_day_charges_exit_cost_for_previous_holdings(self) -> None:
        result, detail = simulate_day(
            pd.Timestamp("2026-01-01", tz="UTC"),
            10_000.0,
            {},
            {"A": 0.20},
            {},
            pd.DataFrame(),
            {},
            BASE_COST,
        )
        self.assertAlmostEqual(result["cost"], 0.20 * BASE_COST)
        self.assertAlmostEqual(result["net"], -0.20 * BASE_COST)
        self.assertEqual(detail["endingWeights"], {})

    def test_three_x_scales_notional_without_changing_selection(self) -> None:
        target = {"A": 0.20, "B": -0.15, "C": 0.10}
        scaled = scale_target_weights(target, 3.0)
        self.assertEqual(set(scaled), set(target))
        self.assertAlmostEqual(scaled["A"], 0.60)
        self.assertAlmostEqual(scaled["B"], -0.45)
        self.assertAlmostEqual(sum(abs(weight) for weight in scaled.values()), 1.35)

    def test_invalid_leverage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scale_target_weights({"A": 0.20}, 0.0)


if __name__ == "__main__":
    unittest.main()
