from __future__ import annotations

import unittest
from decimal import Decimal

from live.plan import PositionSnapshot, close_all_orders, plan_orders, target_contracts
from live.test_instruments import spec


PRICE = 100_000.0
BTC = "BTC-USDT-SWAP"
# ctVal 0.01 at 100k => 1000 USDT per contract, lot 0.1 => 100 USDT per lot.
SPECS = {BTC: spec(BTC)}
PRICES = {BTC: PRICE}


def held(contracts: str) -> dict[str, PositionSnapshot]:
    return {BTC: PositionSnapshot(BTC, Decimal(contracts), PRICE)}


class TargetTests(unittest.TestCase):
    def test_a_long_weight_becomes_a_positive_contract_target(self) -> None:
        # 10_000 equity at weight 0.05 => 500 USDT => 0.5 contracts.
        self.assertEqual(target_contracts(SPECS[BTC], PRICE, 10_000.0, 0.05), Decimal("0.5"))

    def test_a_short_weight_becomes_a_negative_contract_target(self) -> None:
        self.assertEqual(target_contracts(SPECS[BTC], PRICE, 10_000.0, -0.05), Decimal("-0.5"))

    def test_a_weight_too_small_to_trade_becomes_no_position(self) -> None:
        # 500 equity at 0.1 => 50 USDT, under the 100 USDT minimum order.
        self.assertEqual(target_contracts(SPECS[BTC], PRICE, 500.0, 0.1), Decimal(0))


class PlanTests(unittest.TestCase):
    def test_opening_from_flat_buys_the_target(self) -> None:
        orders, skips = plan_orders({BTC: 0.05}, {}, SPECS, PRICES, 10_000.0)
        self.assertEqual(skips, [])
        self.assertEqual(len(orders), 1)
        self.assertEqual((orders[0].side, orders[0].contracts), ("buy", Decimal("0.5")))
        self.assertFalse(orders[0].reduce_only)
        self.assertAlmostEqual(orders[0].notional_usdt, 500.0)

    def test_only_the_difference_is_traded_not_the_whole_position(self) -> None:
        orders, _ = plan_orders({BTC: 0.05}, held("0.3"), SPECS, PRICES, 10_000.0)
        self.assertEqual((orders[0].side, orders[0].contracts), ("buy", Decimal("0.2")))

    def test_shrinking_a_position_is_marked_reduce_only(self) -> None:
        orders, _ = plan_orders({BTC: 0.02}, held("0.5"), SPECS, PRICES, 10_000.0)
        self.assertEqual((orders[0].side, orders[0].contracts), ("sell", Decimal("0.3")))
        self.assertTrue(orders[0].reduce_only)

    def test_dropping_out_of_the_book_closes_the_position(self) -> None:
        orders, _ = plan_orders({}, held("0.5"), SPECS, PRICES, 10_000.0)
        self.assertEqual((orders[0].side, orders[0].contracts), ("sell", Decimal("0.5")))
        self.assertTrue(orders[0].reduce_only)

    def test_a_side_flip_closes_first_then_opens(self) -> None:
        orders, _ = plan_orders({BTC: -0.05}, held("0.5"), SPECS, PRICES, 10_000.0)
        self.assertEqual([order.reason for order in orders], ["FLIP_CLOSE", "FLIP_OPEN"])
        self.assertEqual((orders[0].side, orders[0].contracts), ("sell", Decimal("0.5")))
        self.assertTrue(orders[0].reduce_only)
        self.assertEqual((orders[1].side, orders[1].contracts), ("sell", Decimal("0.5")))
        self.assertFalse(orders[1].reduce_only)

    def test_an_unchanged_position_produces_no_order(self) -> None:
        orders, skips = plan_orders({BTC: 0.05}, held("0.5"), SPECS, PRICES, 10_000.0)
        self.assertEqual(orders, [])
        self.assertEqual(skips, [])

    def test_a_delta_below_the_minimum_order_is_skipped_not_rounded_up(self) -> None:
        # Target 0.55 floors to 0.5, which equals the holding, so nothing moves.
        orders, _ = plan_orders({BTC: 0.055}, held("0.5"), SPECS, PRICES, 10_000.0)
        self.assertEqual(orders, [])

    def test_a_holding_with_no_spec_is_reported_rather_than_silently_ignored(self) -> None:
        orders, skips = plan_orders({}, held("0.5"), {}, PRICES, 10_000.0)
        self.assertEqual(orders, [])
        self.assertEqual(len(skips), 1)
        self.assertIn("cannot size an exit", skips[0].reason)

    def test_an_exit_too_small_to_place_is_flagged_as_an_exit(self) -> None:
        tiny = {BTC: PositionSnapshot(BTC, Decimal("0.05"), PRICE)}
        orders, skips = plan_orders({}, tiny, SPECS, PRICES, 10_000.0)
        self.assertEqual(orders, [])
        self.assertIn("exit of 0.05 contracts", skips[0].reason)

    def test_no_price_is_skipped_rather_than_sized_at_zero(self) -> None:
        orders, skips = plan_orders({BTC: 0.05}, {}, SPECS, {}, 10_000.0)
        self.assertEqual(orders, [])
        self.assertEqual(skips[0].reason, "no usable price")

    def test_a_small_account_produces_no_orders_at_all(self) -> None:
        # 500 USDT over six positions is ~83 each, below BTC's 100 USDT minimum.
        orders, skips = plan_orders({BTC: 1 / 6}, {}, SPECS, PRICES, 500.0)
        self.assertEqual(orders, [])
        self.assertEqual(skips, [])


class CloseAllTests(unittest.TestCase):
    def test_every_position_is_flattened_reduce_only(self) -> None:
        positions = {
            BTC: PositionSnapshot(BTC, Decimal("0.5"), PRICE),
            "ETH-USDT-SWAP": PositionSnapshot("ETH-USDT-SWAP", Decimal("-0.4"), 4_000.0),
        }
        specs = {BTC: spec(BTC), "ETH-USDT-SWAP": spec("ETH-USDT-SWAP", contract_value="0.1")}
        orders = close_all_orders(positions, specs, "HALT")
        self.assertEqual([order.side for order in orders], ["sell", "buy"])
        self.assertTrue(all(order.reduce_only for order in orders))
        self.assertTrue(all(order.reason == "HALT" for order in orders))


if __name__ == "__main__":
    unittest.main()
