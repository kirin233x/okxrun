from __future__ import annotations

import unittest
from decimal import Decimal

from live.instruments import (
    InstrumentSpec,
    UnsupportedInstrument,
    contracts_for_notional,
    format_size,
    min_order_notional,
    notional_for_contracts,
    parse_instrument,
    round_to_lot,
    tradable_universe,
)


def spec(
    inst_id: str = "BTC-USDT-SWAP",
    contract_value: str = "0.01",
    lot_size: str = "0.1",
    min_size: str = "0.1",
    contract_type: str = "linear",
    settle: str = "USDT",
    state: str = "live",
) -> InstrumentSpec:
    return InstrumentSpec(
        inst_id=inst_id,
        contract_value=Decimal(contract_value),
        contract_multiplier=Decimal("1"),
        lot_size=Decimal(lot_size),
        min_size=Decimal(min_size),
        tick_size=Decimal("0.1"),
        contract_type=contract_type,
        settle_currency=settle,
        state=state,
    )


class ParseTests(unittest.TestCase):
    def test_parses_the_fields_that_decide_order_size(self) -> None:
        parsed = parse_instrument(
            {
                "instId": "ETH-USDT-SWAP",
                "ctVal": "0.1",
                "ctMult": "1",
                "lotSz": "0.01",
                "minSz": "0.01",
                "tickSz": "0.01",
                "ctType": "linear",
                "settleCcy": "USDT",
                "state": "live",
            }
        )
        self.assertEqual(parsed.contract_value, Decimal("0.1"))
        self.assertEqual(parsed.min_size, Decimal("0.01"))
        self.assertTrue(parsed.tradable)

    def test_a_suspended_instrument_is_not_tradable(self) -> None:
        self.assertFalse(spec(state="suspend").tradable)


class SizingTests(unittest.TestCase):
    def test_notional_uses_contract_value_not_raw_contract_count(self) -> None:
        # 1 contract of 0.01 BTC at 100k is 1000 USDT, not 100k.
        self.assertAlmostEqual(notional_for_contracts(spec(), 100_000.0, Decimal("1")), 1_000.0)

    def test_size_is_floored_onto_the_lot_grid(self) -> None:
        # 250 USDT at 100k buys 0.0025 BTC = 0.25 contracts, floored to 0.2.
        self.assertEqual(contracts_for_notional(spec(), 100_000.0, 250.0), Decimal("0.2"))

    def test_flooring_never_exceeds_the_requested_notional(self) -> None:
        for notional in (83.0, 199.0, 250.0, 999.0):
            size = contracts_for_notional(spec(), 100_000.0, notional)
            self.assertLessEqual(notional_for_contracts(spec(), 100_000.0, size), notional)

    def test_a_position_below_the_minimum_order_is_refused_not_rounded_up(self) -> None:
        # 0.1 contracts of 0.01 BTC at 100k is a 100 USDT minimum; 83 cannot buy it.
        self.assertEqual(contracts_for_notional(spec(), 100_000.0, 83.0), Decimal(0))

    def test_min_order_notional_is_what_blocks_a_small_account(self) -> None:
        self.assertAlmostEqual(min_order_notional(spec(), 100_000.0), 100.0)

    def test_inverse_contracts_are_refused_rather_than_mis_sized(self) -> None:
        with self.assertRaises(UnsupportedInstrument):
            contracts_for_notional(spec(contract_type="inverse"), 100_000.0, 250.0)

    def test_non_usdt_settlement_is_refused(self) -> None:
        with self.assertRaises(UnsupportedInstrument):
            contracts_for_notional(spec(settle="BTC"), 100_000.0, 250.0)

    def test_zero_or_negative_price_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            contracts_for_notional(spec(), 0.0, 250.0)

    def test_round_to_lot_floors_and_ignores_sign(self) -> None:
        self.assertEqual(round_to_lot(spec(), Decimal("-0.37")), Decimal("0.3"))

    def test_format_size_avoids_scientific_notation(self) -> None:
        self.assertEqual(format_size(Decimal("0.0000001")), "0.0000001")
        self.assertEqual(format_size(Decimal("10")), "10")
        self.assertEqual(format_size(Decimal("-0.30")), "0.3")


class UniverseTests(unittest.TestCase):
    def test_instruments_whose_minimum_exceeds_the_target_are_rejected(self) -> None:
        specs = {
            "BTC-USDT-SWAP": spec("BTC-USDT-SWAP", contract_value="0.01"),
            "DOGE-USDT-SWAP": spec("DOGE-USDT-SWAP", contract_value="1000", lot_size="1", min_size="1"),
        }
        prices = {"BTC-USDT-SWAP": 100_000.0, "DOGE-USDT-SWAP": 0.2}
        usable, rejected = tradable_universe(specs, prices, target_notional_usdt=208.0)
        # BTC needs 100 USDT minimum and fits; DOGE needs 200 and also fits.
        self.assertEqual(usable, ["BTC-USDT-SWAP", "DOGE-USDT-SWAP"])
        self.assertEqual(rejected, {})

    def test_a_contract_too_large_for_the_account_is_reported_with_a_reason(self) -> None:
        specs = {"BTC-USDT-SWAP": spec(min_size="1", lot_size="1")}
        usable, rejected = tradable_universe(specs, {"BTC-USDT-SWAP": 100_000.0}, 208.0)
        self.assertEqual(usable, [])
        self.assertIn("min order 1000.00 USDT exceeds target 208.00 USDT", rejected["BTC-USDT-SWAP"])

    def test_missing_price_and_dead_instruments_are_rejected(self) -> None:
        specs = {"A-USDT-SWAP": spec("A-USDT-SWAP"), "B-USDT-SWAP": spec("B-USDT-SWAP", state="suspend")}
        usable, rejected = tradable_universe(specs, {"B-USDT-SWAP": 10.0}, 1_000.0)
        self.assertEqual(usable, [])
        self.assertEqual(rejected["A-USDT-SWAP"], "no price")
        self.assertIn("suspend", rejected["B-USDT-SWAP"])


if __name__ == "__main__":
    unittest.main()
