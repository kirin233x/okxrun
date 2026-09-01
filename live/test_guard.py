from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from live.guard import AccountView, Guard, OrderIntent
from live.settings import Settings


UNIVERSE = {"BTC-USDT-SWAP", "ETH-USDT-SWAP"}


def settings(state_dir: Path, **overrides: object) -> Settings:
    base = dict(
        api_key="k",
        api_secret="s",
        passphrase="p",
        simulated=False,
        dry_run=False,
        leverage=1.0,
        variant_id="r9.2",
        max_gross_leverage=2.0,
        max_instrument_notional_usdt=200.0,
        max_order_notional_usdt=150.0,
        max_orders_per_hour=10,
        min_equity_usdt=50.0,
        daily_kill_loss=0.03,
        rebalance_minute_utc=2,
        fast_loop_seconds=60,
        state_dir=state_dir,
        halt_file=state_dir / "HALT",
        rest_base="https://example.invalid",
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def intent(**overrides: object) -> OrderIntent:
    base = dict(
        inst_id="BTC-USDT-SWAP",
        side="buy",
        contracts=Decimal("0.1"),
        notional_usdt=100.0,
        reduce_only=False,
        reason="REBALANCE_OPEN",
    )
    base.update(overrides)
    return OrderIntent(**base)  # type: ignore[arg-type]


def view(**overrides: object) -> AccountView:
    base = dict(
        equity_usdt=500.0,
        gross_notional_after_usdt=500.0,
        instrument_notional_after_usdt=100.0,
        orders_in_last_hour=0,
    )
    base.update(overrides)
    return AccountView(**base)  # type: ignore[arg-type]


class GuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.state = Path(self._temp.name)
        self.guard = Guard(settings(self.state), UNIVERSE)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_a_normal_order_is_allowed(self) -> None:
        self.assertTrue(self.guard.check(intent(), view()).allowed)

    def test_an_instrument_outside_the_universe_is_refused(self) -> None:
        verdict = self.guard.check(intent(inst_id="DOGE-USDT-SWAP"), view())
        self.assertFalse(verdict.allowed)
        self.assertIn("not in the configured universe", verdict.reason)

    def test_an_oversized_order_is_refused(self) -> None:
        verdict = self.guard.check(intent(notional_usdt=151.0), view())
        self.assertFalse(verdict.allowed)
        self.assertIn("exceeds cap", verdict.reason)

    def test_too_much_in_one_instrument_is_refused(self) -> None:
        verdict = self.guard.check(intent(), view(instrument_notional_after_usdt=201.0))
        self.assertFalse(verdict.allowed)
        self.assertIn("over cap", verdict.reason)

    def test_too_much_gross_leverage_is_refused(self) -> None:
        verdict = self.guard.check(intent(), view(gross_notional_after_usdt=1_001.0))
        self.assertFalse(verdict.allowed)
        self.assertIn("gross leverage", verdict.reason)

    def test_equity_below_the_floor_stops_new_risk(self) -> None:
        verdict = self.guard.check(intent(), view(equity_usdt=49.0))
        self.assertFalse(verdict.allowed)
        self.assertIn("below floor", verdict.reason)

    def test_the_hourly_order_cap_is_enforced(self) -> None:
        verdict = self.guard.check(intent(), view(orders_in_last_hour=10))
        self.assertFalse(verdict.allowed)
        self.assertIn("rate limit", verdict.reason)

    def test_a_halt_file_stops_new_risk(self) -> None:
        (self.state / "HALT").write_text("stop", encoding="utf-8")
        verdict = self.guard.check(intent(), view())
        self.assertFalse(verdict.allowed)
        self.assertIn("HALT", verdict.reason)


class ClosesAreNeverBlockedTests(unittest.TestCase):
    """A cap that could block an exit would trap the account in a bad position."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.state = Path(self._temp.name)
        self.guard = Guard(settings(self.state), UNIVERSE)
        self.closing = intent(reduce_only=True, reason="POSITION_STOP")

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_a_close_passes_the_notional_caps(self) -> None:
        self.assertTrue(
            self.guard.check(
                intent(reduce_only=True, notional_usdt=10_000.0),
                view(instrument_notional_after_usdt=10_000.0, gross_notional_after_usdt=10_000.0),
            ).allowed
        )

    def test_a_close_passes_while_halted(self) -> None:
        (self.state / "HALT").write_text("stop", encoding="utf-8")
        self.assertTrue(self.guard.check(self.closing, view()).allowed)

    def test_a_close_passes_below_the_equity_floor(self) -> None:
        self.assertTrue(self.guard.check(self.closing, view(equity_usdt=1.0)).allowed)

    def test_a_close_still_respects_the_rate_limit(self) -> None:
        # The rate limit is the one cap that also binds closes: a loop firing
        # exits in a tight cycle is a bug, and the kill switch is not a reason
        # to let it run unbounded.
        self.assertFalse(self.guard.check(self.closing, view(orders_in_last_hour=10)).allowed)

    def test_a_close_outside_the_universe_is_still_refused(self) -> None:
        self.assertFalse(self.guard.check(intent(inst_id="XXX", reduce_only=True), view()).allowed)


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.state = Path(self._temp.name)
        self.guard = Guard(settings(self.state), UNIVERSE)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_a_zero_size_order_is_refused(self) -> None:
        self.assertFalse(self.guard.check(intent(contracts=Decimal(0)), view()).allowed)

    def test_an_unknown_side_is_refused(self) -> None:
        self.assertFalse(self.guard.check(intent(side="short"), view()).allowed)


if __name__ == "__main__":
    unittest.main()
