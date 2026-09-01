from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from live.store import PositionState, Store


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self._temp.name) / "test.sqlite3")

    def tearDown(self) -> None:
        self.store.close()
        self._temp.cleanup()

    def test_events_come_back_newest_first(self) -> None:
        self.store.log("A", {"n": 1})
        self.store.log("B", {"n": 2})
        events = self.store.recent_events(10)
        self.assertEqual([event["kind"] for event in events], ["B", "A"])
        self.assertEqual(events[0]["payload"], {"n": 2})

    def test_events_can_be_filtered_by_kind(self) -> None:
        self.store.log("PREFLIGHT", {"n": 1})
        self.store.log("SIGNAL", {"n": 2})
        self.assertEqual(len(self.store.recent_events(10, kind="PREFLIGHT")), 1)

    def test_an_order_id_is_remembered_so_a_retry_can_detect_it(self) -> None:
        self.store.record_order("abc", "BTC-USDT-SWAP", "buy", Decimal("0.1"), 100.0, False, "OPEN")
        self.assertTrue(self.store.has_order("abc"))
        self.assertFalse(self.store.has_order("def"))

    def test_rejected_orders_do_not_consume_the_rate_limit(self) -> None:
        self.store.record_order("a", "BTC-USDT-SWAP", "buy", Decimal("0.1"), 100.0, False, "OPEN")
        self.store.record_order(
            "b", "BTC-USDT-SWAP", "buy", Decimal("0.1"), 100.0, False, "OPEN", status="REJECTED"
        )
        self.assertEqual(self.store.orders_in_last_hour(), 1)

    def test_old_orders_fall_out_of_the_rate_limit_window(self) -> None:
        self.store.record_order("a", "BTC-USDT-SWAP", "buy", Decimal("0.1"), 100.0, False, "OPEN")
        future = datetime.now(timezone.utc) + timedelta(hours=2)
        self.assertEqual(self.store.orders_in_last_hour(now=future), 0)

    def test_finishing_an_order_keeps_its_response(self) -> None:
        self.store.record_order("a", "BTC-USDT-SWAP", "buy", Decimal("0.1"), 100.0, False, "OPEN")
        self.store.finish_order("a", "SENT", {"ordId": "42"})
        order = self.store.recent_orders(1)[0]
        self.assertEqual(order["status"], "SENT")
        self.assertIn("42", order["response"])

    def test_position_state_round_trips(self) -> None:
        state = PositionState("BTC-USDT-SWAP", "2026-09-01", 1, 100.0, 0.04, 105.0, 99.0)
        self.store.save_position_state(state)
        loaded = self.store.position_state("BTC-USDT-SWAP")
        assert loaded is not None
        self.assertEqual(loaded.direction, 1)
        self.assertAlmostEqual(loaded.peak, 105.0)
        self.store.drop_position_state("BTC-USDT-SWAP")
        self.assertIsNone(self.store.position_state("BTC-USDT-SWAP"))


class TradingDayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self._temp.name) / "test.sqlite3")

    def tearDown(self) -> None:
        self.store.close()
        self._temp.cleanup()

    def test_the_opening_equity_anchor_survives_a_restart_mid_drawdown(self) -> None:
        # This is what makes the 3% kill switch mean anything: a restart after
        # losing 2% must not re-anchor and hand the day a fresh loss budget.
        self.assertAlmostEqual(self.store.open_trading_day("2026-09-01", 500.0), 500.0)
        self.assertAlmostEqual(self.store.open_trading_day("2026-09-01", 490.0), 500.0)

    def test_a_new_day_takes_a_new_anchor(self) -> None:
        self.store.open_trading_day("2026-09-01", 500.0)
        self.assertAlmostEqual(self.store.open_trading_day("2026-09-02", 480.0), 480.0)

    def test_a_killed_day_stays_killed(self) -> None:
        self.store.open_trading_day("2026-09-01", 500.0)
        self.assertFalse(self.store.day_killed("2026-09-01"))
        self.store.mark_day_killed("2026-09-01")
        self.assertTrue(self.store.day_killed("2026-09-01"))
        self.assertFalse(self.store.day_killed("2026-09-02"))


if __name__ == "__main__":
    unittest.main()
