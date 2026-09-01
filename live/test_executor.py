from __future__ import annotations

import unittest
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd

from live.executor import Account, client_order_id, position_snapshots, trading_day
from live.marketdata import append_pending_day
from live.plan import PositionSnapshot
from live.test_instruments import spec
from strategy import build_daily_inputs, compute_signals


BTC = "BTC-USDT-SWAP"


class PositionSnapshotTests(unittest.TestCase):
    def test_net_mode_positions_keep_their_sign(self) -> None:
        rows = [{"instId": BTC, "pos": "-0.5", "posSide": "net", "markPx": "100000"}]
        snapshots = position_snapshots(rows, {})
        self.assertEqual(snapshots[BTC].contracts, Decimal("-0.5"))

    def test_long_short_mode_sides_are_normalised_to_a_signed_position(self) -> None:
        rows = [
            {"instId": BTC, "pos": "0.5", "posSide": "short", "markPx": "100000"},
            {"instId": "ETH-USDT-SWAP", "pos": "0.4", "posSide": "long", "markPx": "4000"},
        ]
        snapshots = position_snapshots(rows, {})
        self.assertEqual(snapshots[BTC].contracts, Decimal("-0.5"))
        self.assertEqual(snapshots["ETH-USDT-SWAP"].contracts, Decimal("0.4"))

    def test_flat_and_empty_rows_are_dropped(self) -> None:
        rows = [{"instId": BTC, "pos": "0", "posSide": "net"}, {"instId": "X", "pos": "", "posSide": "net"}]
        self.assertEqual(position_snapshots(rows, {}), {})

    def test_a_missing_mark_falls_back_to_the_ticker_price(self) -> None:
        rows = [{"instId": BTC, "pos": "0.5", "posSide": "net"}]
        self.assertAlmostEqual(position_snapshots(rows, {BTC: 99_000.0})[BTC].mark_price, 99_000.0)


class AccountTests(unittest.TestCase):
    def test_weights_are_signed_notional_over_equity(self) -> None:
        account = Account(
            equity_usdt=10_000.0,
            positions={BTC: PositionSnapshot(BTC, Decimal("-0.5"), 100_000.0)},
        )
        # 0.5 contracts * 0.01 BTC * 100k = 500 USDT short => -0.05.
        self.assertAlmostEqual(account.weights({BTC: spec(BTC)})[BTC], -0.05)

    def test_gross_notional_adds_both_sides(self) -> None:
        account = Account(
            equity_usdt=10_000.0,
            positions={
                BTC: PositionSnapshot(BTC, Decimal("-0.5"), 100_000.0),
                "ETH-USDT-SWAP": PositionSnapshot("ETH-USDT-SWAP", Decimal("1"), 4_000.0),
            },
        )
        specs = {BTC: spec(BTC), "ETH-USDT-SWAP": spec("ETH-USDT-SWAP", contract_value="0.1")}
        self.assertAlmostEqual(account.gross_notional(specs), 500.0 + 400.0)

    def test_zero_equity_yields_no_weights_rather_than_dividing_by_zero(self) -> None:
        account = Account(0.0, {BTC: PositionSnapshot(BTC, Decimal("1"), 100_000.0)})
        self.assertEqual(account.weights({BTC: spec(BTC)}), {})


class ClientOrderIdTests(unittest.TestCase):
    def test_the_same_order_yields_the_same_id_so_a_retry_is_idempotent(self) -> None:
        first = client_order_id("2026-09-01", BTC, "REBALANCE_OPEN", 0)
        again = client_order_id("2026-09-01", BTC, "REBALANCE_OPEN", 0)
        self.assertEqual(first, again)

    def test_different_days_instruments_and_sequences_differ(self) -> None:
        ids = {
            client_order_id("2026-09-01", BTC, "REBALANCE_OPEN", 0),
            client_order_id("2026-09-02", BTC, "REBALANCE_OPEN", 0),
            client_order_id("2026-09-01", "ETH-USDT-SWAP", "REBALANCE_OPEN", 0),
            client_order_id("2026-09-01", BTC, "POSITION_STOP", 0),
            client_order_id("2026-09-01", BTC, "REBALANCE_OPEN", 1),
        }
        self.assertEqual(len(ids), 5)

    def test_the_id_fits_okx_limits(self) -> None:
        generated = client_order_id("2026-09-01", BTC, "REBALANCE_OPEN", 0)
        self.assertTrue(generated.isalnum())
        self.assertLessEqual(len(generated), 32)


class TradingDayTests(unittest.TestCase):
    def test_the_day_is_the_utc_date(self) -> None:
        self.assertEqual(trading_day(datetime(2026, 9, 1, 23, 59, tzinfo=timezone.utc)), "2026-09-01")

    def test_a_non_utc_clock_is_converted_not_truncated(self) -> None:
        from datetime import timedelta

        tokyo = timezone(timedelta(hours=9))
        # 2026-09-02 08:00 in Tokyo is still 2026-09-01 in UTC.
        self.assertEqual(trading_day(datetime(2026, 9, 2, 8, 0, tzinfo=tokyo)), "2026-09-01")


class PendingDayAlignmentTests(unittest.TestCase):
    """The executor must score day D from bars ending D-1, like the backtest."""

    def _frames(self, days: int) -> dict[str, pd.DataFrame]:
        index = pd.date_range("2026-01-01", periods=days * 24, freq="1h", tz="UTC")
        rng = np.random.default_rng(7)
        frames = {}
        for slot, inst_id in enumerate([BTC, "ETH-USDT-SWAP", "SOL-USDT-SWAP"]):
            close = 100.0 * np.exp(np.cumsum(rng.normal(1e-5 * slot, 0.004, len(index))))
            frames[inst_id] = pd.DataFrame(
                {
                    "open": close,
                    "high": close * 1.001,
                    "low": close * 0.999,
                    "close": close,
                    "volumeQuote": 1e6,
                },
                index=index,
            )
        return frames

    def test_the_pending_day_is_added_once_and_left_empty(self) -> None:
        daily, _ = build_daily_inputs(self._frames(40))
        today = daily[BTC].index.max() + pd.Timedelta(days=1)
        extended, closes = append_pending_day(daily, today)
        self.assertIn(today, closes.index)
        self.assertTrue(pd.isna(closes.loc[today, BTC]))
        self.assertEqual(len(extended[BTC]), len(daily[BTC]) + 1)

    def test_appending_twice_does_not_duplicate_the_row(self) -> None:
        daily, _ = build_daily_inputs(self._frames(40))
        today = daily[BTC].index.max() + pd.Timedelta(days=1)
        once, _ = append_pending_day(daily, today)
        twice, closes = append_pending_day(once, today)
        self.assertEqual(len(twice[BTC]), len(daily[BTC]) + 1)
        self.assertEqual(list(closes.index).count(today), 1)

    def test_signals_for_the_pending_day_read_the_last_completed_day(self) -> None:
        daily, _ = build_daily_inputs(self._frames(40))
        last_complete = daily[BTC].index.max()
        today = last_complete + pd.Timedelta(days=1)
        extended, closes = append_pending_day(daily, today)
        signals = compute_signals(closes, extended)
        # btcPrior at the pending day is the close of the last completed day:
        # scoring D from D-1 exactly as the backtest does.
        self.assertAlmostEqual(
            float(signals["btcPrior"].loc[today]), float(daily[BTC].loc[last_complete, "close"])
        )
        self.assertFalse(np.isnan(float(signals["score"].loc[today, BTC])))


if __name__ == "__main__":
    unittest.main()
