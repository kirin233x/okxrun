"""The trading loop.

Two cadences:

* a daily rebalance just after 00:00 UTC, which scores the universe exactly as
  the backtest does and moves the book to the resulting target;
* a fast loop every minute, which carries the ATR stop, the trailing stop and
  the intraday kill switch.

The exchange is always the source of truth for what is held. Local state only
remembers the day's entry anchor and high-water mark, so a restart re-reads
positions rather than trusting anything it wrote earlier.

Known divergence from the backtest, by construction: the backtest evaluates
stops against each hourly bar's high and low, while the executor sees the last
price once a minute. Fills will differ from the simulation on fast moves.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from strategy import R9, R91, R92, StrategyVariant, compute_signals, stop_fraction_from_atr, stop_price, target_for_day

from .guard import AccountView, Guard, OrderIntent
from .instruments import InstrumentSpec, parse_instrument, tradable_universe
from .marketdata import append_pending_day, instrument_specs, load_inputs, mark_prices
from .okx import OKXClient, OKXError
from .plan import PlanSkip, PlannedOrder, PositionSnapshot, close_all_orders, plan_orders
from .settings import ROOT, Settings
from .store import PositionState, Store


UNIVERSE_FILE = ROOT / "research" / "universe.json"
VARIANTS: dict[str, StrategyVariant] = {R9.id: R9, R91.id: R91, R92.id: R92}
ASSETS_PER_SIDE = 3
SIDES = 2


def load_universe() -> list[str]:
    payload = json.loads(Path(UNIVERSE_FILE).read_text(encoding="utf-8"))
    return list(payload["universe"])


def trading_day(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y-%m-%d")


def client_order_id(day: str, inst_id: str, reason: str, sequence: int) -> str:
    """Deterministic, so a retried send reuses the id instead of double-filling."""
    digest = hashlib.sha1(f"{day}|{inst_id}|{reason}|{sequence}".encode("utf-8")).hexdigest()
    return f"okxrun{digest[:20]}"


def position_snapshots(rows: list[dict[str, Any]], prices: dict[str, float]) -> dict[str, PositionSnapshot]:
    """Signed positions from the account, keyed by instrument."""
    snapshots: dict[str, PositionSnapshot] = {}
    for row in rows:
        inst_id = str(row.get("instId"))
        raw = str(row.get("pos") or "0")
        if raw in {"", "0"}:
            continue
        contracts = Decimal(raw)
        side = str(row.get("posSide") or "net")
        if side == "long":
            contracts = abs(contracts)
        elif side == "short":
            contracts = -abs(contracts)
        mark = row.get("markPx") or row.get("last")
        price = float(mark) if mark else prices.get(inst_id, 0.0)
        if contracts == 0:
            continue
        snapshots[inst_id] = PositionSnapshot(inst_id=inst_id, contracts=contracts, mark_price=price)
    return snapshots


def describe_order(order: PlannedOrder) -> dict[str, Any]:
    return {
        "instId": order.inst_id,
        "side": order.side,
        "contracts": str(order.contracts),
        "notionalUsdt": round(order.notional_usdt, 2),
        "reduceOnly": order.reduce_only,
        "reason": order.reason,
    }


@dataclass
class Account:
    equity_usdt: float
    positions: dict[str, PositionSnapshot]

    def gross_notional(self, specs: dict[str, InstrumentSpec]) -> float:
        from .instruments import notional_for_contracts

        total = 0.0
        for inst_id, position in self.positions.items():
            spec = specs.get(inst_id)
            if spec is None or position.mark_price <= 0:
                continue
            total += notional_for_contracts(spec, position.mark_price, position.contracts)
        return total

    def weights(self, specs: dict[str, InstrumentSpec]) -> dict[str, float]:
        from .instruments import notional_for_contracts

        if self.equity_usdt <= 0:
            return {}
        weights: dict[str, float] = {}
        for inst_id, position in self.positions.items():
            spec = specs.get(inst_id)
            if spec is None or position.mark_price <= 0:
                continue
            notional = notional_for_contracts(spec, position.mark_price, position.contracts)
            sign = 1.0 if position.contracts > 0 else -1.0
            weights[inst_id] = sign * notional / self.equity_usdt
        return weights


@dataclass
class RebalancePlan:
    day: str
    today: pd.Timestamp
    account: Account
    signals: dict[str, Any]
    detail: dict[str, Any]
    target: dict[str, float]
    previous_weights: dict[str, float]
    prices: dict[str, float]
    orders: list[PlannedOrder]
    skips: list[PlanSkip]

    def describe(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "equityUsdt": self.account.equity_usdt,
            "signal": self.detail,
            "targetWeights": self.target,
            "currentWeights": self.previous_weights,
            "orders": [describe_order(order) for order in self.orders],
            "skipped": [{"instId": skip.inst_id, "reason": skip.reason} for skip in self.skips],
        }


class Executor:
    def __init__(self, settings: Settings, client: OKXClient, store: Store) -> None:
        self._settings = settings
        self._client = client
        self._store = store
        self._universe = load_universe()
        self._guard = Guard(settings, set(self._universe))
        self._specs: dict[str, InstrumentSpec] = {}
        self._variant = VARIANTS.get(settings.variant_id, R92)
        self._last_rebalanced_day: str | None = None

    # -------------------------------------------------------------- preflight

    def preflight(self) -> None:
        """Fail fast on the account settings that would mis-size every order."""
        config = self._client.account_config()
        position_mode = str(config.get("posMode"))
        if position_mode != "net_mode":
            raise RuntimeError(
                f"account is in {position_mode!r}; this executor assumes net_mode "
                "(one signed position per instrument). Change it in the OKX UI before trading."
            )
        self._specs = {
            inst_id: parse_instrument(row)
            for inst_id, row in instrument_specs(self._client, set(self._universe)).items()
        }
        missing = sorted(set(self._universe) - set(self._specs))
        equity = self.equity()
        prices = mark_prices(self._client, set(self._universe))
        target_notional = equity * self._settings.leverage / (ASSETS_PER_SIDE * SIDES)
        usable, rejected = tradable_universe(self._specs, prices, target_notional)
        self._store.log(
            "PREFLIGHT",
            {
                "posMode": position_mode,
                "equityUsdt": equity,
                "leverage": self._settings.leverage,
                "variant": self._variant.id,
                "dryRun": self._settings.dry_run,
                "simulated": self._settings.simulated,
                "perPositionNotionalUsdt": target_notional,
                "usableUniverse": usable,
                "rejected": rejected,
                "missingSpecs": missing,
            },
        )
        if len(usable) < ASSETS_PER_SIDE * SIDES:
            raise RuntimeError(
                f"only {len(usable)} of {len(self._universe)} instruments can be traded at "
                f"{target_notional:.2f} USDT per position; need at least {ASSETS_PER_SIDE * SIDES}. "
                f"Rejected: {rejected}"
            )

    # ---------------------------------------------------------------- account

    def equity(self) -> float:
        balance = self._client.balance()
        total = balance.get("totalEq")
        if total in (None, ""):
            raise RuntimeError("balance response carried no totalEq")
        return float(total)

    def account(self) -> Account:
        prices = mark_prices(self._client, set(self._universe))
        positions = position_snapshots(self._client.positions("SWAP"), prices)
        return Account(equity_usdt=self.equity(), positions=positions)

    # --------------------------------------------------------------- ordering

    def _submit(self, order: PlannedOrder, account: Account, day: str, sequence: int) -> bool:
        """Guard, log, and send one order. Returns whether it was sent."""
        from .instruments import notional_for_contracts

        spec = self._specs[order.inst_id]
        signed = order.contracts if order.side == "buy" else -order.contracts
        held = account.positions.get(order.inst_id)
        current = held.contracts if held else Decimal(0)
        price = held.mark_price if held and held.mark_price > 0 else 0.0
        after = current + signed
        instrument_after = notional_for_contracts(spec, price, after) if price > 0 else order.notional_usdt
        gross_after = account.gross_notional(self._specs) + (
            order.notional_usdt if not order.reduce_only else -order.notional_usdt
        )

        intent = OrderIntent(
            inst_id=order.inst_id,
            side=order.side,
            contracts=order.contracts,
            notional_usdt=order.notional_usdt,
            reduce_only=order.reduce_only,
            reason=order.reason,
        )
        view = AccountView(
            equity_usdt=account.equity_usdt,
            gross_notional_after_usdt=max(gross_after, 0.0),
            instrument_notional_after_usdt=abs(instrument_after),
            orders_in_last_hour=self._store.orders_in_last_hour(),
        )
        verdict = self._guard.check(intent, view)
        order_id = client_order_id(day, order.inst_id, order.reason, sequence)

        if not verdict.allowed:
            self._store.record_order(
                order_id,
                order.inst_id,
                order.side,
                order.contracts,
                order.notional_usdt,
                order.reduce_only,
                order.reason,
                status="REJECTED",
            )
            self._store.log(
                "ORDER_REJECTED",
                {"clOrdId": order_id, "instId": order.inst_id, "side": order.side,
                 "contracts": str(order.contracts), "notionalUsdt": order.notional_usdt,
                 "reason": order.reason, "guard": verdict.reason},
            )
            return False

        if self._settings.dry_run:
            self._store.record_order(
                order_id, order.inst_id, order.side, order.contracts,
                order.notional_usdt, order.reduce_only, order.reason, status="DRY_RUN",
            )
            self._store.log(
                "ORDER_DRY_RUN",
                {"clOrdId": order_id, "instId": order.inst_id, "side": order.side,
                 "contracts": str(order.contracts), "notionalUsdt": order.notional_usdt,
                 "reduceOnly": order.reduce_only, "reason": order.reason},
            )
            return False

        if self._store.has_order(order_id):
            existing = self._client.order_by_client_id(order.inst_id, order_id)
            if existing is not None:
                self._store.log("ORDER_ALREADY_PLACED", {"clOrdId": order_id, "state": existing.get("state")})
                return False

        from .instruments import format_size

        self._store.record_order(
            order_id, order.inst_id, order.side, order.contracts,
            order.notional_usdt, order.reduce_only, order.reason, status="PENDING",
        )
        try:
            response = self._client.place_order(
                inst_id=order.inst_id,
                side=order.side,
                size=format_size(order.contracts),
                client_order_id=order_id,
                reduce_only=order.reduce_only,
            )
        except OKXError as error:
            self._store.finish_order(order_id, "FAILED", {"code": error.code, "msg": error.message})
            self._store.log(
                "ORDER_FAILED",
                {"clOrdId": order_id, "instId": order.inst_id, "code": error.code, "msg": error.message},
            )
            return False
        self._store.finish_order(order_id, "SENT", response)
        self._store.log(
            "ORDER_SENT",
            {"clOrdId": order_id, "instId": order.inst_id, "side": order.side,
             "contracts": str(order.contracts), "notionalUsdt": order.notional_usdt,
             "reduceOnly": order.reduce_only, "reason": order.reason, "ordId": response.get("ordId")},
        )
        return True

    def _execute(self, orders: list[PlannedOrder], account: Account, day: str) -> int:
        sent = 0
        for sequence, order in enumerate(orders):
            if self._submit(order, account, day, sequence):
                sent += 1
        return sent

    # -------------------------------------------------------------- rebalance

    def build_plan(self, now: datetime) -> RebalancePlan:
        """Everything the rebalance would do, computed but not sent.

        ``plan`` on the CLI and ``rebalance`` in the loop both go through here,
        so what you inspect is exactly what would be traded.
        """
        day = trading_day(now)
        today = pd.Timestamp(day, tz="UTC")
        account = self.account()

        _, daily, _ = load_inputs(self._client, self._universe)
        daily, closes = append_pending_day(daily, today)
        signals = compute_signals(closes, daily)

        previous_weights = account.weights(self._specs)
        target, detail = target_for_day(
            today, signals, previous_weights, self._variant, self._settings.leverage
        )
        prices = mark_prices(self._client, set(self._universe))
        if detail.get("tradable", False):
            orders, skips = plan_orders(target, account.positions, self._specs, prices, account.equity_usdt)
        else:
            orders, skips = [], []
        return RebalancePlan(
            day=day,
            today=today,
            account=account,
            signals=signals,
            detail=detail,
            target=target,
            previous_weights=previous_weights,
            prices=prices,
            orders=orders,
            skips=skips,
        )

    def rebalance(self, now: datetime) -> None:
        day = trading_day(now)
        if self._store.day_killed(day):
            self._store.log("REBALANCE_SKIPPED", {"day": day, "why": "kill switch already fired today"})
            self._last_rebalanced_day = day
            return

        plan = self.build_plan(now)
        opening_equity = self._store.open_trading_day(day, plan.account.equity_usdt)
        self._store.log(
            "SIGNAL",
            {"day": day, "detail": plan.detail, "target": plan.target,
             "previousWeights": plan.previous_weights,
             "equityUsdt": plan.account.equity_usdt, "openingEquityUsdt": opening_equity},
        )
        if not plan.detail.get("tradable", False):
            self._last_rebalanced_day = day
            return

        if plan.skips:
            self._store.log("PLAN_SKIPS", {"day": day, "skips": [(s.inst_id, s.reason) for s in plan.skips]})
        self._store.log("PLAN", {"day": day, "orders": [describe_order(o) for o in plan.orders]})
        self._execute(plan.orders, plan.account, day)
        self._reset_stop_anchors(day, plan.target, plan.prices, plan.signals, plan.today)
        self._last_rebalanced_day = day

    def _reset_stop_anchors(
        self,
        day: str,
        target: dict[str, float],
        prices: dict[str, float],
        signals: dict[str, Any],
        today: pd.Timestamp,
    ) -> None:
        """Re-anchor every stop to today's opening price.

        This mirrors the backtest, where each simulated day re-enters at the
        day's first bar and resets the peak/trough the trailing stop rides. The
        stop is therefore an intraday stop that resets at 00:00 UTC, not a stop
        measured from the original entry.
        """
        atr = signals["atr"]
        held = set(target) | set(self._store.all_position_state())
        for inst_id in sorted(held):
            weight = target.get(inst_id, 0.0)
            price = prices.get(inst_id, 0.0)
            if weight == 0 or price <= 0:
                self._store.drop_position_state(inst_id)
                continue
            try:
                atr_value = float(atr.loc[today, inst_id])
            except (KeyError, TypeError, ValueError):
                atr_value = float("nan")
            fraction = stop_fraction_from_atr(atr_value)
            if not math.isfinite(fraction):
                self._store.drop_position_state(inst_id)
                continue
            self._store.save_position_state(
                PositionState(
                    inst_id=inst_id,
                    trading_day=day,
                    direction=1 if weight > 0 else -1,
                    entry=price,
                    stop_fraction=fraction,
                    peak=price,
                    trough=price,
                )
            )

    # -------------------------------------------------------------- fast loop

    def fast_tick(self, now: datetime) -> None:
        day = trading_day(now)
        account = self.account()

        if self._guard.halted():
            if account.positions:
                self._store.log("HALT", {"day": day, "positions": sorted(account.positions)})
                self._execute(close_all_orders(account.positions, self._specs, "HALT"), account, day)
            return

        opening_equity = self._store.open_trading_day(day, account.equity_usdt)
        self._store.mark_equity(day, account.equity_usdt, account.gross_notional(self._specs))
        loss = 1.0 - (account.equity_usdt / opening_equity if opening_equity > 0 else 1.0)
        if loss >= self._settings.daily_kill_loss and account.positions:
            self._store.log(
                "DAILY_KILL",
                {"day": day, "openingEquityUsdt": opening_equity,
                 "equityUsdt": account.equity_usdt, "loss": loss},
            )
            self._execute(close_all_orders(account.positions, self._specs, "DAILY_KILL"), account, day)
            self._store.mark_day_killed(day)
            for inst_id in list(self._store.all_position_state()):
                self._store.drop_position_state(inst_id)
            return

        prices = mark_prices(self._client, set(self._universe))
        for inst_id, state in sorted(self._store.all_position_state().items()):
            position = account.positions.get(inst_id)
            price = prices.get(inst_id, 0.0)
            if position is None or price <= 0:
                continue
            peak = max(state.peak, price)
            trough = min(state.trough, price)
            if peak != state.peak or trough != state.trough:
                state.peak, state.trough = peak, trough
                self._store.save_position_state(state)
            stop = stop_price(state.entry, state.direction, state.stop_fraction, peak, trough)
            breached = price <= stop if state.direction > 0 else price >= stop
            if not breached:
                continue
            self._store.log(
                "POSITION_STOP",
                {"day": day, "instId": inst_id, "price": price, "stop": stop,
                 "entry": state.entry, "direction": state.direction,
                 "stopFraction": state.stop_fraction, "peak": peak, "trough": trough},
            )
            self._execute(
                close_all_orders({inst_id: position}, self._specs, "POSITION_STOP"), account, day
            )
            self._store.drop_position_state(inst_id)

    # ------------------------------------------------------------------- loop

    def due_for_rebalance(self, now: datetime) -> bool:
        day = trading_day(now)
        if self._last_rebalanced_day == day:
            return False
        minutes = now.astimezone(timezone.utc).hour * 60 + now.astimezone(timezone.utc).minute
        return minutes >= self._settings.rebalance_minute_utc

    def run(self) -> None:
        self.preflight()
        self._store.log("STARTED", {"dryRun": self._settings.dry_run, "simulated": self._settings.simulated})
        while True:
            now = datetime.now(timezone.utc)
            try:
                if self.due_for_rebalance(now):
                    self.rebalance(now)
                self.fast_tick(now)
            except Exception as error:  # noqa: BLE001 - a loop that dies stops managing risk
                self._store.log("TICK_ERROR", {"error": repr(error)})
            time.sleep(self._settings.fast_loop_seconds)
