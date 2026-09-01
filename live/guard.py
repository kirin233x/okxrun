"""Hard risk limits, enforced locally before anything is signed.

Every rule here is checked against numbers the executor has already computed,
so a sizing bug, a bad price or a runaway loop is rejected on this machine
instead of reaching the exchange.

One invariant matters above all others: **a risk-reducing order is never
blocked**. If the caps could stop a close, a tripped limit would trap the
account in exactly the position it was meant to protect.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .settings import Settings


@dataclass(frozen=True)
class OrderIntent:
    inst_id: str
    side: str  # "buy" or "sell"
    contracts: Decimal
    notional_usdt: float
    reduce_only: bool
    reason: str


@dataclass(frozen=True)
class GuardVerdict:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.allowed


ALLOWED = GuardVerdict(True)


@dataclass(frozen=True)
class AccountView:
    """The state a guard decision is made against."""

    equity_usdt: float
    gross_notional_after_usdt: float
    instrument_notional_after_usdt: float
    orders_in_last_hour: int


class Guard:
    def __init__(self, settings: Settings, universe: set[str]) -> None:
        self._settings = settings
        self._universe = set(universe)

    def halted(self) -> bool:
        return Path(self._settings.halt_file).exists()

    def check(self, intent: OrderIntent, account: AccountView) -> GuardVerdict:
        settings = self._settings

        # Applies to every order, closes included: an instrument that is not in
        # the configured universe is a bug or a compromised config, and the
        # executor has no business touching it either way.
        if intent.inst_id not in self._universe:
            return GuardVerdict(False, f"{intent.inst_id} is not in the configured universe")
        if intent.side not in {"buy", "sell"}:
            return GuardVerdict(False, f"unknown side {intent.side!r}")
        if intent.contracts <= 0:
            return GuardVerdict(False, "size must be positive")
        if account.orders_in_last_hour >= settings.max_orders_per_hour:
            return GuardVerdict(
                False,
                f"order rate limit reached ({account.orders_in_last_hour}/{settings.max_orders_per_hour} per hour)",
            )

        # Everything below adds or maintains risk. Closes stop here.
        if intent.reduce_only:
            return ALLOWED

        if self.halted():
            return GuardVerdict(False, f"HALT file present at {settings.halt_file}")
        if account.equity_usdt < settings.min_equity_usdt:
            return GuardVerdict(
                False,
                f"equity {account.equity_usdt:.2f} USDT below floor {settings.min_equity_usdt:.2f} USDT",
            )
        if intent.notional_usdt > settings.max_order_notional_usdt:
            return GuardVerdict(
                False,
                f"order notional {intent.notional_usdt:.2f} USDT exceeds cap "
                f"{settings.max_order_notional_usdt:.2f} USDT",
            )
        if account.instrument_notional_after_usdt > settings.max_instrument_notional_usdt:
            return GuardVerdict(
                False,
                f"{intent.inst_id} notional would reach {account.instrument_notional_after_usdt:.2f} USDT, "
                f"over cap {settings.max_instrument_notional_usdt:.2f} USDT",
            )
        if account.equity_usdt <= 0:
            return GuardVerdict(False, "equity is not positive")
        gross_leverage = account.gross_notional_after_usdt / account.equity_usdt
        if gross_leverage > settings.max_gross_leverage:
            return GuardVerdict(
                False,
                f"gross leverage would reach {gross_leverage:.2f}x, over cap "
                f"{settings.max_gross_leverage:.2f}x",
            )
        return ALLOWED
