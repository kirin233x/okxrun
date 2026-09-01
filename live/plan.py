"""Turn target weights into the orders that close the gap.

Pure: given the target book, what the exchange says we hold, and the contract
specs, produce the exact list of orders. No network, no clock, no database —
so the arithmetic that decides how much money moves can be tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .instruments import (
    InstrumentSpec,
    UnsupportedInstrument,
    contracts_for_notional,
    notional_for_contracts,
    round_to_lot,
)


@dataclass(frozen=True)
class PositionSnapshot:
    """A position as reported by the exchange. ``contracts`` is signed."""

    inst_id: str
    contracts: Decimal
    mark_price: float


@dataclass(frozen=True)
class PlannedOrder:
    inst_id: str
    side: str  # "buy" or "sell"
    contracts: Decimal
    notional_usdt: float
    reduce_only: bool
    reason: str


@dataclass(frozen=True)
class PlanSkip:
    inst_id: str
    reason: str


def _side(delta: Decimal) -> str:
    return "buy" if delta > 0 else "sell"


def target_contracts(
    spec: InstrumentSpec,
    price: float,
    equity_usdt: float,
    weight: float,
) -> Decimal:
    """Signed contract target for a weight, floored onto the lot grid.

    Returns 0 when the position would be too small to trade, which is how an
    instrument whose minimum order exceeds its share of the book drops out
    instead of being silently oversized.
    """
    notional = abs(equity_usdt * weight)
    size = contracts_for_notional(spec, price, notional)
    if size == 0:
        return Decimal(0)
    return size if weight > 0 else -size


def plan_orders(
    targets: dict[str, float],
    positions: dict[str, PositionSnapshot],
    specs: dict[str, InstrumentSpec],
    prices: dict[str, float],
    equity_usdt: float,
) -> tuple[list[PlannedOrder], list[PlanSkip]]:
    """Orders that move ``positions`` to ``targets``, plus what was skipped."""
    orders: list[PlannedOrder] = []
    skips: list[PlanSkip] = []

    for inst_id in sorted(set(targets) | set(positions)):
        spec = specs.get(inst_id)
        current = positions[inst_id].contracts if inst_id in positions else Decimal(0)
        if spec is None:
            if current != 0:
                skips.append(PlanSkip(inst_id, "held but no contract spec available; cannot size an exit"))
            else:
                skips.append(PlanSkip(inst_id, "no contract spec available"))
            continue

        price = prices.get(inst_id) or (positions[inst_id].mark_price if inst_id in positions else 0.0)
        if price <= 0:
            skips.append(PlanSkip(inst_id, "no usable price"))
            continue

        weight = targets.get(inst_id, 0.0)
        try:
            wanted = target_contracts(spec, price, equity_usdt, weight)
        except UnsupportedInstrument as error:
            skips.append(PlanSkip(inst_id, str(error)))
            continue

        if wanted == current:
            continue

        # A side flip has to be two orders: close the old exposure, then open
        # the new one. Netting them into a single oversized order would leave
        # the account momentarily unhedged in the wrong direction if the second
        # leg were ever rejected.
        if current != 0 and wanted != 0 and (current > 0) != (wanted > 0):
            orders.append(
                PlannedOrder(
                    inst_id=inst_id,
                    side=_side(-current),
                    contracts=abs(current),
                    notional_usdt=notional_for_contracts(spec, price, current),
                    reduce_only=True,
                    reason="FLIP_CLOSE",
                )
            )
            orders.append(
                PlannedOrder(
                    inst_id=inst_id,
                    side=_side(wanted),
                    contracts=abs(wanted),
                    notional_usdt=notional_for_contracts(spec, price, wanted),
                    reduce_only=False,
                    reason="FLIP_OPEN",
                )
            )
            continue

        delta = wanted - current
        size = round_to_lot(spec, abs(delta))
        reducing = abs(wanted) < abs(current)
        if size < spec.min_size:
            # Worth calling out loudly when it blocks an exit: a position too
            # small to close is a stuck position, not a no-op.
            what = "exit" if wanted == 0 else "adjustment"
            skips.append(
                PlanSkip(
                    inst_id,
                    f"{what} of {abs(delta)} contracts is below the {spec.min_size} minimum order size",
                )
            )
            continue

        orders.append(
            PlannedOrder(
                inst_id=inst_id,
                side=_side(delta),
                contracts=size,
                notional_usdt=notional_for_contracts(spec, price, size),
                reduce_only=reducing,
                reason="REBALANCE_REDUCE" if reducing else "REBALANCE_OPEN",
            )
        )

    return orders, skips


def close_all_orders(
    positions: dict[str, PositionSnapshot],
    specs: dict[str, InstrumentSpec],
    reason: str,
) -> list[PlannedOrder]:
    """Flatten everything. Used by the kill switch and by HALT."""
    orders: list[PlannedOrder] = []
    for inst_id in sorted(positions):
        position = positions[inst_id]
        spec = specs.get(inst_id)
        if position.contracts == 0 or spec is None:
            continue
        orders.append(
            PlannedOrder(
                inst_id=inst_id,
                side=_side(-position.contracts),
                contracts=abs(position.contracts),
                notional_usdt=notional_for_contracts(spec, position.mark_price, position.contracts),
                reduce_only=True,
                reason=reason,
            )
        )
    return orders
