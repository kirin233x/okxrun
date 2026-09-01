"""Contract specs and order sizing.

OKX perpetuals trade in contracts, not coins, and every instrument has its own
contract value and lot step. This module is the only place that converts a USDT
notional into a size the exchange will accept, and it is deliberately pure so
the conversion can be tested without an account.

All rounding is toward zero: it is always safe to send slightly less than the
target, never more.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any


@dataclass(frozen=True)
class InstrumentSpec:
    inst_id: str
    contract_value: Decimal  # ctVal, denominated in contract_value_currency
    contract_multiplier: Decimal  # ctMult
    lot_size: Decimal  # lotSz, the size increment in contracts
    min_size: Decimal  # minSz, the smallest order in contracts
    tick_size: Decimal  # tickSz, the price increment
    contract_type: str  # "linear" or "inverse"
    settle_currency: str
    state: str

    @property
    def tradable(self) -> bool:
        return self.state == "live"

    @property
    def coins_per_contract(self) -> Decimal:
        return self.contract_value * self.contract_multiplier


def parse_instrument(payload: dict[str, Any]) -> InstrumentSpec:
    return InstrumentSpec(
        inst_id=str(payload["instId"]),
        contract_value=Decimal(str(payload["ctVal"])),
        contract_multiplier=Decimal(str(payload.get("ctMult") or "1")),
        lot_size=Decimal(str(payload["lotSz"])),
        min_size=Decimal(str(payload["minSz"])),
        tick_size=Decimal(str(payload["tickSz"])),
        contract_type=str(payload.get("ctType") or "linear"),
        settle_currency=str(payload.get("settleCcy") or ""),
        state=str(payload.get("state") or ""),
    )


class UnsupportedInstrument(ValueError):
    """Raised for contracts this executor refuses to size."""


def _require_linear_usdt(spec: InstrumentSpec) -> None:
    # Inverse contracts are collateralised in the base coin and invert the
    # notional maths. The strategy assumes USDT-margined linear swaps, so
    # refuse rather than silently mis-size.
    if spec.contract_type != "linear":
        raise UnsupportedInstrument(f"{spec.inst_id} is {spec.contract_type}, only linear swaps are supported")
    if spec.settle_currency and spec.settle_currency != "USDT":
        raise UnsupportedInstrument(f"{spec.inst_id} settles in {spec.settle_currency}, only USDT is supported")


def notional_for_contracts(spec: InstrumentSpec, price: float, contracts: Decimal) -> float:
    """USDT notional represented by ``contracts`` at ``price``."""
    _require_linear_usdt(spec)
    return float(abs(contracts) * spec.coins_per_contract * Decimal(str(price)))


def min_order_notional(spec: InstrumentSpec, price: float) -> float:
    """Smallest USDT notional the exchange will accept for this instrument."""
    return notional_for_contracts(spec, price, spec.min_size)


def contracts_for_notional(spec: InstrumentSpec, price: float, notional_usdt: float) -> Decimal:
    """Largest lot-aligned contract count worth no more than ``notional_usdt``.

    Returns 0 when the notional cannot cover one minimum order, which is the
    common case for a small account on a large-denomination contract.
    """
    _require_linear_usdt(spec)
    if price <= 0:
        raise ValueError(f"{spec.inst_id}: price must be positive, got {price}")
    if spec.lot_size <= 0:
        raise ValueError(f"{spec.inst_id}: lotSz must be positive, got {spec.lot_size}")
    if notional_usdt <= 0:
        return Decimal(0)

    raw = Decimal(str(notional_usdt)) / (spec.coins_per_contract * Decimal(str(price)))
    lots = (raw / spec.lot_size).to_integral_value(rounding=ROUND_DOWN)
    contracts = lots * spec.lot_size
    if contracts < spec.min_size:
        return Decimal(0)
    return contracts


def round_to_lot(spec: InstrumentSpec, contracts: Decimal) -> Decimal:
    """Round an unsigned contract count down onto the lot grid."""
    if spec.lot_size <= 0:
        raise ValueError(f"{spec.inst_id}: lotSz must be positive, got {spec.lot_size}")
    lots = (abs(contracts) / spec.lot_size).to_integral_value(rounding=ROUND_DOWN)
    return lots * spec.lot_size


def format_size(contracts: Decimal) -> str:
    """Render a size for the API without scientific notation or a stray sign."""
    normalised = abs(contracts).normalize()
    if normalised == normalised.to_integral_value():
        normalised = normalised.quantize(Decimal(1))
    return format(normalised, "f")


def tradable_universe(
    specs: dict[str, InstrumentSpec],
    prices: dict[str, float],
    target_notional_usdt: float,
) -> tuple[list[str], dict[str, str]]:
    """Split a universe into what this account can actually trade, and why not.

    ``target_notional_usdt`` is the typical per-position size; an instrument
    whose minimum order exceeds it is unusable at this account size, and
    including it would silently shrink the book to fewer than the intended
    number of positions.
    """
    usable: list[str] = []
    rejected: dict[str, str] = {}
    for inst_id, spec in sorted(specs.items()):
        price = prices.get(inst_id)
        if price is None or price <= 0:
            rejected[inst_id] = "no price"
            continue
        if not spec.tradable:
            rejected[inst_id] = f"state={spec.state!r}"
            continue
        try:
            floor_notional = min_order_notional(spec, price)
        except UnsupportedInstrument as error:
            rejected[inst_id] = str(error)
            continue
        if floor_notional > target_notional_usdt:
            rejected[inst_id] = (
                f"min order {floor_notional:.2f} USDT exceeds target {target_notional_usdt:.2f} USDT"
            )
            continue
        usable.append(inst_id)
    return usable, rejected
