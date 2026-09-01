"""Shared R9 strategy core.

Both the offline backtest (``research/``) and the live executor import their
signal, sizing and stop logic from this package, so the two can never drift
apart. Nothing here touches the network, the filesystem or an exchange
account: every function is pure, given already-loaded price frames.
"""

from __future__ import annotations

from .config import (
    ASSETS_PER_SIDE,
    BTC_INSTRUMENT,
    BTC_SHOCK_Z,
    DAILY_KILL_LOSS,
    HIGH_LOOKBACK_DAYS,
    LONG_FORMATION_DAYS,
    MAX_ASSET_WEIGHT,
    MAX_STOP,
    MIN_REBALANCE_DELTA,
    MIN_STOP,
    R9,
    R91,
    R92,
    R92_MIN_REBALANCE_DELTA,
    R92_RANK_EXIT_BUFFER,
    RANK_EXIT_BUFFER,
    SHORT_FORMATION_DAYS,
    STOP_ATR_MULTIPLIER,
    TRAIL_DISTANCE,
    TRAIL_TRIGGER,
    VOLATILITY_DAYS,
    StrategyVariant,
)
from .portfolio import (
    apply_rebalance_threshold,
    buffered_selection,
    capped_inverse_vol_weights,
    scale_target_weights,
    target_for_day,
)
from .risk import stop_fraction_from_atr, stop_price
from .signals import compute_signals, cross_sectional_rank

__all__ = [
    "ASSETS_PER_SIDE",
    "BTC_INSTRUMENT",
    "BTC_SHOCK_Z",
    "DAILY_KILL_LOSS",
    "HIGH_LOOKBACK_DAYS",
    "LONG_FORMATION_DAYS",
    "MAX_ASSET_WEIGHT",
    "MAX_STOP",
    "MIN_REBALANCE_DELTA",
    "MIN_STOP",
    "R9",
    "R91",
    "R92",
    "R92_MIN_REBALANCE_DELTA",
    "R92_RANK_EXIT_BUFFER",
    "RANK_EXIT_BUFFER",
    "SHORT_FORMATION_DAYS",
    "STOP_ATR_MULTIPLIER",
    "StrategyVariant",
    "TRAIL_DISTANCE",
    "TRAIL_TRIGGER",
    "VOLATILITY_DAYS",
    "apply_rebalance_threshold",
    "buffered_selection",
    "capped_inverse_vol_weights",
    "compute_signals",
    "cross_sectional_rank",
    "scale_target_weights",
    "stop_fraction_from_atr",
    "stop_price",
    "target_for_day",
]
