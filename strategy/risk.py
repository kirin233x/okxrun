"""Stop placement, shared by the backtest and the live fast loop.

Kept free of any position object so the executor can call it with whatever it
reconciled from the exchange.
"""

from __future__ import annotations

import numpy as np

from .config import MAX_STOP, MIN_STOP, STOP_ATR_MULTIPLIER, TRAIL_DISTANCE, TRAIL_TRIGGER


def stop_fraction_from_atr(atr_percent: float) -> float:
    """ATR-scaled stop distance, clamped to the 3%-6% band.

    Returns NaN when ATR is missing; callers must treat that as "no position".
    """
    return float(np.clip(STOP_ATR_MULTIPLIER * atr_percent, MIN_STOP, MAX_STOP))


def stop_price(entry: float, direction: int, stop_fraction: float, peak: float, trough: float) -> float:
    """Effective stop for a position, tightened once the trail trigger is hit.

    ``peak``/``trough`` must be the confirmed extremes so far, never the
    in-progress bar: using an unconfirmed high would let the trail ratchet on a
    price that has not printed.
    """
    fixed = entry * (1.0 - direction * stop_fraction)
    if direction > 0 and peak >= entry * (1.0 + TRAIL_TRIGGER):
        return max(fixed, peak * (1.0 - TRAIL_DISTANCE))
    if direction < 0 and trough <= entry * (1.0 - TRAIL_TRIGGER):
        return min(fixed, trough * (1.0 + TRAIL_DISTANCE))
    return fixed
