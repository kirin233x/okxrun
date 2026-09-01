"""Selection, weighting and the daily target book."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ASSETS_PER_SIDE,
    BTC_SHOCK_Z,
    MAX_ASSET_WEIGHT,
    RANK_EXIT_BUFFER,
    R9,
    StrategyVariant,
)


def capped_inverse_vol_weights(volatility: pd.Series, gross: float) -> dict[str, float]:
    valid = volatility.replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[valid > 0]
    if valid.empty or gross <= 0:
        return {}
    raw = (1.0 / valid) / (1.0 / valid).sum() * gross
    capped = raw.clip(upper=MAX_ASSET_WEIGHT)
    return {str(inst_id): float(weight) for inst_id, weight in capped.items() if weight > 0}


def buffered_selection(
    ranked: pd.Series,
    previous_weights: dict[str, float],
    exit_buffer: int = RANK_EXIT_BUFFER,
) -> tuple[list[str], list[str]]:
    """Enter the top/bottom three, but keep a holding until it leaves the buffer."""
    ascending = list(ranked.index)
    descending = list(reversed(ascending))
    long_buffer = set(descending[:exit_buffer])
    short_buffer = set(ascending[:exit_buffer])
    retained_longs = [inst_id for inst_id in descending if previous_weights.get(inst_id, 0.0) > 0 and inst_id in long_buffer]
    retained_shorts = [inst_id for inst_id in ascending if previous_weights.get(inst_id, 0.0) < 0 and inst_id in short_buffer]

    longs = retained_longs[:ASSETS_PER_SIDE]
    shorts = retained_shorts[:ASSETS_PER_SIDE]
    longs.extend(inst_id for inst_id in descending if inst_id not in longs and len(longs) < ASSETS_PER_SIDE)
    shorts.extend(inst_id for inst_id in ascending if inst_id not in shorts and len(shorts) < ASSETS_PER_SIDE)
    return longs, shorts


def apply_rebalance_threshold(
    target: dict[str, float],
    previous_weights: dict[str, float],
    threshold: float,
) -> dict[str, float]:
    """Skip same-side trims below the threshold; exits are always honoured."""
    if threshold <= 0:
        return target
    adjusted: dict[str, float] = {}
    for inst_id, proposed in target.items():
        previous = previous_weights.get(inst_id, 0.0)
        same_side = previous * proposed > 0
        if same_side and abs(proposed - previous) < threshold:
            retained = math.copysign(min(abs(previous), MAX_ASSET_WEIGHT), proposed)
            adjusted[inst_id] = retained
        else:
            adjusted[inst_id] = proposed
    gross = sum(abs(weight) for weight in adjusted.values())
    if gross > 1.0:
        adjusted = {inst_id: weight / gross for inst_id, weight in adjusted.items()}
    return adjusted


def scale_target_weights(target: dict[str, float], leverage: float) -> dict[str, float]:
    if not math.isfinite(leverage) or leverage <= 0:
        raise ValueError("leverage must be a positive finite number")
    return {inst_id: weight * leverage for inst_id, weight in target.items()}


def target_for_day(
    date: pd.Timestamp,
    signals: dict[str, pd.DataFrame | pd.Series],
    previous_weights: dict[str, float] | None = None,
    variant: StrategyVariant = R9,
    leverage: float = 1.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Target book for ``date``, plus the reasoning behind it.

    ``previous_weights`` are leverage-scaled, matching what the caller holds;
    they are normalised back to 1x before the rank buffer and the rebalance
    threshold see them, so both behave identically at any leverage.
    """
    previous_weights = previous_weights or {}
    normalized_previous = {inst_id: weight / leverage for inst_id, weight in previous_weights.items()}
    score = signals["score"].loc[date].dropna()  # type: ignore[union-attr]
    volatility = signals["volatility"].loc[date].reindex(score.index).dropna()  # type: ignore[union-attr]
    ranked = score.reindex(volatility.index).dropna().sort_values()
    if len(ranked) < ASSETS_PER_SIDE * 2:
        return {}, {"regime": "INSUFFICIENT_DATA", "tradable": False}
    if variant.rank_exit_buffer:
        longs, shorts = buffered_selection(ranked, normalized_previous, variant.rank_exit_buffer)
    else:
        shorts = list(ranked.index[:ASSETS_PER_SIDE])
        longs = list(ranked.index[-ASSETS_PER_SIDE:])
    btc_prior = float(signals["btcPrior"].loc[date])  # type: ignore[union-attr]
    btc_ma = float(signals["btcMa"].loc[date])  # type: ignore[union-attr]
    btc_return_1 = float(signals["btcReturn1"].loc[date])  # type: ignore[union-attr]
    btc_return_7 = float(signals["btcReturn7"].loc[date])  # type: ignore[union-attr]
    btc_shock_z = float(signals["btcShockZ"].loc[date])  # type: ignore[union-attr]
    if not all(math.isfinite(value) for value in (btc_prior, btc_ma, btc_return_1, btc_return_7, btc_shock_z)):
        return {}, {"regime": "INSUFFICIENT_BTC_DATA", "tradable": False}

    if btc_shock_z > BTC_SHOCK_Z and variant.up_shock_flat:
        long_gross, short_gross = 0.0, 0.0
        regime = "UP_SHOCK_FLAT"
    elif abs(btc_shock_z) > BTC_SHOCK_Z:
        long_gross, short_gross = (1.0, 0.0) if btc_return_1 > 0 else (0.0, 1.0)
        regime = "UP_SHOCK" if btc_return_1 > 0 else "DOWN_SHOCK"
    elif btc_prior > btc_ma and btc_return_7 > 0:
        long_gross, short_gross = 0.75, 0.25
        regime = "UP_TREND"
    elif btc_prior < btc_ma and btc_return_7 < 0:
        long_gross, short_gross = 0.25, 0.75
        regime = "DOWN_TREND"
    else:
        long_gross, short_gross = 0.50, 0.50
        regime = "NEUTRAL"

    long_weights = capped_inverse_vol_weights(volatility.reindex(longs), long_gross)
    short_weights = capped_inverse_vol_weights(volatility.reindex(shorts), short_gross)
    normalized_target = long_weights | {inst_id: -weight for inst_id, weight in short_weights.items()}
    normalized_target = apply_rebalance_threshold(
        normalized_target,
        normalized_previous,
        variant.min_rebalance_delta,
    )
    target = scale_target_weights(normalized_target, leverage)
    detail = {
        "regime": regime,
        "tradable": True,
        "variant": variant.id,
        "btcReturn1": btc_return_1,
        "btcReturn7": btc_return_7,
        "btcShockZ": btc_shock_z,
        "leverage": leverage,
        "longGrossTarget": long_gross * leverage,
        "shortGrossTarget": short_gross * leverage,
        "grossAfterCap": float(sum(abs(weight) for weight in target.values())),
        "longs": longs,
        "shorts": shorts,
        "rankBuffer": variant.rank_exit_buffer or None,
        "minRebalanceDelta": variant.min_rebalance_delta,
    }
    return target, detail
