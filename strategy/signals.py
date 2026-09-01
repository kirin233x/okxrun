"""Cross-sectional momentum signals and the BTC regime inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BTC_INSTRUMENT,
    HIGH_LOOKBACK_DAYS,
    LONG_FORMATION_DAYS,
    SHORT_FORMATION_DAYS,
    VOLATILITY_DAYS,
)


def cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True, method="average")


def compute_signals(closes: pd.DataFrame, daily: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame | pd.Series]:
    """Build every input ``target_for_day`` needs.

    Every series is shifted by one day so a decision made for date D only ever
    reads bars that closed on D-1. Live and backtest both rely on that: the
    executor must not act on a candle that has not been confirmed.
    """
    returns = closes.pct_change(fill_method=None)
    return_30 = closes.pct_change(LONG_FORMATION_DAYS, fill_method=None).shift(1)
    return_7 = closes.pct_change(SHORT_FORMATION_DAYS, fill_method=None).shift(1)
    prior_close = closes.shift(1)
    prior_high = closes.rolling(HIGH_LOOKBACK_DAYS).max().shift(1)
    near_high = prior_close / prior_high - 1.0
    score = (
        0.60 * cross_sectional_rank(return_30)
        + 0.25 * cross_sectional_rank(return_7)
        + 0.15 * cross_sectional_rank(near_high)
    )
    volatility = returns.rolling(VOLATILITY_DAYS).std(ddof=0).shift(1)

    atr_percent: dict[str, pd.Series] = {}
    for inst_id, frame in daily.items():
        previous_close = frame["close"].shift(1)
        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous_close).abs(),
                (frame["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_percent[inst_id] = (true_range / previous_close).rolling(VOLATILITY_DAYS).mean().shift(1)
    atr = pd.concat(atr_percent, axis=1).reindex(closes.index)

    btc_prior = closes[BTC_INSTRUMENT].shift(1)
    btc_ma = closes[BTC_INSTRUMENT].rolling(20).mean().shift(1)
    btc_return_1 = returns[BTC_INSTRUMENT].shift(1)
    btc_return_7 = closes[BTC_INSTRUMENT].pct_change(7, fill_method=None).shift(1)
    btc_vol = returns[BTC_INSTRUMENT].rolling(20).std(ddof=0).shift(1)
    btc_shock_z = btc_return_1 / btc_vol.replace(0, np.nan)
    return {
        "score": score,
        "volatility": volatility,
        "atr": atr,
        "btcPrior": btc_prior,
        "btcMa": btc_ma,
        "btcReturn1": btc_return_1,
        "btcReturn7": btc_return_7,
        "btcShockZ": btc_shock_z,
    }
