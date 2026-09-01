"""Tunable constants and variant definitions for the R9 family.

These values define the strategy. Changing one changes what both the backtest
and the live executor do, which is the point: a parameter must never be
adjusted in one place only.
"""

from __future__ import annotations

from dataclasses import dataclass


BTC_INSTRUMENT = "BTC-USDT-SWAP"

# Formation window: 60% 30-day rank + 25% 7-day rank + 15% near-20-day-high rank.
LONG_FORMATION_DAYS = 30
SHORT_FORMATION_DAYS = 7
HIGH_LOOKBACK_DAYS = 20
VOLATILITY_DAYS = 20

# Selection and sizing.
ASSETS_PER_SIDE = 3
MAX_ASSET_WEIGHT = 0.20

# BTC regime tilt.
BTC_SHOCK_Z = 1.5

# Position risk.
STOP_ATR_MULTIPLIER = 1.5
MIN_STOP = 0.03
MAX_STOP = 0.06
TRAIL_TRIGGER = 0.04
TRAIL_DISTANCE = 0.025
DAILY_KILL_LOSS = 0.03

# Turnover control.
RANK_EXIT_BUFFER = 6
MIN_REBALANCE_DELTA = 0.05
R92_RANK_EXIT_BUFFER = 8
R92_MIN_REBALANCE_DELTA = 0.10


@dataclass(frozen=True)
class StrategyVariant:
    id: str
    name: str
    up_shock_flat: bool = False
    rank_exit_buffer: int = 0
    min_rebalance_delta: float = 0.0


R9 = StrategyVariant("r9", "R9 基线")
R91 = StrategyVariant(
    "r9.1",
    "R9.1 暴涨回避 + 排名缓冲 + 调仓阈值",
    up_shock_flat=True,
    rank_exit_buffer=RANK_EXIT_BUFFER,
    min_rebalance_delta=MIN_REBALANCE_DELTA,
)
R92 = StrategyVariant(
    "r9.2",
    "R9.2 扩大排名缓冲 + 提高调仓阈值",
    up_shock_flat=True,
    rank_exit_buffer=R92_RANK_EXIT_BUFFER,
    min_rebalance_delta=R92_MIN_REBALANCE_DELTA,
)
