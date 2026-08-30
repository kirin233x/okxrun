# okxrun

Research-only backtests for the R9.2 and R10 OKX perpetual-swap momentum strategies.

## Current candidate: R9.2

R9.2 keeps the R9.1 signal and risk model, while reducing unnecessary turnover:

- enter the top/bottom three momentum assets
- retain positions until they leave the top/bottom eight ranks
- ignore same-side normalized target changes below 10%
- use 20-day inverse-volatility weights
- rebalance daily at 00:00 UTC
- use ATR stops, trailing exits, and a 3% intraday portfolio kill switch

The recommended research leverage is 2.5x. The 3x profile is retained as an aggressive comparison.

## Validation snapshot

Using OKX public data through 2026-08-29 UTC:

| Profile | One week | One month | One year | One-year max drawdown | Stress-cost one year |
| --- | ---: | ---: | ---: | ---: | ---: |
| R9.2 2.5x | +2.39% | +13.89% | +179.29% | -24.45% | +40.96% |
| R9.2 3x | +4.59% | +21.48% | +186.55% | -26.67% | +39.36% |

These are historical simulations, not forecasts. The model has survivorship bias, hourly-bar execution approximations, and no exchange liquidation-tier simulation.

## Scope

- Uses only OKX public market-data endpoints.
- Contains no API keys, passphrases, account data, order history, local cache, or live-trading execution.
- R10 remains an experimental comparison and is not the current candidate.
- Generated market caches and backtest reports are intentionally excluded from version control.

## Setup

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --project research
uv run --project research python -m unittest research.test_r9_momentum research.test_r10_momentum
```

## Run R9.2

Recommended 2.5x research profile:

```bash
uv run --project research python research/r9_momentum_backtest.py --refresh --leverage 2.5
```

Aggressive 3x comparison:

```bash
uv run --project research python research/r9_momentum_backtest.py --refresh --leverage 3
```

## Run R10

```bash
uv run --project research python research/r10_momentum_backtest.py --refresh-market --refresh-funding
```
