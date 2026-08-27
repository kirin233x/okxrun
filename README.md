# okxrun

Research-only backtests for the R9.1 and R10 OKX perpetual-swap momentum strategies.

## Scope

- Uses only OKX public market-data endpoints.
- Contains no API keys, passphrases, account data, order history, local cache, or live-trading execution.
- R9.1 is the current high-risk candidate; R10 is an experimental dual-speed variant.
- Backtest results do not guarantee future returns.

## Setup

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --project research
uv run --project research python -m unittest research.test_r9_momentum research.test_r10_momentum
```

## Run

Download public candles and run R9.1 at 3x:

```bash
uv run --project research python research/r9_momentum_backtest.py --refresh --leverage 3
```

Run R10:

```bash
uv run --project research python research/r10_momentum_backtest.py --refresh
```

Generated market caches and reports are intentionally excluded from version control.
