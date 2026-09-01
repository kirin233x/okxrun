# okxrun

Live execution of the R9.2 OKX perpetual-swap momentum strategy, plus the
research backtests it came from.

The strategy is deterministic code. No model decides what to trade at runtime:
the executor scores the universe, computes a target book, and moves to it.
Parameter changes go through git.

```
strategy/   signals, selection, sizing, stops   — shared by backtest and executor
live/       OKX client, order planning, risk guard, event log, the loop
portal/     read-only dashboard over the event log
research/   backtests (unchanged, heavier dependencies)
```

`strategy/` exists so the two sides cannot drift: the executor scores a day
with the same code the backtest scored it with.

## Quick start on a VPS

```bash
git clone https://github.com/kirin233x/okxrun.git && cd okxrun
cp .env.example .env && $EDITOR .env     # fill in OKX keys, set your limits
sudo ./scripts/install.sh
```

That installs two systemd services (`okxrun-executor`, `okxrun-portal`), both
`Restart=always`. The portal binds to `127.0.0.1:8787` — reach it over an ssh
tunnel or behind an authenticating reverse proxy, never bound to `0.0.0.0`.

**`OKX_DRY_RUN=1` is the default.** Nothing is sent until you change it.

### Before switching it off

```bash
uv run --project . python -m live preflight   # account checks + what is tradable
uv run --project . python -m live plan        # today's target book and the orders
```

`preflight` refuses to continue if the account is not in **net mode**, and
reports which instruments your balance can actually trade. `plan` prints the
exact orders a rebalance would send — it goes through the same code path as the
live rebalance, so what you read is what would be traded.

### Stopping

```bash
touch state/HALT      # flattens every position and blocks new risk
rm state/HALT && sudo systemctl restart okxrun-executor
```

## How it runs

**Daily rebalance, shortly after 00:00 UTC.** Fetch confirmed hourly candles,
build daily bars, score the universe, compute the R9.2 target book, diff it
against the positions the exchange reports, and send only the difference.

**Fast loop, every 60s.** Carry the ATR stop, the trailing stop and the 3%
intraday kill switch.

The exchange is the source of truth for positions. On restart the executor
re-reads them rather than trusting anything it wrote. SQLite holds the event
log and the day's stop anchors, not the position book.

### Account sizing

The executor sizes off the sub-account's own balance (`totalEq`). There is no
capital setting: whatever is in the account is what it trades. Point the API
key at a sub-account holding only what you intend to risk.

### Risk limits

Every limit in `.env` is enforced locally, before a request is signed, so a
sizing bug is rejected on your machine rather than at the exchange:

| Setting | What it stops |
| --- | --- |
| `OKXRUN_MAX_GROSS_LEVERAGE` | total exposure over the account |
| `OKXRUN_MAX_ORDER_NOTIONAL_USDT` | one oversized order |
| `OKXRUN_MAX_INSTRUMENT_NOTIONAL_USDT` | too much in one coin |
| `OKXRUN_MAX_ORDERS_PER_HOUR` | a runaway loop |
| `OKXRUN_MIN_EQUITY_USDT` | trading a drained account |
| `OKXRUN_DAILY_KILL_LOSS` | the day's loss budget |
| `state/HALT` | everything, immediately |

**A risk-reducing order is never blocked.** If a cap could stop a close, a
tripped limit would trap the account in the position it was meant to protect.
Only the rate limit and the universe whitelist apply to exits.

## Known divergences from the backtest

These are properties of the design, not bugs. Read them before trusting a
comparison between live results and the reported backtest numbers.

- **Stops are intraday and reset at 00:00 UTC.** The backtest re-enters each
  simulated day at that day's opening bar and resets the peak/trough the
  trailing stop rides. The executor reproduces that: it re-anchors every stop
  to the day's opening price. A stop is not measured from your original entry.
- **Stop granularity.** The backtest tests each hourly bar's high and low; the
  executor sees the last price once a minute. Fills will differ on fast moves.
- **Minimum order sizes.** OKX perpetuals trade in contracts. An instrument
  whose smallest order exceeds its share of the book is dropped from the
  universe rather than oversized — visible in `preflight` and on the portal.
  At a small balance this can leave fewer than three positions per side, which
  is a different strategy from the one that was backtested.
- **No liquidation modelling.** The backtest simulates neither the exchange's
  margin tiers nor auto-deleveraging.
- **Survivorship bias.** The backtest replays a fixed, currently-liquid
  20-instrument universe over history.

The reported backtest returns are historical simulations, not forecasts, and
should not be expected to reproduce live.

## Research

Backtests keep their own, heavier project (xgboost, scikit-learn):

```bash
uv sync --project research
uv run --project research python -m unittest research.test_r9_momentum research.test_r10_momentum
uv run --project research python research/r9_momentum_backtest.py --refresh --leverage 2.5
```

### R9.2

- enter the top/bottom three momentum assets
- retain positions until they leave the top/bottom eight ranks
- ignore same-side normalized target changes below 10%
- 20-day inverse-volatility weights, rebalanced daily at 00:00 UTC
- ATR stops, trailing exits, and a 3% intraday portfolio kill switch

Validation snapshot on OKX public data through 2026-08-29 UTC:

| Profile | One week | One month | One year | One-year max drawdown | Stress-cost one year |
| --- | ---: | ---: | ---: | ---: | ---: |
| R9.2 2.5x | +2.39% | +13.89% | +179.29% | -24.45% | +40.96% |
| R9.2 3x | +4.59% | +21.48% | +186.55% | -26.67% | +39.36% |

R10 remains an experimental comparison and is not the current candidate.

## Tests

```bash
uv run --project . python -m unittest discover -s live -t .      # 77 tests
uv run --project research python -m unittest research.test_r9_momentum research.test_r10_momentum
```

The live tests cover the arithmetic that decides how much money moves —
contract sizing, order planning, the risk guard, the day-anchor state — and run
without an account or network access.

## Security

- Create the API key on the sub-account, enable **Trade**, leave **Withdraw**
  off, and bind it to the VPS's outbound IP.
- `.env` is chmod 600 and gitignored.
- The portal opens the database read-only and holds no credentials; it cannot
  place an order.
- The systemd units run with `ProtectSystem=strict` and can write only to
  `state/`.
