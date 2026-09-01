"""CLI: python -m live <command>

    preflight  check the account and report which instruments this balance can
               actually trade, without placing anything
    plan       compute today's target book and print the orders it implies
    run        the trading loop (honours OKX_DRY_RUN)
    status     recent events and orders from the local log
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from .executor import Executor
from .okx import OKXClient
from .settings import ConfigError, load_settings
from .store import Store


def _dump(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="live")
    parser.add_argument("command", choices=("preflight", "plan", "run", "status"))
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    store = Store(settings.database)
    try:
        if args.command == "status":
            _dump({"events": store.recent_events(50), "orders": store.recent_orders(25)})
            return 0

        executor = Executor(settings, OKXClient(settings), store)
        if args.command == "preflight":
            executor.preflight()
            _dump(store.recent_events(1, kind="PREFLIGHT"))
            return 0
        if args.command == "plan":
            executor.preflight()
            _dump(executor.build_plan(datetime.now(timezone.utc)).describe())
            return 0

        executor.run()
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
