"""Runtime configuration, read from the environment.

Everything that decides how much money can move lives here and is validated at
startup, so a typo in .env fails loudly before the first order rather than
halfway through a rebalance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_DIR = ROOT / "state"
LIVE_REST = "https://www.okx.com"


class ConfigError(RuntimeError):
    """Raised when .env is missing or internally inconsistent."""


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise ConfigError(f"{name} is required; copy .env.example to .env and fill it in")
    return value


def _float(name: str, default: str) -> float:
    raw = os.environ.get(name) or default
    try:
        return float(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from error


def _int(name: str, default: str) -> int:
    raw = os.environ.get(name) or default
    try:
        return int(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from error


def _bool(name: str, default: str) -> bool:
    raw = (os.environ.get(name) or default).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean, got {raw!r}")


def load_dotenv(path: Path | None = None) -> None:
    """Load .env into os.environ without overwriting anything already set.

    Deliberately minimal: KEY=VALUE, '#' comments, optional surrounding quotes.
    A real dotenv library would be another dependency on the box that holds the
    trading keys.
    """
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    api_key: str
    api_secret: str
    passphrase: str
    simulated: bool
    dry_run: bool

    leverage: float
    variant_id: str

    # Hard limits. These are the reason it is safe to hand over an API key:
    # every one of them is enforced locally, before a request is signed.
    max_gross_leverage: float
    max_instrument_notional_usdt: float
    max_order_notional_usdt: float
    max_orders_per_hour: int
    min_equity_usdt: float
    daily_kill_loss: float

    rebalance_minute_utc: int
    fast_loop_seconds: int

    state_dir: Path
    halt_file: Path
    rest_base: str

    @property
    def database(self) -> Path:
        return self.state_dir / "okxrun.sqlite3"


def load_settings() -> Settings:
    load_dotenv()
    dry_run = _bool("OKX_DRY_RUN", "1")
    # Required even in dry run: reading the balance and open positions is a
    # signed call, and a dry run that cannot see the real account would be
    # planning against imaginary state.
    api_key = _env("OKX_API_KEY")
    api_secret = _env("OKX_API_SECRET")
    passphrase = _env("OKX_PASSPHRASE")

    state_dir = Path(os.environ.get("OKXRUN_STATE_DIR") or DEFAULT_STATE_DIR)
    settings = Settings(
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase,
        simulated=_bool("OKX_SIMULATED", "0"),
        dry_run=dry_run,
        leverage=_float("OKXRUN_LEVERAGE", "1.0"),
        variant_id=os.environ.get("OKXRUN_VARIANT") or "r9.2",
        max_gross_leverage=_float("OKXRUN_MAX_GROSS_LEVERAGE", "3.0"),
        max_instrument_notional_usdt=_float("OKXRUN_MAX_INSTRUMENT_NOTIONAL_USDT", "400"),
        max_order_notional_usdt=_float("OKXRUN_MAX_ORDER_NOTIONAL_USDT", "400"),
        max_orders_per_hour=_int("OKXRUN_MAX_ORDERS_PER_HOUR", "40"),
        min_equity_usdt=_float("OKXRUN_MIN_EQUITY_USDT", "50"),
        daily_kill_loss=_float("OKXRUN_DAILY_KILL_LOSS", "0.03"),
        rebalance_minute_utc=_int("OKXRUN_REBALANCE_MINUTE_UTC", "2"),
        fast_loop_seconds=_int("OKXRUN_FAST_LOOP_SECONDS", "60"),
        state_dir=state_dir,
        halt_file=Path(os.environ.get("OKXRUN_HALT_FILE") or (state_dir / "HALT")),
        rest_base=os.environ.get("OKX_REST_BASE") or LIVE_REST,
    )
    validate(settings)
    return settings


def validate(settings: Settings) -> None:
    if settings.leverage <= 0:
        raise ConfigError("OKXRUN_LEVERAGE must be positive")
    if settings.leverage > settings.max_gross_leverage:
        raise ConfigError(
            f"OKXRUN_LEVERAGE ({settings.leverage}) exceeds "
            f"OKXRUN_MAX_GROSS_LEVERAGE ({settings.max_gross_leverage})"
        )
    if settings.max_order_notional_usdt <= 0 or settings.max_instrument_notional_usdt <= 0:
        raise ConfigError("notional caps must be positive")
    if settings.max_orders_per_hour <= 0:
        raise ConfigError("OKXRUN_MAX_ORDERS_PER_HOUR must be positive")
    if not 0 < settings.daily_kill_loss < 1:
        raise ConfigError("OKXRUN_DAILY_KILL_LOSS must be between 0 and 1")
    if not 0 <= settings.rebalance_minute_utc < 60:
        raise ConfigError("OKXRUN_REBALANCE_MINUTE_UTC must be a minute of the hour")
    if settings.fast_loop_seconds < 5:
        raise ConfigError("OKXRUN_FAST_LOOP_SECONDS below 5s would hammer the API")
