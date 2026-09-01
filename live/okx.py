"""Minimal OKX v5 REST client.

Only the endpoints this executor needs, written against urllib so the machine
holding the trading keys does not need an HTTP stack pulled from PyPI.

Private requests are signed per OKX v5: base64(HMAC-SHA256(timestamp + METHOD +
requestPath + body)), where requestPath includes the query string and body is
the exact string sent.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from .settings import Settings


PUBLIC_TIMEOUT = 45
RETRIES = 5


class OKXError(RuntimeError):
    """An error OKX reported in the response envelope."""

    def __init__(self, code: str, message: str, path: str, data: Any = None) -> None:
        super().__init__(f"OKX {code} on {path}: {message}")
        self.code = code
        self.message = message
        self.path = path
        self.data = data


def _timestamp() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class OKXClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base = settings.rest_base.rstrip("/")

    # ------------------------------------------------------------------ core

    def _headers(self, method: str, path: str, body: str, private: bool) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "okxrun/1.0",
            "Connection": "close",
        }
        if self._settings.simulated:
            headers["x-simulated-trading"] = "1"
        if not private:
            return headers
        timestamp = _timestamp()
        message = f"{timestamp}{method.upper()}{path}{body}"
        signature = base64.b64encode(
            hmac.new(
                self._settings.api_secret.encode("utf-8"),
                message.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        headers.update(
            {
                "OK-ACCESS-KEY": self._settings.api_key,
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": self._settings.passphrase,
            }
        )
        return headers

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, str] | None = None,
        payload: dict[str, Any] | list[Any] | None = None,
        private: bool = False,
    ) -> list[dict[str, Any]]:
        if params:
            path = f"{path}?{urllib.parse.urlencode(params)}"
        body = json.dumps(payload, separators=(",", ":")) if payload is not None else ""
        last_error: Exception | None = None
        for attempt in range(RETRIES):
            request = urllib.request.Request(
                f"{self._base}{path}",
                method=method.upper(),
                data=body.encode("utf-8") if body else None,
                # Headers are rebuilt each attempt: the signature covers a
                # timestamp, and OKX rejects one that has drifted too far.
                headers=self._headers(method, path, body, private),
            )
            try:
                with urllib.request.urlopen(request, timeout=PUBLIC_TIMEOUT) as response:
                    envelope = json.load(response)
                break
            except urllib.error.HTTPError as error:
                # OKX returns a JSON envelope with a useful code on 4xx too.
                try:
                    envelope = json.loads(error.read().decode("utf-8"))
                    break
                except Exception:  # noqa: BLE001 - fall through to the retry path
                    last_error = error
            except (urllib.error.URLError, http.client.RemoteDisconnected, TimeoutError) as error:
                last_error = error
            time.sleep(min(1.5 * (attempt + 1), 6))
        else:
            raise RuntimeError(f"OKX request failed after {RETRIES} attempts: {last_error}") from last_error

        code = str(envelope.get("code"))
        if code != "0":
            raise OKXError(code, str(envelope.get("msg")), path, envelope.get("data"))
        return list(envelope.get("data") or [])

    # ---------------------------------------------------------------- public

    def instruments(self, inst_type: str = "SWAP") -> list[dict[str, Any]]:
        return self._request("GET", "/api/v5/public/instruments", {"instType": inst_type})

    def tickers(self, inst_type: str = "SWAP") -> list[dict[str, Any]]:
        return self._request("GET", "/api/v5/market/tickers", {"instType": inst_type})

    def candles(self, inst_id: str, bar: str, limit: int = 100, after: int | None = None) -> list[list[str]]:
        params = {"instId": inst_id, "bar": bar, "limit": str(limit)}
        if after is not None:
            params["after"] = str(after)
        rows = self._request("GET", "/api/v5/market/history-candles", params)
        return [list(row) for row in rows]  # type: ignore[arg-type]

    # --------------------------------------------------------------- private

    def account_config(self) -> dict[str, Any]:
        data = self._request("GET", "/api/v5/account/config", private=True)
        if not data:
            raise OKXError("empty", "account config returned no rows", "/api/v5/account/config")
        return data[0]

    def balance(self) -> dict[str, Any]:
        data = self._request("GET", "/api/v5/account/balance", private=True)
        if not data:
            raise OKXError("empty", "balance returned no rows", "/api/v5/account/balance")
        return data[0]

    def positions(self, inst_type: str = "SWAP") -> list[dict[str, Any]]:
        return self._request("GET", "/api/v5/account/positions", {"instType": inst_type}, private=True)

    def set_leverage(self, inst_id: str, leverage: str, margin_mode: str = "cross") -> list[dict[str, Any]]:
        return self._request(
            "POST",
            "/api/v5/account/set-leverage",
            payload={"instId": inst_id, "lever": leverage, "mgnMode": margin_mode},
            private=True,
        )

    def order_by_client_id(self, inst_id: str, client_order_id: str) -> dict[str, Any] | None:
        """Look up one order by our own id, for idempotent retries."""
        try:
            data = self._request(
                "GET",
                "/api/v5/trade/order",
                {"instId": inst_id, "clOrdId": client_order_id},
                private=True,
            )
        except OKXError as error:
            # 51603 is "order does not exist", which is the answer, not a failure.
            if error.code == "51603":
                return None
            raise
        return data[0] if data else None

    def place_order(
        self,
        inst_id: str,
        side: str,
        size: str,
        client_order_id: str,
        reduce_only: bool,
        margin_mode: str = "cross",
    ) -> dict[str, Any]:
        payload = {
            "instId": inst_id,
            "tdMode": margin_mode,
            "side": side,
            "ordType": "market",
            "sz": size,
            "clOrdId": client_order_id,
        }
        if reduce_only:
            payload["reduceOnly"] = "true"
        data = self._request("POST", "/api/v5/trade/order", payload=payload, private=True)
        if not data:
            raise OKXError("empty", "order placement returned no rows", "/api/v5/trade/order")
        row = data[0]
        if str(row.get("sCode", "0")) != "0":
            raise OKXError(str(row.get("sCode")), str(row.get("sMsg")), "/api/v5/trade/order", row)
        return row
