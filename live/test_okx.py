from __future__ import annotations

import base64
import hashlib
import hmac
import tempfile
import unittest
from pathlib import Path

from live.okx import OKXClient
from live.test_guard import settings


class SigningTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.settings = settings(Path(self._temp.name), api_key="key", api_secret="secret", passphrase="pass")
        self.client = OKXClient(self.settings)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _expected(self, timestamp: str, method: str, path: str, body: str) -> str:
        return base64.b64encode(
            hmac.new(b"secret", f"{timestamp}{method}{path}{body}".encode("utf-8"), hashlib.sha256).digest()
        ).decode("utf-8")

    def test_public_requests_carry_no_credentials(self) -> None:
        headers = self.client._headers("GET", "/api/v5/market/tickers", "", private=False)
        self.assertNotIn("OK-ACCESS-SIGN", headers)
        self.assertNotIn("OK-ACCESS-KEY", headers)

    def test_private_requests_are_signed_over_timestamp_method_path_and_body(self) -> None:
        path = "/api/v5/trade/order"
        body = '{"instId":"BTC-USDT-SWAP"}'
        headers = self.client._headers("POST", path, body, private=True)
        timestamp = headers["OK-ACCESS-TIMESTAMP"]
        self.assertEqual(headers["OK-ACCESS-SIGN"], self._expected(timestamp, "POST", path, body))
        self.assertEqual(headers["OK-ACCESS-KEY"], "key")
        self.assertEqual(headers["OK-ACCESS-PASSPHRASE"], "pass")

    def test_the_query_string_is_part_of_the_signature(self) -> None:
        # Signing the bare path while sending a query string is the classic
        # cause of a 401 that looks like a clock problem.
        plain = "/api/v5/account/positions"
        with_query = "/api/v5/account/positions?instType=SWAP"
        headers = self.client._headers("GET", with_query, "", private=True)
        timestamp = headers["OK-ACCESS-TIMESTAMP"]
        self.assertEqual(headers["OK-ACCESS-SIGN"], self._expected(timestamp, "GET", with_query, ""))
        self.assertNotEqual(headers["OK-ACCESS-SIGN"], self._expected(timestamp, "GET", plain, ""))

    def test_the_method_is_upper_cased_in_the_signature(self) -> None:
        headers = self.client._headers("get", "/api/v5/account/balance", "", private=True)
        timestamp = headers["OK-ACCESS-TIMESTAMP"]
        self.assertEqual(headers["OK-ACCESS-SIGN"], self._expected(timestamp, "GET", "/api/v5/account/balance", ""))

    def test_the_timestamp_is_iso8601_with_milliseconds(self) -> None:
        timestamp = self.client._headers("GET", "/x", "", private=True)["OK-ACCESS-TIMESTAMP"]
        self.assertRegex(timestamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


class SimulatedHeaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.state = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_demo_trading_adds_the_simulated_header(self) -> None:
        client = OKXClient(settings(self.state, simulated=True))
        self.assertEqual(client._headers("GET", "/x", "", private=False)["x-simulated-trading"], "1")

    def test_live_trading_omits_it(self) -> None:
        client = OKXClient(settings(self.state, simulated=False))
        self.assertNotIn("x-simulated-trading", client._headers("GET", "/x", "", private=False))


if __name__ == "__main__":
    unittest.main()
