"""Read-only portal over the executor's event log.

Runs as a separate process from the executor and opens the SQLite file in
read-only mode. It holds no API credentials and can place no orders: the worst
a compromised portal can do is show you the data it already shows you.

    uvicorn portal.app:app --host 127.0.0.1 --port 8787
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from live.settings import ROOT, load_dotenv
import os


load_dotenv()
STATE_DIR = Path(os.environ.get("OKXRUN_STATE_DIR") or (ROOT / "state"))
DATABASE = STATE_DIR / "okxrun.sqlite3"
HALT_FILE = Path(os.environ.get("OKXRUN_HALT_FILE") or (STATE_DIR / "HALT"))

app = FastAPI(title="okxrun portal", docs_url=None, redoc_url=None)


def _connect() -> sqlite3.Connection:
    # Read-only URI: the portal physically cannot write to the executor's state.
    connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _rows(connection: sqlite3.Connection, sql: str, *args: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(sql, args).fetchall()]


def _latest_payload(connection: sqlite3.Connection, kind: str) -> dict[str, Any] | None:
    rows = _rows(
        connection, "SELECT ts, payload FROM events WHERE kind = ? ORDER BY id DESC LIMIT 1", kind
    )
    if not rows:
        return None
    return {"ts": rows[0]["ts"], **json.loads(rows[0]["payload"])}


@app.get("/api/state")
def state() -> dict[str, Any]:
    if not DATABASE.exists():
        return {"ready": False, "reason": f"no database at {DATABASE}"}
    with _connect() as connection:
        equity = _rows(
            connection,
            "SELECT ts, equity_usdt, gross_notional_usdt FROM equity_marks ORDER BY ts DESC LIMIT 2880",
        )
        orders = _rows(connection, "SELECT * FROM orders ORDER BY ts DESC LIMIT 100")
        events = [
            {"ts": row["ts"], "kind": row["kind"], "payload": json.loads(row["payload"])}
            for row in _rows(connection, "SELECT ts, kind, payload FROM events ORDER BY id DESC LIMIT 200")
        ]
        days = _rows(connection, "SELECT * FROM trading_days ORDER BY trading_day DESC LIMIT 30")
        positions = _rows(connection, "SELECT * FROM position_state ORDER BY inst_id")
        return {
            "ready": True,
            "halted": HALT_FILE.exists(),
            "preflight": _latest_payload(connection, "PREFLIGHT"),
            "signal": _latest_payload(connection, "SIGNAL"),
            "plan": _latest_payload(connection, "PLAN"),
            "equity": list(reversed(equity)),
            "orders": orders,
            "events": events,
            "tradingDays": days,
            "stopState": positions,
        }


PAGE = """
<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>okxrun</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='13' font-size='13'>&#128200;</text></svg>">
<style>
  :root { color-scheme: light dark; --fg:#111; --muted:#666; --bg:#fafafa; --card:#fff; --line:#e3e3e3;
          --pos:#0a7d33; --neg:#b3261e; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8e8e8; --muted:#9a9a9a; --bg:#161616; --card:#1f1f1f; --line:#333; --pos:#4ade80; --neg:#f87171; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif; }
  header { padding:16px 20px; border-bottom:1px solid var(--line); display:flex; gap:14px; align-items:baseline; flex-wrap:wrap; }
  h1 { font-size:16px; margin:0; font-weight:650; letter-spacing:-0.01em; }
  main { padding:20px; display:grid; gap:16px; max-width:1100px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted);
             margin:0 0 10px; font-weight:600; }
  .row { display:flex; gap:24px; flex-wrap:wrap; }
  .stat b { display:block; font-size:22px; font-variant-numeric:tabular-nums; font-weight:600; }
  .stat span { color:var(--muted); font-size:12px; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  th,td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { color:var(--muted); font-weight:600; font-size:12px; }
  .scroll { overflow-x:auto; }
  .pos { color:var(--pos); } .neg { color:var(--neg); }
  .pill { font-size:11px; padding:2px 8px; border-radius:99px; border:1px solid var(--line); color:var(--muted); }
  .pill.alarm { color:#fff; background:var(--neg); border-color:var(--neg); }
  pre { margin:0; font-size:12px; white-space:pre-wrap; word-break:break-word; color:var(--muted); }
</style>
<header>
  <h1>okxrun</h1>
  <span id="mode" class="pill">loading</span>
  <span id="halt" class="pill" hidden>HALTED</span>
  <span id="updated" style="margin-left:auto;color:var(--muted);font-size:12px"></span>
</header>
<main>
  <div class="card"><h2>Account</h2><div class="row" id="stats"></div></div>
  <div class="card"><h2>Tradable universe</h2><div id="universe"></div></div>
  <div class="card"><h2>Target vs held</h2><div class="scroll"><table id="book"></table></div></div>
  <div class="card"><h2>Latest signal</h2><pre id="signal"></pre></div>
  <div class="card"><h2>Orders</h2><div class="scroll"><table id="orders"></table></div></div>
  <div class="card"><h2>Events</h2><div class="scroll"><table id="events"></table></div></div>
</main>
<script>
const fmt = (n, d=2) => (n===null||n===undefined||Number.isNaN(n)) ? "-" : Number(n).toFixed(d);
const el = id => document.getElementById(id);

function table(node, columns, rows) {
  const head = "<tr>" + columns.map(c => `<th>${c.label}</th>`).join("") + "</tr>";
  const body = rows.map(r => "<tr>" + columns.map(c => `<td>${c.get(r)}</td>`).join("") + "</tr>").join("");
  node.innerHTML = head + body;
}

async function refresh() {
  let data;
  try { data = await (await fetch("/api/state")).json(); }
  catch { el("updated").textContent = "portal unreachable"; return; }
  if (!data.ready) { el("mode").textContent = data.reason; return; }

  const pre = data.preflight || {};
  // Only one of these three states is actually moving money; say which plainly.
  const posture = pre.dryRun ? "DRY RUN · nothing sent"
                : pre.simulated ? "LIVE · demo account"
                : "LIVE · real money";
  el("mode").textContent = [posture, pre.variant || "", pre.leverage ? pre.leverage + "x" : ""]
    .filter(Boolean).join(" · ");
  el("mode").className = "pill" + (!pre.dryRun && !pre.simulated ? " alarm" : "");
  el("halt").hidden = !data.halted;
  el("updated").textContent = "updated " + new Date().toLocaleTimeString();

  const equity = data.equity || [];
  const now = equity.length ? equity[equity.length-1].equity_usdt : (pre.equityUsdt ?? null);
  const day = (data.tradingDays || [])[0];
  const open = day ? day.opening_equity_usdt : null;
  const dayPct = (now !== null && open) ? (now/open - 1) * 100 : null;
  el("stats").innerHTML = `
    <div class="stat"><b>${fmt(now)}</b><span>equity USDT</span></div>
    <div class="stat"><b class="${dayPct===null?"":dayPct>=0?"pos":"neg"}">${dayPct===null?"-":fmt(dayPct)+"%"}</b><span>today</span></div>
    <div class="stat"><b>${fmt(equity.length ? equity[equity.length-1].gross_notional_usdt : null)}</b><span>gross notional</span></div>
    <div class="stat"><b>${day && day.killed ? "YES" : "no"}</b><span>kill switch fired</span></div>`;

  // At a small balance this is the section that explains a thin book: an
  // instrument whose minimum order exceeds its share simply cannot be traded.
  const usable = pre.usableUniverse || [];
  const rejected = pre.rejected || {};
  const rejectedRows = Object.entries(rejected)
    .map(([k, v]) => `<tr><td>${k}</td><td style="white-space:normal;color:var(--muted)">${v}</td></tr>`).join("");
  el("universe").innerHTML =
    `<p style="margin:0 0 8px">${usable.length} tradable at ${fmt(pre.perPositionNotionalUsdt)} USDT per position`
    + `${Object.keys(rejected).length ? `, ${Object.keys(rejected).length} excluded` : ""}.</p>`
    + `<div class="scroll"><table>${usable.map(u => `<tr><td>${u}</td><td class="pos">tradable</td></tr>`).join("")}${rejectedRows}</table></div>`;

  const signal = data.signal || {};
  const target = signal.target || {};
  const current = signal.previousWeights || {};
  const names = [...new Set([...Object.keys(target), ...Object.keys(current)])].sort();
  table(el("book"),
    [
      {label:"instrument", get:r=>r},
      {label:"target", get:r=>{const v=target[r]; return v===undefined?"-":`<span class="${v>=0?"pos":"neg"}">${fmt(v*100)}%</span>`;}},
      {label:"held", get:r=>{const v=current[r]; return v===undefined?"-":`<span class="${v>=0?"pos":"neg"}">${fmt(v*100)}%</span>`;}},
    ], names);

  el("signal").textContent = JSON.stringify(signal.detail || {}, null, 1);

  table(el("orders"),
    [
      {label:"time", get:o=>o.ts.replace("T"," ").slice(0,19)},
      {label:"instrument", get:o=>o.inst_id},
      {label:"side", get:o=>`<span class="${o.side==="buy"?"pos":"neg"}">${o.side}</span>`},
      {label:"contracts", get:o=>o.contracts},
      {label:"USDT", get:o=>fmt(o.notional_usdt)},
      {label:"reduce", get:o=>o.reduce_only?"yes":""},
      {label:"why", get:o=>o.reason},
      {label:"status", get:o=>o.status},
    ], data.orders || []);

  table(el("events"),
    [
      {label:"time", get:e=>e.ts.replace("T"," ").slice(0,19)},
      {label:"kind", get:e=>e.kind},
      {label:"detail", get:e=>{
        const s = JSON.stringify(e.payload);
        return s.length > 160 ? s.slice(0,160)+"…" : s;
      }},
    ], data.events || []);
}

refresh();
setInterval(refresh, 15000);
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE
