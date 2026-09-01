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


PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OKX 动量策略</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='13' font-size='13'>&#128200;</text></svg>">
<style>
  :root {
    color-scheme: dark;
    --bg: #0b0d10;
    --bg-2: #12151a;
    --card: #171b22;
    --card-2: #1d232c;
    --line: #2a3140;
    --fg: #e8edf5;
    --muted: #8b95a8;
    --dim: #5c6576;
    --pos: #3dd68c;
    --pos-bg: rgba(61, 214, 140, 0.12);
    --neg: #ff6b6b;
    --neg-bg: rgba(255, 107, 107, 0.12);
    --warn: #f5c15a;
    --warn-bg: rgba(245, 193, 90, 0.12);
    --live: #ff6b6b;
    --accent: #7aa2ff;
    --shadow: 0 12px 40px rgba(0,0,0,.35);
    --radius: 14px;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.55 "PingFang SC", "Hiragino Sans GB", "Noto Sans SC", "Microsoft YaHei", ui-sans-serif, system-ui, sans-serif; }
  body { min-height: 100vh; }
  header {
    position: sticky; top: 0; z-index: 10;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    padding: 14px 22px;
    background: rgba(11,13,16,.86);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--line);
  }
  .brand { display: flex; align-items: baseline; gap: 10px; }
  h1 { margin: 0; font-size: 17px; font-weight: 650; letter-spacing: .02em; }
  .sub { color: var(--muted); font-size: 12px; }
  .spacer { flex: 1; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; padding: 4px 10px; border-radius: 999px;
    border: 1px solid var(--line); color: var(--muted); background: var(--card);
    white-space: nowrap;
  }
  .pill .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--dim); }
  .pill.live { color: #fff; background: #8b1e1e; border-color: #b3261e; }
  .pill.live .dot { background: #ffb4b4; }
  .pill.dry { color: var(--warn); background: var(--warn-bg); border-color: transparent; }
  .pill.halt { color: #fff; background: var(--neg); border-color: var(--neg); }
  .pill.ok { color: var(--pos); background: var(--pos-bg); border-color: transparent; }
  main { padding: 20px 22px 40px; display: grid; gap: 16px; max-width: 1240px; margin: 0 auto; }
  .kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
  .card {
    background: linear-gradient(180deg, var(--card) 0%, var(--bg-2) 100%);
    border: 1px solid var(--line); border-radius: var(--radius);
    padding: 16px 18px; box-shadow: var(--shadow);
  }
  .kpi { padding: 16px 18px; }
  .kpi .label { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
  .kpi .value { font-size: 26px; font-weight: 650; font-variant-numeric: tabular-nums; letter-spacing: -0.03em; }
  .kpi .hint { color: var(--dim); font-size: 12px; margin-top: 4px; }
  .spark { width: 100%; height: 36px; margin-top: 10px; display: block; }
  .grid-2 { display: grid; grid-template-columns: 1.35fr .65fr; gap: 16px; }
  .card h2 { margin: 0 0 12px; font-size: 13px; font-weight: 650; color: var(--muted); letter-spacing: .06em; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }
  th { color: var(--dim); font-size: 12px; font-weight: 600; }
  td { font-size: 13px; }
  tr:last-child td { border-bottom: 0; }
  .scroll { overflow-x: auto; }
  .pos { color: var(--pos); } .neg { color: var(--neg); }
  .tag {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600;
  }
  .tag.long { background: var(--pos-bg); color: var(--pos); }
  .tag.short { background: var(--neg-bg); color: var(--neg); }
  .tag.neutral { background: var(--card-2); color: var(--muted); }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; }
  .chip { padding: 5px 10px; border-radius: 8px; background: var(--card-2); border: 1px solid var(--line); font-size: 12px; }
  .chip.out { color: var(--muted); }
  .signal-rows { display: grid; gap: 8px; }
  .signal-row { display: flex; justify-content: space-between; gap: 12px; font-size: 13px; }
  .signal-row span { color: var(--muted); }
  .signal-row b { font-weight: 600; text-align: right; }
  .empty { color: var(--dim); padding: 12px 0; }
  .status-sent { color: var(--pos); }
  .status-rejected, .status-failed { color: var(--neg); }
  .status-pending { color: var(--warn); }
  .muted { color: var(--muted); }
  .detail { color: var(--muted); white-space: normal; max-width: 520px; }
  @media (max-width: 980px) {
    .kpis, .grid-2 { grid-template-columns: 1fr 1fr; }
  }
  @media (max-width: 640px) {
    .kpis, .grid-2 { grid-template-columns: 1fr; }
    header { padding: 12px 14px; }
    main { padding: 14px; }
    .kpi .value { font-size: 22px; }
  }
</style>
<header>
  <div class="brand">
    <h1>OKX 动量策略</h1>
    <span class="sub">只读监控 · 每 15 秒刷新</span>
  </div>
  <span id="mode" class="pill"><span class="dot"></span>加载中</span>
  <span id="halt" class="pill halt" hidden>已紧急停止</span>
  <span class="spacer"></span>
  <span id="updated" class="sub"></span>
</header>
<main>
  <section class="kpis">
    <div class="card kpi"><div class="label">账户权益</div><div class="value" id="eq">—</div><div class="hint" id="eqHint">USDT</div><svg class="spark" id="spark" viewBox="0 0 100 36" preserveAspectRatio="none"></svg></div>
    <div class="card kpi"><div class="label">今日盈亏</div><div class="value" id="pnl">—</div><div class="hint" id="pnlHint">相对开盘权益</div></div>
    <div class="card kpi"><div class="label">总名义</div><div class="value" id="gross">—</div><div class="hint" id="grossHint">多空绝对值之和</div></div>
    <div class="card kpi"><div class="label">实际杠杆</div><div class="value" id="lev">—</div><div class="hint" id="levHint">总名义 / 权益</div></div>
    <div class="card kpi"><div class="label">运行状态</div><div class="value" id="run">—</div><div class="hint" id="runHint">仓位模式 · 策略</div></div>
  </section>
  <section class="grid-2">
    <div class="card">
      <h2>当前仓位</h2>
      <div class="scroll"><table id="book"></table></div>
    </div>
    <div class="card">
      <h2>今日信号</h2>
      <div id="signal" class="signal-rows"></div>
    </div>
  </section>
  <div class="card">
    <h2>最近订单</h2>
    <div class="scroll"><table id="orders"></table></div>
  </div>
  <div class="card">
    <h2>可交易标的</h2>
    <div id="universe"></div>
  </div>
  <div class="card">
    <h2>事件日志</h2>
    <div class="scroll"><table id="events"></table></div>
  </div>
</main>
<script>
const KIND = {
  PREFLIGHT: "启动检查", STARTED: "执行器启动", SIGNAL: "每日信号", PLAN: "下单计划",
  PLAN_SKIPS: "跳过标的", ORDER_SENT: "订单已发送", ORDER_REJECTED: "订单被拒",
  ORDER_FAILED: "下单失败", ORDER_DRY_RUN: "演练下单", ORDER_ALREADY_PLACED: "订单已存在",
  TICK_ERROR: "循环异常", POSITION_STOP: "止损平仓", DAILY_KILL: "日内熔断",
  HALT: "紧急停止", REBALANCE_SKIPPED: "跳过再平衡",
};
const REASON = {
  REBALANCE_OPEN: "开仓", REBALANCE_CLOSE: "减仓", REBALANCE_REDUCE: "减仓",
  POSITION_STOP: "止损", HALT: "紧急停止", DAILY_KILL: "日内熔断",
};
const STATUS = { SENT: "已发送", REJECTED: "已拒绝", PENDING: "发送中", FAILED: "失败", DRY_RUN: "演练" };
const REGIME = {
  NEUTRAL: "中性", UP_TREND: "上升趋势", DOWN_TREND: "下降趋势",
  UP_SHOCK: "向上冲击", DOWN_SHOCK: "向下冲击", UP_SHOCK_FLAT: "冲击后空仓",
  INSUFFICIENT_DATA: "数据不足", INSUFFICIENT_BTC_DATA: "BTC 数据不足",
};
const fmt = (n, d=2) => (n===null || n===undefined || Number.isNaN(Number(n))) ? "—" : Number(n).toFixed(d);
const el = id => document.getElementById(id);
const coin = id => String(id || "").replace("-USDT-SWAP", "");
const cls = (n) => n > 0 ? "pos" : n < 0 ? "neg" : "";
function zhTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso).replace("T", " ").slice(0, 19);
  return d.toLocaleString("zh-CN", { timeZone: "Asia/Shanghai", hour12: false,
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function table(node, columns, rows, emptyText) {
  if (!rows || !rows.length) {
    node.innerHTML = `<tr><td class="empty" colspan="${columns.length}">${emptyText || "暂无数据"}</td></tr>`;
    return;
  }
  node.innerHTML = "<tr>" + columns.map(c => `<th>${c.label}</th>`).join("") + "</tr>"
    + rows.map(r => "<tr>" + columns.map(c => `<td>${c.get(r)}</td>`).join("") + "</tr>").join("");
}
function sparkline(svg, series) {
  if (!svg) return;
  const vals = (series || []).map(x => Number(x.equity_usdt)).filter(n => Number.isFinite(n));
  if (vals.length < 2) { svg.innerHTML = ""; return; }
  const min = Math.min(...vals), max = Math.max(...vals), span = max - min || 1;
  const w = 100, h = 36, p = 2;
  const pts = vals.map((v, i) => {
    const x = p + (i / (vals.length - 1)) * (w - p * 2);
    const y = h - p - ((v - min) / span) * (h - p * 2);
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  const up = vals[vals.length - 1] >= vals[0];
  svg.innerHTML = `<polyline fill="none" stroke="${up ? "#3dd68c" : "#ff6b6b"}" stroke-width="1.6" points="${pts}" />`;
}
function rejectReason(text) {
  const m = String(text || "").match(/min order ([\d.]+) USDT exceeds target ([\d.]+) USDT/);
  if (m) return `最小下单 ${m[1]} USDT，超过目标仓 ${m[2]} USDT`;
  return text || "";
}
function summarize(kind, payload) {
  const p = payload || {};
  if (kind === "ORDER_SENT" || kind === "ORDER_REJECTED" || kind === "ORDER_FAILED" || kind === "ORDER_DRY_RUN") {
    const side = p.side === "buy" ? "买" : p.side === "sell" ? "卖" : "";
    const extra = p.guard ? ` · ${p.guard}` : p.msg ? ` · ${p.msg}` : "";
    return `${side}${coin(p.instId || p.inst_id)} ${p.contracts || ""} 张 · ${fmt(p.notionalUsdt || p.notional_usdt)} USDT${extra}`;
  }
  if (kind === "SIGNAL") {
    const d = p.detail || p;
    return `${REGIME[d.regime] || d.regime || ""} · 多 ${(d.longs||[]).map(coin).join(" / ") || "无"} · 空 ${(d.shorts||[]).map(coin).join(" / ") || "无"}`;
  }
  if (kind === "PREFLIGHT") return `权益 ${fmt(p.equityUsdt)} · 杠杆 ${p.leverage}x · 单向持仓`;
  if (kind === "STARTED") return p.dryRun ? "演练模式，不会发单" : "实盘模式，订单会发到交易所";
  if (kind === "PLAN") return `${(p.orders || []).length} 笔计划订单`;
  if (kind === "TICK_ERROR") return String(p.error || JSON.stringify(p));
  if (kind === "POSITION_STOP") return `${coin(p.instId)} 触发止损 @ ${fmt(p.price, 4)}`;
  if (kind === "DAILY_KILL") return `亏损 ${fmt((p.loss || 0) * 100)}%`;
  const s = JSON.stringify(p);
  return s.length > 140 ? s.slice(0, 140) + "…" : s;
}

async function refresh() {
  let data;
  try { data = await (await fetch("/api/state")).json(); }
  catch { el("updated").textContent = "面板暂时连不上"; return; }
  if (!data.ready) { el("mode").innerHTML = `<span class="dot"></span>${data.reason}`; return; }

  const pre = data.preflight || {};
  const live = !pre.dryRun && !pre.simulated;
  const posture = pre.dryRun ? "演练 · 不会发单" : pre.simulated ? "模拟盘" : "实盘";
  el("mode").className = "pill " + (live ? "live" : "dry");
  el("mode").innerHTML = `<span class="dot"></span>${posture} · ${pre.variant || "r9.2"} · ${pre.leverage || "—"}x`;
  el("halt").hidden = !data.halted;
  el("updated").textContent = "更新于 " + new Date().toLocaleTimeString("zh-CN", { hour12: false });

  const equity = data.equity || [];
  const last = equity.length ? equity[equity.length - 1] : null;
  const now = last ? last.equity_usdt : (pre.equityUsdt ?? null);
  const day = (data.tradingDays || [])[0];
  const open = day ? day.opening_equity_usdt : null;
  const dayPct = (now !== null && open) ? (now / open - 1) * 100 : null;
  const gross = last ? last.gross_notional_usdt : null;
  const levNow = (now && gross) ? gross / now : null;

  el("eq").textContent = fmt(now);
  el("eqHint").textContent = "USDT · 开盘 " + fmt(open);
  el("pnl").textContent = dayPct === null ? "—" : ((dayPct >= 0 ? "+" : "") + fmt(dayPct) + "%");
  el("pnl").className = "value " + (dayPct === null ? "" : cls(dayPct));
  el("pnlHint").textContent = day && day.killed ? "今日熔断已触发" : "相对开盘权益";
  el("gross").textContent = fmt(gross);
  el("grossHint").textContent = "USDT · 多空绝对值之和";
  el("lev").textContent = levNow === null ? "—" : fmt(levNow) + "x";
  el("levHint").textContent = "设定 " + (pre.leverage || "—") + "x · 上限见配置";
  el("run").textContent = data.halted ? "已停止" : "运行中";
  el("run").className = "value " + (data.halted ? "neg" : "pos");
  el("runHint").textContent = (pre.posMode === "net_mode" ? "单向持仓" : (pre.posMode || "仓位模式未知"))
    + " · " + (pre.variant || "");
  sparkline(el("spark"), equity.slice(-180));

  const signal = data.signal || {};
  const target = signal.target || {};
  const stops = {};
  (data.stopState || []).forEach(s => { stops[s.inst_id] = s; });
  const names = [...new Set([...Object.keys(target), ...Object.keys(stops)])].sort();
  table(el("book"), [
    {label: "标的", get: r => coin(r)},
    {label: "方向", get: r => {
      const t = target[r];
      const s = stops[r];
      const dir = t !== undefined ? t : (s ? s.direction : 0);
      if (dir > 0) return `<span class="tag long">多</span>`;
      if (dir < 0) return `<span class="tag short">空</span>`;
      return `<span class="tag neutral">平</span>`;
    }},
    {label: "目标权重", get: r => {
      const v = target[r];
      return v === undefined ? "—" : `<span class="${cls(v)}">${fmt(v * 100, 1)}%</span>`;
    }},
    {label: "入场价", get: r => stops[r] ? fmt(stops[r].entry, 4) : "—"},
    {label: "止损幅度", get: r => stops[r] ? fmt((stops[r].stop_fraction || 0) * 100, 1) + "%" : "—"},
    {label: "当日高/低", get: r => stops[r] ? `${fmt(stops[r].peak, 4)} / ${fmt(stops[r].trough, 4)}` : "—"},
  ], names, "还没有仓位");

  const d = signal.detail || {};
  const longs = (d.longs || []).map(coin).join("、") || "无";
  const shorts = (d.shorts || []).map(coin).join("、") || "无";
  el("signal").innerHTML = d.regime ? [
    ["市场状态", REGIME[d.regime] || d.regime],
    ["做多", longs],
    ["做空", shorts],
    ["多头目标", fmt((d.longGrossTarget || 0) * 100, 0) + "% 权益"],
    ["空头目标", fmt((d.shortGrossTarget || 0) * 100, 0) + "% 权益"],
    ["总名义", fmt(d.grossAfterCap, 2) + "x"],
    ["BTC 1日", fmt((d.btcReturn1 || 0) * 100) + "%"],
    ["信号时间", zhTime(signal.ts)],
  ].map(([k,v]) => `<div class="signal-row"><span>${k}</span><b>${v}</b></div>`).join("")
    : `<div class="empty">还没有当日信号</div>`;

  table(el("orders"), [
    {label: "时间", get: o => zhTime(o.ts)},
    {label: "标的", get: o => coin(o.inst_id)},
    {label: "方向", get: o => `<span class="${o.side==="buy"?"pos":"neg"}">${o.side==="buy"?"买入":"卖出"}</span>`},
    {label: "张数", get: o => o.contracts},
    {label: "名义 USDT", get: o => fmt(o.notional_usdt)},
    {label: "原因", get: o => REASON[o.reason] || o.reason},
    {label: "状态", get: o => `<span class="status-${(o.status||"").toLowerCase()}">${STATUS[o.status] || o.status}</span>`},
  ], data.orders || [], "还没有订单");

  const usable = pre.usableUniverse || [];
  const rejected = pre.rejected || {};
  const chips = usable.map(u => `<span class="chip">${coin(u)}</span>`).join("")
    + Object.entries(rejected).map(([k,v]) => `<span class="chip out">${coin(k)} · ${rejectReason(v)}</span>`).join("");
  el("universe").innerHTML =
    `<p class="muted" style="margin:0 0 10px">当前每只目标约 ${fmt(pre.perPositionNotionalUsdt)} USDT；可交易 ${usable.length} 个`
    + `${Object.keys(rejected).length ? `，排除 ${Object.keys(rejected).length} 个（最小下单量大于目标仓）` : ""}。</p>`
    + `<div class="chips">${chips || '<span class="empty">尚未完成启动检查</span>'}</div>`;

  table(el("events"), [
    {label: "时间", get: e => zhTime(e.ts)},
    {label: "类型", get: e => KIND[e.kind] || e.kind},
    {label: "说明", get: e => `<span class="detail">${summarize(e.kind, e.payload)}</span>`},
  ], data.events || [], "还没有事件");
}
refresh();
setInterval(refresh, 15000);
</script>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE
