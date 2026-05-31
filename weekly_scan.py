"""Weekly portfolio scan — cost-optimized to stay under $0.50 per full run.

Cost controls applied (from the post-mortem of the first paid runs):
  - Haiku scan: `max_uses=1` on web_search so each position costs exactly one search.
  - Sonnet fee validation: only fires for positions flagged TRIM/SELL AND ≥3% of portfolio.
    Small positions get the preliminary verdict as-is; fee drag dominates anyway.
  - Opportunity hunt: `max_uses=5` total — the model is instructed to think harder
    per query rather than search more.
  - `max_retries=0` on the client. 429s and 400s fail loud instead of doubling spend.
  - Live cost tracker reads `response.usage` + counts `server_tool_use` blocks and
    prints estimated spend at the end of every run.

Optional integrations:
  - Telegram: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to get the report
    pushed to your phone when the scan finishes.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import anthropic
import requests
from dotenv import load_dotenv

from portfolio_scan import find_positions_file, parse_positions_xls

load_dotenv()

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

# Per-position web_search budget on the Haiku scan. The model is also instructed in the
# system prompt to use exactly one query; max_uses is the hard cap.
HAIKU_SEARCH_BUDGET = 1
# Total web_search budget for the free-form opportunity hunt.
OPPORTUNITY_SEARCH_BUDGET = 5

HAIKU_WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    # Haiku 4.5 can't drive programmatic tool calling, so route the tool directly.
    "allowed_callers": ["direct"],
    "max_uses": HAIKU_SEARCH_BUDGET,
}
SONNET_FEE_WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    # The fee call needs to look up Swissquote pricing + FX spread once or twice.
    "max_uses": 3,
}
SONNET_OPP_WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": OPPORTUNITY_SEARCH_BUDGET,
}

# Pricing per 1M tokens. See platform.claude.com/docs/en/pricing.
PRICES_PER_TOKEN = {
    HAIKU: {"input": 1.0e-6, "output": 5.0e-6, "cache_read": 0.1e-6, "cache_write": 1.25e-6},
    SONNET: {"input": 3.0e-6, "output": 15.0e-6, "cache_read": 0.3e-6, "cache_write": 3.75e-6},
}
WEB_SEARCH_COST_PER_QUERY = 0.01  # $10 / 1,000 queries.

# Positions below this fraction of total portfolio value skip the Sonnet fee-validation
# step. The preliminary Haiku verdict stands. Fee math is dominated by Swissquote's
# minimum commission anyway, so revalidating a 0.5% position isn't worth a Sonnet call.
MATERIAL_POSITION_THRESHOLD = 0.03


# ──────────────────────────── system prompts ────────────────────────────────


SCANNER_SYSTEM = """You are a buy-side equity analyst doing rapid weekly portfolio triage.

For each position the user gives you:
1. Call web_search EXACTLY ONCE to find the current market price and the single most material headline from the last 14 days.
2. Render a verdict.

Hard rules:
- Use web_search at most ONE time. If the single search doesn't give you enough, output your best assessment anyway based on what you found plus the cost-basis and P&L the user supplied.
- No follow-up searches. No clarifications. No browsing.
- Be terse. 1-2 sentence reasoning. No disclaimers, no boilerplate.

Verdicts:
- HOLD: thesis intact, no action
- ADD: thesis strengthening or pullback into support
- TRIM: take partial profit OR partial de-risk on thesis impairment
- SELL: full exit — thesis broken or capital better deployed

Output structured JSON. Schema is enforced."""

OPPORTUNITY_SYSTEM = f"""You are a sharp generalist investor surfacing 3 high-conviction ideas to buy now.

CRITICAL BUDGET CONSTRAINT: You have a hard cap of {OPPORTUNITY_SEARCH_BUDGET} web_search queries TOTAL across the entire response. Spend them deliberately:
- Plan your searches BEFORE issuing the first one.
- Each query should cover a broad-but-pointed theme (e.g. "Q1 2026 unloved small-cap industrial with insider buying"), not single-ticker confirmation.
- Lean on what you already know about market structure, sector dynamics, and recent earnings reactions. The searches are for fresh, dated facts you genuinely need to verify or discover — not background knowledge.

Required mix in your 3 picks:
- At least 1 SHORT-TERM CATALYST: earnings / regulatory / data / M&A inflection inside 4-12 weeks
- At least 1 LONG-TERM STRUCTURAL: positioned in front of a multi-year secular shift the market hasn't priced. Examples in hindsight: Western Digital pre-storage-cycle, Dell pre-AI-server, EMCOR pre-data-center buildout.

Hard rules:
- NO mega-caps (NVDA, AAPL, MSFT, AMZN, GOOG, META, TSLA) unless you have a genuinely non-consensus angle.
- NO names already in the user's portfolio (the user will tell you which).
- NO diversification boilerplate, NO "consult your advisor", NO disclaimers.
- Be specific: ticker, current price, entry range, thesis in 3-5 sentences, key risk, timing.

Output structured JSON. Schema is enforced."""


# ──────────────────────────── output schemas ────────────────────────────────


SCANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "current_price": {"type": "number", "description": "Latest market price from web_search, in trade currency"},
        "top_headline": {"type": "string", "description": "Single most material headline from the last 14 days (or 'no recent news' if none surfaced)"},
        "verdict": {"type": "string", "enum": ["HOLD", "ADD", "TRIM", "SELL"]},
        "reasoning": {"type": "string", "description": "1-2 sentence rationale"},
    },
    "required": ["current_price", "top_headline", "verdict", "reasoning"],
    "additionalProperties": False,
}

FEE_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "swissquote_fees_summary": {"type": "string"},
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "preliminary_verdict": {"type": "string"},
                    "final_verdict": {"type": "string", "enum": ["HOLD", "TRIM", "SELL"]},
                    "estimated_trading_fee_chf": {"type": "number"},
                    "estimated_fx_cost_chf": {"type": "number"},
                    "estimated_net_pnl_chf": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "symbol",
                    "preliminary_verdict",
                    "final_verdict",
                    "estimated_trading_fee_chf",
                    "estimated_fx_cost_chf",
                    "estimated_net_pnl_chf",
                    "rationale",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["swissquote_fees_summary", "decisions"],
    "additionalProperties": False,
}

OPPORTUNITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "type": {"type": "string", "enum": ["short_term_catalyst", "long_term_structural"]},
                    "current_price": {"type": "number"},
                    "entry_price": {"type": "string"},
                    "thesis": {"type": "string"},
                    "key_risk": {"type": "string"},
                    "catalyst_timeline": {"type": "string"},
                },
                "required": ["ticker", "type", "current_price", "entry_price", "thesis", "key_risk", "catalyst_timeline"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["picks"],
    "additionalProperties": False,
}


# ──────────────────────────── cost tracker ──────────────────────────────────


class CostTracker:
    """Aggregates token + web_search spend across all API calls in a run."""

    def __init__(self) -> None:
        self.total: float = 0.0
        self.calls: list[dict[str, Any]] = []

    def record(self, model: str, response: Any, label: str) -> float:
        usage = getattr(response, "usage", None)
        prices = PRICES_PER_TOKEN.get(model, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
        input_tok = getattr(usage, "input_tokens", 0) or 0
        output_tok = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

        tok_cost = (
            input_tok * prices["input"]
            + output_tok * prices["output"]
            + cache_read * prices["cache_read"]
            + cache_write * prices["cache_write"]
        )

        # Count web_search invocations by scanning content blocks.
        ws_count = sum(
            1
            for b in response.content
            if getattr(b, "type", None) == "server_tool_use" and getattr(b, "name", None) == "web_search"
        )
        ws_cost = ws_count * WEB_SEARCH_COST_PER_QUERY

        cost = tok_cost + ws_cost
        self.total += cost
        self.calls.append({
            "label": label,
            "model": model,
            "input": input_tok,
            "output": output_tok,
            "cache_read": cache_read,
            "web_searches": ws_count,
            "cost": cost,
        })
        return cost

    def summary(self) -> str:
        n = len(self.calls)
        ws = sum(c["web_searches"] for c in self.calls)
        return (
            f"Estimated spend: ${self.total:.4f} "
            f"({n} API calls, {ws} web searches)"
        )


# ──────────────────────────── API helpers ───────────────────────────────────


def last_text_block(response: Any) -> str:
    texts = [b.text for b in response.content if b.type == "text"]
    return texts[-1] if texts else ""


def parse_json_or_fallback(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {**fallback, "_raw": text[:500]}


def scan_position(client: anthropic.Anthropic, pos: dict[str, Any], cost: CostTracker) -> dict[str, Any]:
    pnl = pos["gv_pct_chf"]
    pct = pos["position_pct"]
    user = (
        f"Position: {pos['symbol']}  ({pos['section']})\n"
        f"Quantity: {pos['anzahl']}\n"
        f"Cost basis: {pos['einstandskurs']} {pos['waehrung']}\n"
        f"Recorded price: {pos['preis']} {pos['waehrung']}\n"
        f"Portfolio weight: {pct:.2%}   Unrealized P&L (CHF): {pnl:+.2%}\n\n"
        "Use web_search ONCE for the current price and the single biggest headline from the last 14 days. "
        "Then return your verdict."
    )
    response = client.messages.create(
        model=HAIKU,
        max_tokens=500,
        system=SCANNER_SYSTEM,
        tools=[HAIKU_WEB_SEARCH_TOOL],
        output_config={"format": {"type": "json_schema", "schema": SCANNER_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    cost.record(HAIKU, response, f"scan {pos['symbol']}")
    return parse_json_or_fallback(
        last_text_block(response),
        fallback={"verdict": "UNKNOWN", "top_headline": "parse failure", "current_price": pos["preis"], "reasoning": ""},
    )


def decide_flagged(
    client: anthropic.Anthropic, flagged: list[dict[str, Any]], cost: CostTracker
) -> dict[str, Any]:
    payload = [
        {
            "symbol": s["position"]["symbol"],
            "quantity": s["position"]["anzahl"],
            "cost_basis": s["position"]["einstandskurs"],
            "currency": s["position"]["waehrung"],
            "current_price": s.get("current_price", s["position"]["preis"]),
            "portfolio_weight_pct": s["position"]["position_pct"],
            "preliminary_verdict": s.get("verdict"),
            "preliminary_reasoning": s.get("reasoning", ""),
        }
        for s in flagged
    ]
    user = (
        "Before I act on the following TRIM/SELL flags I need fee math validated against "
        "Swissquote's current schedule. Account denominated in CHF.\n\n"
        "Step 1: web_search Swissquote's current trading commission tiers (per-trade fees by "
        "order value, including minimums and exchange surcharges) and their FX spread when "
        "converting USD/EUR back to CHF. Budget: 3 searches max.\n\n"
        "Step 2: For each position compute trading fee, FX cost, and net realized P&L in CHF "
        "vs cost basis.\n\n"
        "Step 3: Confirm or downgrade the verdict. A TRIM/SELL only stands if (a) the thesis "
        "is genuinely impaired, (b) it's a loss cut freeing capital, or (c) gains are large "
        "enough that fees are immaterial. Otherwise downgrade to HOLD.\n\n"
        f"Positions:\n```json\n{json.dumps(payload, indent=2)}\n```\n\n"
        "Return structured JSON."
    )
    response = client.messages.create(
        model=SONNET,
        max_tokens=1800,
        tools=[SONNET_FEE_WEB_SEARCH_TOOL],
        output_config={"format": {"type": "json_schema", "schema": FEE_DECISION_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    cost.record(SONNET, response, "fee decision")
    return parse_json_or_fallback(
        last_text_block(response),
        fallback={"swissquote_fees_summary": "parse failure", "decisions": []},
    )


def opportunity_hunt(
    client: anthropic.Anthropic, owned_tickers: list[str], cost: CostTracker
) -> dict[str, Any]:
    held = ", ".join(sorted(owned_tickers))
    user = (
        f"User's current holdings (do NOT suggest any of these): {held}\n\n"
        f"Surface 3 picks per the rules in your system prompt. Remember: {OPPORTUNITY_SEARCH_BUDGET} web searches MAX. "
        "Think hard between searches. Specific tickers, entry levels, timing — no fluff."
    )
    response = client.messages.create(
        model=SONNET,
        max_tokens=2500,
        system=OPPORTUNITY_SYSTEM,
        tools=[SONNET_OPP_WEB_SEARCH_TOOL],
        output_config={"format": {"type": "json_schema", "schema": OPPORTUNITY_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    cost.record(SONNET, response, "opportunity hunt")
    return parse_json_or_fallback(
        last_text_block(response),
        fallback={"picks": []},
    )


# ──────────────────────────── report rendering ──────────────────────────────


def render_report(
    scans: list[dict[str, Any]],
    decisions: dict[str, Any] | None,
    skipped_small: list[dict[str, Any]],
    opps: dict[str, Any],
    cost: CostTracker,
    run_date: str,
) -> str:
    L: list[str] = []
    L.append(f"# Weekly Portfolio Scan — {run_date}\n")
    L.append(f"_{cost.summary()}_\n")

    L.append("## Portfolio Check\n")
    L.append("| Symbol | Section | Weight | Price | Verdict | Top Headline |")
    L.append("|--------|---------|--------|-------|---------|--------------|")
    for s in scans:
        pos = s["position"]
        sym = pos["symbol"]
        section = pos["section"]
        weight = f"{pos['position_pct']:.1%}"
        price = s.get("current_price", "?")
        verdict = s.get("verdict", "?")
        headline = (s.get("top_headline") or "").replace("|", "/").replace("\n", " ")
        headline = headline[:90] + ("…" if len(headline) > 90 else "")
        L.append(f"| {sym} | {section} | {weight} | {price} | {verdict} | {headline} |")

    L.append("\n### Reasoning\n")
    for s in scans:
        sym = s["position"]["symbol"]
        verdict = s.get("verdict", "?")
        reasoning = s.get("reasoning", "")
        L.append(f"- **{sym}** ({verdict}): {reasoning}")

    if decisions and decisions.get("decisions"):
        L.append("\n## Fee-Validated Decisions (Sonnet, positions ≥3% weight)\n")
        L.append(f"**Swissquote fee summary**: {decisions.get('swissquote_fees_summary', '')}\n")
        for d in decisions["decisions"]:
            L.append(f"### {d['symbol']}: {d['preliminary_verdict']} → **{d['final_verdict']}**")
            L.append(
                f"- Trading fee: {d['estimated_trading_fee_chf']:.2f} CHF | "
                f"FX cost: {d['estimated_fx_cost_chf']:.2f} CHF | "
                f"**Net P&L: {d['estimated_net_pnl_chf']:+.2f} CHF**"
            )
            L.append(f"- {d['rationale']}\n")

    if skipped_small:
        L.append("\n### Flagged TRIM/SELL but below 3% — fee validation skipped\n")
        for s in skipped_small:
            pos = s["position"]
            L.append(
                f"- **{pos['symbol']}** ({pos['position_pct']:.1%} weight): "
                f"preliminary verdict {s.get('verdict')} stands. {s.get('reasoning','')}"
            )

    if not decisions and not skipped_small:
        L.append("\n## Fee-Validated Decisions\n\nNo TRIM/SELL flags — fee validation skipped.\n")

    L.append("\n## Top 3 Opportunities (Sonnet, free-form)\n")
    for p in opps.get("picks", []):
        kind = p["type"].replace("_", " ").title()
        L.append(f"### {p['ticker']} — {kind}")
        L.append(f"- **Current**: {p['current_price']}  |  **Entry**: {p['entry_price']}  |  **Timing**: {p['catalyst_timeline']}")
        L.append(f"- **Thesis**: {p['thesis']}")
        L.append(f"- **Key risk**: {p['key_risk']}\n")

    return "\n".join(L)


def short_summary(scans: list[dict[str, Any]], opps: dict[str, Any], cost: CostTracker, run_date: str) -> str:
    counts: dict[str, int] = {}
    for s in scans:
        v = s.get("verdict", "?")
        counts[v] = counts.get(v, 0) + 1
    counts_str = "  ".join(f"{v}: {n}" for v, n in sorted(counts.items()))
    top_tickers = ", ".join(p["ticker"] for p in opps.get("picks", []))
    return (
        f"📊 Weekly scan — {run_date}\n"
        f"{counts_str}\n"
        f"Opportunities: {top_tickers or 'none'}\n"
        f"{cost.summary()}"
    )


# ──────────────────────────── Telegram ──────────────────────────────────────


def send_telegram(summary: str, report_path: Path) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram not configured (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env). Skipping notification.")
        return
    base = f"https://api.telegram.org/bot{token}"
    try:
        r = requests.post(
            f"{base}/sendMessage",
            data={"chat_id": chat_id, "text": summary},
            timeout=20,
        )
        r.raise_for_status()
        with open(report_path, "rb") as f:
            r = requests.post(
                f"{base}/sendDocument",
                data={"chat_id": chat_id},
                files={"document": (report_path.name, f, "text/markdown")},
                timeout=30,
            )
            r.raise_for_status()
        print("Telegram notification sent.")
    except requests.RequestException as e:
        print(f"Telegram notification failed (run still completed): {e}")


# ──────────────────────────── main ──────────────────────────────────────────


def main() -> None:
    # max_retries=0 — a 429 or 400 should fail loud, not silently double the spend.
    client = anthropic.Anthropic(max_retries=0)
    cost = CostTracker()
    positions_path = find_positions_file()
    positions = parse_positions_xls(positions_path)
    run_date = date.today().isoformat()

    print(f"Weekly scan — {run_date}")
    print(f"Loaded {len(positions)} positions from {positions_path}")
    print(f"Budget guards: max_uses=1 (Haiku), max_uses={OPPORTUNITY_SEARCH_BUDGET} (opportunity), Sonnet fee gate ≥{MATERIAL_POSITION_THRESHOLD:.0%}\n")

    # Phase 1 — Haiku scans (one web_search each).
    scans: list[dict[str, Any]] = []
    for i, pos in enumerate(positions, 1):
        print(f"[{i:>2}/{len(positions)}] Haiku scan {pos['symbol']:<6} ({pos['position_pct']:>5.1%}) ...", end=" ", flush=True)
        try:
            result = scan_position(client, pos, cost)
        except anthropic.APIError as e:
            print(f"FAILED: {type(e).__name__}: {getattr(e, 'message', e)}")
            result = {"verdict": "ERROR", "top_headline": str(e)[:200], "current_price": pos["preis"], "reasoning": "API error"}
        result["position"] = pos
        scans.append(result)
        print(result.get("verdict", "?"))

    # Phase 2 — Sonnet fee validation, gated on material positions only.
    flagged = [s for s in scans if s.get("verdict") in ("TRIM", "SELL")]
    material = [s for s in flagged if (s["position"].get("position_pct") or 0) >= MATERIAL_POSITION_THRESHOLD]
    skipped_small = [s for s in flagged if s not in material]

    decisions: dict[str, Any] | None = None
    if material:
        print(f"\nSonnet validating {len(material)} material TRIM/SELL (≥{MATERIAL_POSITION_THRESHOLD:.0%} weight) with Swissquote fee math ...")
        try:
            decisions = decide_flagged(client, material, cost)
            print("  done.")
        except anthropic.APIError as e:
            print(f"  FAILED: {type(e).__name__}: {getattr(e, 'message', e)}")
            decisions = {"swissquote_fees_summary": f"API error: {e}", "decisions": []}
        # Brief pause so the opportunity hunt isn't stacked into the same ITPM window.
        time.sleep(15)
    else:
        if flagged:
            print(f"\n{len(flagged)} TRIM/SELL flagged but all below {MATERIAL_POSITION_THRESHOLD:.0%} — Sonnet fee step skipped.")
        else:
            print("\nNothing flagged TRIM/SELL — Sonnet fee step skipped.")

    # Phase 3 — opportunity hunt.
    print(f"\nSonnet running opportunity hunt (max {OPPORTUNITY_SEARCH_BUDGET} searches) ...")
    owned = [p["symbol"] for p in positions]
    try:
        opps = opportunity_hunt(client, owned, cost)
        print(f"  done — {len(opps.get('picks', []))} picks.\n")
    except anthropic.APIError as e:
        print(f"  FAILED: {type(e).__name__}: {getattr(e, 'message', e)}\n")
        opps = {"picks": []}

    # Render + save.
    report = render_report(scans, decisions, skipped_small, opps, cost, run_date)
    summary = short_summary(scans, opps, cost, run_date)

    print("=" * 70)
    print(report)
    print("=" * 70)
    print(cost.summary())

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / f"weekly_{run_date}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {out_path}")

    send_telegram(summary, out_path)


if __name__ == "__main__":
    main()
