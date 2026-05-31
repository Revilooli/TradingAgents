"""Weekly portfolio scan — actionable order specs straight to Telegram.

Pipeline:
  1. N Haiku scans (one web_search each) → preliminary HOLD/ADD/TRIM/SELL.
  2. One Sonnet call ingests ALL flagged (ADD/TRIM/SELL) positions, web_searches
     Swissquote fees + FX spread, and returns ready-to-place order specs: limit
     price, share count, fee estimate, net CHF. Downgrades to HOLD any case where
     fee drag eats the trade.
  3. One Sonnet call generates 3 free-form opportunity picks, each with a consensus
     analyst price target found via web_search and an expected-upside %.
  4. Telegram message built from the order specs — scannable, plain-text, ready to
     copy into Swissquote without further research. Full markdown report attached
     as a file.

Cost controls:
  - Haiku: max_uses=1 on web_search per position.
  - Sonnet fee call: max_uses=4 (fees + light price refresh).
  - Sonnet opportunity call: max_uses=5 (the model is told to plan and think harder
    rather than search more).
  - max_retries=0 on the client — 429s and 400s fail loud, no silent double spend.
  - Live cost tracker reads response.usage + counts server_tool_use blocks.

Integrations:
  - Telegram: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable.
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
    # Now covers fee lookup + light price-refresh for any position with stale recent_price.
    "max_uses": 4,
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

# Telegram messages have a 4096-char limit. Leave margin for safety; chunk if exceeded.
TELEGRAM_MAX_CHARS = 3800


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

OPPORTUNITY_SYSTEM = f"""You are a sharp generalist investor surfacing 3 high-conviction ideas to buy now. Output must be IMMEDIATELY actionable — the user reads it on Telegram and places limit orders without further research.

CRITICAL BUDGET CONSTRAINT: You have a hard cap of {OPPORTUNITY_SEARCH_BUDGET} web_search queries TOTAL across the entire response. Spend them deliberately:
- Plan your searches BEFORE issuing the first one.
- Each query should cover a broad-but-pointed theme OR confirm specifics (current price + consensus analyst PT) on a shortlisted candidate.
- Lean on what you already know about market structure, sector dynamics, and recent earnings reactions. Searches are for fresh dated facts (current price, consensus PT, recent catalysts), not background knowledge.

For each of the 3 picks you MUST surface (via web_search where needed):
- Current market price in trade currency
- A consensus or median sell-side analyst price target (search e.g. "TICKER analyst price target" / "TICKER consensus PT")
- expected_upside_pct = (target - current) / current
- A "why_now" sentence: what makes THIS WEEK the buy moment

Required mix in your 3 picks:
- At least 1 SHORT-TERM CATALYST: earnings / regulatory / data / M&A inflection inside 4-12 weeks
- At least 1 LONG-TERM STRUCTURAL: positioned in front of a multi-year secular shift the market hasn't priced. Examples in hindsight: Western Digital pre-storage-cycle, Dell pre-AI-server, EMCOR pre-data-center buildout.

Hard rules:
- NO mega-caps (NVDA, AAPL, MSFT, AMZN, GOOG, META, TSLA) unless you have a genuinely non-consensus angle.
- NO names already in the user's portfolio (the user will tell you which).
- NO diversification boilerplate, NO "consult your advisor", NO disclaimers.
- limit_entry_price = current_price × ~0.99 unless wider entry makes sense given recent volatility.

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

ACTIONABLE_DECISIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "swissquote_fees_summary": {"type": "string", "description": "1-2 sentence summary of fee schedule"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "action": {"type": "string", "enum": ["BUY", "TRIM", "SELL", "HOLD"]},
                    "currency": {"type": "string", "description": "Trade currency: USD / EUR / CHF / etc."},
                    "current_price": {"type": "number", "description": "Current market price in trade currency"},
                    "limit_price": {"type": "number", "description": "Suggested limit price in trade currency. For BUY: slightly below current. For TRIM/SELL: slightly above current."},
                    "shares": {"type": "integer", "description": "Number of shares to transact. For BUY: sized to ~500 CHF order. For TRIM: a partial position size. For SELL: full position."},
                    "estimated_fee_chf": {"type": "number", "description": "Total Swissquote trading commission + FX spread for this order, in CHF"},
                    "estimated_net_chf": {"type": "number", "description": "For BUY: total CHF cash leaving the account incl fees. For TRIM/SELL: net CHF proceeds AFTER fees."},
                    "reason": {"type": "string", "description": "ONE clear sentence: why this action now. No disclaimers."},
                },
                "required": ["symbol", "action", "currency", "current_price", "limit_price", "shares", "estimated_fee_chf", "estimated_net_chf", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["swissquote_fees_summary", "actions"],
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
                    "currency": {"type": "string", "description": "Trade currency: USD / EUR / CHF / etc."},
                    "current_price": {"type": "number"},
                    "limit_entry_price": {"type": "number", "description": "Suggested limit-buy price (slightly below current)"},
                    "analyst_target_price": {"type": "number", "description": "Consensus or median sell-side analyst price target found via web_search"},
                    "expected_upside_pct": {"type": "number", "description": "(analyst_target_price - current_price) / current_price"},
                    "thesis": {"type": "string", "description": "3-5 sentence investment thesis"},
                    "why_now": {"type": "string", "description": "ONE sentence: why act this week vs holding off"},
                    "key_risk": {"type": "string"},
                    "catalyst_timeline": {"type": "string"},
                },
                "required": ["ticker", "type", "currency", "current_price", "limit_entry_price", "analyst_target_price", "expected_upside_pct", "thesis", "why_now", "key_risk", "catalyst_timeline"],
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


def decide_actionable(
    client: anthropic.Anthropic, flagged: list[dict[str, Any]], cost: CostTracker
) -> dict[str, Any]:
    """One Sonnet call returns concrete order parameters for every flagged position.

    Replaces the prior fee-validation-only scope. Each output row carries everything the
    user needs to place an order on Swissquote without re-researching: limit price, share
    count, fee estimate, net CHF impact, and a one-sentence reason. Covers TRIM, SELL,
    and BUY (the BUY action corresponds to a Haiku ADD verdict).
    """
    payload = []
    for s in flagged:
        pos = s["position"]
        # Derive an implied CHF/unit from the xls snapshot so the model can size BUY
        # orders to ~500 CHF without needing a fresh FX lookup.
        chf_per_unit = None
        try:
            qty = float(pos["anzahl"]) or 0
            tw = float(pos.get("totalwert_chf") or 0)
            if qty:
                chf_per_unit = tw / qty
        except (TypeError, ValueError):
            chf_per_unit = None

        payload.append({
            "symbol": pos["symbol"],
            "current_quantity_held": pos["anzahl"],
            "cost_basis": pos["einstandskurs"],
            "currency": pos["waehrung"],
            "recent_price": s.get("current_price", pos["preis"]),
            "implied_chf_per_unit": chf_per_unit,
            "portfolio_weight_pct": pos["position_pct"],
            "preliminary_verdict": s.get("verdict"),  # ADD / TRIM / SELL
            "preliminary_reasoning": s.get("reasoning", ""),
        })

    user = (
        "I need fully-actionable order parameters for the positions below — concrete enough "
        "that I can open Swissquote and place limit orders without doing any additional "
        "research. Account is denominated in CHF.\n\n"
        "Step 1: web_search Swissquote's current trading commission schedule (per-trade fees "
        "by order value with minimums/surcharges) AND their FX spread on USD/EUR↔CHF "
        "conversion. Budget: ≤4 searches total for fees and any price refresh you need.\n\n"
        "Step 2: For each position, output an `action` row:\n"
        "  - For preliminary verdict ADD → action='BUY'. Size `shares` so the gross order is "
        "≈500 CHF. Use `implied_chf_per_unit` from the payload for sizing if non-CHF.\n"
        "  - For preliminary verdict TRIM → action='TRIM'. Size `shares` at roughly 30-50% of "
        "current_quantity_held.\n"
        "  - For preliminary verdict SELL → action='SELL'. `shares` = current_quantity_held.\n"
        "  - For ANY position where fee drag eats the case (small positions where the "
        "Swissquote minimum dominates, or marginal-conviction TRIMs), downgrade to "
        "action='HOLD' and explain in `reason`.\n\n"
        "Step 3: Set `limit_price` in trade currency:\n"
        "  - BUY: ~1% below current_price (tighter if low-vol, wider if high-vol).\n"
        "  - TRIM/SELL: ~1% above current_price.\n\n"
        "Step 4: Compute `estimated_fee_chf` (commission + FX spread) and `estimated_net_chf`:\n"
        "  - BUY: total CHF cash LEAVING the account incl. fees (positive number, close to 500).\n"
        "  - TRIM/SELL: net CHF proceeds AFTER fees (positive number — what hits the account).\n\n"
        "Step 5: `reason` is ONE sentence. No disclaimers, no boilerplate.\n\n"
        f"Positions:\n```json\n{json.dumps(payload, indent=2, default=str)}\n```\n\n"
        "Return structured JSON."
    )
    response = client.messages.create(
        model=SONNET,
        max_tokens=2500,
        tools=[SONNET_FEE_WEB_SEARCH_TOOL],
        output_config={"format": {"type": "json_schema", "schema": ACTIONABLE_DECISIONS_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    cost.record(SONNET, response, "actionable decisions")
    return parse_json_or_fallback(
        last_text_block(response),
        fallback={"swissquote_fees_summary": "parse failure", "actions": []},
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
        weight = f"{pos['position_pct']:.1%}"
        price = s.get("current_price", "?")
        verdict = s.get("verdict", "?")
        headline = (s.get("top_headline") or "").replace("|", "/").replace("\n", " ")
        headline = headline[:90] + ("…" if len(headline) > 90 else "")
        L.append(f"| {pos['symbol']} | {pos['section']} | {weight} | {price} | {verdict} | {headline} |")

    L.append("\n### Reasoning\n")
    for s in scans:
        L.append(f"- **{s['position']['symbol']}** ({s.get('verdict', '?')}): {s.get('reasoning', '')}")

    actions = (decisions or {}).get("actions", [])
    if actions:
        L.append("\n## Actionable Orders (Sonnet, fee-validated)\n")
        L.append(f"_Swissquote fees: {(decisions or {}).get('swissquote_fees_summary', '')}_\n")
        for a in actions:
            L.append(f"### {a['symbol']} — **{a['action']}**")
            L.append(
                f"- Current: {a['current_price']} {a['currency']}  |  "
                f"Limit: **{a['limit_price']} {a['currency']}**  |  "
                f"Shares: **{a['shares']}**"
            )
            if a["action"] == "BUY":
                L.append(f"- Total cost: ≈**{a['estimated_net_chf']:.0f} CHF** (incl. {a['estimated_fee_chf']:.2f} CHF fees)")
            elif a["action"] in ("TRIM", "SELL"):
                L.append(f"- Net proceeds: ≈**{a['estimated_net_chf']:.0f} CHF** (after {a['estimated_fee_chf']:.2f} CHF fees)")
            else:
                L.append(f"- Downgraded to HOLD ({a['estimated_fee_chf']:.2f} CHF fee drag)")
            L.append(f"- {a['reason']}\n")
    else:
        L.append("\n## Actionable Orders\n\nNo actionable verdicts this week.\n")

    L.append("\n## Top 3 Opportunities (Sonnet)\n")
    for p in opps.get("picks", []):
        kind = p["type"].replace("_", " ").title()
        L.append(f"### {p['ticker']} — {kind}")
        upside = p.get("expected_upside_pct", 0) or 0
        L.append(
            f"- Current: {p['current_price']} {p.get('currency','')}  |  "
            f"Limit entry: **{p['limit_entry_price']} {p.get('currency','')}**  |  "
            f"PT: {p.get('analyst_target_price','?')} (**{upside:+.0%}** upside)"
        )
        L.append(f"- **Why now**: {p.get('why_now','')}")
        L.append(f"- Thesis: {p['thesis']}")
        L.append(f"- Risk: {p['key_risk']}  |  Timing: {p['catalyst_timeline']}\n")

    return "\n".join(L)


def _fmt_price(p: float | int | str | None, ccy: str = "") -> str:
    """Compact price formatter — 2dp for prices, no trailing junk."""
    if p is None:
        return "?"
    try:
        return f"{float(p):.2f} {ccy}".strip()
    except (TypeError, ValueError):
        return f"{p} {ccy}".strip()


def build_telegram_message(
    decisions: dict[str, Any] | None,
    opps: dict[str, Any],
    cost: CostTracker,
    run_date: str,
) -> str:
    """Produce the actionable Telegram message. Scannable, no fluff, order-ready."""
    L: list[str] = []
    L.append(f"📊 Weekly Scan — {run_date}")
    L.append(f"{cost.summary()}")
    L.append("")

    actions = (decisions or {}).get("actions", [])
    buys = [a for a in actions if a["action"] == "BUY"]
    sells = [a for a in actions if a["action"] in ("TRIM", "SELL")]

    if buys:
        L.append(f"🟢 BUY / ADD ({len(buys)})")
        L.append("━" * 22)
        for a in buys:
            L.append(f"{a['symbol']}  {_fmt_price(a['current_price'], a['currency'])}")
            L.append(f"  → Limit buy {_fmt_price(a['limit_price'], a['currency'])}")
            L.append(f"  → {a['shares']} shares (≈{a['estimated_net_chf']:.0f} CHF incl. {a['estimated_fee_chf']:.0f} CHF fees)")
            L.append(f"  → {a['reason']}")
            L.append("")

    if sells:
        L.append(f"🔴 TRIM / SELL ({len(sells)})")
        L.append("━" * 22)
        for a in sells:
            L.append(f"{a['symbol']}  {_fmt_price(a['current_price'], a['currency'])}")
            L.append(f"  → Limit {a['action'].lower()} {_fmt_price(a['limit_price'], a['currency'])}")
            L.append(f"  → {a['shares']} shares  →  net ≈{a['estimated_net_chf']:.0f} CHF (after {a['estimated_fee_chf']:.0f} CHF fees)")
            L.append(f"  → {a['reason']}")
            L.append("")

    picks = opps.get("picks", [])
    if picks:
        L.append(f"💡 OPPORTUNITIES ({len(picks)})")
        L.append("━" * 22)
        for p in picks:
            kind = "short-term" if p["type"] == "short_term_catalyst" else "long-term"
            upside = p.get("expected_upside_pct", 0) or 0
            ccy = p.get("currency", "")
            L.append(f"{p['ticker']}  {_fmt_price(p['current_price'], ccy)}  ({kind})")
            L.append(f"  → Entry: {_fmt_price(p['limit_entry_price'], ccy)}")
            L.append(f"  → PT: {_fmt_price(p.get('analyst_target_price'), ccy)}  ({upside:+.0%} upside)")
            L.append(f"  → Why now: {p.get('why_now','')}")
            L.append("")

    if not buys and not sells and not picks:
        L.append("✅ All clear — no actionable items this week.")

    return "\n".join(L).rstrip()


def chunk_for_telegram(message: str, limit: int = TELEGRAM_MAX_CHARS) -> list[str]:
    """Split a long message at line boundaries so each chunk stays under limit."""
    if len(message) <= limit:
        return [message]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in message.split("\n"):
        if current_len + len(line) + 1 > limit and current:
            chunks.append("\n".join(current))
            current = [line]
            current_len = len(line) + 1
        else:
            current.append(line)
            current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# ──────────────────────────── Telegram ──────────────────────────────────────


def send_telegram(message: str, report_path: Path) -> None:
    """Send the actionable message (chunked if >4K chars) + full markdown report as a file."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram not configured (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env). Skipping notification.")
        return
    base = f"https://api.telegram.org/bot{token}"
    try:
        for chunk in chunk_for_telegram(message):
            r = requests.post(
                f"{base}/sendMessage",
                data={"chat_id": chat_id, "text": chunk, "disable_web_page_preview": "true"},
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
    print(f"Budget guards: max_uses=1 (Haiku), max_uses={OPPORTUNITY_SEARCH_BUDGET} (opportunity), all actionable verdicts → 1 Sonnet call\n")

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

    # Phase 2 — Sonnet builds full order specs for every actionable verdict.
    flagged = [s for s in scans if s.get("verdict") in ("ADD", "TRIM", "SELL")]
    decisions: dict[str, Any] | None = None
    if flagged:
        print(f"\nSonnet building order specs for {len(flagged)} actionable positions (ADD/TRIM/SELL) ...")
        try:
            decisions = decide_actionable(client, flagged, cost)
            print("  done.")
        except anthropic.APIError as e:
            print(f"  FAILED: {type(e).__name__}: {getattr(e, 'message', e)}")
            decisions = {"swissquote_fees_summary": f"API error: {e}", "actions": []}
        # Brief pause so the opportunity hunt isn't stacked into the same ITPM window.
        time.sleep(15)
    else:
        print("\nNothing flagged ADD/TRIM/SELL — Sonnet order step skipped.")

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
    report = render_report(scans, decisions, opps, cost, run_date)
    telegram_msg = build_telegram_message(decisions, opps, cost, run_date)

    print("=" * 70)
    print(report)
    print("=" * 70)
    print(cost.summary())

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / f"weekly_{run_date}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {out_path}")

    send_telegram(telegram_msg, out_path)


if __name__ == "__main__":
    main()
