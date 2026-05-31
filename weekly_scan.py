"""Weekly portfolio scan: Haiku per-position research + Sonnet fee-aware decisions + free-form opportunity hunt.

Cost shape per run (rough): ~$0.50–$1.00 dominated by web_search calls.
- N Haiku scans (one per position) with server-side web_search → preliminary HOLD/ADD/TRIM/SELL.
- 1 Sonnet call IF any positions flagged TRIM/SELL: looks up current Swissquote fees via
  web_search, computes net-of-fees P&L, validates each flagged position.
- 1 Sonnet call: free-form opportunity hunt — top 3 picks mixing short-term catalysts and
  long-term structural setups, no pre-supplied watchlist.

Structured outputs (`output_config.format`) enforce parseable JSON on every call — no prefills,
no brittle regex. Server-side web_search runs inside the model's turn; we only retry once on
`stop_reason == "pause_turn"` (the server's 10-iteration cap).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from portfolio_scan import find_positions_file, parse_positions_xls

load_dotenv()

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}

SCANNER_SYSTEM = """You are a buy-side equity analyst doing weekly portfolio maintenance.

For each position the user gives you:
1. Use web_search to find the current market price and the 1-3 most material news items from the last 14 days (earnings, guidance, M&A, regulatory, sector moves, insider activity).
2. Assess the position against its cost basis and current unrealized P&L: is the thesis intact, strengthening, weakening, or broken?
3. Output a structured verdict.

Verdict definitions:
- HOLD: thesis intact, no action warranted
- ADD: thesis strengthening, conviction higher than at entry, or pullback into support
- TRIM: take partial profit OR de-risk on partial thesis impairment (keep core exposure)
- SELL: full exit — thesis broken, dead money, or capital better deployed elsewhere

Be terse and substantive. Skip disclaimers ("not investment advice", "consult a professional"). Skip sector-rotation boilerplate. Output structured JSON only — the schema is enforced."""

SCANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "current_price": {"type": "number", "description": "Latest market price from web_search, in trade currency"},
        "key_news": {"type": "string", "description": "1-3 sentence summary of material news from the last 14 days"},
        "verdict": {"type": "string", "enum": ["HOLD", "ADD", "TRIM", "SELL"]},
        "reasoning": {"type": "string", "description": "2-4 sentence rationale tying news and price action to the verdict"},
    },
    "required": ["current_price", "key_news", "verdict", "reasoning"],
    "additionalProperties": False,
}

FEE_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "swissquote_fees_summary": {
            "type": "string",
            "description": "Summary of Swissquote's current trading commission tiers and FX spread from web_search",
        },
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
                    "entry_price": {"type": "string", "description": "Entry level or range, e.g. '12.50' or '11-13'"},
                    "thesis": {"type": "string", "description": "3-5 sentence investment thesis"},
                    "key_risk": {"type": "string"},
                    "catalyst_timeline": {"type": "string", "description": "When the thesis pays off — e.g. 'Q1 2026 earnings', '12-24 months', 'multi-year'"},
                },
                "required": ["ticker", "type", "current_price", "entry_price", "thesis", "key_risk", "catalyst_timeline"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["picks"],
    "additionalProperties": False,
}


def call_with_pause_retry(client: anthropic.Anthropic, **kwargs: Any) -> Any:
    """Call messages.create; if the server-side web_search loop hits its 10-iteration cap
    (`stop_reason == "pause_turn"`), append the assistant content and re-send once. Do NOT
    add a "continue" user message — the API resumes automatically from the trailing
    server_tool_use block."""
    response = client.messages.create(**kwargs)
    if response.stop_reason == "pause_turn":
        messages = list(kwargs["messages"])
        messages.append({"role": "assistant", "content": response.content})
        response = client.messages.create(**{**kwargs, "messages": messages})
    return response


def last_text_block(response: Any) -> str:
    """Server-side tools interleave server_tool_use / web_search_tool_result / text blocks.
    The final structured-output JSON lives in the LAST text block."""
    texts = [b.text for b in response.content if b.type == "text"]
    return texts[-1] if texts else ""


def parse_json_or_fallback(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {**fallback, "_raw": text[:500]}


def scan_position(client: anthropic.Anthropic, pos: dict[str, Any]) -> dict[str, Any]:
    pnl = pos["gv_pct_chf"]
    user = (
        f"Position: {pos['symbol']}\n"
        f"Section: {pos['section']}\n"
        f"Quantity: {pos['anzahl']}\n"
        f"Cost basis: {pos['einstandskurs']} {pos['waehrung']}\n"
        f"Recorded price: {pos['preis']} {pos['waehrung']}\n"
        f"Unrealized P&L (CHF): {pnl:+.2%}\n\n"
        "Search the web for the current market price and material news from the last 14 days. "
        "Return your structured assessment."
    )
    response = call_with_pause_retry(
        client,
        model=HAIKU,
        max_tokens=600,
        system=[{"type": "text", "text": SCANNER_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        tools=[WEB_SEARCH_TOOL],
        output_config={"format": {"type": "json_schema", "schema": SCANNER_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    return parse_json_or_fallback(
        last_text_block(response),
        fallback={"verdict": "UNKNOWN", "key_news": "parse failure", "current_price": pos["preis"], "reasoning": ""},
    )


def decide_flagged(client: anthropic.Anthropic, flagged: list[dict[str, Any]]) -> dict[str, Any]:
    payload = [
        {
            "symbol": s["position"]["symbol"],
            "quantity": s["position"]["anzahl"],
            "cost_basis": s["position"]["einstandskurs"],
            "currency": s["position"]["waehrung"],
            "current_price": s.get("current_price", s["position"]["preis"]),
            "preliminary_verdict": s.get("verdict"),
            "preliminary_reasoning": s.get("reasoning", ""),
        }
        for s in flagged
    ]
    user = (
        "I have positions flagged for TRIM or SELL by an upstream analyst. Before I act, "
        "I need fee math validated against Swissquote's actual fee schedule. Account is "
        "denominated in CHF.\n\n"
        "Step 1: Use web_search to find Swissquote's CURRENT trading commission tiers (per-trade "
        "fees by order value, including any minimums and exchange surcharges) and their FX "
        "spread when converting USD/EUR proceeds back to CHF.\n\n"
        "Step 2: For each position below, compute:\n"
        "  - Gross proceeds = quantity × current_price (in trade currency)\n"
        "  - Trading commission (in CHF)\n"
        "  - FX conversion cost back to CHF (if non-CHF)\n"
        "  - Estimated net realized P&L in CHF vs. cost basis (after all costs)\n\n"
        "Step 3: Validate or override the preliminary verdict. A TRIM/SELL only stands if:\n"
        "  (a) the thesis is genuinely impaired enough that fee drag is acceptable, OR\n"
        "  (b) it's a loss cut to free capital for higher-conviction redeployment, OR\n"
        "  (c) gains are large enough that fees are immaterial.\n"
        "Otherwise downgrade to HOLD.\n\n"
        f"Positions:\n```json\n{json.dumps(payload, indent=2)}\n```\n\n"
        "Return structured JSON."
    )
    response = call_with_pause_retry(
        client,
        model=SONNET,
        max_tokens=2000,
        tools=[WEB_SEARCH_TOOL],
        output_config={"format": {"type": "json_schema", "schema": FEE_DECISION_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    return parse_json_or_fallback(
        last_text_block(response),
        fallback={"swissquote_fees_summary": "parse failure", "decisions": []},
    )


def opportunity_hunt(client: anthropic.Anthropic, owned_tickers: list[str]) -> dict[str, Any]:
    held = ", ".join(sorted(owned_tickers))
    user = (
        "You're a sharp generalist investor doing free-form idea generation. Surface 3 genuinely "
        "compelling ideas to buy right now.\n\n"
        "Mix at least one of each:\n"
        "- SHORT-TERM CATALYST: earnings / data / regulatory / M&A inflection within 4-12 weeks\n"
        "- LONG-TERM STRUCTURAL: companies positioned in front of multi-year secular shifts that "
        "the market hasn't priced in yet. Examples of what this looks like in hindsight: Western "
        "Digital before the storage upcycle, Dell before the AI server tailwind, EMCOR before the "
        "data-center buildout. Both were widely overlooked when the setup was forming.\n\n"
        "Use web_search aggressively. Look for:\n"
        "- Recent earnings reactions where the stock disconnected from improving fundamentals\n"
        "- Insider buying clusters (Form 4 filings)\n"
        "- Sector rotations actively underway (capital flows, breadth changes)\n"
        "- Beaten-down names with stabilizing fundamentals (failure-of-shorts setups)\n"
        "- Quiet structural winners in unsexy industries\n\n"
        "Hard rules:\n"
        "- NO mega-cap obvious calls (NVDA, AAPL, MSFT, AMZN, GOOG, META, TSLA) unless you have "
        "a genuinely non-consensus angle.\n"
        f"- AVOID names already in the user's portfolio: {held}\n"
        "- NO diversification boilerplate, NO 'consider your risk tolerance', NO disclaimers.\n"
        "- Specific tickers, specific entry levels, specific timing.\n\n"
        "Return structured JSON with exactly 3 picks."
    )
    response = call_with_pause_retry(
        client,
        model=SONNET,
        max_tokens=2800,
        tools=[WEB_SEARCH_TOOL],
        output_config={"format": {"type": "json_schema", "schema": OPPORTUNITY_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    return parse_json_or_fallback(
        last_text_block(response),
        fallback={"picks": []},
    )


def render_report(
    scans: list[dict[str, Any]],
    decisions: dict[str, Any] | None,
    opps: dict[str, Any],
    run_date: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# Weekly Portfolio Scan — {run_date}\n")

    lines.append("## Portfolio Check\n")
    lines.append("| Symbol | Section | Price | Verdict | Key News |")
    lines.append("|--------|---------|-------|---------|----------|")
    for s in scans:
        pos = s["position"]
        sym = pos["symbol"]
        section = pos["section"]
        price = s.get("current_price", "?")
        verdict = s.get("verdict", "?")
        news = (s.get("key_news") or "").replace("|", "/").replace("\n", " ")
        news = news[:100] + ("…" if len(news) > 100 else "")
        lines.append(f"| {sym} | {section} | {price} | {verdict} | {news} |")

    lines.append("\n### Reasoning\n")
    for s in scans:
        sym = s["position"]["symbol"]
        verdict = s.get("verdict", "?")
        reasoning = s.get("reasoning", "")
        lines.append(f"- **{sym}** ({verdict}): {reasoning}")

    if decisions and decisions.get("decisions"):
        lines.append("\n## Fee-Validated Decisions (Sonnet)\n")
        lines.append(f"**Swissquote fee summary**: {decisions.get('swissquote_fees_summary', '')}\n")
        for d in decisions["decisions"]:
            lines.append(f"### {d['symbol']}: {d['preliminary_verdict']} → **{d['final_verdict']}**")
            lines.append(
                f"- Trading fee: {d['estimated_trading_fee_chf']:.2f} CHF | "
                f"FX cost: {d['estimated_fx_cost_chf']:.2f} CHF | "
                f"**Net P&L: {d['estimated_net_pnl_chf']:+.2f} CHF**"
            )
            lines.append(f"- {d['rationale']}\n")
    else:
        lines.append("\n## Fee-Validated Decisions\n\nNo positions flagged for TRIM/SELL — fee validation skipped.\n")

    lines.append("\n## Top 3 Opportunities (Sonnet, free-form)\n")
    for p in opps.get("picks", []):
        kind = p["type"].replace("_", " ").title()
        lines.append(f"### {p['ticker']} — {kind}")
        lines.append(f"- **Current price**: {p['current_price']}  |  **Entry**: {p['entry_price']}  |  **Timing**: {p['catalyst_timeline']}")
        lines.append(f"- **Thesis**: {p['thesis']}")
        lines.append(f"- **Key risk**: {p['key_risk']}\n")

    return "\n".join(lines)


def main() -> None:
    client = anthropic.Anthropic()
    positions_path = find_positions_file()
    positions = parse_positions_xls(positions_path)
    run_date = date.today().isoformat()

    print(f"Weekly scan — {run_date}")
    print(f"Loaded {len(positions)} positions from {positions_path}\n")

    scans: list[dict[str, Any]] = []
    for i, pos in enumerate(positions, 1):
        print(f"[{i:>2}/{len(positions)}] Haiku scan {pos['symbol']:<6} ...", end=" ", flush=True)
        try:
            result = scan_position(client, pos)
        except anthropic.APIStatusError as e:
            print(f"API error ({e.status_code}): {e.message}")
            result = {"verdict": "ERROR", "key_news": str(e), "current_price": pos["preis"], "reasoning": ""}
        result["position"] = pos
        scans.append(result)
        print(result.get("verdict", "?"))

    flagged = [s for s in scans if s.get("verdict") in ("TRIM", "SELL")]
    decisions: dict[str, Any] | None = None
    if flagged:
        print(f"\nSonnet validating {len(flagged)} flagged TRIM/SELL with Swissquote fee math ...")
        decisions = decide_flagged(client, flagged)
        print("  done.")
    else:
        print("\nNothing flagged TRIM/SELL — skipping Sonnet fee validation.")

    print("\nSonnet running free-form opportunity hunt (top 3) ...")
    owned = [p["symbol"] for p in positions]
    opps = opportunity_hunt(client, owned)
    print(f"  done — {len(opps.get('picks', []))} picks.\n")

    report = render_report(scans, decisions, opps, run_date)
    print("=" * 70)
    print(report)
    print("=" * 70)

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / f"weekly_{run_date}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
