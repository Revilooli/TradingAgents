"""Run a full TradingAgents analysis across a Swissquote portfolio export.

Reads positions.xls (Swissquote's German-locale XLS dump), pulls the
Aktien + ETFs sections, drops crypto and subtotal rows, then analyzes
each symbol one at a time.

Cost shape: Haiku everywhere EXCEPT the Portfolio Manager (final
buy/sell/hold), which uses Sonnet. The Research Manager normally takes
``deep_think_llm`` too, so we monkey-patch it back down to Haiku so Sonnet
fires exactly once per ticker — at the final decision.
"""

from datetime import date
from pathlib import Path

import xlrd
from dotenv import load_dotenv

# Patch BEFORE importing TradingAgentsGraph so the swap is in effect when
# GraphSetup runs.
from tradingagents.graph import setup as _setup_mod
from tradingagents.llm_clients import create_llm_client

load_dotenv()

QUICK_MODEL = "claude-haiku-4-5-20251001"
DEEP_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 500
TRADE_DATE = date.today().isoformat()

# Swissquote column layout (0-indexed). Row 0 of the sheet labels these.
COL_SECTION = 0   # "Aktien" / "ETFs" / "Kryptowährungen" on header rows
COL_SYMBOL = 1
COL_ANZAHL = 2
COL_EINSTAND = 3
COL_PREIS = 7
COL_WAEHRUNG = 8
COL_GV_PCT = 10
COL_TOTALWERT_CHF = 11
COL_POSITION_PCT = 12


def parse_positions_xls(path: Path) -> list[dict]:
    """Extract Aktien + ETFs positions from a Swissquote XLS export.

    Skips the Kryptowährungen section, all ``Zwischensumme`` subtotals,
    and the ``Gesamt`` grand-total row. German umlauts in section headers
    arrive as mojibake from xlrd, so we match on ASCII-clean prefixes
    ("Aktien", "ETFs", "Krypto").
    """
    wb = xlrd.open_workbook(path)
    sheet = wb.sheet_by_index(0)
    positions: list[dict] = []
    section: str | None = None

    for i in range(1, sheet.nrows):  # skip header row 0
        row = [sheet.cell_value(i, j) for j in range(sheet.ncols)]
        section_marker = str(row[COL_SECTION]).strip()

        if section_marker == "Aktien":
            section = "aktien"
            continue
        if section_marker == "ETFs":
            section = "etfs"
            continue
        if section_marker.startswith("Krypto"):  # mojibake-safe
            section = "skip"
            continue
        if section_marker == "Gesamt":
            break

        if section not in ("aktien", "etfs"):
            continue

        symbol = str(row[COL_SYMBOL]).strip()
        if not symbol or symbol.startswith("Zwischensumme"):
            continue

        positions.append({
            "symbol": symbol,
            "section": section,
            "anzahl": row[COL_ANZAHL],
            "einstandskurs": row[COL_EINSTAND],
            "preis": row[COL_PREIS],
            "waehrung": str(row[COL_WAEHRUNG]).strip(),
            "gv_pct_chf": row[COL_GV_PCT],
            "totalwert_chf": row[COL_TOTALWERT_CHF],
            "position_pct": row[COL_POSITION_PCT],
        })

    return positions


def find_positions_file() -> Path:
    """Locate positions.xls — script dir, then TradingAgents/ subdir."""
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir / "positions.xls", script_dir / "TradingAgents" / "positions.xls"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"positions.xls not found in {script_dir} or {script_dir / 'TradingAgents'}"
    )


def main():
    positions_path = find_positions_file()
    positions = parse_positions_xls(positions_path)
    tickers = [p["symbol"] for p in positions]
    print(f"Loaded {len(positions)} positions from {positions_path}")
    for p in positions:
        print(f"  [{p['section']:6}] {p['symbol']:<6} {p['anzahl']:>10} @ "
              f"{p['einstandskurs']} → {p['preis']} {p['waehrung']}  "
              f"G&V {p['gv_pct_chf']:+.4%}")

    # Force the Research Manager onto Haiku regardless of the deep LLM passed in.
    # Without this, deep_think_llm=Sonnet would fire Sonnet twice per ticker
    # (Research Manager + Portfolio Manager) instead of just at the final call.
    haiku_for_rm = create_llm_client(
        "anthropic", QUICK_MODEL, max_tokens=MAX_TOKENS
    ).get_llm()
    orig_create_research_manager = _setup_mod.create_research_manager
    _setup_mod.create_research_manager = lambda _llm: orig_create_research_manager(haiku_for_rm)

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["quick_think_llm"] = QUICK_MODEL
    config["deep_think_llm"] = DEEP_MODEL  # used by Portfolio Manager only (RM is patched)
    config["max_debate_rounds"] = 1
    config["max_tokens"] = MAX_TOKENS
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }

    ta = TradingAgentsGraph(debug=False, config=config)

    results = {}
    for ticker in tickers:
        print(f"\n{'=' * 60}\nAnalyzing {ticker} on {TRADE_DATE}\n{'=' * 60}")
        try:
            _, decision = ta.propagate(ticker, TRADE_DATE)
            results[ticker] = decision
            print(f"{ticker}: {decision}")
        except Exception as e:
            results[ticker] = f"ERROR: {e}"
            print(f"{ticker}: failed — {e}")

    print("\n\nSUMMARY")
    print("=" * 60)
    for ticker, decision in results.items():
        print(f"{ticker:<6} {decision}")


if __name__ == "__main__":
    main()
