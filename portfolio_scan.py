"""Run a full TradingAgents analysis across a watchlist, one ticker at a time.

Cost shape: Haiku everywhere EXCEPT the Portfolio Manager (final
buy/sell/hold), which uses Sonnet. The Research Manager normally takes
``deep_think_llm`` too, so we monkey-patch it back down to Haiku so Sonnet
fires exactly once per ticker — at the final decision.
"""

from datetime import date

from dotenv import load_dotenv

# Patch BEFORE importing TradingAgentsGraph so the swap is in effect when
# GraphSetup runs.
from tradingagents.graph import setup as _setup_mod
from tradingagents.llm_clients import create_llm_client

load_dotenv()

QUICK_MODEL = "claude-haiku-4-5-20251001"
DEEP_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 500

WATCHLIST = ["RCAT", "VST", "TSM", "NBIS", "ASML", "AAPL", "ABBV", "NVO", "ANIK", "BGC"]
TRADE_DATE = date.today().isoformat()

# Force the Research Manager onto Haiku regardless of the deep LLM passed in.
# Without this, deep_think_llm=Sonnet would fire Sonnet twice per ticker
# (Research Manager + Portfolio Manager) instead of just at the final call.
_haiku_for_rm = create_llm_client(
    "anthropic", QUICK_MODEL, max_tokens=MAX_TOKENS
).get_llm()
_orig_create_research_manager = _setup_mod.create_research_manager
_setup_mod.create_research_manager = lambda _llm: _orig_create_research_manager(_haiku_for_rm)

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402

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
for ticker in WATCHLIST:
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
