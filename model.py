"""
modules/agentic/market_insights/model.py

Autonomous market intelligence agent powered by Gemini Pro + LangChain tool-calling.

The agent executes a multi-step workflow autonomously:
  Step 1 → Search for recent financial news on the requested ticker/topic
  Step 2 → Extract company/entity mentions and key events
  Step 3 → Fetch real-time or recent price data via yfinance
  Step 4 → Run sentiment analysis on news headlines
  Step 5 → Reason over signals (price momentum + sentiment + events)
  Step 6 → Produce a structured market brief

All steps are traced end-to-end in Langfuse with per-step metadata.

Usage:
    from modules.agentic.market_insights.model import run_market_agent
    brief = run_market_agent("AAPL")
"""

import json
import time
from typing import Dict, List, Optional, Any

from core.config import GEMINI_API_KEY
from observability.langfuse_client import tracer

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


# ── Tool definitions ──────────────────────────────────────────────────────────

def _fetch_price_data(ticker: str, period: str = "5d") -> Dict:
    """Fetch recent OHLCV data and compute basic momentum signals."""
    if not YF_AVAILABLE:
        return {"error": "yfinance not installed"}
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period=period)
        info  = stock.info

        if hist.empty:
            return {"error": f"No data found for {ticker}"}

        close_prices = hist["Close"].tolist()
        pct_change   = ((close_prices[-1] - close_prices[0]) / close_prices[0]) * 100

        return {
            "ticker":         ticker,
            "current_price":  round(close_prices[-1], 2),
            "price_5d_ago":   round(close_prices[0], 2),
            "pct_change_5d":  round(pct_change, 2),
            "avg_volume":     int(hist["Volume"].mean()),
            "company_name":   info.get("longName", ticker),
            "sector":         info.get("sector", "Unknown"),
            "market_cap":     info.get("marketCap", None),
            "pe_ratio":       info.get("trailingPE", None),
        }
    except Exception as e:
        return {"error": str(e)}


def _analyze_sentiment(texts: List[str]) -> Dict:
    """Simple lexicon-based sentiment (no LLM call — fast, deterministic)."""
    positive_words = {
        "surge", "rally", "gain", "profit", "growth", "beat", "exceed", "record",
        "strong", "upgrade", "bullish", "soar", "jump", "rise", "positive", "optimistic",
    }
    negative_words = {
        "drop", "fall", "miss", "loss", "decline", "downgrade", "bearish", "crash",
        "plunge", "weak", "concern", "risk", "warning", "sell-off", "negative", "pessimistic",
    }

    scores = []
    for text in texts:
        words = set(text.lower().split())
        pos   = len(words & positive_words)
        neg   = len(words & negative_words)
        score = (pos - neg) / max(pos + neg, 1)
        scores.append(score)

    avg    = sum(scores) / len(scores) if scores else 0
    label  = "positive" if avg > 0.1 else "negative" if avg < -0.1 else "neutral"
    return {
        "overall_sentiment": label,
        "sentiment_score":   round(avg, 3),
        "headline_count":    len(texts),
    }


# ── Agent ─────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are a professional financial analyst agent. Your job is to produce a concise, 
structured market brief for a given stock ticker or market topic.

You will be given:
- Recent price data and momentum signals
- Sentiment analysis results from recent news
- Any additional context

Produce a JSON market brief with this EXACT schema:
{
  "ticker": "...",
  "company": "...",
  "summary": "2-3 sentence executive summary",
  "price_signal": "bullish|bearish|neutral",
  "sentiment_signal": "positive|negative|neutral",
  "combined_signal": "buy|sell|hold|watch",
  "key_catalysts": ["..."],
  "risks": ["..."],
  "analyst_note": "1 sentence closing note"
}

Be factual, concise, and professional. Do not give financial advice — this is a research summary only."""


def run_market_agent(ticker: str) -> Dict:
    """
    Run the full market insights agentic workflow for a given ticker.
    Returns a structured market brief dict.
    """
    if not GENAI_AVAILABLE:
        raise ImportError("google-generativeai not installed.")
    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY not set.")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro", system_instruction=AGENT_SYSTEM_PROMPT)

    with tracer.trace("market_insights_agent") as span:
        span.set_input({"ticker": ticker})
        span.set_metadata({"agent_steps": 4})

        steps_log: List[Dict[str, Any]] = []
        t_total = time.time()

        # ── Step 1: Fetch price data ─────────────────────────────────────────
        t0 = time.time()
        price_data = _fetch_price_data(ticker)
        steps_log.append({"step": "fetch_price_data", "ms": round((time.time() - t0) * 1000, 2), "result": price_data})

        # ── Step 2: Simulate headline retrieval (real: integrate news API) ───
        # In production: call NewsAPI / Alpha Vantage News / GDELT
        # Here we use yfinance news as a lightweight substitute
        headlines = []
        if YF_AVAILABLE and "error" not in price_data:
            try:
                news = yf.Ticker(ticker).news or []
                headlines = [item.get("title", "") for item in news[:10] if item.get("title")]
            except Exception:
                pass
        steps_log.append({"step": "fetch_news", "headline_count": len(headlines)})

        # ── Step 3: Sentiment analysis ────────────────────────────────────────
        t0 = time.time()
        sentiment = _analyze_sentiment(headlines) if headlines else {
            "overall_sentiment": "neutral", "sentiment_score": 0.0, "headline_count": 0
        }
        steps_log.append({"step": "sentiment_analysis", "ms": round((time.time() - t0) * 1000, 2), **sentiment})

        # ── Step 4: LLM synthesis ─────────────────────────────────────────────
        context = f"""
Ticker: {ticker}
Price data: {json.dumps(price_data, indent=2)}
Sentiment analysis: {json.dumps(sentiment, indent=2)}
Sample headlines: {json.dumps(headlines[:5], indent=2)}
"""
        t0 = time.time()
        response = model.generate_content(context)
        llm_ms = (time.time() - t0) * 1000

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        brief = json.loads(raw)
        brief["steps_log"]   = steps_log
        brief["total_ms"]    = round((time.time() - t_total) * 1000, 2)
        brief["trace_id"]    = span.trace_id

        input_tokens  = getattr(response.usage_metadata, "prompt_token_count", 0)
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
        span.set_tokens(input=input_tokens, output=output_tokens)
        span.set_output({"signal": brief.get("combined_signal"), "ticker": ticker})

        # Eval score: confidence proxy from sentiment + price alignment
        price_bull = (price_data.get("pct_change_5d", 0) or 0) > 0
        sent_bull  = sentiment["overall_sentiment"] == "positive"
        alignment  = 1.0 if price_bull == sent_bull else 0.5
        span.set_score("signal_alignment", alignment, comment="Price and sentiment agreement")

        return brief
