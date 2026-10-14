"""
Market-data and news tools.

Concepts used (from class):
- yfinance + @tool decorator                          (Day 17 - finGraph AI Agent)
- TavilySearch / DuckDuckGoSearchRun web search tools (Day 15 - tools demo)
- pandas for indicators                                (numpy_pandas classes)
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# 1. Plain Python helpers (easy to test on their own)
# ---------------------------------------------------------------------------
def fetch_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Daily OHLCV history from Yahoo Finance."""
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'. Check the symbol (e.g. AAPL, TCS.NS).")
    return df


def _clean(value, digits: int = 2):
    """Round numbers and turn NaN into None so they are JSON friendly."""
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        return value
    return round(float(value), digits)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_indicators(df: pd.DataFrame) -> dict:
    """Technical summary of a price history DataFrame."""
    close = df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi = compute_rsi(close)
    daily_ret = close.pct_change().dropna()

    last = close.iloc[-1]
    prev = close.iloc[-2] if len(close) > 1 else last

    def pct_back(days):
        return _clean((last / close.iloc[-days - 1] - 1) * 100) if len(close) > days else None

    summary = {
        "last_close": _clean(last),
        "previous_close": _clean(prev),
        "day_change_pct": _clean((last / prev - 1) * 100),
        "return_1m_pct": pct_back(21),
        "return_3m_pct": pct_back(63),
        "return_period_pct": _clean((last / close.iloc[0] - 1) * 100),
        "period_high": _clean(close.max()),
        "period_low": _clean(close.min()),
        "sma_20": _clean(sma20.iloc[-1]),
        "sma_50": _clean(sma50.iloc[-1]),
        "rsi_14": _clean(rsi.iloc[-1], 1),
        "annual_volatility_pct": _clean(daily_ret.std() * np.sqrt(252) * 100),
        "avg_volume": int(df["Volume"].tail(20).mean()) if "Volume" in df else None,
        "data_points": int(len(close)),
        "as_of": str(close.index[-1].date()),
    }

    # Simple rule-based trend label (gives the LLM a grounded starting point)
    if summary["sma_20"] and summary["sma_50"]:
        if last > summary["sma_20"] > summary["sma_50"]:
            summary["trend"] = "Uptrend (price > SMA20 > SMA50)"
        elif last < summary["sma_20"] < summary["sma_50"]:
            summary["trend"] = "Downtrend (price < SMA20 < SMA50)"
        else:
            summary["trend"] = "Sideways / mixed"
    else:
        summary["trend"] = "Not enough data for SMA50"

    summary["quant_signal"] = quant_signal(summary)
    return summary


def quant_signal(s: dict) -> dict:
    """Transparent score from -3 to +3 using trend, momentum and RSI."""
    score, reasons = 0, []
    if s.get("sma_20") and s.get("sma_50"):
        if s["last_close"] > s["sma_50"]:
            score += 1; reasons.append("Price above 50-day average")
        else:
            score -= 1; reasons.append("Price below 50-day average")
    if s.get("return_1m_pct") is not None:
        if s["return_1m_pct"] > 3:
            score += 1; reasons.append("Positive 1-month momentum")
        elif s["return_1m_pct"] < -3:
            score -= 1; reasons.append("Negative 1-month momentum")
    rsi = s.get("rsi_14")
    if rsi is not None:
        if rsi > 70:
            score -= 1; reasons.append(f"RSI {rsi} = overbought")
        elif rsi < 30:
            score += 1; reasons.append(f"RSI {rsi} = oversold")
    label = "BUY" if score >= 2 else "SELL" if score <= -2 else "HOLD"
    return {"score": score, "label": label, "reasons": reasons}


def chart_data(df: pd.DataFrame, rows: int = 180) -> list:
    """Last N rows of close + moving averages as a list of dicts (safe to keep in graph state)."""
    out = pd.DataFrame({
        "Close": df["Close"],
        "SMA20": df["Close"].rolling(20).mean(),
        "SMA50": df["Close"].rolling(50).mean(),
    }).tail(rows)
    out.index = out.index.strftime("%Y-%m-%d")
    out = out.round(2).reset_index().rename(columns={"index": "Date", "Date": "Date"})
    return out.where(pd.notnull(out), None).to_dict(orient="records")


def fetch_fundamentals(ticker: str) -> dict:
    """Company fundamentals (same yfinance fields as the finGraph class notebook)."""
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        info = {}
    keys = {
        "longName": "name", "sector": "sector", "industry": "industry", "currency": "currency",
        "marketCap": "market_cap", "trailingPE": "pe_ratio", "forwardPE": "forward_pe",
        "priceToBook": "price_to_book", "dividendYield": "dividend_yield",
        "fiftyTwoWeekHigh": "52w_high", "fiftyTwoWeekLow": "52w_low",
        "debtToEquity": "debt_to_equity", "ebitda": "ebitda", "totalRevenue": "total_revenue",
        "profitMargins": "profit_margin", "recommendationKey": "analyst_consensus",
    }
    data = {new: info.get(old) for old, new in keys.items()}
    data["name"] = data.get("name") or ticker
    return data


def _parse_yf_news(item: dict) -> dict:
    """yfinance has changed its news format over versions - handle both."""
    c = item.get("content", item)
    url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link")
    provider = c.get("provider")
    source = provider.get("displayName") if isinstance(provider, dict) else c.get("publisher", "Yahoo Finance")
    published = c.get("pubDate") or c.get("providerPublishTime")
    if isinstance(published, (int, float)):
        published = datetime.fromtimestamp(published).strftime("%Y-%m-%d")
    return {
        "title": c.get("title", ""),
        "content": c.get("summary") or c.get("description") or c.get("title", ""),
        "source": source or "Yahoo Finance",
        "url": url or "",
        "published": str(published or "")[:10],
    }


def fetch_news(ticker: str, company_name: str = "", max_items: int = 8) -> list:
    """
    Collect recent news about the company.
    Order: Tavily (if TAVILY_API_KEY set) -> Yahoo Finance news -> DuckDuckGo.
    """
    query = f"{company_name or ticker} ({ticker}) stock news"
    items = []

    if os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_tavily import TavilySearch
            res = TavilySearch(max_results=max_items, topic="news").invoke({"query": query})
            for r in (res.get("results", []) if isinstance(res, dict) else []):
                items.append({
                    "title": r.get("title", ""), "content": r.get("content", ""),
                    "source": "Tavily", "url": r.get("url", ""), "published": r.get("published_date", "")[:10] if r.get("published_date") else "",
                })
        except Exception as e:
            print("Tavily search failed:", e)

    if not items:
        try:
            for raw in (yf.Ticker(ticker).news or [])[:max_items]:
                parsed = _parse_yf_news(raw)
                if parsed["title"]:
                    items.append(parsed)
        except Exception as e:
            print("Yahoo news failed:", e)

    if not items:
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            text = DuckDuckGoSearchRun().invoke(query)
            if text:
                items.append({"title": f"Web search: {query}", "content": text, "source": "DuckDuckGo", "url": "", "published": ""})
        except Exception as e:
            print("DuckDuckGo search failed:", e)

    return items[:max_items]


# ---------------------------------------------------------------------------
# 2. LangChain tools (used by the chat / ReAct agent with ToolNode)
# ---------------------------------------------------------------------------
@tool
def get_stock_price(ticker: str) -> dict:
    """Return the latest closing price, daily change and 1-month return for a stock ticker.
    Example: AAPL, MSFT, TSLA, RELIANCE.NS, TCS.NS, INFY.NS"""
    try:
        s = compute_indicators(fetch_price_history(ticker, "3mo"))
        return {k: s[k] for k in ("last_close", "previous_close", "day_change_pct", "return_1m_pct", "as_of")}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_technical_indicators(ticker: str, period: str = "6mo") -> dict:
    """Return technical indicators (SMA20, SMA50, RSI14, volatility, trend, quant signal) for a ticker.
    period can be 1mo, 3mo, 6mo, 1y, 2y."""
    try:
        return compute_indicators(fetch_price_history(ticker, period))
    except Exception as e:
        return {"error": str(e)}


@tool
def get_company_fundamentals(ticker: str) -> dict:
    """Return company fundamentals such as sector, market cap, P/E ratio, debt to equity, EBITDA and revenue."""
    return fetch_fundamentals(ticker)


@tool
def search_stock_news(ticker: str, company_name: str = "") -> list:
    """Return recent news headlines and summaries about a company/stock."""
    return [{"title": n["title"], "summary": n["content"][:400], "source": n["source"]}
            for n in fetch_news(ticker, company_name, max_items=5)]


STOCK_TOOLS = [get_stock_price, get_technical_indicators, get_company_fundamentals, search_stock_news]
