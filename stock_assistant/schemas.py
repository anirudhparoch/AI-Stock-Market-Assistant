"""
State and output schemas.

Concepts used (from class):
- TypedDict + Annotated + add_messages for LangGraph state   (Day 16)
- Pydantic BaseModel + Field + Literal for structured output (Day 10 - PydanticOutputParser, Day 18 - bwa)
"""
from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ---------------- Structured LLM outputs ----------------
class QueryIntent(BaseModel):
    ticker: str = Field(description="Yahoo Finance ticker symbol in UPPER CASE, e.g. AAPL, MSFT, TSLA, "
                                    "RELIANCE.NS, TCS.NS. Use an empty string if no company/stock is mentioned.")
    company_name: str = Field(description="Company name if known, else empty string.")
    intent: Literal["full_analysis", "price_check", "news_only", "general"] = Field(
        description="full_analysis = should I buy/sell/hold, analyse, outlook, recommendation; "
                    "price_check = only price / technical numbers; "
                    "news_only = only latest news / sentiment; "
                    "general = finance concept question, chit-chat, or a follow-up question about the "
                    "previous answer (e.g. 'what risks did you mention?').")


class NewsSentiment(BaseModel):
    sentiment: Literal["Positive", "Neutral", "Negative"] = Field(description="Overall news sentiment for the stock")
    score: float = Field(description="Sentiment score between -1 (very negative) and 1 (very positive)")
    key_points: List[str] = Field(description="3 to 5 short key points from the news, each citing [n]")


class Recommendation(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"] = Field(description="Simulated recommendation")
    confidence: int = Field(description="Confidence from 0 to 100")
    time_horizon: str = Field(description="e.g. 'Short term (1-3 months)'")
    justification: List[str] = Field(description="3 to 5 reasons grounded in the given data")
    risks: List[str] = Field(description="2 to 4 key risks")


# ---------------- LangGraph shared state ----------------
class StockState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]   # conversation memory (chat)
    query: str               # user question
    ticker_hint: str         # optional ticker typed in the UI
    period: str              # price history period: 1mo, 3mo, 6mo, 1y, 2y
    ticker: str
    company_name: str
    intent: str
    price_summary: dict
    price_chart: list
    fundamentals: dict
    news_items: list
    news_context: str
    news_sentiment: dict
    analysis: str
    recommendation: dict
    final_report: str
    error: str
