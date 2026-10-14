"""
LangGraph multi-agent workflow.

                         ┌──────────────┐
            START ─────► │  supervisor  │
                         └──────┬───────┘
         route_after_supervisor │ (conditional edge)
     ┌──────────────────────────┼──────────────────────────┐
     ▼                          ▼                          ▼
┌─────────────┐          ┌─────────────┐            ┌────────────┐
│ market_data │          │  news_rag   │◄──┐        │    chat    │◄─┐
└──────┬──────┘          └──────┬──────┘   │        └─────┬──────┘  │
       │ route_after_market     │ route_after_news   tools_condition│
       ├──► news_rag ───────────┘          │              ▼         │
       │                        ▼          │        ┌──────────┐    │
       │                  ┌──────────┐     │        │  tools   │────┘
       │                  │ analyst  │     │        └──────────┘
       │                  └────┬─────┘     │
       │                       ▼           │
       │                 ┌─────────────┐   │
       │                 │ recommender │   │
       │                 └──────┬──────┘   │
       ▼                        ▼          │
     ┌───────────────────────────────┐     │
     │            report             │◄────┘ (news_only)
     └───────────────┬───────────────┘
                     ▼
                    END
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .agents import make_agents
from .config import get_llm
from .schemas import StockState
from .tools import STOCK_TOOLS


# ---------------- Conditional-edge router functions ----------------
def route_after_supervisor(state: StockState) -> str:
    intent = state.get("intent")
    if intent in ("full_analysis", "price_check"):
        return "market_data"
    if intent == "news_only":
        return "news_rag"
    return "chat"


def route_after_market(state: StockState) -> str:
    if state.get("error"):
        return "report"            # invalid ticker / no data -> skip the LLM agents
    if state.get("intent") == "price_check":
        return "report"
    return "news_rag"


def route_after_news(state: StockState) -> str:
    return "report" if state.get("intent") == "news_only" else "analyst"


def build_graph(llm=None, with_memory: bool = True):
    llm = llm or get_llm()
    agents = make_agents(llm)

    builder = StateGraph(StockState)

    # Nodes (each node = one specialised agent)
    builder.add_node("supervisor", agents["supervisor"])
    builder.add_node("market_data", agents["market_data"])
    builder.add_node("news_rag", agents["news_rag"])
    builder.add_node("analyst", agents["analyst"])
    builder.add_node("recommender", agents["recommender"])
    builder.add_node("report", agents["report"])
    builder.add_node("chat", agents["chat"])
    builder.add_node("tools", ToolNode(STOCK_TOOLS))

    # Edges
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_after_supervisor,
                                  {"market_data": "market_data", "news_rag": "news_rag", "chat": "chat"})
    builder.add_conditional_edges("market_data", route_after_market,
                                  {"news_rag": "news_rag", "report": "report"})
    builder.add_conditional_edges("news_rag", route_after_news,
                                  {"analyst": "analyst", "report": "report"})
    builder.add_edge("analyst", "recommender")
    builder.add_edge("recommender", "report")
    builder.add_edge("report", END)

    # ReAct loop for general chat questions (same pattern as the Day 16 chatbot)
    builder.add_conditional_edges("chat", tools_condition)   # -> "tools" or END
    builder.add_edge("tools", "chat")

    return builder.compile(checkpointer=InMemorySaver() if with_memory else None,
                           name="stock_market_multi_agent")


def ask(graph, query: str, thread_id: str = "1", ticker_hint: str = "", period: str = "6mo") -> dict:
    """Convenience helper: run one question through the graph."""
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"query": query, "ticker_hint": ticker_hint, "period": period}, config=config)
