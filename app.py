"""
Streamlit web app for the AI Stock Market Assistant.

Run:   streamlit run app.py
"""
import uuid

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Stock Market Assistant", page_icon="📈", layout="wide")

# When deployed on Streamlit Community Cloud, API keys come from "Secrets" instead of .env
import os  # noqa: E402
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from stock_assistant.config import provider_label  # noqa: E402
from stock_assistant.graph import build_graph  # noqa: E402

NODE_LABELS = {
    "supervisor": "🧭 Supervisor agent understood the question",
    "market_data": "📊 Market-data agent fetched prices & indicators",
    "news_rag": "📰 News RAG agent retrieved & scored the news",
    "analyst": "🧠 Analyst agent wrote the insights",
    "recommender": "🎯 Recommendation agent decided BUY / SELL / HOLD",
    "report": "📝 Report assembled",
    "chat": "💬 Chat agent responded",
    "tools": "🛠️ Tool called",
}


@st.cache_resource(show_spinner="Loading the multi-agent graph…")
def load_graph():
    return build_graph()


def run_graph(graph, payload: dict, thread_id: str) -> dict:
    """Stream node-by-node updates so the user can watch the agents work."""
    config = {"configurable": {"thread_id": thread_id}}
    with st.status("Agents are working…", expanded=True) as status:
        for update in graph.stream(payload, config=config, stream_mode="updates"):
            for node in update:
                st.write(NODE_LABELS.get(node, node))
        status.update(label="Done", state="complete", expanded=False)
    return graph.get_state(config).values


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.title("📈 Stock Assistant")
    st.caption(f"LLM: `{provider_label()}`")
    st.markdown(
        "**Ticker examples**  \n"
        "US: `AAPL`, `MSFT`, `NVDA`, `TSLA`  \n"
        "India (NSE): `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`"
    )
    period = st.selectbox("Price history period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    if st.button("🔄 New chat"):
        st.session_state.pop("chat_thread", None)
        st.session_state.pop("chat_history", None)
    st.info("Educational project. Recommendations are simulated, not financial advice.", icon="⚠️")

try:
    graph = load_graph()
except Exception as e:
    st.error(f"Could not start the LLM: {e}\n\nCheck your `.env` file (see README).")
    st.stop()

tab_analyze, tab_chat, tab_graph = st.tabs(["🔍 Analyze a stock", "💬 Chat with the assistant", "🕸️ Agent graph"])

# ---------------------------------------------------------------- Tab 1: Analyze
with tab_analyze:
    c1, c2 = st.columns([1, 2])
    ticker = c1.text_input("Ticker", value="AAPL").strip().upper()
    question = c2.text_input("Question (optional)", value="Should I buy, sell or hold this stock?")

    if st.button("Run analysis", type="primary") and ticker:
        result = run_graph(
            graph,
            {"query": f"{question} Stock: {ticker}", "ticker_hint": ticker, "period": period},
            thread_id=f"analyze-{uuid.uuid4()}",
        )
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if result:
        if result.get("error"):
            st.error(result["error"])
        else:
            p, rec = result.get("price_summary", {}), result.get("recommendation", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Last close", p.get("last_close"), f"{p.get('day_change_pct')}%")
            m2.metric("RSI (14)", p.get("rsi_14"))
            m3.metric("News sentiment", result.get("news_sentiment", {}).get("sentiment", "-"))
            m4.metric("Simulated call", rec.get("action", "-"), f"{rec.get('confidence', '-')}% confidence",
                      delta_color="off")
            if result.get("price_chart"):
                df = pd.DataFrame(result["price_chart"]).set_index("Date")
                st.line_chart(df)
            st.markdown(result.get("final_report", ""))
            with st.expander("Retrieved news context (RAG)"):
                st.text(result.get("news_context", ""))

# ---------------------------------------------------------------- Tab 2: Chat
with tab_chat:
    st.caption("Ask anything: *“Analyse TCS.NS”*, *“Latest news on Nvidia”*, *“What is the RSI of MSFT?”*, "
               "*“What is a P/E ratio?”*. The assistant remembers the conversation.")
    st.session_state.setdefault("chat_thread", str(uuid.uuid4()))
    st.session_state.setdefault("chat_history", [])

    for role, text in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(text)

    if prompt := st.chat_input("Ask about a stock…"):
        st.session_state["chat_history"].append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            res = run_graph(graph, {"query": prompt, "ticker_hint": "", "period": period},
                            thread_id=st.session_state["chat_thread"])
            answer = res.get("final_report") or res["messages"][-1].content
            st.markdown(answer)
        st.session_state["chat_history"].append(("assistant", answer))

# ---------------------------------------------------------------- Tab 3: Graph
with tab_graph:
    st.markdown("**Multi-agent workflow built with LangGraph** (conditional edges are dotted).")
    try:
        st.image(graph.get_graph().draw_mermaid_png())
    except Exception:
        st.code(graph.get_graph().draw_mermaid(), language="text")
