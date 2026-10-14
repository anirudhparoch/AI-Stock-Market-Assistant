# Assignment 1.1 – AI Stock Market Assistant (LangChain + LangGraph)

A multi-agent AI assistant that analyses stock data, retrieves financial news with RAG, reasons with an LLM and gives a **simulated** BUY / SELL / HOLD recommendation with justification.

> ⚠️ Educational project. Recommendations are simulated, **not financial advice**.

## Deliverables
| Deliverable | File |
|---|---|
| Jupyter Notebook | `AI_Stock_Market_Assistant.ipynb` |
| Deployable application | `app.py` (Streamlit) + `stock_assistant/` package |
| Presentation | `presentation/AI_Stock_Market_Assistant.pptx` (+ PDF) |

## Architecture
```
START → supervisor ──(conditional)──┬─► market_data ──(conditional)──┬─► news_rag ──(conditional)──┬─► analyst → recommender → report → END
                                    │                                 └─► report (price only / error) └─► report (news only)
                                    ├─► news_rag
                                    └─► chat ⇄ tools  (ReAct loop with tools_condition)
```
| Agent | Responsibility |
|---|---|
| Supervisor | Reads the question, extracts the ticker, picks the route (`PydanticOutputParser`) |
| Market Data | yfinance prices, SMA20/50, RSI14, volatility, fundamentals, rule-based quant signal |
| News RAG | Fetches news (Tavily → Yahoo Finance → DuckDuckGo), FAISS retrieval, sentiment |
| Analyst | LLM reasoning over price, fundamentals and news |
| Recommender | Simulated BUY/SELL/HOLD, confidence, reasons and risks (Pydantic) |
| Report | Builds the final markdown answer |
| Chat (ReAct) | General questions; calls tools via `ToolNode` |

**State:** `StockState` (TypedDict) with an `add_messages` reducer, persisted per thread with `InMemorySaver` (conversation memory).

## Tech stack (all from class)
LangChain (`ChatGroq`, `ChatPromptTemplate`, `@tool`, LCEL runnables, `PydanticOutputParser`) · LangGraph (`StateGraph`, conditional edges, `ToolNode`, `tools_condition`, `InMemorySaver`) · FAISS · HuggingFace `all-MiniLM-L6-v2` embeddings · yfinance · Tavily · Streamlit.

## Setup on Mac (M1, 16 GB RAM)
1. Open this folder in VS Code: **File → Open Folder… → Assignment_1_1_AI_Stock_Market_Assistant**
2. Open the terminal (**Ctrl + `**) and run:
   ```bash
   bash setup_mac.sh
   ```
   This creates `.venv`, installs `requirements.txt` (~2 GB, mostly PyTorch for embeddings) and creates `.env`.
   *Needs Python 3.11 or 3.12. If you don't have it:* `brew install python@3.12` *or* `uv python install 3.12`.
3. Get a free Groq key at <https://console.groq.com/keys> and paste it into `.env`:
   ```
   GROQ_API_KEY=gsk_...
   ```
   Optional: `TAVILY_API_KEY` (better news). Without it, Yahoo Finance news is used.
4. **Notebook:** open `AI_Stock_Market_Assistant.ipynb` → *Select Kernel* (top right) → choose `.venv` → **Run All**.
5. **App:**
   ```bash
   source .venv/bin/activate
   streamlit run app.py
   ```
   or press **F5** in VS Code (launch config "Run Streamlit app"). It opens at <http://localhost:8501>.

### Run fully offline (optional)
Install [Ollama](https://ollama.com), run `ollama pull qwen3.5:2b`, then set `LLM_PROVIDER=ollama` in `.env`. A 2B model uses ~2–3 GB RAM, fine on 16 GB.

### Memory / performance notes
- Embedding model is ~90 MB and runs on CPU; total app RAM ≈ 1–1.5 GB.
- Groq free tier has per-minute token limits. A full analysis uses 4 LLM calls; if you hit a rate limit the client retries automatically. You can also set `GROQ_MODEL=llama-3.1-8b-instant`.


### Free API keys used (all set in `.env`)
| Service | Used for | Free key |
|---|---|---|
| Groq (`openai/gpt-oss-120b`) | Main LLM (`LLM_PROVIDER=groq`) | https://console.groq.com/keys |
| Google Gemini (`gemini-3.6-flash`) | Backup LLM: set `LLM_PROVIDER=gemini` if Groq hits its daily limit | https://aistudio.google.com/apikey |
| Hugging Face | Faster download of the local embedding model | https://huggingface.co/settings/tokens |
| Tavily (Stock Assistant only) | News search | https://app.tavily.com |
No paid service (e.g. OpenAI) is used anywhere.

## Deploy (Streamlit Community Cloud)
Push this folder to GitHub (`.env` is git-ignored) → <https://share.streamlit.io> → *New app* → main file `app.py` → *Advanced settings → Secrets*:
```toml
GROQ_API_KEY = "gsk_..."
GEMINI_API_KEY = "..."
HUGGINGFACE_API_KEY = "hf_..."
TAVILY_API_KEY = "tvly-..."
```

## Project structure
```
├── AI_Stock_Market_Assistant.ipynb   # step-by-step notebook
├── app.py                            # Streamlit app
├── stock_assistant/
│   ├── config.py      # .env, LLM + embeddings
│   ├── tools.py       # yfinance + news tools (@tool)
│   ├── news_rag.py    # FAISS RAG over news
│   ├── schemas.py     # LangGraph state + Pydantic schemas
│   ├── agents.py      # the specialised agents (nodes)
│   └── graph.py       # LangGraph workflow + conditional edges
├── presentation/
├── requirements.txt · .env.example · setup_mac.sh
```

## Sample questions
- *Should I buy Apple stock right now?*
- *What is the current price and RSI of TCS?*
- *Latest news sentiment on Nvidia*
- *What is a P/E ratio and what is Microsoft's?*
