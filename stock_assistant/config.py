"""
Configuration: loads API keys from .env and creates the LLM + embedding model.

Concepts used (from class):
- python-dotenv + os.environ for API keys            (Day 9/10)
- ChatGroq "openai/gpt-oss-20b"                      (Day 10, 13-18)
- ChatOllama for a fully local model                 (Day 13 - QA_chatbot_ollama)
- HuggingFaceEmbeddings "all-MiniLM-L6-v2"          (Day 13/14 - QA_chatbot, hybrid_search)
"""
import os
import re
import warnings
from functools import lru_cache

# --- Mac (Apple Silicon) safety flags -------------------------------------
# FAISS and PyTorch both ship their own OpenMP library; on macOS this can crash
# with "OMP: Error #15". These flags avoid that and silence tokenizer warnings.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv  # noqa: E402

warnings.filterwarnings("ignore")
load_dotenv()
# Hugging Face token (free) -> faster, rate-limit-free download of the embedding model
if os.getenv("HUGGINGFACE_API_KEY") and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_KEY")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()        # groq | gemini (free keys) | ollama (free, local)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")   # same model used in class
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_llm(temperature: float = 0.2):
    """Return a LangChain chat model based on LLM_PROVIDER in the .env file."""
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)

    if LLM_PROVIDER == "gemini":
        # Google Gemini - free key from https://aistudio.google.com/apikey  (same class as Day 9/15/17 notebooks)
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError("GEMINI_API_KEY is missing. Add it to your .env file.")
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature, google_api_key=key, max_retries=3)

    # Default: Groq (free API key from https://console.groq.com)
    from langchain_groq import ChatGroq
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is missing. Add it to your .env file.")
    kwargs = dict(model=GROQ_MODEL, temperature=temperature, max_retries=5)
    if "gpt-oss" in GROQ_MODEL:
        kwargs["reasoning_effort"] = "low"   # fewer hidden reasoning tokens -> stays inside free-tier limits
    return ChatGroq(**kwargs)


@lru_cache(maxsize=1)
def get_embeddings():
    """Small local embedding model (~90 MB). Runs comfortably on an M1 with 16 GB RAM."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def strip_think(text) -> str:
    """Some local reasoning models (e.g. qwen3 on Ollama) wrap their thoughts in <think> tags."""
    text = text if isinstance(text, str) else getattr(text, "content", str(text))
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def provider_label() -> str:
    model = {"groq": GROQ_MODEL, "gemini": GEMINI_MODEL, "ollama": OLLAMA_MODEL}.get(LLM_PROVIDER, GROQ_MODEL)
    return f"{LLM_PROVIDER}:{model}"
