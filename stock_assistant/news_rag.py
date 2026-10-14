"""
RAG over financial news.

Pipeline (same steps as the Day 13/14 QA chatbot):
    news articles -> Documents -> RecursiveCharacterTextSplitter -> HuggingFace embeddings
    -> FAISS vector store -> retriever -> context for the LLM
"""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_embeddings


def news_to_documents(news_items: list) -> list:
    docs = []
    for n in news_items:
        text = f"{n.get('title', '')}\n{n.get('content', '')}".strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"title": n.get("title", ""), "source": n.get("source", ""),
                          "url": n.get("url", ""), "published": n.get("published", "")},
            ))
    return docs


def build_news_vectorstore(news_items: list, embeddings=None):
    """Chunk + embed + index the news in FAISS."""
    docs = news_to_documents(news_items)
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings or get_embeddings())


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata.get('source', '')} {d.metadata.get('published', '')}) {d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def retrieve_news_context(news_items: list, question: str, k: int = 4, embeddings=None) -> tuple:
    """Return (context_text, retrieved_docs) for the question."""
    vs = build_news_vectorstore(news_items, embeddings)
    if vs is None:
        return "No recent news found.", []
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(question)
    return format_docs(docs), docs
