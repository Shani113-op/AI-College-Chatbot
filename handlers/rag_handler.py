import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import retrieve_context

def safe_retrieve_context(user_text, top_k=2):
    """
    Retrieve relevant docs + metadata for a user query.
    Falls back to [] on errors.
    Returns: list of dicts → [{ "doc": str, "metadata": {...} }, ...]
    """
    try:
        results = retrieve_context(user_text, top_k=top_k)
        formatted = [{"doc": doc, "metadata": meta} for doc, meta in results]
        return formatted
    except Exception as e:
        print(f"⚠️ RAG retrieval failed: {e}")
        return []
