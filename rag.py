import os
import json
import requests
import chromadb

os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Ollama config
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "phi"

# ChromaDB client
chroma_client = chromadb.PersistentClient(path="college_db")

collection = chroma_client.get_or_create_collection(
    name="colleges",
    embedding_function=None
)

def embed_text(text: str):
    """Get embeddings from Ollama"""
    try:
        resp = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
        resp.raise_for_status()
        return resp.json().get("embedding", [])
    except Exception as e:
        print(f"⚠️ Embedding failed: {e}")
        return []

def college_to_text(college: dict) -> str:
    """Convert JSON college entry to readable text"""
    return (
        f"{college.get('name', 'Unknown')} - {college.get('location', 'Unknown')}. "
        f"Fees: {college.get('fees', 'NA')}, "
        f"Placement: {college.get('placementRate', 'NA')}, "
        f"Cutoffs: {college.get('cutOffs', 'NA')}, "
        f"Deadline: {college.get('admissionDeadline', 'NA')}, "
        f"Website: {college.get('website', 'NA')}"
    )

def build_vector_db(college_file="data/staticColleges.json"):
    """Rebuild collection from JSON dataset"""
    try:
        chroma_client.delete_collection("colleges")
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name="colleges",
        embedding_function=None
    )

    with open(college_file, "r", encoding="utf-8") as f:
        colleges = json.load(f)

    for idx, college in enumerate(colleges):
        doc_text = college_to_text(college)
        vector = embed_text(doc_text)
        if vector:
            collection.add(
                documents=[doc_text],
                embeddings=[vector],
                ids=[str(idx)],
                metadatas=[{"name": college.get("name"), "location": college.get("location")}]
            )

    print(f"✅ Stored {len(colleges)} colleges in vector DB")

def retrieve_context(query, top_k=2):
    """Retrieve similar college entries for a query"""
    query_vec = embed_text(query)
    if not query_vec:
        return []

    results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return list(zip(docs, metas))
