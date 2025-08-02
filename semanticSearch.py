# === semanticSearch.py ===
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# --------- Config ---------
PINECONE_API_KEY = os.getenv("pcsk_Ssru2_DFFfktbQquCFujGEmGsAJ2KkRaJBvSiHBExHmSG2mxgCEH7ZHnabamVLwFXdn7e")
INDEX_NAME = "insurance-index"
MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize Sentence Transformer (once)
model = SentenceTransformer(MODEL_NAME)

# Initialize Pinecone client (only if API key present)
index = None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        print(f"[Pinecone] Connected to index: {INDEX_NAME}")
    except Exception as e:
        print(f"[Pinecone] Connection failed: {e}")
        index = None

# --------- Local Embeddings Loader ---------
def load_embeddings_and_chunks(embed_file="embeddings.npy", chunk_file="chunks.pkl"):
    embeddings = np.load(embed_file)
    with open(chunk_file, "rb") as f:
        chunks = pickle.load(f)
    print(f"[Local] Loaded embeddings shape {embeddings.shape} and {len(chunks)} chunks")
    return embeddings, chunks

# --------- Local Semantic Search ---------
def local_semantic_search(query, embeddings, chunks, top_n=3):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = np.dot(embeddings, query_embedding)
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [
        {
            "doc": chunks[idx]["doc"],
            "chunk": chunks[idx]["chunk"],
            "score": float(similarities[idx])
        }
        for idx in top_indices
    ]

# --------- Pinecone Semantic Search ---------
def pinecone_semantic_search(query, top_n=3):
    if not index:
        raise RuntimeError("Pinecone index not initialized.")
    query_embedding = model.encode([query])[0]
    results = index.query(vector=query_embedding.tolist(), top_k=top_n, include_metadata=True)
    return [
        {
            "doc": match["metadata"].get("file", "unknown"),
            "chunk": match["metadata"].get("text", ""),
            "score": match["score"]
        }
        for match in results["matches"]
    ]

# --------- Smart Wrapper (auto Pinecone -> fallback) ---------
def semantic_search(query, embeddings=None, chunks=None, top_n=3):
    """
    Uses Pinecone if available, otherwise local search.
    """
    try:
        if index:
            return pinecone_semantic_search(query, top_n=top_n)
        else:
            if embeddings is None or chunks is None:
                raise RuntimeError("Local embeddings/chunks not loaded.")
            return local_semantic_search(query, embeddings, chunks, top_n=top_n)
    except Exception as e:
        print(f"[semantic_search] Pinecone failed ({e}), using local search.")
        if embeddings is None or chunks is None:
            raise RuntimeError("No fallback data provided.")
        return local_semantic_search(query, embeddings, chunks, top_n=top_n)

# --------- CLI Testing ---------
if __name__ == "__main__":
    try:
        from embedding import load_embeddings_and_chunks as load_local
        embeddings, chunks = load_local()
    except:
        embeddings, chunks = None, None

    query = "Is knee surgery covered for a 46-year-old with a 3-month-old policy?"
    results = semantic_search(query, embeddings, chunks, top_n=3)
    print("\nTop relevant chunks:")
    for r in results:
        print(f"\n[Doc: {r['doc']}] (score: {r['score']:.4f})\n{r['chunk'][:400]}...")
