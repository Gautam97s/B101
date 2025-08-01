import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

def load_embeddings_and_chunks(embed_file="embeddings.npy", chunk_file="chunks.pkl"):
    """
    Loads embeddings and chunk metadata.
    """
    embeddings = np.load(embed_file)
    with open(chunk_file, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded embeddings shape {embeddings.shape} and {len(chunks)} chunks")
    return embeddings, chunks

def semantic_search(query, embeddings, chunks, top_n=3, model_name="all-MiniLM-L6-v2"):
    """
    Perform semantic search to find the most relevant chunks.
    Args:
        query (str): User query
        embeddings (np.ndarray): Embeddings of all chunks
        chunks (list): List of dicts [{"doc": ..., "chunk": ...}]
        top_n (int): Number of top results to return
    Returns:
        list of dicts: [{"doc": ..., "chunk": ..., "score": float}]
    """
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = np.dot(embeddings, query_embedding)

    top_indices = similarities.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "doc": chunks[idx]["doc"],
            "chunk": chunks[idx]["chunk"],
            "score": float(similarities[idx])
        })
    return results

if __name__ == "__main__":
    # Example usage
    embeddings, chunks = load_embeddings_and_chunks()
    query = "Is knee surgery covered for a 46-year-old with a 3-month-old policy?"
    results = semantic_search(query, embeddings, chunks, top_n=3)

    print("\nTop relevant chunks:")
    for r in results:
        print(f"\n[Doc: {r['doc']}] (score: {r['score']:.4f})\n{r['chunk'][:400]}...")
