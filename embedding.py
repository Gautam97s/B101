# === embedding.py ===
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def get_chunk_embeddings(chunks: list, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings

def save_embeddings_and_chunks(embeddings, chunks, embed_file="embeddings.npy", chunk_file="chunks.pkl"):
    np.save(embed_file, embeddings)
    with open(chunk_file, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved embeddings ({embeddings.shape}) and {len(chunks)} chunks")

def load_embeddings_and_chunks(embed_file="embeddings.npy", chunk_file="chunks.pkl"):
    embeddings = np.load(embed_file)
    with open(chunk_file, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded embeddings {embeddings.shape} and {len(chunks)} chunks")
    return embeddings, chunks
