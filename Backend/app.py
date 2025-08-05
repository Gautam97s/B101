from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import os
import uuid
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from chunking import prepare_chunks
from embedding import load_embeddings_and_chunks
from semanticSearch import semantic_search
from llm_query import query_llm
from fastapi.middleware.cors import CORSMiddleware

# --------- Environment ---------
load_dotenv()
API_TOKEN = os.getenv("TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "insurance-index"

# --------- FastAPI App ---------
app = FastAPI(root_path="/api/v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Local Fallback Data ---------
local_embeddings, local_chunks = load_embeddings_and_chunks()

# --------- Pinecone Init ---------
pc = None
index = None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,  # for all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        index = pc.Index(INDEX_NAME)
        print(f"[Pinecone] Connected to index: {INDEX_NAME}")
    except Exception as e:
        print(f"[Pinecone] Init failed: {e}")
        index = None

# --------- Embedding Model ---------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

class HackathonRequest(BaseModel):
    documents: str | None = None
    questions: List[str]

@app.post("/hackrx/run")
def run_hackathon(request: HackathonRequest, authorization: str = Header(None)):
    # --- Auth check ---
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    namespace = "default"
    answers = []

    # --- Handle dynamic document URL ---
    if request.documents and index:
        namespace = f"dynamic-{uuid.uuid4().hex[:8]}"
        try:
            response = requests.get(request.documents, timeout=10)
            response.raise_for_status()
            pdf_file = BytesIO(response.content)
            reader = PdfReader(pdf_file)
            text = "".join([page.extract_text() or "" for page in reader.pages])

            # --- Chunk & embed ---
            chunks = prepare_chunks({"url_doc": text})
            texts = [c["chunk"] for c in chunks]
            embeddings = embed_model.encode(texts, normalize_embeddings=True)

            # --- Upsert to Pinecone ---
            vectors = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                vectors.append((
                    f"url_chunk_{i}",
                    emb.tolist(),
                    {"file": "url_doc", "text": chunk["chunk"]}
                ))
            index.upsert(vectors=vectors, namespace=namespace)
            print(f"[Pinecone] Inserted {len(vectors)} vectors to namespace={namespace}")
        except Exception as e:
            print(f"Error processing document URL: {e}")
            namespace = "default"

    for q in request.questions:
        if index:
            # --- Query Pinecone ---
            query_emb = embed_model.encode([q])[0]
            results = index.query(
                vector=query_emb.tolist(),
                top_k=3,
                include_metadata=True,
                namespace=namespace
            )
            top_chunks = [
                {
                    "doc": match["metadata"].get("file", "unknown"),
                    "chunk": match["metadata"].get("text", ""),
                    "score": match["score"]
                }
                for match in results.get("matches", [])
            ]
        else:
            # --- Fallback to local search ---
            print("[Fallback] Using local semantic search")
            top_chunks = semantic_search(q, local_embeddings, local_chunks, top_n=3)

        # --- Build context for LLM ---
        context = "\n".join([f"[Source: {c['doc']}]\n{c['chunk']}" for c in top_chunks])
        llm_result = query_llm(q, context)
        answers.append(
            llm_result["justification"] if llm_result and "justification" in llm_result
            else "No relevant information found."
        )

    return {"answers": answers}
