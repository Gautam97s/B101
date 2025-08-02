from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import os
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from chunking import prepare_chunks
from embedding import get_chunk_embeddings, load_embeddings_and_chunks
from semanticSearch import semantic_search
from llm_query import query_llm

# Load environment variables
load_dotenv()
API_TOKEN = os.getenv("TOKEN")

app = FastAPI(root_path="/api/v1")

# Preload local embeddings & chunks
local_embeddings, local_chunks = load_embeddings_and_chunks()

class HackathonRequest(BaseModel):
    documents: str | None = None
    questions: List[str]

@app.post("/hackrx/run")
def run_hackathon(request: HackathonRequest, authorization: str = Header(None)):
    # === Authentication check ===
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Default to using local embeddings unless we dynamically process a doc
    use_local = False
    chunks, embeddings = None, None

    # --- Handle dynamic document URL ---
    if request.documents:
        try:
            response = requests.get(request.documents, timeout=10)
            response.raise_for_status()

            # Extract text
            pdf_file = BytesIO(response.content)
            reader = PdfReader(pdf_file)
            text = "".join([page.extract_text() or "" for page in reader.pages])

            # Dynamic chunking & embeddings (not stored in Pinecone for this run)
            chunks = prepare_chunks({"url_doc": text})
            embeddings = get_chunk_embeddings(chunks)
        except Exception as e:
            print(f"Error processing document URL: {e}")
            use_local = True
    else:
        use_local = True

    # Use precomputed local embeddings if no document URL or error occurred
    if use_local:
        print("Falling back to local precomputed embeddings.")
        chunks, embeddings = local_chunks, local_embeddings

    answers = []
    for q in request.questions:
        # --- Semantic search (Pinecone -> fallback) ---
        top_chunks = semantic_search(q, embeddings, chunks, top_n=3)

        # --- Build context for LLM ---
        context = "\n".join([f"[Source: {c['doc']}]\n{c['chunk']}" for c in top_chunks])

        # --- LLM reasoning ---
        llm_result = query_llm(q, context)
        answers.append(llm_result["justification"] if llm_result and "justification" in llm_result
                       else "No relevant information found.")

    return {"answers": answers}
