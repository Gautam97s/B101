from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import os
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from chunking import prepare_chunks
from embedding import get_chunk_embeddings
from semanticSearch import semantic_search
from llm_query import query_llm

# Load environment variables
load_dotenv()
API_TOKEN = os.getenv("TOKEN")

app = FastAPI(root_path="/api/v1")

# === Input model ===
class HackathonRequest(BaseModel):
    documents: str   # URL to the policy PDF
    questions: List[str]

@app.post("/hackrx/run")
def run_hackathon(request: HackathonRequest, authorization: str = Header(None)):
    # === Auth check (skip if no TOKEN in .env) ===
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # === Download the document ===
    try:
        response = requests.get(request.documents)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Document download failed: {str(e)}")

    # === Extract text from PDF ===
    try:
        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF text extraction failed: {str(e)}")

    # === Chunk + Embed dynamically ===
    chunks = prepare_chunks({"url_doc": text})
    embeddings = get_chunk_embeddings(chunks)

    answers = []
    for q in request.questions:
        # Semantic search to get top relevant chunks
        top_chunks = semantic_search(q, embeddings, chunks, top_n=3)

        # Build context for LLM
        context = ""
        for c in top_chunks:
            context += f"\n[Source: {c['doc']}]\n{c['chunk']}\n"

        # Query LLM to generate answer
        llm_result = query_llm(q, context)

        # Extract only "justification" or fallback
        if llm_result and "justification" in llm_result:
            answers.append(llm_result["justification"])
        else:
            answers.append("No relevant information found.")

    return {"answers": answers}
