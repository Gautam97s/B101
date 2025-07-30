import json
import re
import requests
from semanticSearch import semantic_search, load_embeddings_and_chunks

def query_llm(user_query, context, model_name="llama3"):
    """
    Send query + retrieved context to Ollama and return parsed JSON
    """
    prompt = f"""
Based on the following insurance policy clauses and this query:
Query: {user_query}

Policy Context (with sources):
{context}

ONLY return a valid JSON object in this format (no extra text):
{{
  "decision": "approved" or "rejected",
  "justification": "Concise reasoning",
  "cited_clauses": [
    "Copy-paste the exact sentences or clauses from the provided context that support the decision."
  ],
  "payout_amount": "amount if specified, else null",
  "source_docs": ["List of source document names used"]
}}
If no relevant clause is found, set cited_clauses to ["No direct clause found"].
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt},
            stream=True
        )
        raw = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                raw += data.get("response", "")
    except Exception as e:
        print("Error calling Ollama API:", e)
        return None

    # --- Try JSON parsing ---
    raw = re.sub(r'(\d),(\d)', r'\1\2', raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            cleaned = re.sub(r'(\d),(\d)', r'\1\2', match.group(0))
            return json.loads(cleaned)
        print("JSON extraction failed.\nRaw output:\n", raw)
        return None

def process_query(user_query, top_n=5):
    """
    Full pipeline: semantic search -> build context -> LLM reasoning
    """
    embeddings, chunks = load_embeddings_and_chunks()
    top_chunks = semantic_search(user_query, embeddings, chunks, top_n=top_n)

    # Build context with doc names
    context = ""
    source_docs = []
    for c in top_chunks:
        context += f"\n[Source: {c['doc']}]\n{c['chunk']}\n"
        if c["doc"] not in source_docs:
            source_docs.append(c["doc"])

    result = query_llm(user_query, context)

    # Ensure source_docs present
    if result and "source_docs" not in result:
        result["source_docs"] = source_docs

    # print("\n--- Semantic Search Results ---")
    # for c in top_chunks:
    #     print(f"\n[Doc: {c['doc']}] (score: {c['score']:.4f})\n{c['chunk'][:300]}...")

    print("\n--- Final LLM Decision ---")
    print(json.dumps(result, indent=2) if result else "No valid JSON decision returned.")
    return result

if __name__ == "__main__":
    query = "Normal delivery expenses claim under maternity benefit"
    process_query(query)
