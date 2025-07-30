# === build_embeddings.py ===
from extract import extract_from_multiple_pdfs
from chunking import prepare_chunks
from embedding import get_chunk_embeddings, save_embeddings_and_chunks

if __name__ == "__main__":
    pdf_files = [
        "BAJHLIP23020V012223.pdf",
        "CHOTGDP23004V012223.pdf",
        "EDLHLGA23009V012223.pdf",
        "HDFHLIP23024V072223.pdf",
        "ICIHLIP22012V012223.pdf"
    ]

    # Step 1: Extract text
    all_texts = extract_from_multiple_pdfs(pdf_files)

    # Step 2: Chunk
    all_chunks = prepare_chunks(all_texts)
    print(f"Total chunks created: {len(all_chunks)}")

    # Step 3: Embeddings
    embeddings = get_chunk_embeddings(all_chunks)

    # Step 4: Save
    save_embeddings_and_chunks(embeddings, all_chunks)
