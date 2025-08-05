import os
from extract import extract_from_multiple_pdfs
from chunking import prepare_chunks
from embedding import get_chunk_embeddings, save_embeddings_and_chunks

if __name__ == "__main__":
    pdf_dir = "Docs"

    # Get all .pdf files in Docs/
    pdf_files = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("No PDF files found in 'Docs/' folder.")
        exit(1)

    print(f"Found {len(pdf_files)} PDF(s).")

    # Step 1: Extract text
    all_texts = extract_from_multiple_pdfs(pdf_files)

    # Step 2: Chunk
    all_chunks = prepare_chunks(all_texts)
    print(f"Total chunks created: {len(all_chunks)}")

    # Step 3: Embeddings
    embeddings = get_chunk_embeddings(all_chunks)

    # Step 4: Save
    save_embeddings_and_chunks(embeddings, all_chunks)
