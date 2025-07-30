# === extract.py ===
import os
import PyPDF2

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a single PDF file.
    """
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_from_multiple_pdfs(pdf_paths: list) -> dict:
    """
    Extract text from multiple PDFs.
    Returns: { "filename.pdf": "full extracted text", ... }
    """
    all_texts = {}
    for path in pdf_paths:
        file_name = os.path.basename(path)
        print(f"Extracting text from {file_name}...")
        text = extract_text_from_pdf(path)
        all_texts[file_name] = text
    return all_texts
