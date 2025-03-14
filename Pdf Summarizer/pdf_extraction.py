#pdf_extraction.py
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text.strip()
