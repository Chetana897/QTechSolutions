#app.py
import streamlit as st
import nltk
from pdf_extraction import extract_text_from_pdf
from text_preprocessing import preprocess_text
from chunking import chunk_text
from summarization import extractive_summary

# Download necessary NLTK data for tokenization
nltk.download("punkt")

st.title("Enhanced PDF Summarization Tool")

# File uploader widget for PDF files
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    # Save the uploaded PDF to a temporary file
    temp_pdf_path = "temp.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Step 1: Extract text from PDF
    raw_text = extract_text_from_pdf(temp_pdf_path)
    st.write("Extracted text length:", len(raw_text))

    # Step 2: Preprocess the extracted text
    processed_text = preprocess_text(raw_text)
    st.subheader("Processed Text Preview")
    st.text_area("PDF Text", processed_text, height=400)

    # Step 3: Chunk the text (with overlap for context preservation)
    chunks = chunk_text(processed_text, max_length=1000, overlap=100)
    st.write("Number of chunks:", len(chunks))

    # Step 4: Summarize each chunk (internally, not displayed)
    chunk_summaries = [extractive_summary(chunk, num_sentences=5) for chunk in chunks]

    # Step 5: Combine chunk summaries and generate a final summary.
    # Increase the number of sentences for the final summary if needed.
    combined_summary_text = " ".join(chunk_summaries)
    final_summary = extractive_summary(combined_summary_text, num_sentences=10)  # Adjust num_sentences here

    st.subheader("Final Summary")
    st.write(final_summary)
