#chunking.py
def chunk_text(text, max_length=1000, overlap=100):

    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + max_length
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += max_length - overlap
    return chunks
