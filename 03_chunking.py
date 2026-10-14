from importlib import import_module

documents = import_module("01_documents").documents
build_search_text = import_module("02_preprocessing").build_search_text

CHUNK_SIZE = 30
CHUNK_OVERLAP = 10


def chunk_fixed(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_chunks():
    rows = []
    for doc in documents:
        for index, text in enumerate(chunk_fixed(doc["text"])):
            rows.append({
                "chunk_id": f"doc{doc['document_id']}_chunk{index}",
                "document_id": doc["document_id"],
                "title": doc["title"],
                "effective_date": doc["effective_date"],
                "is_current": doc["is_current"],
                "chunk_text": text,
                "search_text": build_search_text(doc["title"], doc["department"], doc["doc_type"], text),
            })
    return rows


if __name__ == "__main__":
    chunks = build_chunks()
    print("chunks:", len(chunks), "| current:", sum(c["is_current"] for c in chunks))
