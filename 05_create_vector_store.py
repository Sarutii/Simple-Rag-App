from importlib import import_module
from pathlib import Path

import chromadb
from chromadb.config import Settings

vectors = import_module("04_vector_representation")

DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "course_support_docs"


def create_vector_store():
    embeddings = vectors.encode_chunks()

    client = chromadb.PersistentClient(
        path=str(DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(COLLECTION_NAME)

    collection.upsert(
        ids=[c["chunk_id"] for c in vectors.chunks],
        documents=[c["chunk_text"] for c in vectors.chunks],
        metadatas=[
            {
                "document_id": c["document_id"],
                "title": c["title"],
                "is_current": c["is_current"],
                "effective_date": c["effective_date"],
            }
            for c in vectors.chunks
        ],
        embeddings=embeddings.tolist(),
    )
    return collection


if __name__ == "__main__":
    collection = create_vector_store()
    print("chroma collection count:", collection.count())
