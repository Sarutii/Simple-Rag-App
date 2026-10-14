from importlib import import_module

import numpy as np

store = import_module("05_create_vector_store")
tokenize = import_module("02_preprocessing").tokenize
vectors = import_module("04_vector_representation")


def hybrid_search(collection, query, k=10, alpha=None):

    data = collection.get(include=["documents", "metadatas", "embeddings"])

    if not data["ids"]:
        return []

    bm25_corpus = [tokenize(f"{m['title']} {doc}") for doc, m in zip(data["documents"], data["metadatas"])]
    bm25 = vectors.build_bm25(bm25_corpus)
    bm_scores = np.array(bm25.get_scores(tokenize(query)))

    query_embedding = vectors.encode_query(query)
    sem_scores = np.array(data["embeddings"]) @ query_embedding

    score_kwargs = {} if alpha is None else {"alpha": alpha}
    scores = vectors.hybrid_score(bm_scores, sem_scores, **score_kwargs)
    order = np.argsort(-scores)[:k]

    return [{
        "chunk_id": data["ids"][i],
        "document_id": data["metadatas"][i]["document_id"],
        "title": data["metadatas"][i]["title"],
        "is_current": data["metadatas"][i]["is_current"],
        "effective_date": data["metadatas"][i]["effective_date"],
        "chunk_text": data["documents"][i],
        "score": float(scores[i]),
    } for i in order]


if __name__ == "__main__":
    collection = store.create_vector_store()
    for row in hybrid_search(collection, "How much does an official transcript cost right now?", k=5):
        status = "CURRENT" if row["is_current"] else "OUTDATED"
        print(f"{row['score']:.3f}  doc{row['document_id']}  {status}  {row['title']}")



# result = collection.query(
#     query_embeddings=[embed_text(query)],
#     n_results=k,
#     include=["documents", "metadatas", "distances"],
# )