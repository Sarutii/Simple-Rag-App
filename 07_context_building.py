from importlib import import_module

store = import_module("05_create_vector_store")
retrieve = import_module("06_retrieve")

# chromadb has to make its first native call before pandas enters this
# process, or this import order segfaults on this setup - store is already
# imported above, pandas comes after.
import re
import pandas as pd


def build_context_package(
    collection,
    query,
    retrieval_k=8,
    alpha=None,  # None = the alpha measured on this corpus (04_vector_representation.ALPHA)
    max_context_chunks=3,
    max_chunks_per_document=1,
    word_budget=150,
    current_only=True,
    min_score_ratio=0.40,
    min_absolute_score=0.05,
):

    rows = retrieve.hybrid_search(collection, query, k=retrieval_k, alpha=alpha)
    candidates = pd.DataFrame(rows)

    if current_only and len(candidates):
        candidates = candidates[candidates["is_current"]].reset_index(drop=True)

    if len(candidates):
        candidates = candidates.sort_values(
            by=[ "score", "effective_date"],
            ascending=[False, False],
        ).reset_index(drop=True)

    max_score = candidates["score"].max() if len(candidates) else 0.0
    selected_rows = []
    seen_texts = set()
    per_document_counts = {}
    used_words = 0

    for _, row in candidates.iterrows():
        # Score filters
        if row["score"] < min_absolute_score:
            continue
        if max_score > 0 and row["score"] < max_score * min_score_ratio:
            continue

        # Deduplication
        normalized = re.sub(r"\s+", " ", row["chunk_text"]).strip().lower()
        if normalized in seen_texts:
            continue

        # Per-document limit
        doc_count = per_document_counts.get(row["document_id"], 0)
        if doc_count >= max_chunks_per_document:
            continue

        # Word budget
        chunk_words = len(row["chunk_text"].split())
        if selected_rows and used_words + chunk_words > word_budget:
            continue

        selected_rows.append(row.to_dict())
        seen_texts.add(normalized)
        per_document_counts[row["document_id"]] = doc_count + 1
        used_words += chunk_words

        if len(selected_rows) >= max_context_chunks:
            break

    # Build labeled context text
    blocks = []
    for position, row in enumerate(selected_rows, start=1):
        currency_label = "CURRENT" if row["is_current"] else "OUTDATED"
        blocks.append(
            f"[Source {position}] {row['title']} | {row['effective_date']} | {currency_label}\n"
            f"{row['chunk_text']}"
        )

    return {
        "query": query,
        "candidates": candidates,
        "selected_df": pd.DataFrame(selected_rows),
        "context_text": "\n\n".join(blocks),
        "used_words": used_words,
        "num_sources": len(selected_rows),
    }


if __name__ == "__main__":
    collection = store.create_vector_store()

    print("--- filter on (current_only=True) ---")
    package = build_context_package(collection, "How much does an official transcript cost right now?")
    print(package["context_text"])
    print("source doc ids:", package["selected_df"]["document_id"].tolist())

    print("\n--- filter off (current_only=False) ---")
    package = build_context_package(
        collection, "How much does an official transcript cost right now?", current_only=False
    )
    print(package["context_text"])
    print("source doc ids:", package["selected_df"]["document_id"].tolist())
