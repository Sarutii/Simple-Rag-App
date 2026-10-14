import os
os.environ.setdefault("USE_TF", "0")

from importlib import import_module

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

chunks = import_module("03_chunking").build_chunks()
tokenize = import_module("02_preprocessing").tokenize

MODEL_NAME = "all-MiniLM-L6-v2"

ALPHA = 0.5


def encode_chunks():
    model = SentenceTransformer(MODEL_NAME)
    return model.encode([c["search_text"] for c in chunks], normalize_embeddings=True)


def encode_query(text):
    model = SentenceTransformer(MODEL_NAME)
    return model.encode([text], normalize_embeddings=True)[0]


def build_bm25(tokenized_corpus):
    return BM25Okapi(tokenized_corpus)


def min_max(scores):
    scores = np.asarray(scores, dtype=float)
    span = scores.max() - scores.min()
    return np.zeros_like(scores) if span == 0 else (scores - scores.min()) / span


def hybrid_score(bm25_scores, embedding_scores, alpha=ALPHA):
    return alpha * min_max(embedding_scores) + (1 - alpha) * min_max(bm25_scores)


if __name__ == "__main__":
    bm25 = build_bm25([tokenize(c["search_text"]) for c in chunks])
    embeddings = encode_chunks()
    sample_scores = bm25.get_scores(tokenize("transcript fee"))
    print("chunk embeddings:", embeddings.shape, "| bm25 docs:", len(sample_scores), "| alpha:", ALPHA)
