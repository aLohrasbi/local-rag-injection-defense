"""Embedding-based retrieval over a small document corpus.

The corpus is a few hundred short emails, so exact cosine similarity with
numpy is sufficient; no vector-index dependency is needed.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .types import Document, RetrievedDoc  # noqa: F401  (re-exported)


class Retriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.docs: list[Document] = []
        self.doc_matrix: np.ndarray | None = None

    def index(self, docs: list[Document]) -> None:
        self.docs = list(docs)
        embeddings = self.model.encode(
            [d.text for d in docs], normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.doc_matrix = np.asarray(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        if self.doc_matrix is None or not self.docs:
            return []
        q = self.model.encode([query], normalize_embeddings=True,
                              show_progress_bar=False)
        scores = (self.doc_matrix @ q.T).ravel()
        order = np.argsort(-scores)[:top_k]
        return [
            RetrievedDoc(doc=self.docs[i], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(order)
        ]
