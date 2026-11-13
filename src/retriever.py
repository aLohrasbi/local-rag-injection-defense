"""Embedding-based retrieval over a small document corpus.

The corpus is a few hundred short emails, so exact cosine similarity with
numpy is sufficient; no vector-index dependency is needed.

The sentence-transformers model is loaded lazily on first use, so importing this
module (and running the unit tests with an overridden `_encode`) needs no torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import Document, RetrievedDoc  # noqa: F401  (re-exported)


class Retriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Any = None
        self.docs: list[Document] = []
        self.doc_matrix: np.ndarray | None = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        """L2-normalised embeddings, one row per text. Tests override this method."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return np.asarray(self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False))

    def index(self, docs: list[Document]) -> None:
        self.docs = list(docs)
        self.doc_matrix = self._encode([d.text for d in docs])

    def _scores(self, query: str) -> np.ndarray:
        assert self.doc_matrix is not None
        q = self._encode([query])
        return (self.doc_matrix @ q.T).ravel()

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        if self.doc_matrix is None or not self.docs:
            return []
        scores = self._scores(query)
        order = np.argsort(-scores)[:top_k]
        return [
            RetrievedDoc(doc=self.docs[i], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(order)
        ]

    def score_doc(self, query: str, doc_id: str) -> RetrievedDoc | None:
        """The real query-document similarity of one indexed doc (rank is a placeholder, 0)."""
        if self.doc_matrix is None:
            return None
        scores = self._scores(query)
        for i, d in enumerate(self.docs):
            if d.doc_id == doc_id:
                return RetrievedDoc(doc=d, score=float(scores[i]), rank=0)
        return None
