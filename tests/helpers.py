"""Test doubles: a torch-free hashing retriever and a config builder. FakeLLM lives in src/llm.py."""

from __future__ import annotations

import re
import time
import zlib
from pathlib import Path

import numpy as np

from src.retriever import Retriever

FIXTURE_TEST = Path(__file__).resolve().parent / "fixtures" / "cases_test.jsonl"


class HashRetriever(Retriever):
    """Bag-of-words hashing embeddings (L2-normalised): deterministic, no model download."""

    DIM = 512

    def __init__(self) -> None:
        super().__init__("hash")

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.DIM))
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, zlib.crc32(tok.encode()) % self.DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SlowRetriever(HashRetriever):
    """Adds a fixed delay to every search, to prove total latency includes retrieval."""

    def __init__(self, delay_s: float = 0.02) -> None:
        super().__init__()
        self.delay_s = delay_s

    def search(self, query, top_k=5):
        time.sleep(self.delay_s)
        return super().search(query, top_k)


def make_config(results_dir: Path, cases_file: Path = FIXTURE_TEST, top_k: int = 3,
                defenses: list[str] | None = None) -> dict:
    return {
        "models": [{"name": "fake-1", "label": "fake-a"}, {"name": "fake-2", "label": "fake-b"}],
        "defenses": defenses or ["none", "delimiter", "filter", "both"],
        "retrieval": {"embedding_model": "hash", "top_k": top_k, "num_distractors": 5},
        "generation": {"temperature": 0.0, "seed": 42, "num_predict": 16,
                       "ollama_url": "http://unused.invalid"},
        "data": {"cases_file": str(cases_file), "filter_action": "quarantine"},
        "paths": {"results_dir": str(results_dir)},
    }
