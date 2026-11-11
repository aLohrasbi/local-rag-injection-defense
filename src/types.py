"""Dependency-free shared data types (importable without numpy/torch)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    text: str
    poisoned: bool = False
    benign_instruction: bool = False  # legitimately contains instructions
    meta: dict = field(default_factory=dict)


@dataclass
class RetrievedDoc:
    doc: Document
    score: float
    rank: int
