"""Minimal local RAG pipeline.

One case flows: question -> embed -> top-k retrieval -> (defense layer) ->
prompt construction -> local LLM -> response. Supports optional forced
retrieval (guarantees the target doc is in top-k) and a trusted-canary positive
control (the injection instruction is placed in the trusted system prompt and
the poisoned doc is indexed as its clean_text).

Latency bookkeeping (B6): `latency_gen_s` times the LLM call only; `latency_total_s`
covers retrieval + forced insertion + defense + prompt construction + generation.
Indexing the case's documents is corpus preparation and is excluded from both.
"""

from __future__ import annotations

import time

from .defenses import RuleFilter, apply_defenses, build_context, system_prompt_for
from .llm import GenerationResult, OllamaClient
from .retriever import Document, RetrievedDoc, Retriever

USER_TEMPLATE = "Context documents:\n{context}\n\nQuestion: {question}\nAnswer:"


def rerank(items: list[RetrievedDoc]) -> list[RetrievedDoc]:
    """Sort by similarity score (descending, stable) and renumber ranks from 1."""
    ordered = sorted(items, key=lambda r: -r.score)
    return [RetrievedDoc(doc=r.doc, score=r.score, rank=i + 1) for i, r in enumerate(ordered)]


class RAGPipeline:
    def __init__(self, llm: OllamaClient, retriever: Retriever,
                 rule_filter: RuleFilter | None = None, top_k: int = 5):
        self.llm = llm
        self.retriever = retriever
        self.rule_filter = rule_filter
        self.top_k = top_k

    def index_case(self, docs: list[dict], clean: bool = False) -> None:
        """Index a case's documents. With clean=True poisoned docs are indexed as `clean_text`
        (B8: the injection must then exist only in the trusted prompt)."""
        indexed = []
        for d in docs:
            text = d["text"]
            if clean and d.get("poisoned"):
                if "clean_text" not in d:
                    raise ValueError(
                        f"doc {d['doc_id']!r} has no clean_text; rebuild the data with "
                        f"scripts/download_data.py before running the positive control")
                text = d["clean_text"]
            indexed.append(Document(
                doc_id=d["doc_id"],
                text=text,
                poisoned=d.get("poisoned", False),
                benign_instruction=d.get("benign_instruction", False),
            ))
        self.retriever.index(indexed)

    def answer(self, question: str, defense: str,
               forced_doc_id: str | None = None,
               trusted_injection: str | None = None
               ) -> tuple[GenerationResult, dict]:
        t_start = time.perf_counter()
        retrieved: list[RetrievedDoc] = self.retriever.search(question, self.top_k)
        forced = False
        if forced_doc_id and all(r.doc.doc_id != forced_doc_id for r in retrieved):
            # B7: use the doc's real similarity score, displace the lowest-ranked hit, re-sort.
            item = self.retriever.score_doc(question, forced_doc_id)
            if item is not None:
                retrieved = rerank(retrieved[:-1] + [item])
                forced = True
        decision = apply_defenses(retrieved, defense, self.rule_filter)
        context = build_context(decision, defense)
        prompt = USER_TEMPLATE.format(context=context, question=question)
        sys_prompt = system_prompt_for(defense)
        if trusted_injection:
            sys_prompt = f"{sys_prompt}\n\n{trusted_injection}"
        t_gen = time.perf_counter()
        result = self.llm.generate(prompt, system=sys_prompt)
        t_end = time.perf_counter()
        info = {
            "docs_retrieved": [r.doc.doc_id for r in retrieved],
            "scores": [r.score for r in retrieved],
            "forced": forced,
            "filter_flagged_docs": list(decision.filter_reasons.keys()),
            "quarantined": [r.doc.doc_id for r in decision.quarantined],
            "latency_gen_s": t_end - t_gen,
            "latency_total_s": t_end - t_start,
        }
        return result, info
