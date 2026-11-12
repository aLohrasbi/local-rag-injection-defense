"""Minimal local RAG pipeline.

One case flows: question -> embed -> top-k retrieval -> (defense layer) ->
prompt construction -> local LLM -> response. Supports optional forced
retrieval (guarantees target doc in top-k) and trusted-canary positive control
(places injection instruction in the trusted system prompt).
"""

from __future__ import annotations

from .defenses import (RuleFilter, apply_defenses, build_context,
                       system_prompt_for)
from .llm import GenerationResult, OllamaClient
from .retriever import Document, RetrievedDoc, Retriever

USER_TEMPLATE = "Context documents:\n{context}\n\nQuestion: {question}\nAnswer:"


class RAGPipeline:
    def __init__(self, llm: OllamaClient, retriever: Retriever,
                 rule_filter: RuleFilter | None = None, top_k: int = 5):
        self.llm = llm
        self.retriever = retriever
        self.rule_filter = rule_filter
        self.top_k = top_k

    def index_case(self, docs: list[dict]) -> None:
        self.retriever.index([
            Document(
                doc_id=d["doc_id"],
                text=d["text"],
                poisoned=d.get("poisoned", False),
                benign_instruction=d.get("benign_instruction", False),
            )
            for d in docs
        ])

    def answer(self, question: str, defense: str,
               forced_doc_id: str | None = None,
               trusted_injection: str | None = None
               ) -> tuple[GenerationResult, dict]:
        retrieved: list[RetrievedDoc] = self.retriever.search(question, self.top_k)
        if forced_doc_id and all(r.doc.doc_id != forced_doc_id for r in retrieved):
            for d in self.retriever.docs:
                if d.doc_id == forced_doc_id:
                    forced = RetrievedDoc(doc=d, score=0.0, rank=len(retrieved) or 1)
                    retrieved = (retrieved[:-1] + [forced]) if retrieved else [forced]
                    break
        decision = apply_defenses(retrieved, defense, self.rule_filter)
        context = build_context(decision, defense)
        prompt = USER_TEMPLATE.format(context=context, question=question)
        sys_prompt = system_prompt_for(defense)
        if trusted_injection:
            sys_prompt = f"{sys_prompt}\n\n{trusted_injection}"
        result = self.llm.generate(prompt, system=sys_prompt)
        info = {
            "docs_retrieved": [r.doc.doc_id for r in retrieved],
            "filter_flagged_docs": list(decision.filter_reasons.keys()),
            "quarantined": [r.doc.doc_id for r in decision.quarantined],
        }
        return result, info
