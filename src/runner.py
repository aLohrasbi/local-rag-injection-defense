"""Experiment runner: models x defenses x cases -> JSONL records + report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .data import answer_is_correct, load_cases
from .defenses import RuleFilter
from .llm import OllamaClient
from .metrics import (CaseRecord, canary_success, is_refusal,
                      summarize_cell, summary_table)
from .rag import RAGPipeline
from .retriever import Retriever


def run_experiment(config: dict, max_cases: int | None = None,
                   dry_run: bool = False) -> dict:
    cases = load_cases(config["data"]["cases_file"])
    if max_cases:
        cases = cases[:max_cases]

    rule_filter = RuleFilter(action=config["data"].get("filter_action", "quarantine"))
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    records_path = results_dir / "results.jsonl"

    records: list[CaseRecord] = []
    with open(records_path, "w", encoding="utf-8") as out:
        for model_cfg in config["models"]:
            llm = OllamaClient(
                model=model_cfg["name"],
                url=config["generation"]["ollama_url"],
                temperature=config["generation"]["temperature"],
                seed=config["generation"]["seed"],
                num_predict=config["generation"]["num_predict"],
                num_ctx=config["generation"].get("num_ctx", 8192),
            )
            if dry_run:
                print(f"[dry-run] would use model {model_cfg['name']}")
                continue
            if not llm.is_up():
                raise RuntimeError(
                    f"Ollama not reachable at {config['generation']['ollama_url']} "
                    f"— start it before running the experiment.")
            if not llm.has_model(model_cfg["name"]):
                raise RuntimeError(
                    f"model {model_cfg['name']!r} is not pulled in Ollama — run: "
                    f"ollama pull {model_cfg['name']}")

            retriever = Retriever(config["retrieval"]["embedding_model"])
            pipeline = RAGPipeline(llm, retriever, rule_filter,
                                   top_k=config["retrieval"]["top_k"])

            for defense in config["defenses"]:
                for case in cases:
                    pipeline.index_case(case["docs"])
                    result, info = pipeline.answer(case["question"], defense)
                    kind = case["kind"]
                    canary = case.get("canary")
                    poisoned_doc = case.get("poisoned_doc")
                    poisoned_retrieved = (poisoned_doc in info["docs_retrieved"]
                                          if poisoned_doc else None)
                    rec = CaseRecord(
                        case_id=case["id"],
                        kind=kind,
                        model=model_cfg["label"],
                        defense=defense,
                        variant=case.get("variant", ""),
                        response=result.text,
                        canary=canary,
                        canary_emitted=(canary_success(result.text, canary)
                                        if canary else None),
                        poisoned_doc=poisoned_doc,
                        poisoned_retrieved=poisoned_retrieved,
                        filter_flagged_docs=info["filter_flagged_docs"],
                        docs_retrieved=info["docs_retrieved"],
                        answer_correct=(answer_is_correct(result.text,
                                                          case.get("reference_answers", []))
                                        if kind == "benign" else None),
                        refusal=is_refusal(result.text),
                        latency_s=result.latency_s,
                    )
                    records.append(rec)
                    out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                print(f"done: {model_cfg['label']} / {defense} "
                      f"({len(cases)} cases)")

    # Aggregate: one summary row per (model, defense)
    summaries = []
    for model_cfg in config["models"]:
        for defense in config["defenses"]:
            cell = [r for r in records
                    if r.model == model_cfg["label"] and r.defense == defense]
            if cell:
                summaries.append(summarize_cell(cell))

    report = summary_table(summaries)
    (results_dir / "report.md").write_text(report, encoding="utf-8")
    return {"records": len(records), "report": report}
