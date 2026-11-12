"""Experiment runner: models x defenses x cases -> JSONL records + report."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from .data import answer_is_correct, load_cases
from .defenses import RuleFilter
from .llm import OllamaClient
from .metrics import (CaseRecord, canary_success, is_refusal,
                      paired_defense_tests, summarize_cell, summary_table)
from .rag import RAGPipeline
from .retriever import Retriever


def run_experiment(config: dict, max_cases: int | None = None,
                   dry_run: bool = False,
                   forced_retrieval: bool = False,
                   positive_control: bool = False,
                   stability_check: bool = False) -> dict:
    cases = load_cases(config["data"]["cases_file"])
    if positive_control:
        cases = [c for c in cases if c.get("kind") == "attack"]
    if max_cases:
        cases = cases[:max_cases]

    seed = int(config["generation"].get("seed", 42))
    rule_filter = RuleFilter(action=config["data"].get("filter_action", "quarantine"))
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    records_path = results_dir / "results.jsonl"

    defenses = ["none"] if positive_control else config["defenses"]
    records: list[CaseRecord] = []
    with open(records_path, "w", encoding="utf-8") as out:
        for model_cfg in config["models"]:
            llm = OllamaClient(
                model=model_cfg["name"],
                url=config["generation"]["ollama_url"],
                temperature=config["generation"]["temperature"],
                seed=seed,
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

            for defense in defenses:
                # Randomize case order per cell under fixed seed to spread drift
                cell_cases = list(cases)
                random.Random(seed).shuffle(cell_cases)
                for case in cell_cases:
                    pipeline.index_case(case["docs"])
                    kind = case["kind"]
                    poisoned_doc = case.get("poisoned_doc")
                    forced_id = poisoned_doc if (forced_retrieval and kind == "attack") else None
                    trusted_inj = case.get("injection_text") if (positive_control and kind == "attack") else None
                    result, info = pipeline.answer(
                        case["question"], defense,
                        forced_doc_id=forced_id,
                        trusted_injection=trusted_inj,
                    )
                    canary = case.get("canary")
                    poisoned_retrieved = (
                        True if trusted_inj else
                        (poisoned_doc in info["docs_retrieved"] if poisoned_doc else None)
                    )
                    answerable = case.get("answerable", bool(case.get("reference_answers")))
                    rec = CaseRecord(
                        case_id=case["id"],
                        kind=kind,
                        model=model_cfg["label"],
                        defense="positive_control" if positive_control else defense,
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
                        answerable=answerable,
                        attack_family=case.get("attack_family", ""),
                    )
                    records.append(rec)
                    out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                print(f"done: {model_cfg['label']} / {defense} "
                      f"({len(cell_cases)} cases)")

            if stability_check and not dry_run and cases:
                sample = cases[:min(10, len(cases))]
                matches = 0
                for c in sample:
                    pipeline.index_case(c["docs"])
                    r2, _ = pipeline.answer(c["question"], defenses[0])
                    orig = next((r.response for r in records
                                 if r.model == model_cfg["label"]
                                 and r.defense == ("positive_control" if positive_control else defenses[0])
                                 and r.case_id == c["id"]), None)
                    if orig == r2.text:
                        matches += 1
                print(f"stability check ({model_cfg['label']}): {matches}/{len(sample)} exact matches")

    summaries = []
    for model_cfg in config["models"]:
        for defense in (["positive_control"] if positive_control else defenses):
            cell = [r for r in records
                    if r.model == model_cfg["label"] and r.defense == defense]
            if cell:
                summaries.append(summarize_cell(cell))

    report = summary_table(summaries)
    contrasts = paired_defense_tests(records)
    if contrasts:
        lines = ["\n### Paired Exact McNemar Tests (vs. `none`, Holm-corrected)\n",
                 "| Model | Defense | Paired N | Fixed (b) | Broke (c) | p (exact) | p (Holm) |",
                 "|---|---|---|---|---|---|---|"]
        for row in contrasts:
            lines.append(
                f"| {row['model']} | {row['defense']} | {row['n_paired']} "
                f"| {row['fixed_b']} | {row['broke_c']} "
                f"| {row['p_raw']:.3f} | {row['p_holm']:.3f} |"
            )
        report += "\n".join(lines) + "\n"

    (results_dir / "report.md").write_text(report, encoding="utf-8")
    return {"records": len(records), "report": report}
