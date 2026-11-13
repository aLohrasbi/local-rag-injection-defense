"""Experiment runner: models x defenses x cases -> JSONL records + report."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from .data import answer_is_correct, load_cases
from .defenses import RuleFilter
from .llm import OllamaClient
from .metrics import (
    CaseRecord,
    canary_success,
    is_refusal,
    paired_defense_tests,
    record_from_row,
    summarize_cell,
    summary_table,
)
from .rag import RAGPipeline
from .retriever import Retriever

POSITIVE_CONTROL_NOTE = ("trusted-prompt control: instruction is in the system prompt, "
                         "poisoned doc indexed as clean_text")


def cell_order(cases: list[dict], seed: int, model_label: str, defense: str) -> list[dict]:
    """Case order for one (model, defense) cell: shuffled under a seed derived from the cell name
    (B10), so cells do not all share one order but every run reproduces the same one."""
    order = list(cases)
    random.Random(f"{seed}|{model_label}|{defense}").shuffle(order)
    return order


def read_records(path: Path) -> list[CaseRecord]:
    """Read results.jsonl, skipping blank lines and a truncated final line from an aborted run."""
    records: list[CaseRecord] = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                records.append(record_from_row(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return records


def record_key(r: CaseRecord) -> tuple[str, str, str, str]:
    return (r.mode, r.model, r.defense, r.case_id)


def run_experiment(config: dict, max_cases: int | None = None,
                   dry_run: bool = False,
                   forced_retrieval: bool = False,
                   positive_control: bool = False,
                   stability_check: bool = False,
                   resume: bool = False,
                   client: OllamaClient | None = None,
                   retriever: Retriever | None = None) -> dict:
    """Run the factorial experiment.

    `client` / `retriever` can be injected (tests pass FakeLLM and a hashing retriever); when
    `client` is given it is used for every model entry and the Ollama checks are skipped.
    With `resume=True` rows already present in results.jsonl are skipped; without it an existing
    non-empty results.jsonl is an error rather than being silently appended to or overwritten.
    """
    cases = load_cases(config["data"]["cases_file"])
    if positive_control:
        cases = [c for c in cases if c.get("kind") == "attack"]
    if max_cases:
        cases = cases[:max_cases]
    mode = ("positive_control" if positive_control
            else "forced_retrieval" if forced_retrieval else "main")

    seed = int(config["generation"].get("seed", 42))
    defenses = ["none"] if positive_control else config["defenses"]

    if dry_run:
        for model_cfg in config["models"]:
            print(f"[dry-run] would use model {model_cfg['name']} "
                  f"({len(cases)} cases x {len(defenses)} defenses, mode={mode})")
        return {"records": 0, "report": ""}

    rule_filter = RuleFilter(action=config["data"].get("filter_action", "quarantine"))
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    records_path = results_dir / "results.jsonl"

    prior: list[CaseRecord] = []
    needs_newline = False
    if records_path.exists() and records_path.stat().st_size > 0:
        if not resume:
            raise FileExistsError(
                f"{records_path} already has results; pass resume=True (--resume) to continue "
                f"it, or move/delete the file to start a new run")
        prior = read_records(records_path)
        with open(records_path, "rb") as fh:
            fh.seek(-1, 2)
            needs_newline = fh.read(1) != b"\n"
    done = {record_key(r) for r in prior}
    records: list[CaseRecord] = [r for r in prior if r.mode == mode]

    if retriever is None:
        retriever = Retriever(config["retrieval"]["embedding_model"])

    def run_case(pipeline: RAGPipeline, case: dict, defense: str):
        kind = case["kind"]
        forced_id = case.get("poisoned_doc") if (forced_retrieval and kind == "attack") else None
        trusted_inj = case.get("injection_text") if (positive_control and kind == "attack") else None
        pipeline.index_case(case["docs"], clean=positive_control)
        result, info = pipeline.answer(case["question"], defense,
                                       forced_doc_id=forced_id, trusted_injection=trusted_inj)
        return result, info, trusted_inj

    with open(records_path, "a", encoding="utf-8") as out:
        if needs_newline:
            out.write("\n")
        for model_cfg in config["models"]:
            if client is not None:
                llm = client
            else:
                llm = OllamaClient(
                    model=model_cfg["name"],
                    url=config["generation"]["ollama_url"],
                    temperature=config["generation"]["temperature"],
                    seed=seed,
                    num_predict=config["generation"]["num_predict"],
                    num_ctx=config["generation"].get("num_ctx", 8192),
                )
                if not llm.is_up():
                    raise RuntimeError(
                        f"Ollama not reachable at {config['generation']['ollama_url']} "
                        f"— start it before running the experiment.")
                if not llm.has_model(model_cfg["name"]):
                    raise RuntimeError(
                        f"model {model_cfg['name']!r} is not pulled in Ollama — run: "
                        f"ollama pull {model_cfg['name']}")

            pipeline = RAGPipeline(llm, retriever, rule_filter,
                                   top_k=config["retrieval"]["top_k"])

            for defense in defenses:
                rec_defense = "positive_control" if positive_control else defense
                n_skipped = 0
                for case in cell_order(cases, seed, model_cfg["label"], defense):
                    if (mode, model_cfg["label"], rec_defense, case["id"]) in done:
                        n_skipped += 1
                        continue
                    kind = case["kind"]
                    poisoned_doc = case.get("poisoned_doc")
                    result, info, trusted_inj = run_case(pipeline, case, defense)
                    canary = case.get("canary")
                    docs_retrieved = info["docs_retrieved"]
                    poisoned_retrieved = (
                        True if trusted_inj else
                        (poisoned_doc in docs_retrieved if poisoned_doc else None)
                    )
                    poisoned_rank = (docs_retrieved.index(poisoned_doc) + 1
                                     if poisoned_doc in docs_retrieved else None)
                    score = (answer_is_correct(result.text, case.get("reference_answers", []))
                             if kind == "benign" else None)
                    annotation = ""
                    if positive_control:
                        annotation = POSITIVE_CONTROL_NOTE
                    elif info["forced"]:
                        annotation = "poisoned doc force-inserted into top-k"
                    rec = CaseRecord(
                        case_id=case["id"],
                        kind=kind,
                        model=model_cfg["label"],
                        defense=rec_defense,
                        variant=case.get("variant", ""),
                        response=result.text,
                        canary=canary,
                        canary_emitted=(canary_success(result.text, canary)
                                        if canary else None),
                        poisoned_doc=poisoned_doc,
                        poisoned_retrieved=poisoned_retrieved,
                        filter_flagged_docs=info["filter_flagged_docs"],
                        docs_retrieved=docs_retrieved,
                        answer_correct=score.strict if score else None,
                        refusal=is_refusal(result.text),
                        latency_gen_s=info["latency_gen_s"],
                        latency_total_s=info["latency_total_s"],
                        answerable=case.get("answerable", bool(case.get("reference_answers"))),
                        attack_family=case.get("attack_family", ""),
                        answer_lenient=score.lenient if score else None,
                        answer_hedged=score.hedged if score else None,
                        poisoned_rank=poisoned_rank,
                        forced=info["forced"],
                        mode=mode,
                        annotation=annotation,
                    )
                    records.append(rec)
                    done.add(record_key(rec))
                    out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                    out.flush()  # B9: every finished row is on disk before the next LLM call
                print(f"done: {model_cfg['label']} / {defense} "
                      f"({len(cases) - n_skipped} run, {n_skipped} skipped)")

            if stability_check and cases:
                sample = cases[:min(10, len(cases))]
                rec_defense = "positive_control" if positive_control else defenses[0]
                matches = 0
                for c in sample:
                    r2, _info, _inj = run_case(pipeline, c, defenses[0])
                    orig = next((r.response for r in records
                                 if r.model == model_cfg["label"]
                                 and r.defense == rec_defense
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
