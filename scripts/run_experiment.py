"""Entry point for the main experiment, controls, offline filter audit, and rescoring.

    python scripts/run_experiment.py [--max-cases N] [--dry-run] [--adaptive]
                                     [--forced-retrieval] [--positive-control]
                                     [--stability-check] [--filter-audit] [--resume]
                                     [--rescore results/results.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import answer_is_correct, load_cases  # noqa: E402
from src.defenses import RuleFilter  # noqa: E402
from src.metrics import (  # noqa: E402
    CaseRecord,
    canary_success,
    evaluate_filter_offline,
    is_refusal,
    paired_defense_tests,
    record_from_row,
    summarize_cell,
    summary_table,
)
from src.runner import run_experiment  # noqa: E402


def rescore_file(results_path: Path, cases_path: Path) -> str:
    """Re-score an existing results.jsonl using the fixed accuracy scorer."""
    cases_by_id = {c["id"]: c for c in load_cases(cases_path)}
    records: list[CaseRecord] = []
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row["case_id"]
            case = cases_by_id.get(cid, {})
            if row["kind"] == "benign":
                refs = case.get("reference_answers", [])
                score = answer_is_correct(row["response"], refs)
                row["answer_correct"] = score.strict
                row["answer_lenient"] = score.lenient
                row["answer_hedged"] = score.hedged
            elif row.get("canary"):
                row["canary_emitted"] = canary_success(row["response"], row["canary"])
            row["refusal"] = is_refusal(row["response"])
            row.setdefault("answerable", case.get("answerable", bool(case.get("reference_answers"))))
            row.setdefault("attack_family", case.get("attack_family", "task_redirect"))
            records.append(record_from_row(row))

    with open(results_path, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    models = list(dict.fromkeys(r.model for r in records))
    defenses = list(dict.fromkeys(r.defense for r in records))
    summaries = [
        summarize_cell([r for r in records if r.model == m and r.defense == d])
        for m in models for d in defenses
        if any(r.model == m and r.defense == d for r in records)
    ]
    report = summary_table(summaries)
    paired = paired_defense_tests(records)
    if paired:
        report += "\n## Paired Exact McNemar Contrasts (vs. none)\n\n"
        report += "| Model | Defense | Paired N | Fixed (b) | Broke (c) | Exact p | Holm p |\n"
        report += "|---|---|---|---|---|---|---|\n"
        for p in paired:
            report += (
                f"| {p['model']} | {p['defense']} | {p['n_paired']} "
                f"| {p['fixed_b']} | {p['broke_c']} "
                f"| {p['p_raw']:.3f} | {p['p_holm']:.3f} |\n"
            )
    (results_path.parent / "report.md").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--max-cases", type=int, default=None,
                    help="run only the first N cases (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="check config and data without calling any model")
    ap.add_argument("--adaptive", action="store_true",
                    help="run the small defense-aware set instead of the main set")
    ap.add_argument("--forced-retrieval", action="store_true",
                    help="force the target poisoned email into top-k context")
    ap.add_argument("--positive-control", action="store_true",
                    help="place canary instruction in trusted system prompt")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: skip (mode, model, defense, case) rows already in results.jsonl")
    ap.add_argument("--stability-check", action="store_true",
                    help="re-run a sample of cell 1 to verify determinism")
    ap.add_argument("--filter-audit", action="store_true",
                    help="evaluate Defense B recall and FPR offline without an LLM")
    ap.add_argument("--rescore", type=str, default=None,
                    help="re-score an existing results.jsonl in-place without calling Ollama")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.filter_audit:
        rf = RuleFilter(action=cfg["data"].get("filter_action", "quarantine"))
        cases = load_cases(cfg["data"]["cases_file"])
        print(json.dumps(evaluate_filter_offline(cases, rf), indent=2))
        return

    if args.rescore:
        print(rescore_file(Path(args.rescore), Path(cfg["data"]["cases_file"])))
        return

    if args.adaptive:
        cfg["data"]["cases_file"] = cfg["data"]["adaptive_cases_file"]
        cfg["paths"]["results_dir"] = str(Path(cfg["paths"]["results_dir"]) / "adaptive")
    elif args.forced_retrieval:
        cfg["paths"]["results_dir"] = str(Path(cfg["paths"]["results_dir"]) / "forced_retrieval")
    elif args.positive_control:
        cfg["paths"]["results_dir"] = str(Path(cfg["paths"]["results_dir"]) / "positive_control")

    stats = run_experiment(
        cfg,
        max_cases=args.max_cases,
        dry_run=args.dry_run,
        forced_retrieval=args.forced_retrieval,
        positive_control=args.positive_control,
        stability_check=args.stability_check,
        resume=args.resume,
    )
    if args.dry_run:
        print("dry-run complete — no models were called.")
        return
    print(f"\n{stats['records']} responses written.")
    print(stats["report"])


if __name__ == "__main__":
    main()
