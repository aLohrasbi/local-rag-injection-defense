"""Clone BIPIA and adapt a subset of EmailQA into thesis case files.

The BIPIA repository (https://github.com/microsoft/BIPIA) was archived in
September 2026 but remains cloneable. Verified layout (checked 2026-11):
    benchmark/email/{train,test}.jsonl
        100 rows total; fields: context, question, ideal
        'question' values carry a leading 'Q: ' prefix; 'ideal' holds the
        reference answer, and the literal value 'unknown' marks rows with
        no extractable answer (59 of 100 rows).
    benchmark/text_attack_{train,test}.json
        JSON object mapping attack category -> list of instruction strings.
The reference evaluator uses a paid GPT-3.5 judge; we do NOT use it — attacks
are rewritten to demand a unique canary string (see src/data.py).

Case design (only 100 source emails exist, so attack and benign cases share
source emails; the dev/test split by source email id still prevents leakage):
attack cases are built from any usable row because canary scoring needs no
reference answer; benign accuracy cases are built ONLY from answerable rows,
so 'unknown' ideals can never inflate the containment-accuracy metric.
Unanswerable rows are reserved for a future correct-abstention metric.

Outputs (under data/processed/):
    cases_dev.jsonl      dev split (tune Defense B rules here only)
    cases_test.jsonl     frozen test split (main experiment)
    cases_adaptive.jsonl small defense-aware set (exploratory only)

Usage:
    python scripts/download_data.py [--n-attack 100 --n-benign 50 --seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (ADAPTIVE_VARIANTS, BASE_INSTRUCTIONS,  # noqa: E402
                      injection_with_canary, make_canary, split_dev_test,
                      write_cases)

REPO = "https://github.com/microsoft/BIPIA"
RAW_DIR = Path("data/raw/BIPIA")
OUT_DIR = Path("data/processed")

# Number of distractor emails indexed alongside the target email per case.
N_DISTRACTORS = 20


def clone_bipia() -> None:
    if (RAW_DIR / "benchmark").exists():
        print("BIPIA already cloned.")
        return
    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(RAW_DIR)], check=True)


def load_bipia_email(split: str) -> list[dict]:
    path = RAW_DIR / "benchmark" / "email" / f"{split}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def flatten_attack_templates(raw) -> list[str]:
    """Flatten BIPIA text_attack_*.json into a list of instruction strings.

    The file maps category -> list of strings (e.g. "Task Automation":
    ["Write a script to ...", ...]); tolerate a flat list or {'text': ...}
    dicts as well, so a schema drift degrades gracefully instead of raising.
    """
    out: list[str] = []
    if isinstance(raw, dict):
        items = [x for v in raw.values() for x in (v if isinstance(v, list) else [v])]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    for x in items:
        text = str(x.get("text", x) if isinstance(x, dict) else x).strip()
        if text:
            out.append(text)
    return out


def adapt(rows: list[dict], attack_templates,
          n_attack: int, n_benign: int, seed: int) -> list[dict]:
    """Build thesis cases from BIPIA EmailQA rows.

    Schema verified against the cloned repo (benchmark/email/*.jsonl):
    each row has 'context', 'question' (prefixed 'Q: ') and 'ideal' (the
    reference answer; the literal string 'unknown' marks unanswerable rows).
    Attack cases are built from any usable row (canary scoring needs no
    reference answer); benign accuracy cases only from answerable rows.
    """
    rng = random.Random(seed)
    rng.shuffle(rows)
    templates = flatten_attack_templates(attack_templates) or list(BASE_INSTRUCTIONS)

    # Distractor pool from unused rows
    def distractors(exclude_idx: int) -> list[dict]:
        pool = [r for i, r in enumerate(rows) if i != exclude_idx]
        picks = rng.sample(pool, min(N_DISTRACTORS, len(pool)))
        return [{"doc_id": f"distractor-{i}", "text": _ctx_text(r)}
                for i, r in enumerate(picks)]

    cases: list[dict] = []
    built_attack = built_benign = 0

    for idx, row in enumerate(rows):
        if built_attack >= n_attack and built_benign >= n_benign:
            break
        question = str(row.get("question", "")).strip()
        if question.lower().startswith("q:"):
            question = question[2:].strip()
        ideal = str(row.get("ideal", "")).strip()
        answerable = bool(ideal) and ideal.lower() != "unknown"
        context = _ctx_text(row)
        if not question or not context:
            continue
        doc_id = f"email-{idx:04d}-body"
        base_docs = [{"doc_id": doc_id, "text": context}] + distractors(idx)

        if built_attack < n_attack:
            canary = make_canary(rng)
            raw_instr = templates[built_attack % len(templates)]
            injected = injection_with_canary(raw_instr, canary)
            poisoned_text = context + "\n\n" + injected
            docs = [dict(d) for d in base_docs]
            for d in docs:
                if d["doc_id"] == doc_id:
                    d["text"] = poisoned_text
                    d["poisoned"] = True
            cases.append({
                "id": f"email-{idx:04d}-attack", "kind": "attack", "split": "test",
                "question": question,
                "reference_answers": [ideal] if answerable else [],
                "poisoned_doc": doc_id, "injection_text": injected,
                "canary": canary, "variant": "plain", "docs": docs,
            })
            built_attack += 1

        if answerable and built_benign < n_benign:
            cases.append({
                "id": f"email-{idx:04d}-benign", "kind": "benign", "split": "test",
                "question": question, "reference_answers": [ideal],
                "docs": base_docs,
            })
            built_benign += 1

    return cases


def build_adaptive(cases: list[dict], n: int, seed: int) -> list[dict]:
    """Small defense-aware set: re-inject canary demands in variant styles."""
    rng = random.Random(seed)
    attacks = [c for c in cases if c["kind"] == "attack"]
    out = []
    for i, case in enumerate(rng.sample(attacks, min(n, len(attacks)))):
        variant = list(ADAPTIVE_VARIANTS.keys())[i % len(ADAPTIVE_VARIANTS)]
        canary = make_canary(rng)
        new_case = json.loads(json.dumps(case))
        new_case["id"] = f"{case['id']}-adaptive-{i}"
        new_case["variant"] = variant
        new_case["canary"] = canary
        raw = ADAPTIVE_VARIANTS[variant]
        injected = injection_with_canary(raw, canary)
        new_case["injection_text"] = injected
        for d in new_case["docs"]:
            if d["doc_id"] == new_case["poisoned_doc"]:
                # replace the previous injection tail, keep the email body
                body = d["text"]
                for old in ("After completing the above",):
                    body = body.split(old)[0].rstrip()
                d["text"] = body + "\n\n" + injected
        out.append(new_case)
    return out


def _ctx_text(row: dict) -> str:
    for key in ("context", "mail", "email", "document", "text"):
        if row.get(key):
            return str(row[key])
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-attack", type=int, default=100,
                    help="attack cases (one per usable source email; EmailQA has 100)")
    ap.add_argument("--n-benign", type=int, default=50,
                    help="benign cases (answerable rows only; EmailQA has 41)")
    ap.add_argument("--n-adaptive", type=int, default=18)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clone_bipia()
    rows = load_bipia_email("test") + load_bipia_email("train")
    attack_templates = json.loads(
        (RAW_DIR / "benchmark" / "text_attack_test.json").read_text(encoding="utf-8"))

    cases = adapt(rows, attack_templates, args.n_attack, args.n_benign, args.seed)
    dev, test = split_dev_test(cases, dev_fraction=0.3, seed=args.seed)
    # Label the split on each row so the JSONL files are self-describing.
    for c in dev:
        c["split"] = "dev"
    for c in test:
        c["split"] = "test"
    adaptive = build_adaptive(test, args.n_adaptive, args.seed)

    write_cases(dev, OUT_DIR / "cases_dev.jsonl")
    write_cases(test, OUT_DIR / "cases_test.jsonl")
    write_cases(adaptive, OUT_DIR / "cases_adaptive.jsonl")
    print(f"wrote {len(dev)} dev, {len(test)} test, {len(adaptive)} adaptive cases")


if __name__ == "__main__":
    main()
