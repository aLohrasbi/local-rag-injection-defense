"""Clone BIPIA and adapt EmailQA into thesis case files with disjoint templates.

Anti-leakage & construct-validity guarantees:
  * Dev (30%) and Test (70%) are disjoint by BOTH source email AND attack
    template (assign_disjoint_templates).
  * Attack cases include both classic override templates (BASE_INSTRUCTIONS)
    and BIPIA task-redirection templates, tagged via 'attack_family'.
  * Answerable vs. unanswerable status is recorded on every case ('answerable').
  * 18 benign emails in the test split (and 8 in dev) carry realistic legitimate
    directives ('benign_instruction': True) so Defense B FPR is directly measurable.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (ADAPTIVE_VARIANTS, BASE_INSTRUCTIONS, BENIGN_DIRECTIVES,  # noqa: E402
                      CANARY_MARKER, assign_disjoint_templates,
                      has_benign_instruction, injection_with_canary,
                      make_canary, split_dev_test, write_cases)

REPO = "https://github.com/microsoft/BIPIA"
RAW_DIR = Path("data/raw/BIPIA")
OUT_DIR = Path("data/processed")
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


def _ctx_text(row: dict) -> str:
    for key in ("context", "mail", "email", "document", "text"):
        if row.get(key):
            return str(row[key])
    return ""


def adapt(rows: list[dict], attack_templates,
          n_attack: int, n_benign: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    templates = flatten_attack_templates(attack_templates) or list(BASE_INSTRUCTIONS)

    def distractors(exclude_idx: int) -> list[dict]:
        pool = [r for i, r in enumerate(rows) if i != exclude_idx]
        picks = rng.sample(pool, min(N_DISTRACTORS, len(pool)))
        out = []
        for i, r in enumerate(picks):
            txt = _ctx_text(r)
            out.append({
                "doc_id": f"distractor-{i}",
                "text": txt,
                "benign_instruction": has_benign_instruction(txt),
            })
        return out

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
        base_docs = [{
            "doc_id": doc_id,
            "text": context,
            "benign_instruction": has_benign_instruction(context),
        }] + distractors(idx)

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
                "id": f"email-{idx:04d}-attack",
                "kind": "attack",
                "split": "test",
                "answerable": answerable,
                "attack_family": "task_redirect",
                "template_id": f"bipia-{built_attack % len(templates)}",
                "question": question,
                "reference_answers": [ideal] if answerable else [],
                "poisoned_doc": doc_id,
                "injection_text": injected,
                "canary": canary,
                "variant": "plain",
                "docs": docs,
            })
            built_attack += 1

        if answerable and built_benign < n_benign:
            b_docs = [dict(d) for d in base_docs]
            if built_benign < len(BENIGN_DIRECTIVES) and not b_docs[0]["benign_instruction"]:
                tail = BENIGN_DIRECTIVES[built_benign % len(BENIGN_DIRECTIVES)]
                b_docs[0]["text"] = b_docs[0]["text"].rstrip() + "\n\n" + tail
                b_docs[0]["benign_instruction"] = True
            cases.append({
                "id": f"email-{idx:04d}-benign",
                "kind": "benign",
                "split": "test",
                "answerable": True,
                "question": question,
                "reference_answers": [ideal],
                "target_doc": doc_id,
                "docs": b_docs,
            })
            built_benign += 1

    return cases


def build_adaptive(cases: list[dict], n: int, seed: int) -> list[dict]:
    """Defense-aware exploratory set built from answerable test attack cases."""
    rng = random.Random(seed)
    attacks = [c for c in cases if c["kind"] == "attack" and c.get("answerable", True)]
    if len(attacks) < n:
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
                base_body = (
                    d["text"].rsplit("\n\n", 1)[0]
                    if CANARY_MARKER in d["text"]
                    else d["text"]
                )
                d["text"] = base_body + "\n\n" + injected
        out.append(new_case)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-attack", type=int, default=100)
    ap.add_argument("--n-benign", type=int, default=50)
    ap.add_argument("--n-adaptive", type=int, default=18)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clone_bipia()
    rows = load_bipia_email("test") + load_bipia_email("train")
    attack_templates = json.loads(
        (RAW_DIR / "benchmark" / "text_attack_test.json").read_text(encoding="utf-8"))
    flat_templates = flatten_attack_templates(attack_templates)

    cases = adapt(rows, flat_templates, args.n_attack, args.n_benign, args.seed)
    dev, test = split_dev_test(cases, dev_fraction=0.3, seed=args.seed)
    assign_disjoint_templates(dev, test, flat_templates, seed=args.seed, min_test_benign_directives=18)
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
