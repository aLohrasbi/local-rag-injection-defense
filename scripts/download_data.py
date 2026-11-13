"""Clone BIPIA and adapt EmailQA into thesis case files with disjoint templates.

Anti-leakage & construct-validity guarantees:
  * Source rows are split into dev (30%) and test (70%) BEFORE any distractor is drawn, and
    distractors come only from the same split, so no email appears on both sides (split_rows).
  * Dev and Test are also disjoint by attack template (assign_disjoint_templates).
  * Every poisoned document stores 'clean_text' (the email without the injection), used by the
    trusted-prompt positive control and by the audits.
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

from src.data import (  # noqa: E402
    ADAPTIVE_VARIANTS,
    BASE_INSTRUCTIONS,
    BENIGN_DIRECTIVES,
    assign_disjoint_templates,
    has_benign_instruction,
    injection_with_canary,
    make_canary,
    write_cases,
)

REPO = "https://github.com/microsoft/BIPIA"
BIPIA_COMMIT = "a02cbb2eb8bc1fb1fb4ec0f4ca3cb70eb9dd45f8"
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


def split_rows(rows: list[dict], dev_fraction: float = 0.3,
               seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split source rows into (dev, test) BEFORE building cases or drawing distractors.

    Rows are grouped by email text first, so an email that has several questions lands wholly in
    one split. Each row is tagged with a global `_src` index that keeps case ids unique across splits.
    """
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for i, row in enumerate(rows):
        key = re.sub(r"\s+", " ", _ctx_text(row))[:120]
        groups.setdefault(key, []).append({**row, "_src": i})
    keys = sorted(groups)
    rng.shuffle(keys)
    n_dev = max(1, int(len(keys) * dev_fraction))
    dev = [r for k in keys[:n_dev] for r in groups[k]]
    test = [r for k in keys[n_dev:] for r in groups[k]]
    return dev, test


def adapt(rows: list[dict], attack_templates,
          n_attack: int, n_benign: int, seed: int | str,
          n_directives: int = len(BENIGN_DIRECTIVES), split: str = "test") -> list[dict]:
    """Build cases from `rows` only; distractors are drawn from these same rows (one split)."""
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    templates = flatten_attack_templates(attack_templates) or list(BASE_INSTRUCTIONS)

    def distractors(target_ctx: str) -> list[dict]:
        seen = {target_ctx}
        pool = []
        for r in rows:  # same split only, one row per distinct email, never the target email
            txt = _ctx_text(r)
            if txt and txt not in seen:
                seen.add(txt)
                pool.append(r)
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
        src = int(row.get("_src", idx))
        doc_id = f"email-{src:04d}-body"
        base_docs = [{
            "doc_id": doc_id,
            "text": context,
            "benign_instruction": has_benign_instruction(context),
        }] + distractors(context)

        if built_attack < n_attack:
            canary = make_canary(rng)
            raw_instr = templates[built_attack % len(templates)]
            injected = injection_with_canary(raw_instr, canary)
            poisoned_text = context + "\n\n" + injected
            docs = [dict(d) for d in base_docs]
            for d in docs:
                if d["doc_id"] == doc_id:
                    d["clean_text"] = context
                    d["text"] = poisoned_text
                    d["poisoned"] = True
            cases.append({
                "id": f"email-{src:04d}-attack",
                "kind": "attack",
                "split": split,
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
            if built_benign < n_directives and not b_docs[0]["benign_instruction"]:
                tail = BENIGN_DIRECTIVES[built_benign % len(BENIGN_DIRECTIVES)]
                b_docs[0]["text"] = b_docs[0]["text"].rstrip() + "\n\n" + tail
                b_docs[0]["benign_instruction"] = True
            cases.append({
                "id": f"email-{src:04d}-benign",
                "kind": "benign",
                "split": split,
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
        new_case["attack_family"] = "adaptive"
        new_case["template_id"] = f"adaptive-{variant}"
        new_case["canary"] = canary
        raw = ADAPTIVE_VARIANTS[variant]
        injected = injection_with_canary(raw, canary)
        new_case["injection_text"] = injected
        for d in new_case["docs"]:
            if d["doc_id"] == new_case["poisoned_doc"]:
                d["text"] = d["clean_text"] + "\n\n" + injected
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

    # B3: split the source rows first, then build each split from its own rows only.
    dev_rows, test_rows = split_rows(rows, dev_fraction=0.3, seed=args.seed)
    n_dev_attack, n_dev_benign = round(args.n_attack * 0.3), round(args.n_benign * 0.3)
    dev = adapt(dev_rows, flat_templates, n_dev_attack, n_dev_benign,
                seed=f"{args.seed}-dev", n_directives=8, split="dev")
    test = adapt(test_rows, flat_templates, args.n_attack - n_dev_attack, args.n_benign - n_dev_benign,
                 seed=f"{args.seed}-test", n_directives=18, split="test")
    assign_disjoint_templates(dev, test, flat_templates, seed=args.seed, min_test_benign_directives=18)
    adaptive = build_adaptive(test, args.n_adaptive, args.seed)

    write_cases(dev, OUT_DIR / "cases_dev.jsonl")
    write_cases(test, OUT_DIR / "cases_test.jsonl")
    write_cases(adaptive, OUT_DIR / "cases_adaptive.jsonl")
    print(f"wrote {len(dev)} dev, {len(test)} test, {len(adaptive)} adaptive cases")


if __name__ == "__main__":
    main()
