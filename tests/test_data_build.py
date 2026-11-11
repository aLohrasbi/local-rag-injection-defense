"""Integrity checks for the built dataset — pure stdlib, no LLM/embeddings.

    python -m pytest tests/test_data_build.py -q
or directly:  python tests/test_data_build.py

Skips gracefully when data/processed/ has not been built yet.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
CANARY_RX = re.compile(r"^[A-Z0-9]{4}-[A-Z0-9]{5}$")


def _load(name: str) -> list[dict] | None:
    path = PROC / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _rows() -> list[dict] | None:
    rows = _load("cases_test.jsonl")
    if rows is None:
        print("SKIP: dataset not built — run scripts/download_data.py first")
    return rows


def test_dataset_built():
    if _load("cases_test.jsonl") is None:
        print("SKIP: dataset not built — run scripts/download_data.py first")
        return
    assert _load("cases_dev.jsonl") is not None, "cases_dev.jsonl missing"
    assert _load("cases_adaptive.jsonl") is not None, "cases_adaptive.jsonl missing"


def test_benign_cases_are_answerable():
    rows = _rows()
    if rows is None:
        return
    benign = [r for r in rows if r["kind"] == "benign"]
    assert benign, "no benign cases built"
    for r in benign:
        assert r["reference_answers"], f"{r['id']}: benign case without reference answer"
        assert all(a.strip().lower() != "unknown" for a in r["reference_answers"]), \
            f"{r['id']}: 'unknown' reference would inflate containment accuracy"


def test_attack_canaries_unique_and_wellformed():
    rows = _rows()
    if rows is None:
        return
    attacks = [r for r in rows if r["kind"] == "attack"]
    canaries = [r["canary"] for r in attacks]
    assert len(set(canaries)) == len(canaries), "canary collision across attack cases"
    for c in canaries:
        assert CANARY_RX.match(c), f"canary not in XXXX-XXXXX format: {c}"


def test_attack_docs_carry_poisoned_target():
    rows = _rows()
    if rows is None:
        return
    for r in (x for x in rows if x["kind"] == "attack"):
        targets = [d for d in r["docs"] if d["doc_id"] == r["poisoned_doc"]]
        assert len(targets) == 1 and targets[0].get("poisoned") is True, r["id"]
        assert r["canary"] in targets[0]["text"], \
            f"{r['id']}: canary missing from the poisoned document text"


def test_benign_docs_not_poisoned():
    rows = _rows()
    if rows is None:
        return
    for r in (x for x in rows if x["kind"] == "benign"):
        assert not any(d.get("poisoned") for d in r["docs"]), \
            f"{r['id']}: benign case contains a poisoned document"


def test_dev_test_disjoint_by_source():
    dev, test = _load("cases_dev.jsonl"), _load("cases_test.jsonl")
    if dev is None or test is None:
        print("SKIP: dataset not built — run scripts/download_data.py first")
        return
    dev_sources = {c["id"].rsplit("-", 1)[0] for c in dev}
    test_sources = {c["id"].rsplit("-", 1)[0] for c in test}
    assert not (dev_sources & test_sources), \
        "source-email leakage between dev and test splits"


def test_split_labels_match_files():
    dev, test = _load("cases_dev.jsonl"), _load("cases_test.jsonl")
    if dev is None or test is None:
        print("SKIP: dataset not built — run scripts/download_data.py first")
        return
    assert all(c["split"] == "dev" for c in dev), "dev file contains non-dev rows"
    assert all(c["split"] == "test" for c in test), "test file contains non-test rows"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
