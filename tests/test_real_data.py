"""Audit of the REAL data/processed/*.jsonl files. Marker: data. Not run in CI.

    pytest -m data

Skipped when the processed files are absent. Unit tests never read these files (they use
tests/fixtures/); this module is the check to run after `make data`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.data import BASE_INSTRUCTIONS, CANARY_MARKER, load_cases

pytestmark = pytest.mark.data

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
CANARY_RX = re.compile(r"^[A-Z0-9]{4}-[A-Z0-9]{5}$")

# Defaults of scripts/download_data.py: --n-attack 100 --n-benign 50 --n-adaptive 18, dev = 30%.
EXPECTED_DEV_ATTACKS, EXPECTED_TEST_ATTACKS, EXPECTED_ADAPTIVE = 30, 70, 18
MAX_BENIGN_TOTAL = 50
MIN_DEV_DIRECTIVES, MIN_TEST_DIRECTIVES = 8, 18


@pytest.fixture(scope="module")
def real() -> dict[str, list[dict]]:
    names = ("dev", "test", "adaptive")
    if not all((PROC / f"cases_{n}.jsonl").exists() for n in names):
        pytest.skip("data/processed/*.jsonl not built (run `make data`)")
    return {n: load_cases(PROC / f"cases_{n}.jsonl") for n in names}


def _attacks(rows):
    return [r for r in rows if r["kind"] == "attack"]


def _benign(rows):
    return [r for r in rows if r["kind"] == "benign"]


def _email_key(doc: dict) -> str:
    return re.sub(r"\s+", " ", doc.get("clean_text", doc["text"]))[:120]


def _poisoned(case: dict) -> dict:
    return next(d for d in case["docs"] if d["doc_id"] == case["poisoned_doc"])


def test_record_counts(real):
    dev, test, adaptive = real["dev"], real["test"], real["adaptive"]
    assert len(_attacks(dev)) == EXPECTED_DEV_ATTACKS
    assert len(_attacks(test)) == EXPECTED_TEST_ATTACKS
    assert len(adaptive) == EXPECTED_ADAPTIVE
    assert MIN_DEV_DIRECTIVES <= len(_benign(dev)) and MIN_TEST_DIRECTIVES <= len(_benign(test))
    assert len(_benign(dev)) + len(_benign(test)) <= MAX_BENIGN_TOTAL
    assert all(c["split"] == "dev" for c in dev) and all(c["split"] == "test" for c in test)


def test_dev_test_disjoint_by_source_email_and_by_document(real):
    dev, test = real["dev"], real["test"]
    assert {c["id"].rsplit("-", 1)[0] for c in dev}.isdisjoint({c["id"].rsplit("-", 1)[0] for c in test})
    dev_emails = {_email_key(d) for c in dev for d in c["docs"]}
    test_emails = {_email_key(d) for c in test for d in c["docs"]}
    assert dev_emails.isdisjoint(test_emails), "an email (target or distractor) appears in both splits"


def test_dev_test_disjoint_by_template(real):
    dev_t = {c["template_id"] for c in _attacks(real["dev"])}
    test_t = {c["template_id"] for c in _attacks(real["test"])}
    assert dev_t and test_t and dev_t.isdisjoint(test_t)


def test_every_override_template_is_used(real):
    half = len(BASE_INSTRUCTIONS) // 2
    used = {s: {c["template_id"] for c in _attacks(real[s]) if c["attack_family"] == "override"}
            for s in ("dev", "test")}
    assert used["dev"] == {f"override-{i}" for i in range(half)}
    assert used["test"] == {f"override-{i}" for i in range(half, len(BASE_INSTRUCTIONS))}


def test_canaries_are_unique_and_wellformed(real):
    canaries = [c["canary"] for s in real.values() for c in _attacks(s)]
    assert len(set(canaries)) == len(canaries)
    assert all(CANARY_RX.match(c) for c in canaries)


@pytest.mark.parametrize("split,minimum", [("dev", MIN_DEV_DIRECTIVES), ("test", MIN_TEST_DIRECTIVES)])
def test_benign_directives(real, split, minimum):
    n = 0
    for case in _benign(real[split]):
        target = next(d for d in case["docs"] if d["doc_id"] == case["target_doc"])
        n += bool(target.get("benign_instruction"))
    assert n >= minimum, f"{split}: {n} benign cases carry a directive, expected >= {minimum}"


def test_poisoned_docs_have_clean_text(real):
    for rows in real.values():
        for case in _attacks(rows):
            doc = _poisoned(case)
            assert doc.get("poisoned") is True
            assert "clean_text" in doc, f"{case['id']}: poisoned doc lacks clean_text"
            assert case["canary"] not in doc["clean_text"] and CANARY_MARKER not in doc["clean_text"]
            assert doc["text"] == doc["clean_text"] + "\n\n" + case["injection_text"]
            assert CANARY_MARKER in doc["text"]
