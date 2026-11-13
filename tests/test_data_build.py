"""Integrity checks for the case-file schema, run on the synthetic tests/fixtures/ dataset.

The fixtures use exactly the JSONL schema of data/processed/, so these checks do not
depend on the real data being built. The real files are audited separately by
tests/test_real_data.py (marker: data).
"""

from __future__ import annotations

import re

CANARY_RX = re.compile(r"^[A-Z0-9]{4}-[A-Z0-9]{5}$")


def _attacks(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["kind"] == "attack"]


def _benign(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["kind"] == "benign"]


def test_fixtures_have_expected_shape(cases_dev, cases_test, cases_adaptive):
    assert len(_attacks(cases_dev)) == 8 and len(_benign(cases_dev)) == 4
    assert len(_attacks(cases_test)) == 12 and len(_benign(cases_test)) == 6
    assert len(cases_adaptive) == 3
    required = {"id", "kind", "split", "question", "reference_answers", "docs"}
    for r in cases_dev + cases_test + cases_adaptive:
        assert required <= set(r), r["id"]


def test_benign_cases_are_answerable(cases_test):
    benign = _benign(cases_test)
    assert benign
    for r in benign:
        assert r["reference_answers"], f"{r['id']}: benign case without reference answer"
        assert all(a.strip().lower() != "unknown" for a in r["reference_answers"])


def test_attack_canaries_unique_and_wellformed(cases_dev, cases_test, cases_adaptive):
    canaries = [r["canary"] for r in _attacks(cases_dev + cases_test)] + [r["canary"] for r in cases_adaptive]
    assert len(set(canaries)) == len(canaries), "canary collision across attack cases"
    assert all(CANARY_RX.match(c) for c in canaries)


def test_attack_docs_carry_poisoned_target(cases_test):
    for r in _attacks(cases_test):
        targets = [d for d in r["docs"] if d["doc_id"] == r["poisoned_doc"]]
        assert len(targets) == 1 and targets[0].get("poisoned") is True, r["id"]
        assert r["canary"] in targets[0]["text"], f"{r['id']}: canary missing from poisoned text"


def test_benign_docs_not_poisoned(cases_test):
    for r in _benign(cases_test):
        assert not any(d.get("poisoned") for d in r["docs"]), r["id"]


def test_dev_test_disjoint_by_source(cases_dev, cases_test):
    dev_sources = {c["id"].rsplit("-", 1)[0] for c in cases_dev}
    test_sources = {c["id"].rsplit("-", 1)[0] for c in cases_test}
    assert not (dev_sources & test_sources)


def test_split_labels_match_files(cases_dev, cases_test):
    assert all(c["split"] == "dev" for c in cases_dev)
    assert all(c["split"] == "test" for c in cases_test)
