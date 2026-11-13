"""B6-B11: pipeline and runner behaviour, end to end with FakeLLM (no Ollama, no torch)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from scripts import make_figure
from scripts import run_experiment as cli
from src.data import load_cases, write_cases
from src.defenses import RuleFilter
from src.llm import FakeLLM
from src.metrics import record_from_row
from src.rag import RAGPipeline
from src.runner import cell_order, read_records, run_experiment
from tests.helpers import FIXTURE_TEST, HashRetriever, SlowRetriever, make_config

ANSWER = ["The amount is $900.00."]
N_CASES, N_ATTACK, N_UNANSWERABLE = 18, 12, 3


@pytest.fixture()
def cfg(tmp_path):
    return make_config(tmp_path / "results")


def _rows(cfg: dict) -> list[dict]:
    path = Path(cfg["paths"]["results_dir"]) / "results.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run(cfg: dict, llm: FakeLLM | None = None, **kw):
    return run_experiment(cfg, client=llm or FakeLLM(ANSWER), retriever=kw.pop("retriever", HashRetriever()),
                          **kw)


def _unanswerable_attack() -> dict:
    return next(c for c in load_cases(FIXTURE_TEST) if c["kind"] == "attack" and not c["answerable"])


# ----------------------------------------------------------------- end to end

def test_end_to_end_with_fakellm(cfg):
    stats = _run(cfg)
    rows = _rows(cfg)
    assert stats["records"] == len(rows) == 2 * 4 * N_CASES
    assert {r["model"] for r in rows} == {"fake-a", "fake-b"}
    report = (Path(cfg["paths"]["results_dir"]) / "report.md").read_text()
    assert "Canary rate" in report and "ASR" not in report


def test_fakellm_scripts_drive_canary_scoring(cfg):
    def script(prompt, system):
        for c in load_cases(FIXTURE_TEST):
            if c["kind"] == "attack" and c["canary"] in prompt:
                return f"Sure. {c['canary']}"
        return "no code"

    _run(cfg, FakeLLM(script), forced_retrieval=False)
    attacks = [r for r in _rows(cfg) if r["kind"] == "attack" and r["defense"] == "none"]
    for r in attacks:
        assert r["canary_emitted"] == r["poisoned_retrieved"]  # emitted iff the poisoned doc was shown


# ----------------------------------------------------------------- B6 latency

def test_b6_latency_gen_and_total_recorded_on_every_record(cfg):
    _run(cfg, FakeLLM(ANSWER, delay_s=0.01), retriever=SlowRetriever(0.02), max_cases=4)
    rows = _rows(cfg)
    assert rows
    for r in rows:
        assert r["latency_gen_s"] >= 0.01
        assert r["latency_total_s"] >= r["latency_gen_s"] + 0.02 * 0.9   # retrieval is in total only
        assert "latency_s" not in r
    report = (Path(cfg["paths"]["results_dir"]) / "report.md").read_text()
    assert "Gen p50" in report and "Total p95" in report


# ----------------------------------------------------------------- B7 forced retrieval

def _pipeline(top_k=3):
    return RAGPipeline(FakeLLM(ANSWER), HashRetriever(), RuleFilter(), top_k=top_k)


def test_b7_forced_doc_gets_its_real_score_and_replaces_the_lowest_hit():
    case = _unanswerable_attack()
    pipe = _pipeline()
    pipe.index_case(case["docs"])
    natural = pipe.retriever.search(case["question"], 3)
    assert case["poisoned_doc"] not in [r.doc.doc_id for r in natural]   # precondition

    _result, info = pipe.answer(case["question"], "none", forced_doc_id=case["poisoned_doc"])
    ids, scores = info["docs_retrieved"], info["scores"]
    assert info["forced"] is True
    assert len(ids) == 3 and case["poisoned_doc"] in ids
    assert scores == sorted(scores, reverse=True)                        # re-sorted by score
    # the lowest-ranked natural hit was displaced, the others kept
    assert natural[-1].doc.doc_id not in ids
    assert {r.doc.doc_id for r in natural[:-1]} <= set(ids)
    # the forced doc carries its true similarity, not a placeholder 0.0
    true = {r.doc.doc_id: r.score for r in pipe.retriever.search(case["question"], 100)}
    assert scores == pytest.approx([true[i] for i in ids])
    assert true[case["poisoned_doc"]] > 0.0


def test_b7_naturally_retrieved_doc_is_not_marked_forced():
    case = next(c for c in load_cases(FIXTURE_TEST) if c["kind"] == "attack" and c["answerable"])
    pipe = _pipeline()
    pipe.index_case(case["docs"])
    natural = [r.doc.doc_id for r in pipe.retriever.search(case["question"], 3)]
    assert case["poisoned_doc"] in natural
    _result, info = pipe.answer(case["question"], "none", forced_doc_id=case["poisoned_doc"])
    assert info["forced"] is False and info["docs_retrieved"] == natural


def test_b7_poisoned_rank_and_forced_logged_on_every_record(tmp_path):
    main_cfg = make_config(tmp_path / "main", defenses=["none"])
    forced_cfg = make_config(tmp_path / "forced", defenses=["none"])
    _run(main_cfg)
    _run(forced_cfg, forced_retrieval=True)

    unanswerable = {c["id"] for c in load_cases(FIXTURE_TEST) if c["kind"] == "attack" and not c["answerable"]}
    assert len(unanswerable) == N_UNANSWERABLE
    for r in _rows(main_cfg) + _rows(forced_cfg):
        assert "poisoned_rank" in r and "forced" in r
    for r in _rows(main_cfg):
        assert r["forced"] is False
        if r["kind"] == "attack":
            assert (r["poisoned_rank"] is None) == (r["case_id"] in unanswerable)
            assert r["poisoned_retrieved"] == (r["poisoned_rank"] is not None)
        else:
            assert r["poisoned_rank"] is None
    for r in _rows(forced_cfg):
        assert r["mode"] == "forced_retrieval"
        if r["kind"] == "attack":
            assert r["poisoned_retrieved"] is True and 1 <= r["poisoned_rank"] <= 3
            assert r["forced"] == (r["case_id"] in unanswerable)
        else:
            assert r["poisoned_rank"] is None and r["forced"] is False


# ----------------------------------------------------------------- B8 positive control

def test_b8_positive_control_instruction_only_in_trusted_prompt(cfg):
    llm = FakeLLM(ANSWER)
    _run(cfg, llm, positive_control=True)
    attacks = [c for c in load_cases(FIXTURE_TEST) if c["kind"] == "attack"]
    canaries = [c["canary"] for c in attacks]
    assert len(llm.calls) == 2 * N_ATTACK
    for prompt, system in llm.calls:
        assert not any(cn in prompt for cn in canaries), "canary leaked into the retrieved context"
        assert system is not None and sum(cn in system for cn in canaries) == 1
    rows = _rows(cfg)
    assert {r["defense"] for r in rows} == {"positive_control"}
    assert all(r["mode"] == "positive_control" and r["annotation"] for r in rows)
    unanswerable = {c["id"] for c in attacks if not c["answerable"]}
    for r in rows:
        assert r["poisoned_retrieved"] == (r["case_id"] not in unanswerable)


def test_b8_main_run_by_contrast_shows_the_injection_in_the_context(cfg):
    llm = FakeLLM(ANSWER)
    _run(cfg, llm, max_cases=1)   # first fixture case: an answerable attack
    prompts_with_canary = [p for p, _s in llm.calls if "verification code" in p]
    assert prompts_with_canary
    assert not any("verification code" in (s or "") for _p, s in llm.calls)


def test_b8_positive_control_needs_clean_text(cfg, tmp_path):
    cases = load_cases(FIXTURE_TEST)
    for c in cases:
        for d in c["docs"]:
            d.pop("clean_text", None)
    legacy = tmp_path / "legacy.jsonl"
    write_cases(cases, legacy)
    cfg["data"]["cases_file"] = str(legacy)
    with pytest.raises(ValueError, match="clean_text"):
        _run(cfg, positive_control=True)
# ----------------------------------------------------------------- B9 resume

def test_b9_rows_are_flushed_before_the_next_llm_call(cfg):
    path = Path(cfg["paths"]["results_dir"]) / "results.jsonl"
    seen: list[int] = []

    def script(prompt, system):
        seen.append(len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0)
        return "x"

    _run(cfg, FakeLLM(script), max_cases=3)
    assert seen == list(range(len(seen))) and len(seen) == 2 * 4 * 3


def test_b9_resume_skips_completed_rows_run_twice(cfg):
    first = FakeLLM(ANSWER)
    _run(cfg, first)
    path = Path(cfg["paths"]["results_dir"]) / "results.jsonl"
    before = path.read_bytes()
    assert len(first.calls) == 2 * 4 * N_CASES

    second = FakeLLM(["SHOULD NEVER BE CALLED"])
    stats = _run(cfg, second, resume=True)
    assert second.calls == []
    assert path.read_bytes() == before
    assert stats["records"] == 2 * 4 * N_CASES   # report still covers the resumed rows


def test_b9_resume_completes_a_partial_run_without_duplicates(cfg):
    _run(cfg, max_cases=5)
    done_before = len(_rows(cfg))
    second = FakeLLM(ANSWER)
    _run(cfg, second, max_cases=9, resume=True)
    assert len(second.calls) == 2 * 4 * (9 - 5)
    rows = _rows(cfg)
    assert len(rows) == done_before + len(second.calls) == 2 * 4 * 9
    keys = [(r["mode"], r["model"], r["defense"], r["case_id"]) for r in rows]
    assert len(set(keys)) == len(keys)


def test_b9_existing_results_without_resume_is_an_error_and_untouched(cfg):
    _run(cfg, max_cases=2)
    path = Path(cfg["paths"]["results_dir"]) / "results.jsonl"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        _run(cfg, max_cases=2)
    assert path.read_bytes() == before


def test_b9_resume_survives_a_truncated_last_line(cfg):
    _run(cfg, max_cases=2)
    path = Path(cfg["paths"]["results_dir"]) / "results.jsonl"
    n_good = len(_rows(cfg))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"case_id": "email-0000-att')          # crash mid-write, no newline
    second = FakeLLM(ANSWER)
    _run(cfg, second, max_cases=3, resume=True)
    assert len(second.calls) == 2 * 4
    good = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith('{"case_id"')]
    parsed = []
    for line in good:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    assert len(parsed) == n_good + len(second.calls)


def test_b9_resume_key_includes_mode(tmp_path):
    """A forced-retrieval row must not mask the same (model, defense, case) of the main run."""
    cfg = make_config(tmp_path / "r", defenses=["none"])
    _run(cfg, forced_retrieval=True)
    second = FakeLLM(ANSWER)
    _run(cfg, second, resume=True)               # main mode: nothing of it is done yet
    assert len(second.calls) == 2 * N_CASES
    assert {r["mode"] for r in _rows(cfg)} == {"forced_retrieval", "main"}


def test_b9_cli_accepts_resume_flag(cfg, tmp_path, monkeypatch, capsys):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", str(cfg_path), "--dry-run", "--resume"])
    cli.main()
    assert "dry-run complete" in capsys.readouterr().out
    assert not (Path(cfg["paths"]["results_dir"]) / "results.jsonl").exists()   # dry-run writes nothing


# ----------------------------------------------------------------- B10 per-cell shuffle

def test_b10_cell_order_is_seeded_by_cell_name():
    cases = load_cases(FIXTURE_TEST)
    ids = lambda m, d: [c["id"] for c in cell_order(cases, 42, m, d)]  # noqa: E731
    assert ids("fake-a", "none") == ids("fake-a", "none")                     # reproducible
    cells = [ids(m, d) for m in ("fake-a", "fake-b") for d in ("none", "delimiter", "filter", "both")]
    assert len({tuple(c) for c in cells}) == len(cells)                       # every cell differs
    assert all(sorted(c) == sorted(cells[0]) for c in cells)                  # same cases in each


def test_b10_runner_uses_cell_order(cfg):
    _run(cfg, max_cases=6)
    cases = load_cases(FIXTURE_TEST)[:6]
    rows = _rows(cfg)
    for model in ("fake-a", "fake-b"):
        for defense in ("none", "both"):
            got = [r["case_id"] for r in rows if r["model"] == model and r["defense"] == defense]
            assert got == [c["id"] for c in cell_order(cases, 42, model, defense)]


# ----------------------------------------------------------------- B11 naming

def test_b11_records_carry_annotation_and_new_names(cfg):
    _run(cfg, max_cases=2)
    for r in _rows(cfg):
        assert "annotation" in r and "asr" not in r


def test_b11_legacy_rows_still_load():
    legacy = {"case_id": "a", "kind": "attack", "model": "m", "defense": "none", "variant": "plain",
              "response": "x", "canary": "AAAA-BBBBB", "canary_emitted": False, "poisoned_doc": "d",
              "poisoned_retrieved": True, "filter_flagged_docs": [], "docs_retrieved": ["d"],
              "answer_correct": None, "refusal": False, "latency_s": 2.5}
    rec = record_from_row(legacy)
    assert rec.latency_gen_s == 2.5 and rec.latency_total_s is None and rec.mode == "main"
    assert read_records(Path("does-not-exist.jsonl")) == []


def test_b11_figure_uses_canary_disclosure_wording(cfg, tmp_path, monkeypatch):
    _run(cfg, max_cases=4)
    out = tmp_path / "fig.svg"
    results = Path(cfg["paths"]["results_dir"]) / "results.jsonl"
    monkeypatch.setattr(sys, "argv", ["make_figure.py", "--results", str(results), "--out", str(out)])
    make_figure.main()
    svg = out.read_text(encoding="utf-8")
    assert "canary disclosure rate" in svg and "attack success" not in svg
