"""Shared fixtures. Unit tests read tests/fixtures/ only — never data/processed/."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import load_cases

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture()
def cases_dev() -> list[dict]:
    return load_cases(FIXTURES / "cases_dev.jsonl")


@pytest.fixture()
def cases_test() -> list[dict]:
    return load_cases(FIXTURES / "cases_test.jsonl")


@pytest.fixture()
def cases_adaptive() -> list[dict]:
    return load_cases(FIXTURES / "cases_adaptive.jsonl")
