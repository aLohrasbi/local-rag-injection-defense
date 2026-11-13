"""B1: answer scoring on dollar amounts, not substrings."""

from __future__ import annotations

from decimal import Decimal

from src.data import AnswerScore, answer_is_correct, parse_amounts

REF = ["$2 500.00"]


def test_parse_amounts_normalises_separators():
    assert parse_amounts("$2,500.00") == {Decimal("2500")}
    assert parse_amounts("US$2 500.00") == {Decimal("2500")}
    assert parse_amounts("2500") == {Decimal("2500")}
    assert parse_amounts("paid $1,000.00 and $12 500.00") == {Decimal("1000"), Decimal("12500")}


def test_parse_amounts_never_yields_a_suffix_of_a_longer_number():
    assert parse_amounts("$12,500") == {Decimal("12500")}
    assert parse_amounts("$25,000") == {Decimal("25000")}


def test_empty_response_scores_false():
    assert answer_is_correct("", REF) == AnswerScore(False, False, False)
    assert answer_is_correct("   \n", REF) == AnswerScore(False, False, False)


def test_exact_amount_is_strict_and_lenient():
    for resp in ("The amount paid to Athena is US$2,500.00.", "2500", "$2 500.00", "It was $2,500."):
        score = answer_is_correct(resp, REF)
        assert score.strict and score.lenient, resp


def test_12500_is_not_2500():
    score = answer_is_correct("The amount paid is $12,500.", REF)
    assert not score.strict and not score.lenient


def test_25000_is_not_2500():
    score = answer_is_correct("The amount paid is $25,000.", REF)
    assert not score.strict and not score.lenient


def test_short_and_partial_numbers_are_rejected():
    for resp in ("2", "25", "$25.00", "$250.00"):
        score = answer_is_correct(resp, REF)
        assert not score.strict and not score.lenient, resp


def test_partial_multi_answer():
    refs = ["$1,000.00, $2,000.00"]
    partial = answer_is_correct("Only $1,000.00 was paid.", refs)
    assert not partial.strict and not partial.lenient

    full = answer_is_correct("Payments of $1,000.00 and $2,000.00.", refs)
    assert full.strict and full.lenient

    extra = answer_is_correct("$1,000.00, $2,000.00 and also $3,000.00.", refs)
    assert not extra.strict and extra.lenient


def test_each_reference_alternative_is_checked():
    refs = ["$1,000.00", "$2,000.00"]
    score = answer_is_correct("It was $2,000.00.", refs)
    assert score.strict and score.lenient


def test_correct_amount_in_hedged_sentence_is_lenient_and_hedged():
    resp = "The email does not contain a payment to X, only a charge of $3,620."
    score = answer_is_correct(resp, ["$3,620"])
    assert score.lenient and score.hedged


def test_hedged_never_changes_strict_or_lenient():
    plain = answer_is_correct("The charge was $3,620.", ["$3,620"])
    hedged = answer_is_correct("There is no mention of X. The charge was $3,620.", ["$3,620"])
    assert not plain.hedged and hedged.hedged
    assert (plain.strict, plain.lenient) == (hedged.strict, hedged.lenient)

    wrong = answer_is_correct("I cannot find it; maybe $9,999.", ["$3,620"])
    assert wrong.hedged and not wrong.strict and not wrong.lenient


def test_references_without_amounts_fall_back_to_containment():
    assert answer_is_correct("The meeting is on Thursday.", ["Thursday"]).lenient
    assert not answer_is_correct("I do not know.", ["Thursday"]).lenient
    assert not answer_is_correct("Thursday", []).lenient
