"""Tests for agent.triage — complexity heuristic."""

from agent.triage import classify


def test_simple_question_is_not_complex():
    t = classify("what time is it?")
    assert not t.complex
    assert t.reasons == []


def test_explicit_build_is_complex():
    t = classify("build me a CLI word counter in tools/wordcount.py with tests")
    assert t.complex
    assert any("build" in r for r in t.reasons)


def test_refactor_is_complex():
    t = classify("refactor the auth module")
    assert t.complex
    assert any("refactor" in r for r in t.reasons)


def test_multi_step_marker():
    t = classify("first, create the file and then run the tests")
    assert t.complex
    assert any("multi-step" in r for r in t.reasons) or any(
        "first" in r for r in t.reasons
    )


def test_long_with_many_ands():
    msg = (
        "Please look at the foo module and also investigate the bar subsystem "
        "and then examine the baz pipeline and see if the logs line up correctly "
        "with the downstream consumers that we have been recently modifying."
    )
    t = classify(msg)
    assert t.complex


def test_short_question_no_false_positive():
    t = classify("what is 2 + 2?")
    assert not t.complex
