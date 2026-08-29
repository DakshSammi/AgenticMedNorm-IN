from __future__ import annotations

from src.evidence.stage5 import compare_component_count, compare_dosage_form, compare_strength


def test_evidence_rule_helpers_distinguish_match_conflict_and_missing():
    assert compare_strength("10 mg", ["10mg"])[0] == "MATCH"
    assert compare_strength("", ["10mg"])[0] == "NOT_COMPARABLE"
    assert compare_strength("10mg", ["20mg"]) == ("CONFLICT", "STRENGTH_CONFLICT")
    assert compare_dosage_form("tab", "tablet")[0] == "MATCH"
    assert compare_component_count(2, 3) == ("CONFLICT", "COMPONENT_COUNT_CONFLICT")
