from __future__ import annotations

import json

import pandas as pd

from src.verification.stage6 import _adequate_for_accept


def test_verification_rejects_hard_conflicts():
    row = pd.Series(
        {
            "resolution_level": "INGREDIENT_ONLY",
            "entity_type": "INGREDIENT",
            "lexical_status": "MATCH",
            "semantic_status": "MATCH",
            "hard_conflicts_json": json.dumps(["STRENGTH_CONFLICT"]),
            "candidate_facts_json": json.dumps({"source_state": "SYNTHETIC"}),
            "provenance_evidence_json": json.dumps({"source_states": ["SYNTHETIC"]}),
        }
    )
    assert _adequate_for_accept(row) == (False, "HARD_FORMULATION_CONFLICT")
