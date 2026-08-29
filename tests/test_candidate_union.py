from __future__ import annotations

from scripts.run_pipeline import candidate_union


def test_candidate_union_is_unique_by_mention_and_candidate():
    trace = [
        {"mention_id": "m1", "raw_medication_text": "Tab A", "branch": "R1_EXACT_FUZZY", "candidate_id": "ENTITY:INGREDIENT:A", "candidate_name": "A", "entity_type": "INGREDIENT", "source_state": "SYNTHETIC"},
        {"mention_id": "m1", "raw_medication_text": "Tab A", "branch": "R2_BM25", "candidate_id": "ENTITY:INGREDIENT:A", "candidate_name": "A", "entity_type": "INGREDIENT", "source_state": "SYNTHETIC"},
        {"mention_id": "m1", "raw_medication_text": "Tab A", "branch": "R1_EXACT_FUZZY", "candidate_id": "ENTITY:INGREDIENT:B", "candidate_name": "B", "entity_type": "INGREDIENT", "source_state": "SYNTHETIC"},
    ]
    union = candidate_union(trace)
    assert len(union) == 2
    by_id = {row["candidate_id"]: row for row in union}
    assert by_id["ENTITY:INGREDIENT:A"]["branches_returned"] == "R1_EXACT_FUZZY|R2_BM25"
    assert by_id["ENTITY:INGREDIENT:A"]["branch_count"] == 2
