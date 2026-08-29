from __future__ import annotations

from src.ranking.stage4 import reciprocal_rank_fusion


def test_unweighted_rrf_uses_k_60_formula():
    score, components = reciprocal_rank_fusion({"R1_EXACT_FUZZY": 1, "R3_BIOMEDICAL_DENSE": 4}, rrf_k=60)
    assert abs(score - ((1 / (60 + 1)) + (1 / (60 + 4)))) < 1e-12
    assert components["R1_EXACT_FUZZY"] == 1 / 61
    assert components["R3_BIOMEDICAL_DENSE"] == 1 / 64
