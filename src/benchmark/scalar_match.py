from typing import Dict, Any, List, Optional
try:
    from rapidfuzz import fuzz
except ImportError:
    class FuzzFallback:
        @staticmethod
        def ratio(s1: str, s2: str) -> float:
            import difflib
            return difflib.SequenceMatcher(None, s1, s2).ratio() * 100.0
        @staticmethod
        def partial_ratio(s1: str, s2: str) -> float:
            if not s1 or not s2:
                return 0.0
            if s1 in s2 or s2 in s1:
                return 100.0
            import difflib
            return difflib.SequenceMatcher(None, s1, s2).real_quick_ratio() * 100.0
    fuzz = FuzzFallback()
from src.schemas.raw_extraction import CanonicalRawDoc
from src.schemas.benchmark import ScalarMatchDetail
from src.benchmark.normalisation import TextNormaliser

class ScalarMatcher:
    def __init__(self, lenient_threshold: float = 90.0):
        self.lenient_threshold = lenient_threshold

    def match_field(self, field_name: str, gt_val: Optional[str], pred_val: Optional[str]) -> ScalarMatchDetail:
        gt_norm = TextNormaliser.normalise(gt_val)
        pred_norm = TextNormaliser.normalise(pred_val)
        
        # Handling nulls
        if not gt_norm and not pred_norm:
            return ScalarMatchDetail(
                field_name=field_name,
                gt_value=gt_val,
                pred_value=pred_val,
                exact_match=True,
                lenient_match=True,
                similarity_score=100.0
            )
            
        if not gt_norm or not pred_norm:
            return ScalarMatchDetail(
                field_name=field_name,
                gt_value=gt_val,
                pred_value=pred_val,
                exact_match=False,
                lenient_match=False,
                similarity_score=0.0
            )
            
        # exact match check
        exact = (gt_norm == pred_norm)
        
        # lenient match check via rapidfuzz
        sim_score = fuzz.ratio(gt_norm, pred_norm)
        lenient = exact or (sim_score >= self.lenient_threshold)
        
        return ScalarMatchDetail(
            field_name=field_name,
            gt_value=gt_val,
            pred_value=pred_val,
            exact_match=exact,
            lenient_match=lenient,
            similarity_score=sim_score
        )

    def match_docs(self, gt_doc: CanonicalRawDoc, pred_doc: CanonicalRawDoc) -> List[ScalarMatchDetail]:
        details = []
        
        # Patient fields
        gt_pat = gt_doc.patient_information
        pred_pat = pred_doc.patient_information
        patient_fields = ["name", "age", "gender", "address", "phone", "patient_identifier", "abha_id"]
        for field in patient_fields:
            gt_val = getattr(gt_pat, field, None)
            pred_val = getattr(pred_pat, field, None)
            details.append(self.match_field(f"patient_info.{field}", gt_val, pred_val))
            
        # Encounter fields
        gt_enc = gt_doc.encounter_information
        pred_enc = pred_doc.encounter_information
        encounter_fields = ["date", "department", "hospital_name", "doctor_name", "visit_type", "fees", "room_or_queue_no"]
        for field in encounter_fields:
            gt_val = getattr(gt_enc, field, None)
            pred_val = getattr(pred_enc, field, None)
            details.append(self.match_field(f"encounter_info.{field}", gt_val, pred_val))
            
        return details
