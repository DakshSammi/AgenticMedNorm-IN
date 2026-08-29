import re
from typing import List, Optional
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
from src.schemas.benchmark import EntityMatchDetail, UnmatchedPredictionRecord
from src.benchmark.normalisation import TextNormaliser

class HallucinationDetector:
    def __init__(self, hallucination_threshold: float = 60.0, gap_threshold: float = 80.0):
        self.hallucination_threshold = hallucination_threshold
        self.gap_threshold = gap_threshold

    def _get_gt_text_pool(self, gt_doc: CanonicalRawDoc) -> List[str]:
        """
        Collect all text fragments present in the ground truth document to form an evidence pool.
        """
        pool = []
        
        # Patient Info
        p = gt_doc.patient_information
        for val in [p.name, p.age, p.gender, p.address, p.phone, p.patient_identifier, p.abha_id]:
            if val:
                pool.append(val)
                
        # Encounter Info
        e = gt_doc.encounter_information
        for val in [e.date, e.department, e.hospital_name, e.doctor_name, e.visit_type, e.fees, e.room_or_queue_no]:
            if val:
                pool.append(val)
                
        # List entities
        for cat_list in [
            gt_doc.complaints_or_diagnosis,
            gt_doc.observations,
            gt_doc.procedures,
            gt_doc.advice,
            gt_doc.allergy_mentions,
            gt_doc.other_notes
        ]:
            for item in cat_list:
                if item.raw_text:
                    pool.append(item.raw_text)
                if item.evidence_text:
                    pool.append(item.evidence_text)
                    
        # Medications
        for med in gt_doc.medications:
            if med.raw_line_text:
                pool.append(med.raw_line_text)
            for val in [med.raw_name, med.raw_dosage, med.raw_route, med.raw_frequency, med.raw_duration, med.raw_instruction, med.raw_timing]:
                if val:
                    pool.append(val)
                    
        # Lab Observations
        for lab in gt_doc.lab_observations:
            if lab.raw_line_text:
                pool.append(lab.raw_line_text)
            if lab.evidence_text:
                pool.append(lab.evidence_text)
            for val in [lab.test_name, lab.result, lab.unit, lab.reference_range]:
                if val:
                    pool.append(val)
                    
        # Follow Up
        if gt_doc.follow_up:
            f = gt_doc.follow_up
            if f.raw_text:
                pool.append(f.raw_text)
            if f.date:
                pool.append(f.date)
            if f.review_after:
                pool.append(f.review_after)
                
        return pool

    def detect_hallucinations(
        self, 
        gt_doc: CanonicalRawDoc, 
        pred_doc: CanonicalRawDoc, 
        entity_alignments: List[EntityMatchDetail]
    ) -> List[UnmatchedPredictionRecord]:
        records = []
        
        # 1. Gather Ground Truth Evidence Pool
        gt_pool = self._get_gt_text_pool(gt_doc)
        gt_pool_norm = [TextNormaliser.normalise(t) for t in gt_pool if t]
        
        # 2. Check Entity False Positives (Unmatched Predictions)
        for align in entity_alignments:
            if align.alignment_status != "FP":
                continue
                
            pred_text = align.pred_raw_text
            if not pred_text:
                continue
                
            pred_norm = TextNormaliser.normalise(pred_text)
            
            # Find best match in ground truth text pool (any category)
            best_score = 0.0
            best_snippet = None
            
            for gt_orig, gt_norm in zip(gt_pool, gt_pool_norm):
                if not gt_norm:
                    continue
                score = fuzz.partial_ratio(pred_norm, gt_norm)
                if score > best_score:
                    best_score = score
                    best_snippet = gt_orig
            
            # Determine classification
            if best_score >= self.gap_threshold:
                # Text is in GT, but mapped to wrong category or is annotation gap
                classification = "annotation_gap_candidate"
                rationale = (
                    f"Predicted text '{pred_text}' matches GT text '{best_snippet}' (similarity {best_score:.1f}%) "
                    f"in a different context or was missed in category '{align.category}'."
                )
            elif best_score < self.hallucination_threshold:
                # Text is not found anywhere in GT
                classification = "likely_hallucination"
                rationale = (
                    f"Predicted text '{pred_text}' is not supported by any ground truth text "
                    f"(best match similarity is only {best_score:.1f}%)."
                )
            else:
                classification = "manual_review_required"
                rationale = (
                    f"Predicted text '{pred_text}' has moderate match with GT '{best_snippet}' "
                    f"(similarity {best_score:.1f}%), requiring human review."
                )
                
            records.append(UnmatchedPredictionRecord(
                category=align.category,
                pred_text=pred_text,
                confidence=1.0,
                evidence_text=pred_text,
                classification=classification,
                rationale=rationale,
                matched_snippet=best_snippet
            ))
            
        # 3. Check specific Scalar Hallucination: Kumar Nagar case
        # "Kumar Nagar predicted as patient name in p2 is flagged as likely hallucination/OCR confusion
        # because GT patient name is empty and address is Kirti Nagar"
        gt_name = gt_doc.patient_information.name
        pred_name = pred_doc.patient_information.name
        gt_address = gt_doc.patient_information.address
        
        if pred_name and not gt_name:
            pred_name_norm = TextNormaliser.normalise(pred_name)
            
            # Check if prediction is similar to GT address (like Kirti Nagar vs Kumar Nagar)
            address_sim = 0.0
            if gt_address:
                gt_addr_norm = TextNormaliser.normalise(gt_address)
                address_sim = fuzz.partial_ratio(pred_name_norm, gt_addr_norm)
                
            if address_sim >= 70.0:
                records.append(UnmatchedPredictionRecord(
                    category="patient_info.name",
                    pred_text=pred_name,
                    confidence=1.0,
                    evidence_text=pred_name,
                    classification="likely_hallucination",
                    rationale=(
                        f"Predicted patient name '{pred_name}' is a likely hallucination / OCR confusion "
                        f"because ground truth patient name is empty and matches ground truth address "
                        f"'{gt_address}' (similarity {address_sim:.1f}%)."
                    ),
                    matched_snippet=gt_address
                ))
            else:
                # General scalar hallucination if predicted name is completely novel
                records.append(UnmatchedPredictionRecord(
                    category="patient_info.name",
                    pred_text=pred_name,
                    confidence=1.0,
                    evidence_text=pred_name,
                    classification="likely_hallucination",
                    rationale="Predicted patient name when ground truth name is empty and no matching fields exist.",
                    matched_snippet=None
                ))
                
        return records
