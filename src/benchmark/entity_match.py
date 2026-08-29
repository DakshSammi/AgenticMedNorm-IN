import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy.optimize import linear_sum_assignment
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
from src.schemas.raw_extraction import CanonicalRawDoc, RawEntityItem, RawMedicationItem, RawLabObservationItem
from src.schemas.benchmark import EntityMatchDetail, CategoryMetrics
from src.benchmark.normalisation import TextNormaliser

class EntityMatcher:
    def __init__(self, exact_threshold: float = 95.0, lenient_threshold: float = 80.0, review_threshold: float = 65.0, medication_weights: Optional[Dict[str, float]] = None):
        self.exact_threshold = exact_threshold / 100.0
        self.lenient_threshold = lenient_threshold / 100.0
        self.review_threshold = review_threshold / 100.0
        self.medication_weights = medication_weights or {
            "name": 0.40,
            "dosage": 0.12,
            "route": 0.12,
            "frequency": 0.12,
            "duration": 0.12,
            "instruction": 0.12
        }

    def compute_string_similarity(self, s1: Optional[str], s2: Optional[str]) -> float:
        norm1 = TextNormaliser.normalise(s1)
        norm2 = TextNormaliser.normalise(s2)
        if not norm1 and not norm2:
            return 1.0
        if not norm1 or not norm2:
            return 0.0
        if norm1 == norm2:
            return 1.0
        return fuzz.ratio(norm1, norm2) / 100.0

    def compute_medication_similarity(self, gt: RawMedicationItem, pred: RawMedicationItem) -> Tuple[float, Dict[str, Any]]:
        # Medication component matching logic using configurable weights
        w_name = self.medication_weights.get("name", 0.40)
        w_dosage = self.medication_weights.get("dosage", 0.12)
        w_route = self.medication_weights.get("route", 0.12)
        w_frequency = self.medication_weights.get("frequency", 0.12)
        w_duration = self.medication_weights.get("duration", 0.12)
        w_instruction = self.medication_weights.get("instruction", 0.12)

        components = {
            "name": (gt.raw_name, pred.raw_name, w_name),
            "dosage": (gt.raw_dosage, pred.raw_dosage, w_dosage),
            "route": (gt.raw_route, pred.raw_route, w_route),
            "frequency": (gt.raw_frequency, pred.raw_frequency, w_frequency),
            "duration": (gt.raw_duration, pred.raw_duration, w_duration),
            "instruction": (gt.raw_instruction, pred.raw_instruction, w_instruction)
        }
        
        scores = {}
        weighted_sum = 0.0
        for comp_name, (gt_v, pred_v, wt) in components.items():
            if not gt_v and not pred_v:
                comp_score = 1.0
            elif not gt_v or not pred_v:
                comp_score = 0.0
            else:
                comp_score = self.compute_string_similarity(gt_v, pred_v)
            
            scores[comp_name] = comp_score
            weighted_sum += comp_score * wt
            
        return weighted_sum, scores

    def compute_lab_observation_similarity(self, gt: RawLabObservationItem, pred: RawLabObservationItem) -> Tuple[float, Dict[str, Any]]:
        # Lab observation component matching logic
        w_name = 0.40
        w_result = 0.30
        w_unit = 0.15
        w_ref = 0.15

        components = {
            "test_name": (gt.test_name, pred.test_name, w_name),
            "result": (gt.result, pred.result, w_result),
            "unit": (gt.unit, pred.unit, w_unit),
            "reference_range": (gt.reference_range, pred.reference_range, w_ref)
        }
        
        scores = {}
        weighted_sum = 0.0
        for comp_name, (gt_v, pred_v, wt) in components.items():
            if not gt_v and not pred_v:
                comp_score = 1.0
            elif not gt_v or not pred_v:
                comp_score = 0.0
            else:
                comp_score = self.compute_string_similarity(gt_v, pred_v)
            
            scores[comp_name] = comp_score
            weighted_sum += comp_score * wt
            
        return weighted_sum, scores

    def align_entities(self, gt_items: List[Any], pred_items: List[Any], category: str) -> List[EntityMatchDetail]:
        details = []
        if not gt_items and not pred_items:
            return details
            
        n_gt = len(gt_items)
        n_pred = len(pred_items)
        
        # 1. Build similarity and assignment matrices
        sim_matrix = np.zeros((n_gt, n_pred))
        assign_matrix = np.zeros((n_gt, n_pred))
        med_subfields_matrix = {} # cache for medications/lab subfields
        
        for i in range(n_gt):
            for j in range(n_pred):
                gt_item = gt_items[i]
                pred_item = pred_items[j]
                
                if category == "medications":
                    # Dynamic similarity score
                    sim, subfields = self.compute_medication_similarity(gt_item, pred_item)
                    sim_matrix[i, j] = sim
                    med_subfields_matrix[(i, j)] = subfields
                    
                    # Assignment based primarily on name/text similarity
                    name_sim = self.compute_string_similarity(gt_item.raw_name, pred_item.raw_name)
                    if not gt_item.raw_name and not pred_item.raw_name:
                        name_sim = self.compute_string_similarity(gt_item.raw_line_text, pred_item.raw_line_text)
                    assign_matrix[i, j] = name_sim
                elif category == "lab_observations":
                    sim, subfields = self.compute_lab_observation_similarity(gt_item, pred_item)
                    sim_matrix[i, j] = sim
                    med_subfields_matrix[(i, j)] = subfields
                    
                    name_sim = self.compute_string_similarity(gt_item.test_name, pred_item.test_name)
                    if not gt_item.test_name and not pred_item.test_name:
                        name_sim = self.compute_string_similarity(gt_item.raw_line_text, pred_item.raw_line_text)
                    assign_matrix[i, j] = name_sim
                else:
                    # String entity matching
                    gt_text = gt_item.raw_text if hasattr(gt_item, "raw_text") else str(gt_item)
                    pred_text = pred_item.raw_text if hasattr(pred_item, "raw_text") else str(pred_item)
                    sim_matrix[i, j] = self.compute_string_similarity(gt_text, pred_text)
                    assign_matrix[i, j] = sim_matrix[i, j]
                    
        # 2. Run Hungarian Bipartite Assignment on assignment matrix (based on name similarity for medications)
        # cost = 1.0 - assignment
        cost_matrix = 1.0 - assign_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_gt = set()
        matched_pred = set()
        
        # 3. Process matches
        for r, c in zip(row_ind, col_ind):
            gt_item = gt_items[r]
            pred_item = pred_items[c]
            similarity = sim_matrix[r, c]
            
            gt_text = gt_item.raw_line_text if hasattr(gt_item, "raw_line_text") else (gt_item.raw_text if hasattr(gt_item, "raw_text") else str(gt_item))
            pred_text = pred_item.raw_line_text if hasattr(pred_item, "raw_line_text") else (pred_item.raw_text if hasattr(pred_item, "raw_text") else str(pred_item))
            
            subfields = med_subfields_matrix.get((r, c)) if category in ["medications", "lab_observations"] else None
            
            # Match thresholds
            exact = (similarity >= self.exact_threshold)
            lenient = (similarity >= self.lenient_threshold)
            
            name_matches = False
            if category == "medications":
                name_sim = self.compute_string_similarity(gt_item.raw_name, pred_item.raw_name)
                if not gt_item.raw_name and not pred_item.raw_name:
                    name_sim = self.compute_string_similarity(gt_item.raw_line_text, pred_item.raw_line_text)
                name_matches = (name_sim >= self.review_threshold)
            elif category == "lab_observations":
                name_sim = self.compute_string_similarity(gt_item.test_name, pred_item.test_name)
                if not gt_item.test_name and not pred_item.test_name:
                    name_sim = self.compute_string_similarity(gt_item.raw_line_text, pred_item.raw_line_text)
                name_matches = (name_sim >= self.review_threshold)
            
            if similarity >= self.review_threshold or (category in ["medications", "lab_observations"] and name_matches):
                alignment_status = "TP_EXACT" if exact else "TP_LENIENT"
                if alignment_status in ["TP_EXACT", "TP_LENIENT"]:
                    lenient = True
                details.append(EntityMatchDetail(
                    category=category,
                    gt_raw_text=gt_text,
                    pred_raw_text=pred_text,
                    exact_match=exact,
                    lenient_match=lenient,
                    similarity_score=similarity * 100.0,
                    alignment_status=alignment_status,
                    medication_subfields=subfields
                ))
                matched_gt.add(r)
                matched_pred.add(c)
                
        # 4. Process unmatched Ground Truths (False Negatives)
        for r in range(n_gt):
            if r not in matched_gt:
                gt_item = gt_items[r]
                gt_text = gt_item.raw_line_text if hasattr(gt_item, "raw_line_text") else (gt_item.raw_text if hasattr(gt_item, "raw_text") else str(gt_item))
                details.append(EntityMatchDetail(
                    category=category,
                    gt_raw_text=gt_text,
                    pred_raw_text=None,
                    exact_match=False,
                    lenient_match=False,
                    similarity_score=0.0,
                    alignment_status="FN"
                ))
                
        # 5. Process unmatched Predictions (False Positives)
        for c in range(n_pred):
            if c not in matched_pred:
                pred_item = pred_items[c]
                pred_text = pred_item.raw_line_text if hasattr(pred_item, "raw_line_text") else (pred_item.raw_text if hasattr(pred_item, "raw_text") else str(pred_item))
                details.append(EntityMatchDetail(
                    category=category,
                    gt_raw_text=None,
                    pred_raw_text=pred_text,
                    exact_match=False,
                    lenient_match=False,
                    similarity_score=0.0,
                    alignment_status="FP"
                ))
                
        return details

    def compute_category_metrics(self, alignments: List[EntityMatchDetail]) -> CategoryMetrics:
        tp_exact = sum(1 for d in alignments if d.alignment_status == "TP_EXACT")
        tp_lenient = sum(1 for d in alignments if d.alignment_status in ["TP_EXACT", "TP_LENIENT"])
        fp = sum(1 for d in alignments if d.alignment_status == "FP")
        fn = sum(1 for d in alignments if d.alignment_status == "FN")
        
        def calculate_prf(tp, fp, fn):
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            return prec, rec, f1

        prec_ex, rec_ex, f1_ex = calculate_prf(tp_exact, fp, fn)
        prec_len, rec_len, f1_len = calculate_prf(tp_lenient, fp, fn)
        
        return CategoryMetrics(
            precision_exact=prec_ex,
            recall_exact=rec_ex,
            f1_exact=f1_ex,
            precision_lenient=prec_len,
            recall_lenient=rec_len,
            f1_lenient=f1_len
        )
