from typing import List, Dict, Any
from src.schemas.benchmark import DocumentBenchmarkResult, CategoryMetrics

class MetricAggregator:
    def __init__(self, config: Dict[str, Any] = None):
        # Allow custom weights for the experimental headline score
        self.config = config or {}
        self.weights = self.config.get("experimental_headline_weights", {
            "schema_parse": 0.10,
            "scalar_field": 0.20,
            "entity_lenient_f1": 0.45,
            "entity_exact_f1": 0.15,
            "hallucination_penalty": 0.10
        })

    def calculate_rates(self, result: DocumentBenchmarkResult, gt_entities_count: int, pred_entities_count: int) -> DocumentBenchmarkResult:
        """
        Populate Hallucination Rate, Missing Entity Rate, and Annotation-Gap Rate.
        """
        hallucinations = result.likely_hallucination_count
        gaps = result.annotation_gap_candidate_count
        
        # Hallucination Rate: likely hallucinations / total predictions (entities + name if hallucinated)
        total_pred_units = pred_entities_count + (1 if any(p.category == "patient_info.name" and p.classification == "likely_hallucination" for p in result.unmatched_predictions) else 0)
        result.hallucination_rate = hallucinations / max(1, total_pred_units) if total_pred_units > 0 else 0.0
        
        # Annotation-Gap Rate: annotation gaps / total predictions
        result.annotation_gap_rate = gaps / max(1, total_pred_units) if total_pred_units > 0 else 0.0
        
        # Missing Entity Rate: total False Negatives across categories / total Ground Truth entities
        total_fn = sum(1 for align in result.entity_alignments if align.alignment_status == "FN")
        result.missing_entity_rate = total_fn / max(1, gt_entities_count) if gt_entities_count > 0 else 0.0
        
        return result

    def compute_experimental_score(self, result: DocumentBenchmarkResult) -> float:
        """
        Calculate the configurable, experimental headline score.
        Formula:
        Overall Score = w1 * Schema + w2 * Scalar + w3 * Lenient_F1 + w4 * Exact_F1 + w5 * (1 - Hallucination_Rate)
        """
        # Average entity F1 scores across categories with elements in GT or Pred
        f1_exact_list = []
        f1_lenient_list = []
        for cat, metrics in result.metrics_by_category.items():
            # Only count categories that had ground truth or predictions to avoid division/averaging bias
            f1_exact_list.append(metrics.f1_exact)
            f1_lenient_list.append(metrics.f1_lenient)
            
        avg_f1_exact = sum(f1_exact_list) / len(f1_exact_list) if f1_exact_list else 0.0
        avg_f1_lenient = sum(f1_lenient_list) / len(f1_lenient_list) if f1_lenient_list else 0.0
        
        w = self.weights
        score = (
            w.get("schema_parse", 0.10) * result.schema_parse_success +
            w.get("scalar_field", 0.20) * result.scalar_accuracy_lenient +
            w.get("entity_lenient_f1", 0.45) * avg_f1_lenient +
            w.get("entity_exact_f1", 0.15) * avg_f1_exact +
            w.get("hallucination_penalty", 0.10) * (1.0 - result.hallucination_rate)
        )
        
        result.experimental_overall_score = max(0.0, min(1.0, score))
        return result.experimental_overall_score

    def aggregate_dataset(self, doc_results: List[DocumentBenchmarkResult]) -> Dict[str, Any]:
        """
        Aggregates results over a collection of documents to produce summary dataset metrics.
        """
        if not doc_results:
            return {}
            
        n_docs = len(doc_results)
        
        # Averages of document-level rates
        avg_schema = sum(r.schema_parse_success for r in doc_results) / n_docs
        avg_scalar_exact = sum(r.scalar_accuracy_exact for r in doc_results) / n_docs
        avg_scalar_lenient = sum(r.scalar_accuracy_lenient for r in doc_results) / n_docs
        
        avg_hallucination_rate = sum(r.hallucination_rate for r in doc_results) / n_docs
        avg_missing_entity_rate = sum(r.missing_entity_rate for r in doc_results) / n_docs
        avg_annotation_gap_rate = sum(r.annotation_gap_rate for r in doc_results) / n_docs
        
        avg_headline = sum(r.experimental_overall_score for r in doc_results) / n_docs
        
        # Categorized mismatch sums
        total_hallucinations = sum(r.likely_hallucination_count for r in doc_results)
        total_gaps = sum(r.annotation_gap_candidate_count for r in doc_results)
        total_review_needed = sum(r.manual_review_required_count for r in doc_results)
        
        # Category-wise aggregated F1s
        cat_summaries = {}
        all_categories = set()
        for r in doc_results:
            all_categories.update(r.metrics_by_category.keys())
            
        for cat in all_categories:
            f1s_exact = []
            f1s_lenient = []
            precs_exact = []
            recs_exact = []
            precs_lenient = []
            recs_lenient = []
            
            for r in doc_results:
                if cat in r.metrics_by_category:
                    m = r.metrics_by_category[cat]
                    f1s_exact.append(m.f1_exact)
                    f1s_lenient.append(m.f1_lenient)
                    precs_exact.append(m.precision_exact)
                    recs_exact.append(m.recall_exact)
                    precs_lenient.append(m.precision_lenient)
                    recs_lenient.append(m.recall_lenient)
                    
            cat_summaries[cat] = {
                "precision_exact": sum(precs_exact) / len(precs_exact) if precs_exact else 0.0,
                "recall_exact": sum(recs_exact) / len(recs_exact) if recs_exact else 0.0,
                "f1_exact": sum(f1s_exact) / len(f1s_exact) if f1s_exact else 0.0,
                "precision_lenient": sum(precs_lenient) / len(precs_lenient) if precs_lenient else 0.0,
                "recall_lenient": sum(recs_lenient) / len(recs_lenient) if recs_lenient else 0.0,
                "f1_lenient": sum(f1s_lenient) / len(f1s_lenient) if f1s_lenient else 0.0,
            }
            
        return {
            "total_documents": n_docs,
            "schema_parse_success_rate": avg_schema,
            "scalar_accuracy_exact": avg_scalar_exact,
            "scalar_accuracy_lenient": avg_scalar_lenient,
            "entity_exact_f1_macro": sum(c["f1_exact"] for c in cat_summaries.values()) / len(cat_summaries) if cat_summaries else 0.0,
            "entity_lenient_f1_macro": sum(c["f1_lenient"] for c in cat_summaries.values()) / len(cat_summaries) if cat_summaries else 0.0,
            "hallucination_rate": avg_hallucination_rate,
            "missing_entity_rate": avg_missing_entity_rate,
            "annotation_gap_rate": avg_annotation_gap_rate,
            "experimental_overall_score": avg_headline,
            "counts": {
                "likely_hallucination_total": total_hallucinations,
                "annotation_gap_candidate_total": total_gaps,
                "manual_review_required_total": total_review_needed
            },
            "metrics_by_category": cat_summaries
        }

    def aggregate_by_document_type(self, doc_results: List[DocumentBenchmarkResult]) -> Dict[str, Any]:
        """
        Group document benchmark results by document type and aggregate each group.
        """
        by_type = {}
        for r in doc_results:
            doc_type = r.document_type or "unknown"
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(r)
            
        aggregated_by_type = {}
        for doc_type, results in by_type.items():
            aggregated_by_type[doc_type] = self.aggregate_dataset(results)
            
        return aggregated_by_type

