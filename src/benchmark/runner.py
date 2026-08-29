import pandas as pd
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any

from src.schemas.benchmark import DocumentBenchmarkResult, CategoryMetrics
from src.adapters.gt_adapter import GTAdapter
from src.adapters.legacy_prediction_adapter import LegacyPredictionAdapter
from src.benchmark.scalar_match import ScalarMatcher
from src.benchmark.entity_match import EntityMatcher
from src.benchmark.hallucination import HallucinationDetector
from src.benchmark.aggregation import MetricAggregator
from src.benchmark.reports import ReportGenerator

class BenchmarkRunner:
    def __init__(self, manifest_path: Path, project_root: Path, config_path: Path = None):
        self.manifest_path = manifest_path
        self.project_root = project_root
        
        # Load config
        self.config = {}
        if config_path and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                
        # Get threshold values
        thresh = self.config.get("thresholds", {})
        self.exact_thresh = thresh.get("exact_match_ratio", 95.0)
        self.lenient_thresh = thresh.get("lenient_match_ratio", 80.0)
        self.review_thresh = thresh.get("review_required_ratio", 65.0)
        self.hallucination_thresh = thresh.get("hallucination_detection_ratio", 60.0)
        self.gap_thresh = thresh.get("annotation_gap_detection_ratio", 80.0)
        
        # Matchers
        self.scalar_matcher = ScalarMatcher(lenient_threshold=self.lenient_thresh)
        med_weights = self.config.get("medication_component_weights")
        self.entity_matcher = EntityMatcher(
            exact_threshold=self.exact_thresh,
            lenient_threshold=self.lenient_thresh,
            review_threshold=self.review_thresh,
            medication_weights=med_weights
        )
        self.hallucination_detector = HallucinationDetector(
            hallucination_threshold=self.hallucination_thresh,
            gap_threshold=self.gap_thresh
        )
        self.aggregator = MetricAggregator(config=self.config)

    def run(self, output_dir: Path) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
            
        df = pd.read_csv(self.manifest_path)
        doc_results: List[DocumentBenchmarkResult] = []
        
        for _, row in df.iterrows():
            doc_id = str(row["document_id"])
            gt_rel = Path(row["gt_path"])
            pred_rel = Path(row["prediction_path"])
            
            gt_abs = self.project_root / gt_rel
            pred_abs = self.project_root / pred_rel
            
            if not gt_abs.exists():
                print(f"Warning: Ground truth file {gt_abs} does not exist. Skipping document {doc_id}.")
                continue
            if not pred_abs.exists():
                print(f"Warning: Prediction file {pred_abs} does not exist. Skipping document {doc_id}.")
                continue
                
            # 1. Parse documents
            schema_success = 1
            try:
                gt_doc = GTAdapter.from_file(gt_abs)
            except Exception as e:
                print(f"Error parsing ground truth {gt_abs}: {e}")
                continue
                
            try:
                # First check if prediction already follows CanonicalRawDoc, otherwise parse using legacy adapter
                with open(pred_abs, "r", encoding="utf-8") as f:
                    pred_data = json.load(f)
                
                if "schema_version" in pred_data and pred_data["schema_version"] == "raw_rx_v2":
                    from src.schemas.raw_extraction import CanonicalRawDoc
                    pred_doc = CanonicalRawDoc(**pred_data)
                else:
                    pred_doc = LegacyPredictionAdapter.from_dict(pred_data, doc_id)
            except Exception as e:
                print(f"Warning: Failed to parse prediction schema perfectly for {doc_id}: {e}")
                schema_success = 0
                # Initialize empty document to continue scoring with failure marker
                from src.schemas.raw_extraction import CanonicalRawDoc
                pred_doc = CanonicalRawDoc(document_id=doc_id)
                
            # 2. Evaluate scalars
            scalar_matches = self.scalar_matcher.match_docs(gt_doc, pred_doc)
            total_scalars = len(scalar_matches)
            scalars_exact = sum(1 for m in scalar_matches if m.exact_match)
            scalars_lenient = sum(1 for m in scalar_matches if m.lenient_match)
            
            accuracy_exact = scalars_exact / total_scalars if total_scalars > 0 else 1.0
            accuracy_lenient = scalars_lenient / total_scalars if total_scalars > 0 else 1.0
            
            # 3. Align entities across lists
            categories = [
                ("complaints_or_diagnosis", "complaints_or_diagnosis"),
                ("observations", "observations"),
                ("medications", "medications"),
                ("procedures", "procedures"),
                ("advice", "advice"),
                ("allergy_mentions", "allergy_mentions"),
                ("other_notes", "other_notes"),
                ("lab_observations", "lab_observations")
            ]
            
            all_alignments = []
            metrics_by_cat = {}
            total_gt_entities = 0
            total_pred_entities = 0
            
            for cat_key, cat_name in categories:
                gt_list = getattr(gt_doc, cat_key, [])
                pred_list = getattr(pred_doc, cat_key, [])
                
                total_gt_entities += len(gt_list)
                total_pred_entities += len(pred_list)
                
                alignments = self.entity_matcher.align_entities(gt_list, pred_list, cat_name)
                all_alignments.extend(alignments)
                
                # Compute specific category metrics
                if gt_list or pred_list:
                    metrics_by_cat[cat_name] = self.entity_matcher.compute_category_metrics(alignments)
                else:
                    metrics_by_cat[cat_name] = CategoryMetrics()
                    
            # 4. Detect hallucinations and gaps
            unmatched_records = self.hallucination_detector.detect_hallucinations(
                gt_doc, pred_doc, all_alignments
            )
            
            hallucination_count = sum(1 for u in unmatched_records if u.classification == "likely_hallucination")
            gap_count = sum(1 for u in unmatched_records if u.classification == "annotation_gap_candidate")
            review_count = sum(1 for u in unmatched_records if u.classification == "manual_review_required")
            
            # Add any lenient matched items that might trigger manual review
            lenient_review_count = sum(1 for a in all_alignments if a.alignment_status == "TP_LENIENT")
            review_count += lenient_review_count
            
            # 5. Populate Result Object
            res = DocumentBenchmarkResult(
                document_id=doc_id,
                document_type=gt_doc.metadata.document_type if (gt_doc.metadata and gt_doc.metadata.document_type) else "unknown",
                schema_parse_success=schema_success,
                scalar_accuracy_exact=accuracy_exact,
                scalar_accuracy_lenient=accuracy_lenient,
                metrics_by_category=metrics_by_cat,
                likely_hallucination_count=hallucination_count,
                annotation_gap_candidate_count=gap_count,
                manual_review_required_count=review_count,
                scalars=scalar_matches,
                entity_alignments=all_alignments,
                unmatched_predictions=unmatched_records,
                model_name=pred_doc.metadata.model_name if pred_doc.metadata else "unknown",
                backend_name=pred_doc.metadata.backend_name if pred_doc.metadata else "unknown",
                latency_ms=pred_doc.metadata.processing_time_ms if pred_doc.metadata else 0.0
            )
            
            # 6. Calculate rates & overall weighted score
            res = self.aggregator.calculate_rates(res, total_gt_entities, total_pred_entities)
            self.aggregator.compute_experimental_score(res)
            
            doc_results.append(res)
            
        # 7. Aggregate dataset results
        summary_metrics = self.aggregator.aggregate_dataset(doc_results)
        
        # Split by document type
        by_type_metrics = self.aggregator.aggregate_by_document_type(doc_results)
        summary_metrics["by_document_type"] = by_type_metrics
        
        # 8. Generate 7 separate report files
        report_gen = ReportGenerator(output_dir)
        report_gen.generate_all(doc_results, summary_metrics)
        
        return summary_metrics
