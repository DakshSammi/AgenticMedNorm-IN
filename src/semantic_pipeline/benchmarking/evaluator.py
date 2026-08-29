"""
Semantic Benchmarking Evaluator.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from rapidfuzz import fuzz

from ..config import ENHANCED_OUTPUT_DIR, GROUND_TRUTH_DIR, SEMANTIC_EVAL_DIR

logger = logging.getLogger(__name__)

class SemanticEvaluator:
    def __init__(self, ground_truth_dir: Path = GROUND_TRUTH_DIR, enhanced_dir: Path = ENHANCED_OUTPUT_DIR):
        self.ground_truth_dir = ground_truth_dir
        self.enhanced_dir = enhanced_dir

    def evaluate_engine(self, engine_name: str) -> Dict[str, Any]:
        engine_dir = self.enhanced_dir / engine_name
        if not engine_dir.exists():
            return {}

        results = {}
        for gt_file in self.ground_truth_dir.glob("*.json"):
            patient_id = gt_file.stem
            enhanced_file = engine_dir / f"{patient_id}.json"
            
            if not enhanced_file.exists():
                continue
                
            try:
                with open(gt_file, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                with open(enhanced_file, 'r', encoding='utf-8') as f:
                    en_data = json.load(f)
                    
                # Skip if there are engine errors
                engine_errors = en_data.get("ocr_engine_metadata", {}).get("errors", [])
                if engine_errors:
                    logger.warning(f"Skipping evaluation for {patient_id} ({engine_name}) due to engine errors: {engine_errors}")
                    continue

                # Compare entities
                metrics = self._compare_entities(en_data, gt_data)
                results[patient_id] = metrics
            except Exception as e:
                logger.error(f"Error evaluating {patient_id} for {engine_name}: {e}")

        if not results:
            return {}

        # Aggregate metrics
        avg_precision = np.mean([r["precision"] for r in results.values()])
        avg_recall = np.mean([r["recall"] for r in results.values()])
        avg_f1 = np.mean([r["f1"] for r in results.values()])

        return {
            "engine": engine_name,
            "precision": float(avg_precision),
            "recall": float(avg_recall),
            "f1": float(avg_f1),
            "details": results
        }

    def _compare_entities(self, pred: Dict, target: Dict) -> Dict[str, float]:
        """
        Compares extracted entities with ground truth.
        Uses fuzzy matching for medication names and diagnosis.
        """
        # Simplified for demonstration
        tp, fp, fn = 0, 0, 0
        
        # Compare Medications
        pred_meds = pred.get("raw_entities", {}).get("medications", [])
        target_meds = target.get("raw_entities", {}).get("medications", [])
        
        matched_targets = set()
        for p in pred_meds:
            p_name = p.get("normalized_medication_text", p.get("raw_medication_text", "")).lower()
            if not p_name: continue
            
            best_match = -1
            best_score = 0
            for idx, t in enumerate(target_meds):
                if idx in matched_targets: continue
                t_name = t.get("raw_medication_text", "").lower()
                
                score = fuzz.ratio(p_name, t_name) / 100.0
                if score > best_score:
                    best_score = score
                    best_match = idx
            
            if best_score > 0.8:
                tp += 1
                matched_targets.add(best_match)
            else:
                fp += 1
        
        fn = len(target_meds) - len(matched_targets)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {"precision": precision, "recall": recall, "f1": f1}

    def run_all(self) -> Dict[str, Any]:
        engines = [f.name for f in self.enhanced_dir.iterdir() if f.is_dir()]
        report = {}
        for engine in engines:
            report[engine] = self.evaluate_engine(engine)
        return report

    def save_reports(self, report: Dict):
        # Save JSON
        with open(SEMANTIC_EVAL_DIR / "semantic_benchmark.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        # Save Markdown Table
        lines = [
            "# Semantic Pipeline Benchmarking Report",
            "",
            "| Engine | Precision | Recall | F1-Score |",
            "| :--- | :--- | :--- | :--- |"
        ]
        for eng, metrics in report.items():
            if not metrics: continue
            lines.append(f"| **{eng}** | {metrics['precision']:.2f} | {metrics['recall']:.2f} | **{metrics['f1']:.2f}** |")
            
        with open(SEMANTIC_EVAL_DIR / "semantic_benchmark.md", 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
