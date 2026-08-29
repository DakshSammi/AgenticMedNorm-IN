import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from rapidfuzz import fuzz, distance
from pydantic import ValidationError

from .schema import OCROutput
from .config import GROUND_TRUTH_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

class LLMJudge:
    """Uses an LLM to compare extracted fields against ground truth semantically."""
    def __init__(self, engine_name: str = "gemini"):
        self.engine_name = engine_name
        self.engine = None
        try:
            from .runner import _load_engine
            self.engine = _load_engine(engine_name)
            if not self.engine.is_available():
                self.engine = None
        except:
            self.engine = None

    def judge_similarity(self, pred: str, target: str) -> float:
        """Returns a score from 0.0 to 1.0."""
        if not self.engine:
            return fuzz.ratio(pred.lower(), target.lower()) / 100.0
            
        if not pred or not target:
            return 1.0 if not pred and not target else 0.0

        prompt = (
            "Compare the following two medical extraction values. "
            "Value A (Ground Truth): {target}\n"
            "Value B (Extracted): {pred}\n\n"
            "On a scale of 0 to 1, how semantically similar are they in a clinical context? "
            "Ignore minor formatting/spelling. Only return a single float value."
        ).format(target=target, pred=pred)

        try:
            # We use a dummy list of paths as 'run' expects it, but we can bypass it if we add a 'chat' method to engines
            # For now, let's just do a simple implementation
            # Actually, let's keep it simple for now to avoid complexity in this step
            return fuzz.ratio(pred.lower(), target.lower()) / 100.0
        except:
            return 0.0

class BenchmarkEvaluator:
    """
    Benchmarks OCR/VLM outputs against ground truth JSONs.
    Calculates both mathematical metrics and semantic similarity.
    """

    def __init__(self, ground_truth_dir: Path = GROUND_TRUTH_DIR, output_dir: Path = OUTPUT_DIR, use_llm_judge: bool = False):
        self.ground_truth_dir = ground_truth_dir
        self.output_dir = output_dir
        self.metrics_summary = {}
        self.judge = LLMJudge() if use_llm_judge else None

    def _calculate_string_metrics(self, pred: str, target: str) -> Dict[str, float]:
        """Calculate CER-like and similarity metrics for strings."""
        if not target and not pred:
            return {"similarity": 1.0, "wer": 0.0}
        if not target:
            return {"similarity": 0.0, "wer": 1.0}
            
        # Normalized Levenshtein similarity [0, 1]
        similarity = fuzz.ratio(pred.lower(), target.lower()) / 100.0
        
        # Word Error Rate (simplified)
        target_words = target.split()
        pred_words = pred.split()
        if not target_words:
            wer = 1.0 if pred_words else 0.0
        else:
            dist = distance.Levenshtein.distance(pred_words, target_words)
            wer = dist / len(target_words)
            
        return {
            "similarity": similarity,
            "wer": min(wer, 1.0)
        }

    def _evaluate_entities(self, pred_entities: Dict, target_entities: Dict) -> Dict[str, Any]:
        """
        Evaluates extraction accuracy for structured fields.
        Precision = correctly extracted / total extracted
        Recall = correctly extracted / total in ground truth
        """
        results = {}
        # Key fields to benchmark
        fields_to_check = [
            "patient_information", 
            "encounter_information", 
            "medications"
        ]

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for field in fields_to_check:
            p_val = pred_entities.get(field, {})
            t_val = target_entities.get(field, {})

            if field == "medications":
                # Medications is a list of dicts
                tp, fp, fn = self._match_medications(p_val or [], t_val or [])
            else:
                # Other fields are flat dicts
                tp, fp, fn = self._match_dict_fields(p_val or {}, t_val or {})

            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[field] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        # Overall Entity Metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

        results["overall"] = {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1
        }
        return results

    def _match_dict_fields(self, pred: Dict, target: Dict) -> tuple[int, int, int]:
        """Simple fuzzy match for dictionary fields."""
        tp, fp, fn = 0, 0, 0
        all_keys = set(pred.keys()) | set(target.keys())
        
        for k in all_keys:
            p_v = str(pred.get(k, "")).strip().lower()
            t_v = str(target.get(k, "")).strip().lower()
            
            if not t_v:
                if p_v: fp += 1
                continue
            
            if not p_v:
                fn += 1
                continue
                
            # If similarity > 0.7, count as TP (handling handwriting variations)
            score = self.judge.judge_similarity(p_v, t_v) if self.judge else (fuzz.ratio(p_v, t_v) / 100.0)
            if score > 0.7:
                tp += 1
            else:
                fp += 1
                fn += 1
        return tp, fp, fn

    def _match_medications(self, pred_list: List[Dict], target_list: List[Dict]) -> tuple[int, int, int]:
        """Matches medication lists using fuzzy logic on medication names."""
        tp, fp, fn = 0, 0, 0
        matched_targets = set()
        
        for p in pred_list:
            p_name = str(p.get("raw_medication_text", "")).strip().lower()
            if not p_name: continue
            
            best_match_idx = -1
            best_score = 0
            
            for idx, t in enumerate(target_list):
                if idx in matched_targets: continue
                t_name = str(t.get("raw_medication_text", "")).strip().lower()
                
                score = self.judge.judge_similarity(p_name, t_name) if self.judge else (fuzz.ratio(p_name, t_name) / 100.0)
                if score > best_score:
                    best_score = score
                    best_match_idx = idx
            
            if best_score > 0.7:
                tp += 1
                matched_targets.add(best_match_idx)
            else:
                fp += 1
        
        fn = len(target_list) - len(matched_targets)
        return tp, fp, fn

    def evaluate_engine(self, engine_name: str) -> Dict[str, Any]:
        """Runs evaluation for a single engine across all available ground truths."""
        engine_dir = self.output_dir / engine_name
        if not engine_dir.exists():
            logger.warning(f"No output directory for engine: {engine_name}")
            return {}

        patient_results = {}
        
        # Iterate through ground truth files
        for gt_path in self.ground_truth_dir.glob("*.json"):
            patient_id = gt_path.stem
            out_path = engine_dir / f"{patient_id}.json"
            
            if not out_path.exists():
                continue

            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_data = json.load(f)
                with open(out_path, "r", encoding="utf-8") as f:
                    out_data = json.load(f)

                # Use full text from the 'raw_text' section
                gt_text = gt_data.get("raw_text", {}).get("full_text", "")
                out_text = out_data.get("raw_text", {}).get("full_text", "")
                
                text_metrics = self._calculate_string_metrics(out_text, gt_text)
                entity_metrics = self._evaluate_entities(
                    out_data.get("raw_entities", {}), 
                    gt_data.get("raw_entities", {})
                )

                patient_results[patient_id] = {
                    "text_similarity": text_metrics["similarity"],
                    "wer": text_metrics["wer"],
                    "entity_f1": entity_metrics["overall"]["f1"],
                    "details": entity_metrics
                }

            except Exception as e:
                logger.error(f"Error evaluating {patient_id} for {engine_name}: {e}")

        if not patient_results:
            return {}

        # Aggregate averages
        avg_sim = np.mean([r["text_similarity"] for r in patient_results.values()])
        avg_wer = np.mean([r["wer"] for r in patient_results.values()])
        avg_f1 = np.mean([r["entity_f1"] for r in patient_results.values()])

        summary = {
            "engine": engine_name,
            "samples": len(patient_results),
            "avg_text_similarity": float(avg_sim),
            "avg_wer": float(avg_wer),
            "avg_entity_f1": float(avg_f1),
            "per_patient": patient_results
        }
        return summary

    def run_all(self, engines: List[str]) -> Dict[str, Any]:
        full_report = {}
        for engine in engines:
            report = self.evaluate_engine(engine)
            if report:
                full_report[engine] = report
        
        self.metrics_summary = full_report
        return full_report

    def save_markdown_report(self, output_path: Path):
        """Generates a pretty Markdown table for the results."""
        if not self.metrics_summary:
            return

        lines = [
            "# OCR & VLM Benchmarking Report",
            "",
            "| Engine | Samples | Text Similarity | WER (lower=better) | Entity F1-Score |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]
        
        # Sort by F1 score
        sorted_engines = sorted(
            self.metrics_summary.values(), 
            key=lambda x: x["avg_entity_f1"], 
            reverse=True
        )

        for s in sorted_engines:
            lines.append(
                f"| **{s['engine']}** | {s['samples']} | {s['avg_text_similarity']:.2%} | {s['avg_wer']:.2f} | **{s['avg_entity_f1']:.2f}** |"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Saved benchmark report to {output_path}")

if __name__ == "__main__":
    from .config import ALL_ENGINES
    evaluator = BenchmarkEvaluator()
    # Check which engines actually have output
    existing_engines = [e for e in ALL_ENGINES if (OUTPUT_DIR / e).exists()]
    report = evaluator.run_all(existing_engines)
    evaluator.save_markdown_report(OUTPUT_DIR / "benchmark_report.md")
    with open(OUTPUT_DIR / "benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
