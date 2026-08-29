import csv
import json
from pathlib import Path
from typing import List, Dict, Any
from src.schemas.benchmark import DocumentBenchmarkResult

class ReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, doc_results: List[DocumentBenchmarkResult], summary_metrics: Dict[str, Any]):
        self._write_per_document_scores(doc_results)
        self._write_per_field_scores(doc_results)
        self._write_entity_alignment_details(doc_results)
        self._write_error_analysis(doc_results)
        self._write_manual_review_queue(doc_results)
        self._write_summary_metrics(summary_metrics)
        self._write_qualitative_examples_html(doc_results)

    def _write_per_document_scores(self, doc_results: List[DocumentBenchmarkResult]):
        file_path = self.output_dir / "per_document_scores.csv"
        headers = [
            "document_id", "document_type", "schema_parse_success", "scalar_accuracy_exact", "scalar_accuracy_lenient",
            "entity_exact_f1_macro", "entity_lenient_f1_macro", "hallucination_rate", "missing_entity_rate",
            "annotation_gap_rate", "experimental_overall_score", "model_name", "backend_name", "latency_ms"
        ]
        
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for r in doc_results:
                f1_exact_list = [m.f1_exact for m in r.metrics_by_category.values()]
                f1_lenient_list = [m.f1_lenient for m in r.metrics_by_category.values()]
                
                avg_f1_exact = sum(f1_exact_list) / len(f1_exact_list) if f1_exact_list else 0.0
                avg_f1_lenient = sum(f1_lenient_list) / len(f1_lenient_list) if f1_lenient_list else 0.0
                
                writer.writerow([
                    r.document_id,
                    r.document_type or "unknown",
                    r.schema_parse_success,
                    f"{r.scalar_accuracy_exact:.4f}",
                    f"{r.scalar_accuracy_lenient:.4f}",
                    f"{avg_f1_exact:.4f}",
                    f"{avg_f1_lenient:.4f}",
                    f"{r.hallucination_rate:.4f}",
                    f"{r.missing_entity_rate:.4f}",
                    f"{r.annotation_gap_rate:.4f}",
                    f"{r.experimental_overall_score:.4f}",
                    r.model_name or "unknown",
                    r.backend_name or "unknown",
                    f"{r.latency_ms:.1f}"
                ])

    def _write_per_field_scores(self, doc_results: List[DocumentBenchmarkResult]):
        file_path = self.output_dir / "per_field_scores.csv"
        headers = ["document_id", "field_name", "gt_value", "pred_value", "exact_match", "lenient_match", "similarity_score"]
        
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for r in doc_results:
                for s in r.scalars:
                    writer.writerow([
                        r.document_id,
                        s.field_name,
                        s.gt_value or "",
                        s.pred_value or "",
                        1 if s.exact_match else 0,
                        1 if s.lenient_match else 0,
                        f"{s.similarity_score:.1f}"
                    ])

    def _write_entity_alignment_details(self, doc_results: List[DocumentBenchmarkResult]):
        file_path = self.output_dir / "entity_alignment_details.csv"
        headers = ["document_id", "category", "gt_raw_text", "pred_raw_text", "exact_match", "lenient_match", "similarity_score", "alignment_status"]
        
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for r in doc_results:
                for e in r.entity_alignments:
                    writer.writerow([
                        r.document_id,
                        e.category,
                        e.gt_raw_text or "",
                        e.pred_raw_text or "",
                        1 if e.exact_match else 0,
                        1 if e.lenient_match else 0,
                        f"{e.similarity_score:.1f}",
                        e.alignment_status
                    ])

    def _write_error_analysis(self, doc_results: List[DocumentBenchmarkResult]):
        file_path = self.output_dir / "error_analysis.csv"
        headers = ["document_id", "category", "pred_text", "confidence", "evidence_text", "classification", "rationale", "matched_snippet"]
        
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for r in doc_results:
                for u in r.unmatched_predictions:
                    writer.writerow([
                        r.document_id,
                        u.category,
                        u.pred_text,
                        f"{u.confidence:.2f}",
                        u.evidence_text or "",
                        u.classification,
                        u.rationale,
                        u.matched_snippet or ""
                    ])

    def _write_manual_review_queue(self, doc_results: List[DocumentBenchmarkResult]):
        file_path = self.output_dir / "manual_review_queue.csv"
        headers = ["document_id", "item_type", "field_or_category", "gt_text", "pred_text", "classification", "reason"]
        
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for r in doc_results:
                # Add TP_LENIENT entity alignments (e.g. Bioflu vs Bioflu e/d) to review queue
                for e in r.entity_alignments:
                    if e.alignment_status == "TP_LENIENT":
                        writer.writerow([
                            r.document_id,
                            "entity_partial_match",
                            e.category,
                            e.gt_raw_text or "",
                            e.pred_raw_text or "",
                            "review_partial_match",
                            f"Partial match with similarity {e.similarity_score:.1f}%."
                        ])
                
                # Add unmatched predictions classified as manual_review_required
                for u in r.unmatched_predictions:
                    if u.classification == "manual_review_required":
                        writer.writerow([
                            r.document_id,
                            "unmatched_prediction",
                            u.category,
                            "",
                            u.pred_text,
                            u.classification,
                            u.rationale
                        ])

    def _write_summary_metrics(self, summary_metrics: Dict[str, Any]):
        file_path = self.output_dir / "summary_metrics.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=4)

    def _write_qualitative_examples_html(self, doc_results: List[DocumentBenchmarkResult]):
        file_path = self.output_dir / "qualitative_examples.html"
        
        html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Pipeline Qualitative Evaluation Diff Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Outfit', sans-serif;
            background-color: #0b0f19;
            color: #e2e8f0;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            font-size: 2.5rem;
            color: #f8fafc;
            border-bottom: 2px solid #1e293b;
            padding-bottom: 10px;
            margin-bottom: 30px;
            font-weight: 700;
            text-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .card {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 35px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #374151;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        .doc-id {
            font-size: 1.5rem;
            font-weight: 600;
            color: #60a5fa;
        }
        .doc-score {
            font-size: 1.1rem;
            background-color: #1e3a8a;
            color: #93c5fd;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 600;
        }
        .diff-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }
        .diff-col {
            background-color: #1f2937;
            border-radius: 8px;
            padding: 18px;
            border: 1px solid #374151;
        }
        .diff-col h3 {
            margin-top: 0;
            margin-bottom: 15px;
            color: #9ca3af;
            font-size: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .diff-item {
            padding: 8px 12px;
            margin-bottom: 8px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .status-tp-exact {
            background-color: rgba(16, 185, 129, 0.15);
            border-left: 4px solid #10b981;
            color: #34d399;
        }
        .status-tp-lenient {
            background-color: rgba(245, 158, 11, 0.15);
            border-left: 4px solid #f59e0b;
            color: #fbbf24;
        }
        .status-fp {
            background-color: rgba(239, 68, 68, 0.15);
            border-left: 4px solid #ef4444;
            color: #f87171;
        }
        .status-fn {
            background-color: rgba(59, 130, 246, 0.15);
            border-left: 4px solid #3b82f6;
            color: #60a5fa;
        }
        .badge {
            font-size: 0.75rem;
            padding: 2px 6px;
            border-radius: 4px;
            text-transform: uppercase;
            font-weight: 700;
        }
        .badge-tp-exact { background-color: #10b981; color: #0b0f19; }
        .badge-tp-lenient { background-color: #f59e0b; color: #0b0f19; }
        .badge-fp { background-color: #ef4444; color: #ffffff; }
        .badge-fn { background-color: #3b82f6; color: #ffffff; }
        .meta-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        .meta-table th, .meta-table td {
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #374151;
        }
        .meta-table th {
            color: #9ca3af;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Pipeline Qualitative Evaluation Diff Report</h1>
        <p style="color: #9ca3af; margin-bottom: 40px;">Interactive side-by-side diff highlighting exactly how predicted variables align with ground truth annotations.</p>
"""
        
        for r in doc_results:
            html_content += f"""
        <div class="card">
            <div class="card-header">
                <span class="doc-id">Document ID: {r.document_id}</span>
                <span class="doc-score">Headline Score: {r.experimental_overall_score*100:.1f}%</span>
            </div>
            
            <div class="diff-grid" style="margin-bottom: 25px;">
                <div class="diff-col">
                    <h3>Scalar Fields (Patient & Encounter Details)</h3>
                    <table class="meta-table">
                        <thead>
                            <tr>
                                <th>Field Name</th>
                                <th>Ground Truth</th>
                                <th>Prediction</th>
                                <th>Match Status</th>
                            </tr>
                        </thead>
                        <tbody>
"""
            for s in r.scalars:
                status_class = "status-tp-exact" if s.exact_match else ("status-tp-lenient" if s.lenient_match else "status-fp")
                badge_text = "EXACT" if s.exact_match else ("LENIENT" if s.lenient_match else "MISMATCH")
                badge_class = "badge-tp-exact" if s.exact_match else ("badge-tp-lenient" if s.lenient_match else "badge-fp")
                
                html_content += f"""
                            <tr class="{status_class}">
                                <td>{s.field_name}</td>
                                <td>{s.gt_value or "<em>null</em>"}</td>
                                <td>{s.pred_value or "<em>null</em>"}</td>
                                <td><span class="badge {badge_class}">{badge_text}</span></td>
                            </tr>
"""
            html_content += """
                        </tbody>
                    </table>
                </div>
                
                <div class="diff-col">
                    <h3>Entity Mismatches & Error Analysis</h3>
                    <table class="meta-table">
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th>Predicted Value</th>
                                <th>Error Classification</th>
                                <th>Fuzzy Rationale</th>
                            </tr>
                        </thead>
                        <tbody>
"""
            if not r.unmatched_predictions:
                html_content += """
                            <tr>
                                <td colspan="4" style="text-align: center; color: #10b981;">No extraction errors or hallucinations detected!</td>
                            </tr>
"""
            for u in r.unmatched_predictions:
                badge_class = "badge-fp" if u.classification == "likely_hallucination" else ("badge-tp-lenient" if u.classification == "annotation_gap_candidate" else "badge-fn")
                html_content += f"""
                            <tr>
                                <td><code>{u.category}</code></td>
                                <td style="color: #f87171;">{u.pred_text}</td>
                                <td><span class="badge {badge_class}">{u.classification}</span></td>
                                <td style="font-size: 0.85rem; color: #9ca3af;">{u.rationale}</td>
                            </tr>
"""
            html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="diff-grid">
                <div class="diff-col" style="grid-column: span 2;">
                    <h3>All Entity Alignments (Bipartite Hungarian Mapping)</h3>
"""
            for e in r.entity_alignments:
                status_class = "status-tp-exact" if e.alignment_status == "TP_EXACT" else (
                    "status-tp-lenient" if e.alignment_status == "TP_LENIENT" else (
                        "status-fp" if e.alignment_status == "FP" else "status-fn"
                    )
                )
                badge_class = "badge-tp-exact" if e.alignment_status == "TP_EXACT" else (
                    "badge-tp-lenient" if e.alignment_status == "TP_LENIENT" else (
                        "badge-fp" if e.alignment_status == "FP" else "badge-fn"
                    )
                )
                
                gt_disp = e.gt_raw_text or "<em>None (Prediction added this)</em>"
                pred_disp = e.pred_raw_text or "<em>None (Prediction missed this)</em>"
                
                html_content += f"""
                    <div class="diff-item {status_class}">
                        <div style="flex: 1;">
                            <span class="badge {badge_class}" style="margin-right: 15px;">{e.alignment_status}</span>
                            <strong style="color: #e2e8f0; font-size: 0.8rem; text-transform: uppercase;">{e.category}</strong>
                            <div style="display: flex; gap: 40px; margin-top: 5px;">
                                <div><span style="color: #9ca3af; font-size: 0.8rem;">GT:</span> {gt_disp}</div>
                                <div><span style="color: #9ca3af; font-size: 0.8rem;">Pred:</span> {pred_disp}</div>
                            </div>
                        </div>
                        <div style="font-weight: bold; font-size: 1.1rem; color: #e2e8f0;">
                            {e.similarity_score:.0f}%
                        </div>
                    </div>
"""
            html_content += """
                </div>
            </div>
        </div>
"""
            
        html_content += """
    </div>
</body>
</html>
"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
