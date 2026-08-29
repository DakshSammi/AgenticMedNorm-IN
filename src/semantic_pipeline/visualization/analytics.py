"""
Visual analytics for semantic pipeline results.
"""

import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ..config import SEMANTIC_EVAL_DIR

logger = logging.getLogger(__name__)

def generate_visual_reports():
    benchmark_file = SEMANTIC_EVAL_DIR / "semantic_benchmark.json"
    if not benchmark_file.exists():
        logger.warning(f"Benchmark file not found: {benchmark_file}")
        return

    with open(benchmark_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to DataFrame
    rows = []
    for eng, metrics in data.items():
        if metrics:
            rows.append({
                "Engine": eng,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"]
            })
    
    if not rows:
        return
        
    df = pd.DataFrame(rows)
    df.set_index("Engine", inplace=True)

    # 1. Bar Chart for F1 Score
    plt.figure(figsize=(12, 6))
    df["F1"].sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Semantic Entity Extraction F1-Score by Engine")
    plt.ylabel("F1 Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(SEMANTIC_EVAL_DIR / "f1_score_comparison.png")
    plt.close()

    # 2. Precision-Recall Scatter
    plt.figure(figsize=(8, 8))
    for idx, row in df.iterrows():
        plt.scatter(row["Recall"], row["Precision"], label=idx, s=100)
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision vs Recall by Engine")
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(SEMANTIC_EVAL_DIR / "precision_recall_scatter.png")
    plt.close()

    logger.info(f"Visual reports saved to {SEMANTIC_EVAL_DIR}")

if __name__ == "__main__":
    generate_visual_reports()
