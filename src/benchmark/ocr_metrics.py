import re
from collections import defaultdict
import difflib
from typing import Dict, Any, List

def edit_distance(seq_a: List[str] | str, seq_b: List[str] | str) -> int:
    """Computes Levenshtein distance between two sequences."""
    prev = list(range(len(seq_b) + 1))
    for i in range(1, len(seq_a) + 1):
        cur = [i] + [0] * len(seq_b)
        for j in range(1, len(seq_b) + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]

def tokens(text: str) -> List[str]:
    """Tokenizes string to alphanumeric terms."""
    return re.findall(r"[A-Za-z0-9]+", text.lower())

def token_metrics(pred: str, gt: str) -> Dict[str, Any]:
    """Computes precision, recall, F1, and numeric recall on token sets."""
    pred_tokens = tokens(pred)
    gt_tokens = tokens(gt)
    pred_counts = defaultdict(int)
    gt_counts = defaultdict(int)
    for token in pred_tokens:
        pred_counts[token] += 1
    for token in gt_tokens:
        gt_counts[token] += 1
    overlap = sum(min(pred_counts[token], gt_counts[token]) for token in set(pred_counts) | set(gt_counts))
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(gt_tokens))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    
    numeric_gt = {token for token in gt_tokens if any(ch.isdigit() for ch in token)}
    numeric_pred = {token for token in pred_tokens if any(ch.isdigit() for ch in token)}
    numeric_recall = len(numeric_gt & numeric_pred) / max(1, len(numeric_gt)) if numeric_gt else ""
    
    return {
        "token_precision": round(precision, 4),
        "token_recall": round(recall, 4),
        "token_f1": round(f1, 4),
        "numeric_token_recall": round(numeric_recall, 4) if numeric_recall != "" else "",
    }

def edit_distance_metrics(pred: str, gt: str) -> Dict[str, Any]:
    """Computes CER, WER, and normalized edit similarity."""
    norm_pred = " ".join(pred.split())
    norm_gt = " ".join(gt.split())
    cer = edit_distance(norm_pred, norm_gt) / max(1, len(norm_gt)) if norm_pred and norm_gt else ""
    wer = edit_distance(norm_pred.split(), norm_gt.split()) / max(1, len(norm_gt.split())) if norm_pred and norm_gt else ""
    similarity = round(difflib.SequenceMatcher(None, norm_pred.lower(), norm_gt.lower()).ratio(), 4) if norm_pred and norm_gt else ""
    
    return {
        "cer": round(cer, 4) if cer != "" else "",
        "wer": round(wer, 4) if wer != "" else "",
        "normalized_edit_similarity": similarity,
    }
