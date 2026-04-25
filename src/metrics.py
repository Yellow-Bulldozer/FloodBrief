"""
FloodBrief - Evaluation metrics.

Computes segmentation metrics (IoU, F1, precision, recall) and
product metrics (bandwidth savings, event retention).
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Pixel-level segmentation metrics
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 2,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        pred: (H, W) predicted class labels.
        target: (H, W) ground truth class labels.
        num_classes: Number of classes.

    Returns:
        (num_classes, num_classes) confusion matrix.
    """
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

    # Flatten
    pred = pred.flatten()
    target = target.flatten()

    # Filter out ignore values (e.g., -1 or 255)
    valid = (target >= 0) & (target < num_classes)
    pred = pred[valid]
    target = target[valid]

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target, pred):
        cm[t, p] += 1

    return cm


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU from confusion matrix."""
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = np.where(union > 0, intersection / union, 0.0)
    return iou


def precision_recall_f1_from_confusion(
    cm: np.ndarray,
    class_idx: int = 1,
) -> Tuple[float, float, float]:
    """Precision, recall, F1 for a specific class from confusion matrix."""
    tp = cm[class_idx, class_idx]
    fp = cm[:, class_idx].sum() - tp
    fn = cm[class_idx, :].sum() - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return float(precision), float(recall), float(f1)


class SegmentationMetrics:
    """
    Accumulator for segmentation metrics across multiple samples.
    """

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.latencies: List[float] = []

    def update(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        latency_ms: Optional[float] = None,
    ):
        """Add a batch of predictions to the accumulator."""
        if pred.ndim == 3:
            # Batch dimension
            for i in range(pred.shape[0]):
                self.confusion_matrix += compute_confusion_matrix(
                    pred[i], target[i], self.num_classes
                )
        else:
            self.confusion_matrix += compute_confusion_matrix(
                pred, target, self.num_classes
            )

        if latency_ms is not None:
            self.latencies.append(latency_ms)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated confusion matrix."""
        cm = self.confusion_matrix

        # Per-class IoU
        per_class_iou = iou_from_confusion(cm)
        miou = float(np.mean(per_class_iou))

        # Flood class metrics (class 1)
        precision, recall, f1 = precision_recall_f1_from_confusion(cm, class_idx=1)

        # Overall accuracy
        total = cm.sum()
        correct = np.diag(cm).sum()
        accuracy = float(correct / total) if total > 0 else 0.0

        results = {
            "mIoU": round(miou, 4),
            "flood_IoU": round(float(per_class_iou[1]), 4),
            "no_flood_IoU": round(float(per_class_iou[0]), 4),
            "precision_flood": round(precision, 4),
            "recall_flood": round(recall, 4),
            "f1_flood": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "confusion_matrix": cm.tolist(),
        }

        # Latency stats
        if self.latencies:
            results["avg_latency_ms"] = round(np.mean(self.latencies), 2)
            results["p95_latency_ms"] = round(np.percentile(self.latencies, 95), 2)
            results["min_latency_ms"] = round(np.min(self.latencies), 2)
            results["max_latency_ms"] = round(np.max(self.latencies), 2)

        return results

    def reset(self):
        """Reset all accumulated metrics."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
        self.latencies = []


# ---------------------------------------------------------------------------
# Baseline metrics
# ---------------------------------------------------------------------------

def majority_class_baseline(targets: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute metrics for the majority-class baseline.
    Predicts "no flood" (class 0) for every pixel.
    """
    metrics = SegmentationMetrics(num_classes=2)
    for target in targets:
        pred = np.zeros_like(target)  # Always predict "no flood"
        metrics.update(pred, target)

    results = metrics.compute()
    results["baseline_name"] = "majority_class"
    results["baseline_description"] = "Predict 'no flood' for every pixel"
    return results


def random_baseline(targets: List[np.ndarray], seed: int = 42) -> Dict[str, float]:
    """
    Compute metrics for a random baseline.
    Randomly assigns flood/no-flood with the dataset's flood prior.
    """
    # Compute flood prior
    all_labels = np.concatenate([t.flatten() for t in targets])
    flood_prior = float(np.mean(all_labels == 1))

    np.random.seed(seed)
    metrics = SegmentationMetrics(num_classes=2)
    for target in targets:
        pred = (np.random.random(target.shape) < flood_prior).astype(np.int64)
        metrics.update(pred, target)

    results = metrics.compute()
    results["baseline_name"] = "random"
    results["baseline_description"] = f"Random with flood prior={flood_prior:.4f}"
    return results


# ---------------------------------------------------------------------------
# Product metrics
# ---------------------------------------------------------------------------

def compute_product_metrics(
    triage_results: list,
    ground_truth_has_flood: List[bool],
) -> Dict[str, float]:
    """
    Compute product-level triage metrics.

    Args:
        triage_results: List of TriageResult objects.
        ground_truth_has_flood: Whether each tile actually contains flood.

    Returns:
        Dict with bandwidth savings, event retention rate, etc.
    """
    total = len(triage_results)
    if total == 0:
        return {}

    # Downlink decisions
    downlinked = sum(1 for r in triage_results if r.downlink_decision == "downlink")
    skipped = total - downlinked

    # Ground truth stats
    gt_flood_positive = sum(1 for v in ground_truth_has_flood if v)
    gt_flood_negative = total - gt_flood_positive

    # True positives: correctly downlinked flood tiles
    tp = sum(
        1 for r, gt in zip(triage_results, ground_truth_has_flood)
        if r.downlink_decision == "downlink" and gt
    )
    # False negatives: skipped tiles that had flood
    fn = sum(
        1 for r, gt in zip(triage_results, ground_truth_has_flood)
        if r.downlink_decision == "skip" and gt
    )
    # False positives: downlinked tiles with no flood
    fp = sum(
        1 for r, gt in zip(triage_results, ground_truth_has_flood)
        if r.downlink_decision == "downlink" and not gt
    )
    # True negatives: correctly skipped non-flood tiles
    tn = sum(
        1 for r, gt in zip(triage_results, ground_truth_has_flood)
        if r.downlink_decision == "skip" and not gt
    )

    # Event retention: what fraction of true flood tiles were downlinked
    event_retention = tp / gt_flood_positive if gt_flood_positive > 0 else 1.0

    # Bandwidth saving: fraction of tiles skipped
    bandwidth_saving = skipped / total if total > 0 else 0.0

    # Downlink precision: of downlinked tiles, how many are truly flood
    downlink_precision = tp / downlinked if downlinked > 0 else 0.0

    return {
        "total_tiles": total,
        "downlinked": downlinked,
        "skipped": skipped,
        "bandwidth_saving_pct": round(100.0 * bandwidth_saving, 1),
        "event_retention_pct": round(100.0 * event_retention, 1),
        "downlink_precision_pct": round(100.0 * downlink_precision, 1),
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "gt_flood_positive": gt_flood_positive,
        "gt_flood_negative": gt_flood_negative,
    }


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

class InferenceTimer:
    """Context manager for measuring inference latency."""

    def __init__(self):
        self.start_time = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
