"""Zero-dependency evaluation metrics for binary segmentation masks.

Usage:
    from eval_metrics import dice_coefficient, iou, precision_recall

All functions accept numpy arrays of any shape and treat positive (>0) pixels
as foreground.
"""

from typing import Tuple

import numpy as np


def _foreground(pred: np.ndarray, gt: np.ndarray):
    return pred > 0, gt > 0


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sørensen–Dice coefficient: 2·|pred ∩ gt| / (|pred| + |gt|).

    Returns 0.0 when both masks are empty.
    """
    pred_pos, gt_pos = _foreground(pred, gt)
    intersection = np.logical_and(pred_pos, gt_pos).sum()
    denom = pred_pos.sum() + gt_pos.sum()
    return float(2 * intersection / denom) if denom > 0 else 0.0


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union (Jaccard index).

    Returns 0.0 when both masks are empty.
    """
    pred_pos, gt_pos = _foreground(pred, gt)
    intersection = np.logical_and(pred_pos, gt_pos).sum()
    union = np.logical_or(pred_pos, gt_pos).sum()
    return float(intersection / union) if union > 0 else 0.0


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Pixel-level precision and recall.

    Returns
    -------
    precision : float
        TP / (TP + FP); 0.0 if pred is empty.
    recall : float
        TP / (TP + FN); 0.0 if gt is empty.
    """
    pred_pos, gt_pos = _foreground(pred, gt)
    tp = np.logical_and(pred_pos, gt_pos).sum()
    fp = np.logical_and(pred_pos, ~gt_pos).sum()
    fn = np.logical_and(~pred_pos, gt_pos).sum()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return precision, recall


def f1_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Harmonic mean of precision and recall (equivalent to Dice)."""
    p, r = precision_recall(pred, gt)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
