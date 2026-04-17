"""
Leave-one-out cross-validation (LOO-CV) across segmentation backends.

Compares classical pipeline, nnU-Net, and Cellpose against the ground-truth
gc.tif masks using Dice, IoU, precision, and recall from eval_metrics.py.

Usage:
    # Evaluate classical baseline only (no GPU required):
    python -m ml.evaluate --backends classical --data test_image/

    # Evaluate all backends (requires trained models):
    python -m ml.evaluate --backends classical nnunet cellpose --data test_image/

    # Evaluate nnU-Net only with saved results directory:
    python -m ml.evaluate --backends nnunet --data test_image/
"""

import argparse
import logging
import os
import sys
from typing import Dict, List

import numpy as np

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval_metrics import dice_coefficient, iou, precision_recall
from ml.data_loader import CellSample, load_all_samples, loocv_splits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-backend prediction functions
# ---------------------------------------------------------------------------

def _predict_classical(sample: CellSample, config=None) -> np.ndarray:
    import seg_util as su
    final_gc, _, _ = su.gc_segment(
        sample.raw, sample.nucleus_mask, config=config, backend="classical"
    )
    return final_gc


def _predict_nnunet(sample: CellSample) -> np.ndarray:
    from ml.predict_nnunet import gc_segment_nnunet
    gc_mask, _, _ = gc_segment_nnunet(sample.raw, sample.nucleus_mask)
    return gc_mask


def _predict_cellpose(sample: CellSample, model_path: str = None) -> np.ndarray:
    from ml.predict_cellpose import gc_segment_cellpose
    gc_mask, _ = gc_segment_cellpose(sample.raw, sample.nucleus_mask, model_path=model_path)
    return gc_mask


_PREDICT_FN = {
    "classical": _predict_classical,
    "nnunet": _predict_nnunet,
    "cellpose": _predict_cellpose,
}


# ---------------------------------------------------------------------------
# LOO-CV evaluation
# ---------------------------------------------------------------------------

def evaluate_backend(
    backend: str,
    samples: List[CellSample],
    **kwargs,
) -> Dict[str, float]:
    """
    Run leave-one-out CV for one backend and return aggregate metrics.

    For nnU-Net: uses the model trained on all OTHER cells (leave-one-out).
    For classical: no training step; just runs inference on each held-out cell.

    Returns
    -------
    dict with keys: dice_mean, dice_std, iou_mean, iou_std,
                    precision_mean, recall_mean, n_samples
    """
    predict_fn = _PREDICT_FN.get(backend)
    if predict_fn is None:
        raise ValueError(f"Unknown backend: {backend!r}. Choose from {list(_PREDICT_FN)}")

    dice_scores, iou_scores, precisions, recalls = [], [], [], []

    for i, (train_samples, val_sample) in enumerate(loocv_splits(samples)):
        logger.info("[%s] LOO fold %d/%d — held out: %s",
                    backend, i + 1, len(samples), val_sample.cell_id)

        try:
            pred = predict_fn(val_sample, **kwargs)
        except Exception as exc:
            logger.error("Prediction failed for %s: %s", val_sample.cell_id, exc)
            continue

        gt = val_sample.gc_mask
        dice_scores.append(dice_coefficient(pred, gt))
        iou_scores.append(iou(pred, gt))
        p, r = precision_recall(pred, gt)
        precisions.append(p)
        recalls.append(r)

        logger.info(
            "  Dice=%.3f  IoU=%.3f  Prec=%.3f  Rec=%.3f",
            dice_scores[-1], iou_scores[-1], p, r,
        )

    if not dice_scores:
        logger.warning("No valid predictions for backend %r", backend)
        return {}

    return {
        "dice_mean": float(np.mean(dice_scores)),
        "dice_std": float(np.std(dice_scores)),
        "iou_mean": float(np.mean(iou_scores)),
        "iou_std": float(np.std(iou_scores)),
        "precision_mean": float(np.mean(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "n_samples": len(dice_scores),
    }


def print_results_table(results: Dict[str, Dict]) -> None:
    header = f"{'Backend':<12} {'Dice':>8} {'±':>5} {'IoU':>8} {'±':>5} {'Prec':>8} {'Rec':>8} {'N':>4}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for backend, m in sorted(results.items(), key=lambda x: -x[1].get("dice_mean", 0)):
        if not m:
            print(f"{backend:<12}  (no results)")
            continue
        print(
            f"{backend:<12} "
            f"{m['dice_mean']:>8.3f} "
            f"{m['dice_std']:>5.3f} "
            f"{m['iou_mean']:>8.3f} "
            f"{m['iou_std']:>5.3f} "
            f"{m['precision_mean']:>8.3f} "
            f"{m['recall_mean']:>8.3f} "
            f"{m['n_samples']:>4d}"
        )
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Stage-level breakdown
# ---------------------------------------------------------------------------

def evaluate_by_stage(
    backend: str,
    samples: List[CellSample],
    **kwargs,
) -> Dict[str, Dict]:
    """Return per-stage Dice scores to detect stage-specific failure modes."""
    from collections import defaultdict
    predict_fn = _PREDICT_FN[backend]

    by_stage = defaultdict(list)
    for sample in samples:
        try:
            pred = predict_fn(sample, **kwargs)
            by_stage[sample.stage].append(dice_coefficient(pred, sample.gc_mask))
        except Exception as exc:
            logger.warning("Stage eval failed for %s: %s", sample.cell_id, exc)

    return {
        stage: {"dice_mean": float(np.mean(scores)), "n": len(scores)}
        for stage, scores in sorted(by_stage.items())
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="LOO-CV evaluation across segmentation backends"
    )
    parser.add_argument(
        "--data", default="test_image",
        help="Path to master_folder (default: test_image/)"
    )
    parser.add_argument(
        "--backends", nargs="+", default=["classical"],
        choices=["classical", "nnunet", "cellpose"],
        help="Backends to evaluate (default: classical)"
    )
    parser.add_argument(
        "--cellpose-model", default=None,
        help="Path to fine-tuned Cellpose model (optional)"
    )
    parser.add_argument(
        "--stage-breakdown", action="store_true",
        help="Print per-stage Dice breakdown"
    )
    args = parser.parse_args()

    samples = load_all_samples(args.data)
    if not samples:
        logger.error("No samples loaded from %s", args.data)
        sys.exit(1)

    logger.info("Loaded %d samples", len(samples))

    all_results = {}
    for backend in args.backends:
        logger.info("Evaluating backend: %s", backend)
        kwargs = {}
        if backend == "cellpose" and args.cellpose_model:
            kwargs["model_path"] = args.cellpose_model
        all_results[backend] = evaluate_backend(backend, samples, **kwargs)

    print_results_table(all_results)

    if args.stage_breakdown:
        for backend in args.backends:
            stage_res = evaluate_by_stage(backend, samples)
            print(f"\nStage breakdown — {backend}:")
            for stage, m in stage_res.items():
                print(f"  {stage}: Dice={m['dice_mean']:.3f}  (n={m['n']})")


if __name__ == "__main__":
    main()
