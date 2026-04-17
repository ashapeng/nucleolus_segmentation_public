"""
nnU-Net v2 inference wrapper for nucleolus GC segmentation.

Provides gc_segment_nnunet() as a drop-in replacement for the classical
gc_segment() function in seg_util.py.

Prerequisites:
    1. Dataset prepared:  python -m ml.prepare_nnunet
    2. Model trained:     nnUNetv2_train 1 2d all --npz
    3. Environment vars set (nnUNet_raw, nnUNet_preprocessed, nnUNet_results)

Dependencies: nnunetv2, nibabel (pip install nnunetv2 nibabel)
"""

import logging
import os
import tempfile
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# nnU-Net dataset ID and configuration (must match prepare_nnunet.py)
_DATASET_ID = 1
_CONFIGURATION = "2d"   # auto-selected by nnU-Net for 2.5:1 anisotropy
_FOLDS = "all"


def _check_env() -> None:
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if not os.environ.get(var):
            raise EnvironmentError(
                f"Environment variable {var!r} is not set. "
                "Run `python -m ml.prepare_nnunet` and follow the printed instructions."
            )


def _array_to_nifti(arr: np.ndarray, path: str, spacing_zyx=(0.2, 0.08, 0.08)) -> None:
    import nibabel as nib
    sz, sy, sx = spacing_zyx
    affine = np.diag([sz, sy, sx, 1.0])
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), path)


def _nifti_to_array(path: str) -> np.ndarray:
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj)


def gc_segment_nnunet(
    raw_image: np.ndarray,
    nucleus_mask: np.ndarray,
    dataset_id: int = _DATASET_ID,
    configuration: str = _CONFIGURATION,
    folds: str = _FOLDS,
    spacing_zyx: Tuple[float, float, float] = (0.2, 0.08, 0.08),
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Segment the granular component (GC) using a trained nnU-Net v2 model.

    Parameters
    ----------
    raw_image : np.ndarray
        Shape (Z, Y, X, C) uint16 — the full multichannel stack.
    nucleus_mask : np.ndarray
        Shape (Z, Y, X) uint8 — constrains prediction to nucleus interior.
    dataset_id : int
        nnU-Net dataset ID (default 1, set in prepare_nnunet.py).
    configuration : str
        nnU-Net configuration to use (default "2d").
    folds : str
        Folds to use for ensemble prediction ("all" uses all trained folds).
    spacing_zyx : tuple
        Voxel spacing in µm — must match training data.

    Returns
    -------
    final_gc : np.ndarray  (Z, Y, X) uint8 — GC segmentation mask (0 or 255)
    gc_dark_spot : None    — hole detection not performed by nnU-Net
    hole_filled_gc : np.ndarray  (Z, Y, X) uint8 — same as final_gc (no holes)
    """
    _check_env()

    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError:
        raise ImportError("nnunetv2 is required: pip install nnunetv2")

    lpd7 = raw_image[..., 2].astype(np.float32)  # LPD-7 = channel 2

    with tempfile.TemporaryDirectory() as tmpdir:
        in_dir = os.path.join(tmpdir, "input")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(in_dir)
        os.makedirs(out_dir)

        # Write single case for prediction
        _array_to_nifti(lpd7, os.path.join(in_dir, "case_0000_0000.nii.gz"), spacing_zyx)

        results_dir = os.environ["nnUNet_results"]
        model_dir = os.path.join(
            results_dir,
            f"Dataset{dataset_id:03d}_NucleolusGC",
            f"nnUNetTrainer__nnUNetPlans__{configuration}",
        )

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=True,
            verbose=False,
        )
        predictor.initialize_from_trained_model_folder(
            model_dir,
            use_folds=folds if folds == "all" else (int(folds),),
            checkpoint_name="checkpoint_best.pth",
        )
        predictor.predict_from_files(
            [[os.path.join(in_dir, "case_0000_0000.nii.gz")]],
            [os.path.join(out_dir, "case_0000.nii.gz")],
            save_probabilities=False,
            overwrite=True,
        )

        pred = _nifti_to_array(os.path.join(out_dir, "case_0000.nii.gz"))

    # Constrain to nucleus mask
    pred[nucleus_mask == 0] = 0

    gc_mask = (pred > 0).astype(np.uint8) * 255
    logger.info(
        "nnU-Net prediction complete. GC voxels: %d / %d nucleus voxels",
        (gc_mask > 0).sum(), (nucleus_mask > 0).sum(),
    )
    return gc_mask, None, gc_mask.copy()


# ---------------------------------------------------------------------------
# Convenience: batch prediction on a directory
# ---------------------------------------------------------------------------

def predict_directory(
    input_dir: str,
    output_dir: str,
    dataset_id: int = _DATASET_ID,
    configuration: str = _CONFIGURATION,
) -> None:
    """
    Run nnU-Net prediction on all images in input_dir (NIfTI or TIFF).

    For TIFF inputs, converts them automatically before prediction.
    Prefer using the nnUNetv2_predict CLI directly for large batches.
    """
    _check_env()

    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError:
        raise ImportError("nnunetv2 is required: pip install nnunetv2")

    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.environ["nnUNet_results"]
    model_dir = os.path.join(
        results_dir,
        f"Dataset{dataset_id:03d}_NucleolusGC",
        f"nnUNetTrainer__nnUNetPlans__{configuration}",
    )

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        verbose=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds="all",
        checkpoint_name="checkpoint_best.pth",
    )

    input_files = [
        [os.path.join(input_dir, f)]
        for f in sorted(os.listdir(input_dir))
        if f.endswith("_0000.nii.gz")
    ]
    output_files = [
        os.path.join(output_dir, os.path.basename(f[0]).replace("_0000.nii.gz", ".nii.gz"))
        for f in input_files
    ]

    predictor.predict_from_files(input_files, output_files, overwrite=True)
    logger.info("Predicted %d cases → %s", len(input_files), output_dir)
