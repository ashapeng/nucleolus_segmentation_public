"""
Cellpose 3 inference wrapper — secondary backend for nucleolus GC segmentation.

Cellpose produces instance-level labels (each nucleolus gets a unique integer ID),
which is useful when cells contain multiple nucleoli or when downstream analysis
needs per-object measurements.

Fine-tuning workflow:
    from ml.predict_cellpose import finetune_cellpose
    finetune_cellpose(master_folder="test_image/", model_dir="ml/models/cellpose_gc")

Inference:
    from ml.predict_cellpose import gc_segment_cellpose
    gc_mask, labels = gc_segment_cellpose(raw_image, nucleus_mask)

Dependencies: cellpose>=3.0  (pip install cellpose)
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Voxel anisotropy: Z_spacing / XY_spacing = 0.2 / 0.08 = 2.5
_ANISOTROPY = 0.2 / 0.08  # 2.5


def gc_segment_cellpose(
    raw_image: np.ndarray,
    nucleus_mask: np.ndarray,
    model_path: Optional[str] = None,
    diameter: Optional[float] = None,
    do_3d: bool = True,
    channel: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment GC nucleoli using Cellpose 3.

    Parameters
    ----------
    raw_image : np.ndarray
        Shape (Z, Y, X, C) uint16.
    nucleus_mask : np.ndarray
        Shape (Z, Y, X) uint8 — prediction constrained to nucleus interior.
    model_path : str, optional
        Path to a fine-tuned Cellpose model. If None, uses the pretrained
        'nuclei' model (zero-shot baseline).
    diameter : float, optional
        Expected object diameter in pixels. None = auto-estimate.
    do_3d : bool
        Run in 3D mode (default True). Set False for 2D max-projection fallback.
    channel : int
        Which channel to use for segmentation (default 2 = LPD-7).

    Returns
    -------
    gc_mask : np.ndarray  (Z, Y, X) uint8  — binary GC mask (0 or 255)
    instance_labels : np.ndarray  (Z, Y, X) int32  — unique ID per nucleolus
    """
    try:
        from cellpose import models as cp_models
    except ImportError:
        raise ImportError("cellpose is required: pip install cellpose")

    lpd7 = raw_image[..., channel].astype(np.float32)

    if model_path and os.path.exists(model_path):
        model = cp_models.CellposeModel(pretrained_model=model_path, gpu=True)
        logger.info("Cellpose: loaded fine-tuned model from %s", model_path)
    else:
        model = cp_models.Cellpose(model_type="nuclei", gpu=True)
        logger.info("Cellpose: using pretrained 'nuclei' model (zero-shot)")

    if do_3d:
        masks, _, _, _ = model.eval(
            lpd7,
            diameter=diameter,
            do_3D=True,
            anisotropy=_ANISOTROPY,
            channels=[0, 0],   # grayscale
            normalize=True,
        )
    else:
        # 2.5D: segment each Z slice and stack
        slice_masks = []
        for z in range(lpd7.shape[0]):
            m, _, _, _ = model.eval(
                lpd7[z],
                diameter=diameter,
                channels=[0, 0],
                normalize=True,
            )
            slice_masks.append(m)
        masks = np.stack(slice_masks, axis=0)

    # Constrain to nucleus mask
    masks[nucleus_mask == 0] = 0

    instance_labels = masks.astype(np.int32)
    gc_mask = (instance_labels > 0).astype(np.uint8) * 255

    n_nucleoli = len(np.unique(instance_labels)) - 1  # exclude background
    logger.info("Cellpose found %d nucleolus instance(s)", n_nucleoli)

    return gc_mask, instance_labels


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune_cellpose(
    master_folder: str,
    model_dir: str = "ml/models/cellpose_gc",
    base_model: str = "nuclei",
    n_epochs: int = 100,
    learning_rate: float = 0.001,
    channel: int = 2,
) -> str:
    """
    Fine-tune Cellpose on the labelled nucleolus dataset.

    Parameters
    ----------
    master_folder : str
        Path to test_image/ directory.
    model_dir : str
        Where to save the fine-tuned model weights.
    base_model : str
        Starting checkpoint: 'nuclei' (recommended) or 'cyto3'.
    n_epochs : int
        Training epochs (default 100; increase to 200 if Dice plateaus).
    learning_rate : float
        Learning rate (default 0.001).
    channel : int
        Image channel to use (default 2 = LPD-7).

    Returns
    -------
    str
        Path to the saved model weights.
    """
    try:
        from cellpose import models as cp_models, io as cp_io
        from cellpose.train import train_seg
    except ImportError:
        raise ImportError("cellpose>=3.0 is required: pip install cellpose")

    from ml.data_loader import load_all_samples
    from ml.augment import augment_sample

    os.makedirs(model_dir, exist_ok=True)

    samples = load_all_samples(master_folder)
    train_images, train_masks = [], []

    for sample in samples:
        img = sample.lpd7       # (Z, Y, X) float32
        msk = sample.gc_mask    # (Z, Y, X) uint8

        # Add original
        train_images.append(img)
        train_masks.append((msk > 0).astype(np.uint16))

        # Augmented copies
        for aug_img, aug_msk in augment_sample(img, msk, n_augmentations=10):
            train_images.append(aug_img.astype(np.float32))
            train_masks.append((aug_msk > 0).astype(np.uint16))

    logger.info(
        "Fine-tuning Cellpose on %d samples (%d cells + augmentations)",
        len(train_images), len(samples),
    )

    model = cp_models.CellposeModel(
        model_type=base_model,
        gpu=True,
    )

    model_path = train_seg(
        net=model.net,
        train_data=train_images,
        train_labels=train_masks,
        channels=[0, 0],
        normalize=True,
        save_path=model_dir,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        SGD=False,
        batch_size=8,
        nimg_per_epoch=max(8, len(train_images)),
        model_name="cellpose_nucleolus_gc",
    )

    logger.info("Cellpose fine-tuning complete. Model saved to: %s", model_path)
    return model_path
