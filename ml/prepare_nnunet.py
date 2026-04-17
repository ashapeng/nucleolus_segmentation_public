"""
Convert the nucleolus dataset from TIFF stacks to nnU-Net v2 format.

nnU-Net v2 directory structure produced:
    nnUNet_raw/
        Dataset001_NucleolusGC/
            imagesTr/
                case_0000.nii.gz   (LPD-7 channel — primary)
                case_0001.nii.gz   (RPOA-2 channel — optional second modality)
            labelsTr/
                case_0000.nii.gz   (binary label: 0=bg, 1=GC)
            dataset.json

Set these environment variables before running nnU-Net commands:
    export nnUNet_raw="<project_root>/nnUNet_raw"
    export nnUNet_preprocessed="<project_root>/nnUNet_preprocessed"
    export nnUNet_results="<project_root>/nnUNet_results"

Then run:
    nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
    nnUNetv2_train 1 2d all       # auto-selects 2d due to 2.5:1 anisotropy
    nnUNetv2_find_best_configuration 1 -c 2d
    nnUNetv2_predict -i nnUNet_raw/Dataset001_NucleolusGC/imagesTr \\
                     -o predictions/ -d 1 -c 2d -f all

Dependencies: nibabel (pip install nibabel)
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)

# Voxel spacing in micrometres: (Z, Y, X)
VOXEL_SPACING_ZYX = (0.2, 0.08, 0.08)

# nnU-Net expects spacing in the NIfTI header as (x, y, z) mm — we use µm directly;
# nnU-Net fingerprinting reads relative spacing ratios, so absolute units don't matter
# as long as they are consistent across the dataset.
DATASET_ID = 1
DATASET_NAME = "NucleolusGC"


def _save_nifti(arr: np.ndarray, path: str, spacing_zyx=(0.2, 0.08, 0.08)) -> None:
    """Save a (Z, Y, X) numpy array as a NIfTI file with correct voxel spacing.

    nibabel convention:
        affine diagonal = (i_spacing, j_spacing, k_spacing, 1)
        for data stored as (Z, Y, X) → (i=Z, j=Y, k=X)
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required: pip install nibabel")

    sz, sy, sx = spacing_zyx
    affine = np.diag([sz, sy, sx, 1.0])
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), path)
    logger.debug("Saved NIfTI: %s  shape=%s", path, arr.shape)


def _save_label_nifti(arr: np.ndarray, path: str, spacing_zyx=(0.2, 0.08, 0.08)) -> None:
    """Save a binary label mask as integer NIfTI (0/1)."""
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required: pip install nibabel")

    sz, sy, sx = spacing_zyx
    affine = np.diag([sz, sy, sx, 1.0])
    label = (arr > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(label, affine), path)
    logger.debug("Saved label NIfTI: %s  nonzero=%d", path, label.sum())


def prepare_dataset(
    master_folder: str,
    output_root: str,
    multichannel: bool = False,
    n_augmentations: int = 10,
) -> str:
    """
    Convert all cells in master_folder to nnU-Net v2 format under output_root.

    Parameters
    ----------
    master_folder : str
        Path to the test_image/ directory.
    output_root : str
        Root directory for nnU-Net data (nnUNet_raw parent).
    multichannel : bool
        If True, include LPD-7 (ch2) + RPOA-2 (ch1) as two modalities.
        If False (default), use only LPD-7.
    n_augmentations : int
        Number of augmented copies per original sample (0 = no augmentation).

    Returns
    -------
    str
        Path to the created Dataset directory.
    """
    from ml.data_loader import load_all_samples
    from ml.augment import augment_sample

    dataset_dir = os.path.join(
        output_root, "nnUNet_raw", f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    )
    images_dir = os.path.join(dataset_dir, "imagesTr")
    labels_dir = os.path.join(dataset_dir, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    samples = load_all_samples(master_folder)
    if not samples:
        raise ValueError(f"No valid samples found in {master_folder}")

    channel_names = {"0": "LPD7"}
    if multichannel:
        channel_names["1"] = "RPOA2"

    case_index = 0

    for sample in samples:
        # Build the list of (image, mask) pairs: original + augmented
        img_3d = sample.lpd7              # (Z, Y, X) float32
        msk_3d = sample.gc_mask           # (Z, Y, X) uint8

        pairs = [(img_3d, msk_3d)]
        if n_augmentations > 0:
            pairs += augment_sample(img_3d, msk_3d, n_augmentations=n_augmentations)

        for img, msk in pairs:
            case_id = f"case_{case_index:04d}"

            # Channel 0: LPD-7
            _save_nifti(img, os.path.join(images_dir, f"{case_id}_0000.nii.gz"))

            # Channel 1 (optional): RPOA-2 — must have same shape as LPD-7 after augmentation
            if multichannel:
                rpoa2 = sample.rpoa2
                if img.shape != rpoa2.shape:
                    # Augmentation changed shape (z-crop); use same crop on rpoa2
                    # For simplicity, skip second channel for augmented copies
                    rpoa2_aug = rpoa2[:img.shape[0]]
                else:
                    rpoa2_aug = rpoa2
                _save_nifti(rpoa2_aug, os.path.join(images_dir, f"{case_id}_0001.nii.gz"))

            # Label
            _save_label_nifti(msk, os.path.join(labels_dir, f"{case_id}.nii.gz"))

            case_index += 1

        logger.info(
            "Prepared %s → %d cases (1 original + %d augmented)",
            sample.cell_id, 1 + n_augmentations, n_augmentations,
        )

    # Write dataset.json
    dataset_json = {
        "channel_names": channel_names,
        "labels": {"background": 0, "nucleolus_gc": 1},
        "numTraining": case_index,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    json_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    logger.info(
        "dataset.json written: %d total cases (%d cells × %d aug + 1 original)",
        case_index, len(samples), n_augmentations,
    )

    # Print run instructions
    _print_instructions(output_root, multichannel)

    return dataset_dir


def _print_instructions(output_root: str, multichannel: bool) -> None:
    abs_root = os.path.abspath(output_root)
    print("\n" + "=" * 60)
    print("nnU-Net v2 dataset prepared successfully.")
    print("=" * 60)
    print("\nStep 1 — Set environment variables:")
    print(f'  export nnUNet_raw="{abs_root}/nnUNet_raw"')
    print(f'  export nnUNet_preprocessed="{abs_root}/nnUNet_preprocessed"')
    print(f'  export nnUNet_results="{abs_root}/nnUNet_results"')
    print("\nStep 2 — Install nnU-Net (if not already):")
    print("  pip install nnunetv2")
    print("\nStep 3 — Fingerprint and preprocess (fully automatic):")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")
    print(
        "\nNOTE: nnU-Net will detect the 2.5:1 voxel anisotropy (Z=0.2µm, XY=0.08µm) "
        "and automatically choose the optimal 2d configuration."
    )
    print("\nStep 4 — Train (all folds = 5-fold CV on the full augmented dataset):")
    print(f"  nnUNetv2_train {DATASET_ID} 2d all --npz")
    print("\nStep 5 — Find best configuration:")
    print(f"  nnUNetv2_find_best_configuration {DATASET_ID} -c 2d")
    print("\nStep 6 — Predict on new images:")
    print(f"  nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d {DATASET_ID} -c 2d -f all")
    print("=" * 60 + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Prepare nnU-Net dataset from nucleolus TIFFs")
    parser.add_argument(
        "--data", default="test_image",
        help="Path to master_folder (default: test_image/)"
    )
    parser.add_argument(
        "--output", default=".",
        help="Root directory where nnUNet_raw/ will be created (default: project root)"
    )
    parser.add_argument(
        "--multichannel", action="store_true",
        help="Include LPD-7 + RPOA-2 as two input modalities"
    )
    parser.add_argument(
        "--augmentations", type=int, default=10,
        help="Number of augmented copies per cell (default: 10; 0 = disabled)"
    )
    args = parser.parse_args()

    prepare_dataset(
        master_folder=args.data,
        output_root=args.output,
        multichannel=args.multichannel,
        n_augmentations=args.augmentations,
    )


if __name__ == "__main__":
    main()
