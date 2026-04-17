"""
Joint image + mask augmentation for 3D fluorescence microscopy data.

All transforms are applied identically to the image and mask to maintain
spatial correspondence. Biologically-valid constraints are enforced:
  - XY rotation: ±180° (nucleolus has no preferred XY orientation)
  - Z-tilt:       ±5°  (small tilt only)
  - Z-flip:       DISABLED (top/bottom of Z-stack is biologically meaningful)
  - XY-flip:      enabled
  - Intensity ops: image only (never applied to mask)

Dependencies: numpy, scipy (both already required by the main project)
"""

import logging
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

def random_rotate_xy(
    image: np.ndarray,
    mask: np.ndarray,
    angle_range: Tuple[float, float] = (-180.0, 180.0),
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate the XY plane by a random angle (applied identically to all Z slices)."""
    if rng is None:
        rng = np.random.default_rng()
    angle = rng.uniform(*angle_range)
    # axes=(1, 2) rotates in the Y-X plane for a (Z, Y, X[, C]) array
    rot_img = ndi.rotate(image, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
    rot_msk = ndi.rotate(mask.astype(np.float32), angle, axes=(1, 2), reshape=False, order=0, mode="nearest")
    return rot_img, (rot_msk > 0.5).astype(mask.dtype)


def random_z_tilt(
    image: np.ndarray,
    mask: np.ndarray,
    tilt_range: Tuple[float, float] = (-5.0, 5.0),
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tilt slightly around the X axis (Z-Y rotation)."""
    if rng is None:
        rng = np.random.default_rng()
    angle = rng.uniform(*tilt_range)
    rot_img = ndi.rotate(image, angle, axes=(0, 1), reshape=False, order=1, mode="nearest")
    rot_msk = ndi.rotate(mask.astype(np.float32), angle, axes=(0, 1), reshape=False, order=0, mode="nearest")
    return rot_img, (rot_msk > 0.5).astype(mask.dtype)


def random_xy_flip(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly flip along Y and/or X axes."""
    if rng is None:
        rng = np.random.default_rng()
    for axis in (1, 2):  # Y=1, X=2  (never flip Z=0)
        if rng.random() > 0.5:
            image = np.flip(image, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
    return image, mask


def random_intensity_scale(
    image: np.ndarray,
    scale_range: Tuple[float, float] = (0.7, 1.3),
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Multiply image intensity by a random per-channel scale factor."""
    if rng is None:
        rng = np.random.default_rng()
    if image.ndim == 4:  # (Z, Y, X, C)
        n_channels = image.shape[-1]
        scales = rng.uniform(*scale_range, size=n_channels)
        return np.clip(image * scales[np.newaxis, np.newaxis, np.newaxis, :], 0, None).astype(image.dtype)
    scale = rng.uniform(*scale_range)
    return np.clip(image * scale, 0, None).astype(image.dtype)


def random_gamma_correction(
    image: np.ndarray,
    gamma_range: Tuple[float, float] = (0.8, 1.2),
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Apply random gamma correction (normalise → gamma → denormalise)."""
    if rng is None:
        rng = np.random.default_rng()
    gamma = rng.uniform(*gamma_range)
    lo, hi = image.min(), image.max()
    if hi == lo:
        return image.copy()
    normalised = (image.astype(np.float32) - lo) / (hi - lo)
    corrected = np.power(normalised, gamma)
    return (corrected * (hi - lo) + lo).astype(image.dtype)


def random_gaussian_noise(
    image: np.ndarray,
    sigma_range: Tuple[float, float] = (0.005, 0.03),
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Add Gaussian noise scaled relative to the image range."""
    if rng is None:
        rng = np.random.default_rng()
    sigma = rng.uniform(*sigma_range)
    image_range = float(image.max() - image.min())
    noise = rng.normal(0, sigma * image_range, size=image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, None).astype(image.dtype)


def random_elastic_deform(
    image: np.ndarray,
    mask: np.ndarray,
    sigma: float = 5.0,
    alpha: float = 80.0,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply elastic deformation in the XY plane only.

    sigma controls smoothness of the displacement field.
    alpha controls the magnitude of deformation.
    Z deformation is intentionally omitted to avoid unrealistic Z-axis warping.
    """
    if rng is None:
        rng = np.random.default_rng()

    z, y, x = image.shape[:3]

    # Generate random 2D displacement fields (same for all Z slices)
    dy = ndi.gaussian_filter(rng.uniform(-1, 1, (y, x)).astype(np.float32), sigma) * alpha
    dx = ndi.gaussian_filter(rng.uniform(-1, 1, (y, x)).astype(np.float32), sigma) * alpha

    yy, xx = np.meshgrid(np.arange(y), np.arange(x), indexing="ij")
    indices_y = np.clip(yy + dy, 0, y - 1).ravel()
    indices_x = np.clip(xx + dx, 0, x - 1).ravel()

    def _warp_slice(slc, order):
        return ndi.map_coordinates(slc, [indices_y, indices_x], order=order, mode="nearest").reshape(y, x)

    # Apply to each Z slice (and each channel if multichannel)
    if image.ndim == 4:
        warped_img = np.stack(
            [np.stack([_warp_slice(image[zi, :, :, c], 1) for c in range(image.shape[-1])], axis=-1)
             for zi in range(z)],
            axis=0,
        )
    else:
        warped_img = np.stack([_warp_slice(image[zi], 1) for zi in range(z)], axis=0)

    warped_msk = np.stack(
        [((_warp_slice(mask[zi].astype(np.float32), 0)) > 0.5).astype(mask.dtype) for zi in range(z)],
        axis=0,
    )
    return warped_img.astype(image.dtype), warped_msk


def random_z_crop(
    image: np.ndarray,
    mask: np.ndarray,
    min_keep_ratio: float = 0.6,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly crop a contiguous subset of Z slices (keeps ≥ min_keep_ratio of depth)."""
    if rng is None:
        rng = np.random.default_rng()
    z = image.shape[0]
    min_z = max(1, int(z * min_keep_ratio))
    keep = rng.integers(min_z, z + 1)
    start = rng.integers(0, z - keep + 1)
    return image[start:start + keep], mask[start:start + keep]


# ---------------------------------------------------------------------------
# Composite augmentation pipeline
# ---------------------------------------------------------------------------

def augment_sample(
    image: np.ndarray,
    mask: np.ndarray,
    n_augmentations: int = 10,
    seed: int = None,
) -> list:
    """
    Generate n_augmentations augmented (image, mask) pairs from one sample.

    Returns a list of (augmented_image, augmented_mask) tuples.
    The original (unmodified) sample is NOT included — call this and then
    prepend the original yourself if desired.
    """
    rng = np.random.default_rng(seed)
    augmented = []

    for _ in range(n_augmentations):
        img = image.copy()
        msk = mask.copy()

        # Spatial transforms (joint)
        img, msk = random_rotate_xy(img, msk, rng=rng)
        if rng.random() > 0.5:
            img, msk = random_z_tilt(img, msk, rng=rng)
        img, msk = random_xy_flip(img, msk, rng=rng)
        if rng.random() > 0.5:
            img, msk = random_elastic_deform(img, msk, rng=rng)
        if rng.random() > 0.3:
            img, msk = random_z_crop(img, msk, rng=rng)

        # Intensity transforms (image only)
        img = random_intensity_scale(img, rng=rng)
        img = random_gamma_correction(img, rng=rng)
        img = random_gaussian_noise(img, rng=rng)

        augmented.append((img, msk))

    return augmented
