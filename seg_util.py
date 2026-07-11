"""Nucleolus / GC segmentation utilities built on AllenCell segmenter primitives."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.measure import label
from skimage.morphology import (
    ball,
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    disk,
    remove_small_holes,
    remove_small_objects,
)

from aicssegmentation.core.pre_processing_utils import (
    image_smoothing_gaussian_3d,
    image_smoothing_gaussian_slice_by_slice,
    intensity_normalization,
    suggest_normalization_param,
)
from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper

import round_numbers as rn

logger = logging.getLogger(__name__)

_GLOBAL_THRESH = {
    "tri": "triangle",
    "triangle": "triangle",
    "med": "median",
    "median": "median",
    "ave": "ave_tri_med",
    "ave_tri_med": "ave_tri_med",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _percentile_clip_and_normalize(img: np.ndarray, percentile: float = 99.9) -> np.ndarray:
    """Clip to given percentile then min-max normalize to [0, 1]."""
    clipped = np.clip(img, None, np.percentile(img, percentile))
    lo, hi = clipped.min(), clipped.max()
    if hi == lo:
        return np.zeros_like(clipped, dtype=np.float64)
    return (clipped - lo) / (hi - lo)


def _to_uint8_mask(arr: np.ndarray) -> np.ndarray:
    """Convert a binary/labelled array to uint8 with positive pixels set to 255."""
    out = arr.astype(np.uint8)
    out[out > 0] = 255
    return out


def _zero_z_cap_slices(vol: np.ndarray) -> np.ndarray:
    """Zero the first and last Z slices in-place and return the volume."""
    vol[0] = 0
    vol[-1] = 0
    return vol


def _largest_label_mask(labeled: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the largest connected component in a labelled image."""
    if labeled.max() == 0:
        return np.zeros(labeled.shape, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # ignore background
    return labeled == int(counts.argmax())


def _round_threshold(value: float) -> float:
    """Round a threshold up to two significant fractional digits."""
    return rn.round_up(value, rn.decimal_num(value, two_digit=True))


def _apply_per_slice(vol: np.ndarray, fn) -> np.ndarray:
    """Apply a 2D function to each Z slice; returns a new array."""
    out = np.empty_like(vol)
    for z in range(vol.shape[0]):
        out[z] = fn(vol[z])
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def image_2d_seg(raw_img: np.ndarray, nucleus_mask: Optional[np.ndarray], sigma_2d: float) -> np.ndarray:
    """Segment the maximal Z-projection of a multi-channel stack with Otsu.

    Parameters
    ----------
    raw_img : ndarray
        Z-stack, shape (Z, Y, X) or (Z, Y, X, C).
    nucleus_mask : ndarray or None
        Optional nucleus mask; when given, Otsu is computed only inside it.
    sigma_2d : float
        Gaussian smoothing sigma applied to the projection.
    """
    if raw_img.ndim == 3:
        raw_img = raw_img[..., np.newaxis]

    max_projection = np.max(raw_img, axis=0)  # (Y, X, C)
    n_channels = max_projection.shape[-1]

    normalized = np.stack(
        [_percentile_clip_and_normalize(max_projection[..., c]) for c in range(n_channels)],
        axis=-1,
    )
    smoothed = np.stack(
        [
            ndi.gaussian_filter(normalized[..., c], sigma=sigma_2d, mode="nearest", truncate=3)
            for c in range(n_channels)
        ],
        axis=-1,
    )

    thresholded = np.zeros_like(smoothed)
    for i in range(n_channels):
        channel = smoothed[..., i]
        if nucleus_mask is not None:
            masked = np.where(nucleus_mask[0] > 0, channel, 0)
            positives = masked[masked > 0]
        else:
            positives = channel[channel > 0]
        if positives.size == 0:
            continue
        cutoff = threshold_otsu(positives)
        thresholded[..., i] = channel > cutoff

    post_seg = np.zeros_like(thresholded, dtype=np.uint8)
    for i in range(n_channels):
        opened = binary_opening(thresholded[..., i], footprint=np.ones((3, 3)))
        filled = remove_small_holes(opened)
        labeled = label(filled, connectivity=2)
        if labeled.max() == 0:
            continue
        post_seg[..., i] = _to_uint8_mask(_largest_label_mask(labeled))
    return post_seg


def bg_subtraction(raw_img: np.ndarray, bg_mask: np.ndarray, clip: bool = False) -> np.ndarray:
    """Subtract mean background intensity (from ``bg_mask``) per channel.

    Negative values after subtraction are left as-is unless ``clip=True``.
    """
    out = raw_img.astype(raw_img.dtype, copy=True)

    if raw_img.ndim == 4:
        for z in range(bg_mask.shape[0]):
            bg_slice = bg_mask[z] > 0
            if not np.any(bg_slice):
                continue
            for ch in range(raw_img.shape[-1]):
                bg_pixels = raw_img[z, ..., ch][bg_slice]
                bg_pixels = bg_pixels[bg_pixels > 0]
                if bg_pixels.size == 0:
                    continue
                out[z, ..., ch] = raw_img[z, ..., ch] - np.mean(bg_pixels)
                if clip:
                    np.clip(out[z, ..., ch], 0, None, out=out[z, ..., ch])

    elif raw_img.ndim == 3:
        bg_2d = np.max(bg_mask, axis=0) > 0
        for ch in range(raw_img.shape[-1]):
            bg_pixels = raw_img[..., ch][bg_2d]
            bg_pixels = bg_pixels[bg_pixels > 0]
            if bg_pixels.size == 0:
                continue
            out[..., ch] = raw_img[..., ch] - np.mean(bg_pixels)
            if clip:
                np.clip(out[..., ch], 0, None, out=out[..., ch])

    return out


def min_max_norm(raw_img: np.ndarray, suggest_norm: bool = False) -> np.ndarray:
    """Min-max intensity normalization via AllenCell ``intensity_normalization``."""
    if suggest_norm:
        low_ratio, up_ratio = suggest_normalization_param(raw_img)
        scaling_param = [low_ratio, up_ratio]
    else:
        scaling_param = [0]
    return intensity_normalization(raw_img.copy(), scaling_param=scaling_param)


def gaussian_smooth_stack(img: np.ndarray, sigma: list) -> np.ndarray:
    """Gaussian smooth a single-channel stack (3D if ``len(sigma)>1``, else slice-wise)."""
    if len(sigma) > 1:
        return image_smoothing_gaussian_3d(img, sigma=sigma, truncate_range=3.0)
    return image_smoothing_gaussian_slice_by_slice(img, sigma=sigma[0], truncate_range=3.0)


def global_otsu(
    img: np.ndarray,
    mask: np.ndarray,
    global_thresh_method: str,
    mini_size: float,
    local_adjust: float = 0.98,
    extra_criteria: bool = False,
    return_object: bool = False,
    keep_largest: bool = False,
):
    """Masked-object hybrid thresholding (global + per-object local Otsu).

    ``global_thresh_method``: ``"triangle"``/``"tri"``, ``"median"``/``"med"``,
    or ``"ave_tri_med"``/``"ave"``.
    """
    method = _GLOBAL_THRESH.get(global_thresh_method)
    if method is None:
        raise ValueError(f"Unknown global_thresh_method: {global_thresh_method!r}")

    if method == "triangle":
        th_low_level = threshold_triangle(img)
    elif method == "median":
        th_low_level = np.percentile(img, 50)
    else:
        th_low_level = (threshold_triangle(img) + np.percentile(img, 50)) / 2

    bw_low_level = remove_small_objects(img > th_low_level, min_size=mini_size, connectivity=1)
    bw_low_level = binary_dilation(bw_low_level, footprint=ball(1))
    _zero_z_cap_slices(bw_low_level)

    bw_high_level = np.zeros_like(bw_low_level)
    lab_low, num_obj = label(bw_low_level, return_num=True, connectivity=1)
    local_cutoff = (0.333 * threshold_otsu(img[mask > 0])) if extra_criteria else None

    for idx in range(1, num_obj + 1):
        single_obj = lab_low == idx
        local_otsu = threshold_otsu(img[single_obj])
        final_otsu = _round_threshold(local_otsu)
        logger.debug(
            "otsu=%.4f  rounded=%.4f  adjusted=%.4f",
            local_otsu, final_otsu, final_otsu * local_adjust,
        )
        if local_cutoff is not None and local_otsu <= local_cutoff:
            continue
        bw_high_level[np.logical_and(img > final_otsu * local_adjust, single_obj)] = 1

    def _close_fill_clean(slice_2d: np.ndarray) -> np.ndarray:
        closed = binary_closing(slice_2d, footprint=disk(2))
        filled = remove_small_holes(
            closed.astype(bool), area_threshold=np.count_nonzero(closed), connectivity=2
        )
        return remove_small_objects(filled, min_size=30, connectivity=2)

    global_mask = _apply_per_slice(bw_high_level, _close_fill_clean)
    _zero_z_cap_slices(global_mask)

    if keep_largest:
        labeled_mask, n_labels = label(global_mask, return_num=True, connectivity=1)
        logger.debug("number of mask objects: %d", n_labels)
        segmented = _largest_label_mask(labeled_mask)
    else:
        segmented = global_mask

    struct2 = ndi.generate_binary_structure(2, 2)

    def _dilate_close_fill(slice_2d: np.ndarray) -> np.ndarray:
        dilated = binary_dilation(slice_2d.astype(bool), footprint=disk(1))
        closed = binary_closing(dilated, footprint=struct2)
        return remove_small_holes(
            closed, area_threshold=np.count_nonzero(closed), connectivity=2
        )

    dilated_mask = _apply_per_slice(segmented, _dilate_close_fill)
    _zero_z_cap_slices(dilated_mask)
    result = dilated_mask > 0

    if return_object:
        return result, bw_low_level
    return result


def segment_spot(
    raw_img: np.ndarray,
    nucleus_mask: np.ndarray,
    nucleolus_mask: np.ndarray,
    LoG_sigma: list,
    mini_size: float,
    invert_raw: bool = False,
    show_param: bool = False,
    arbitray_cutoff: bool = False,  # typo kept for API compatibility
):
    """Detect dark/bright spots with multi-scale LoG filtering inside the nucleolus."""
    del arbitray_cutoff  # API compatibility only
    spot = (np.max(raw_img) - raw_img) if invert_raw else raw_img.copy()

    nucleus_mask_eroded = binary_erosion(nucleus_mask, footprint=ball(2))
    _zero_z_cap_slices(nucleus_mask_eroded)
    eroded_bool = nucleus_mask_eroded.astype(bool)

    param_spot = []
    for sigma in LoG_sigma:
        spot_temp = np.empty_like(spot, dtype=np.float64)
        for z in range(spot.shape[0]):
            spot_temp[z] = -(sigma ** 2) * ndi.gaussian_laplace(spot[z], sigma)

        log_in_mask = spot_temp[eroded_bool]
        p96 = float(np.percentile(log_in_mask, 96))
        param_spot.append([sigma, _round_threshold(p96)])

    cutoff_mean = float(np.mean([p[1] for p in param_spot]))
    logger.debug("spot seg mean cutoff value: %.6f", cutoff_mean)
    updated_param = [[p[0], cutoff_mean] for p in param_spot]
    if show_param:
        logger.info("spot seg parameters: %s", param_spot)
        logger.info("spot seg updated parameters: %s", updated_param)

    spot_by_log = dot_2d_slice_by_slice_wrapper(spot, updated_param)

    struct2 = ndi.generate_binary_structure(2, 2)

    def _open_clean_close(slice_2d: np.ndarray) -> np.ndarray:
        opened = binary_opening(slice_2d, footprint=struct2)
        cleaned = remove_small_objects(opened, min_size=10, connectivity=2)
        return binary_closing(cleaned, footprint=struct2).astype(np.uint8)

    spot_opened = _apply_per_slice(spot_by_log, _open_clean_close)
    spot_in_structure = np.logical_and(spot_opened, nucleolus_mask).astype(np.uint8)

    labeled, n_obj = label(spot_in_structure, return_num=True, connectivity=2)
    for i in range(1, n_obj + 1):
        coords = np.argwhere(labeled == i)
        if len(set(coords[:, 0])) <= 2:
            labeled[labeled == i] = 0
    multi_z = labeled > 0

    size_filtered = remove_small_objects(multi_z, min_size=mini_size, connectivity=2)
    return _to_uint8_mask(size_filtered)


def final_gc_holes(spot_mask: np.ndarray, nucleolus_mask: np.ndarray):
    """Combine nucleolus mask with dark-spot holes; return GC and hole-filled masks."""
    final_gc = nucleolus_mask.copy()
    final_gc[spot_mask > 0] = 0

    hole_filled = np.zeros_like(final_gc)
    for z in range(final_gc.shape[0]):
        hole_filled[z] = remove_small_holes(
            final_gc[z].astype(bool),
            area_threshold=np.count_nonzero(nucleolus_mask[z]),
            connectivity=2,
        )

    return _to_uint8_mask(final_gc), _to_uint8_mask(hole_filled)

def gc_segment(
    raw_image: np.ndarray,
    nucleus_mask: np.ndarray,
    sigma: float = None,
    local_adjust_for_GC: float = None,
    config=None,
    backend: str = "classical",
):
    """End-to-end granular-component segmentation pipeline.

    Parameters
    ----------
    backend : str
        ``"classical"`` (default), ``"nnunet"``, or ``"cellpose"``.

    Returns
    -------
    final_gc, dark_spot, hole_filled_gc : uint8 masks
        For ML backends, ``dark_spot`` is ``None`` and hole-filled equals GC.
    """
    if backend == "nnunet":
        from ml.predict_nnunet import gc_segment_nnunet
        return gc_segment_nnunet(raw_image, nucleus_mask)

    if backend == "cellpose":
        from ml.predict_cellpose import gc_segment_cellpose
        if config is None:
            from config_loader import load_config
            config = load_config()
        model_path = config.get("ml", {}).get("cellpose", {}).get("model_path", None)
        gc_mask, _ = gc_segment_cellpose(raw_image, nucleus_mask, model_path=model_path)
        return gc_mask, None, gc_mask.copy()

    # --- classical backend ---
    if config is None:
        from config_loader import load_config
        config = load_config()
    cfg = config["segmentation"]["gc_segment"]
    if sigma is None:
        sigma = cfg["sigma"]
    if local_adjust_for_GC is None:
        local_adjust_for_GC = cfg["local_adjust"]
    log_sigma = list(
        np.arange(cfg["log_sigma_min"], cfg["log_sigma_max"], cfg["log_sigma_step"], dtype=float)
    )

    raw_img = raw_image.copy()
    nucleus_mask = nucleus_mask.copy()

    normalized = np.stack(
        [min_max_norm(raw_img[..., i]) for i in range(raw_img.shape[-1])],
        axis=-1,
    )
    gc_smoothed = ndi.gaussian_filter(
        normalized[..., 2], sigma=sigma, mode="nearest", truncate=3
    )
    gc_otsu = global_otsu(
        gc_smoothed,
        nucleus_mask,
        global_thresh_method="ave",
        mini_size=cfg["mini_size_otsu"],
        local_adjust=local_adjust_for_GC,
        extra_criteria=False,
        keep_largest=True,
    )
    gc_dark_spot = segment_spot(
        normalized[..., 2],
        nucleus_mask,
        gc_otsu,
        LoG_sigma=log_sigma,
        mini_size=cfg["mini_size_spot"],
        invert_raw=True,
    )
    final_gc, hole_filled_gc = final_gc_holes(gc_dark_spot, gc_otsu)
    return _to_uint8_mask(final_gc), _to_uint8_mask(gc_dark_spot), _to_uint8_mask(hole_filled_gc)


def ball_confocol(radius_xy, radius_z, dtype=np.uint8):
    """Generate an anisotropic 3D ball footprint for Weber-lab confocal voxels."""
    n_xy = 2 * radius_xy + 1
    n_z = 2 * radius_z + 1
    Z, Y, X = np.mgrid[
        -radius_z:radius_z:n_z * 1j,
        -radius_xy:radius_xy:n_xy * 1j,
        -radius_xy:radius_xy:n_xy * 1j,
    ]
    return np.array(X ** 2 + Y ** 2 + Z ** 2 <= radius_xy * radius_z, dtype=dtype)
