"""Mask- and intensity-based measurement utilities for nucleolus analysis."""

from __future__ import annotations

import logging
import os
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import find_contours, label, marching_cubes, mesh_surface_area, regionprops
from skimage.morphology import ball, dilation, disk

from Import_Functions import import_imgs

logger = logging.getLogger(__name__)

SHAPE_PARAMS_3D = [
    "cell_id", "obj_id", "surface_area", "volume",
    "surface_to_volume_ratio", "sphericity", "aspect_ratio", "solidity",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def extract_stage(cell_id: str) -> str:
    """Extract larval stage (L1–L4) from a cell_id string.

    Underscores are common in experiment folder names (e.g. ``20220304_L1/10_2``),
    so this uses an alphanumeric boundary rather than ``\\b`` (which treats ``_``
    as a word character and would miss ``_L1``).

    Raises ValueError if the stage cannot be determined.
    """
    match = re.search(r"(?<![A-Za-z0-9])(L[1-4])(?![A-Za-z0-9])", cell_id)
    if match is None:
        raise ValueError(f"Cannot extract larval stage from cell_id: {cell_id!r}")
    return match.group(1)


def _empty_row_df(columns: List[str], index=0) -> pd.DataFrame:
    """Create a one-row DataFrame with ``None`` placeholders for ``columns``."""
    return pd.DataFrame({c: None for c in columns}, index=[index] if np.isscalar(index) else index)


def _assign_row(df: pd.DataFrame, index, values: dict) -> None:
    """Assign ``values`` (column -> value) into ``df`` at ``index``."""
    for col, val in values.items():
        df.loc[index, col] = val


def _add_stage_column(df: pd.DataFrame, cell_id_col: str = "cell_id") -> pd.DataFrame:
    """Return a copy of ``df`` with a ``stage`` column derived from ``cell_id``."""
    out = df.copy()
    out["stage"] = out[cell_id_col].map(extract_stage)
    return out


def _sort_by_stage(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by="stage", ascending=True).reset_index(drop=True)


def _masked_channels(raw_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out pixels outside ``mask`` for every channel of ``raw_img``."""
    mask_bool = mask > 0
    return np.where(mask_bool[..., np.newaxis], raw_img, 0)


def _to_uint8_binary(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.uint8)
    out[out > 0] = 255
    return out


def _cv_qcd(values: np.ndarray) -> tuple:
    """Return (coefficient of variation, quartile coefficient of dispersion)."""
    mean = np.mean(values)
    cv = round(float(np.std(values) / mean), 3) if mean != 0 else float("nan")
    q1, q3 = np.quantile(values, [0.25, 0.75])
    qcd = round(float((q3 - q1) / (q3 + q1)), 3) if (q3 + q1) != 0 else float("nan")
    return cv, qcd


def _safe_ratio(num: int, denom: int) -> float:
    return float(num / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# shape measurement
# ---------------------------------------------------------------------------

def shape_discriber(
    structure_mask: np.ndarray,
    resolution: List,
    cell_id: str,
    measured_parameters: List,
) -> pd.DataFrame:
    """Measure shape descriptors for each connected object in ``structure_mask``.

    For 3D masks ``resolution`` is ``[z, y, x]``; for 2D it is ``[y, x]``.
    Expected columns typically match ``SHAPE_PARAMS_3D``.
    """
    structure_mask = np.pad(structure_mask, 5, "constant", constant_values=0)

    if np.count_nonzero(structure_mask) == 0:
        df = _empty_row_df(measured_parameters)
        for key in measured_parameters:
            df.loc[0, key] = cell_id if key == "cell_id" else 0
        return df

    labeled_mask, num_objs = label(structure_mask, return_num=True, connectivity=2)
    df = pd.DataFrame(columns=measured_parameters, index=range(1, num_objs + 1))

    is_3d = len(structure_mask.shape) == 3 and len(resolution) == 3
    is_2d = len(structure_mask.shape) == 2 and len(resolution) == 2

    if is_3d:
        pad = int(labeled_mask.shape[0] / 2)
        labeled_mask = np.pad(labeled_mask, pad, "constant")
        voxel_vol = float(np.prod(resolution))

        for i in range(1, num_objs + 1):
            obj_seg = labeled_mask == i
            verts, faces, _, _ = marching_cubes(
                obj_seg.astype(float), spacing=tuple(resolution), allow_degenerate=True
            )
            surface_area = mesh_surface_area(verts, faces)
            vol = float(np.count_nonzero(obj_seg) * voxel_vol)
            props = regionprops(obj_seg.astype(np.uint8))[0]
            equiv_sphere_surf = (np.pi ** (1 / 3)) * ((6 * vol) ** (2 / 3))
            _assign_row(df, i, {
                "cell_id": cell_id,
                "obj_id": i,
                "surface_area": surface_area,
                "volume": vol,
                "surface_to_volume_ratio": surface_area / vol,
                "sphericity": equiv_sphere_surf / surface_area,
                "aspect_ratio": props.axis_minor_length / props.axis_major_length,
                "solidity": props.solidity,
            })

    elif is_2d:
        labeled_mask = np.pad(
            labeled_mask,
            (int(labeled_mask.shape[0] / 2), int(labeled_mask.shape[1] / 2)),
            "constant",
        )
        pixel_area = resolution[0] * resolution[1]

        for i in range(1, num_objs + 1):
            obj_seg = labeled_mask == i
            logger.debug("2D object %d voxel count: %d", i, np.count_nonzero(obj_seg))
            props = regionprops(obj_seg.astype(np.uint8))[0]
            area = float(np.count_nonzero(obj_seg) * pixel_area)
            perimeter = props.perimeter * resolution[0]
            _assign_row(df, i, {
                "cell_id": cell_id,
                "obj_id": i,
                "surface_area": perimeter,
                "volume": area,
                "surface_to_volume_ratio": perimeter / area,
                "sphericity": 4 * np.pi * area / (perimeter ** 2),
                "aspect_ratio": props.axis_minor_length / props.axis_major_length,
                "solidity": props.solidity,
            })

    return df


def batch_measure_shape(
    master_folder: str,
    mask_name: str,
    shape_parameters: List,
    resolution_3d: List = None,
    config=None,
) -> List[pd.DataFrame]:
    """Walk ``master_folder`` and measure shape for every cell's ``mask_name``."""
    if config is None:
        from config_loader import load_config
        config = load_config()
    if resolution_3d is None:
        resolution_3d = config["measurement"]["resolution_3d"]

    all_dfs = []
    for item in os.listdir(master_folder):
        experiment_set_dir = os.path.join(master_folder, item)
        if not os.path.isdir(experiment_set_dir):
            continue
        date = os.path.basename(experiment_set_dir)
        for cell in os.listdir(experiment_set_dir):
            cell_seg_dir = os.path.join(experiment_set_dir, cell)
            cell_id = date + os.sep + os.path.basename(cell_seg_dir)
            logger.info("Processing: %s", cell_seg_dir)
            mask = import_imgs(cell_seg_dir, mask_name)
            all_dfs.append(
                shape_discriber(
                    mask, resolution=resolution_3d, cell_id=cell_id,
                    measured_parameters=shape_parameters,
                )
            )
    return all_dfs


def group_gc_measure_df(
    measurement_dfs: List,
    number_parameters: List,
    size_parameters: List,
):
    """Group per-object measurements into cell- and object-level DataFrames."""
    df = pd.concat(measurement_dfs, axis=0, ignore_index=True)
    df = _add_stage_column(df)

    grouped = df.groupby("cell_id", sort=False)
    number_by_cell_df = grouped.agg(obj_id=("obj_id", "max")).reset_index()
    number_by_cell_df = number_by_cell_df.rename(columns={"obj_id": number_parameters[1]})
    number_by_cell_df["stage"] = number_by_cell_df["cell_id"].map(extract_stage)
    number_by_cell_df = number_by_cell_df[number_parameters]
    number_by_cell_df = _sort_by_stage(number_by_cell_df)

    size_agg = grouped.agg(surface_area=("surface_area", "sum"), volume=("volume", "sum")).reset_index()
    size_agg["surface_to_volume_ratio"] = size_agg["surface_area"] / size_agg["volume"]
    size_agg["stage"] = size_agg["cell_id"].map(extract_stage)
    size_by_cell_df = size_agg[size_parameters]
    size_by_cell_df = _sort_by_stage(size_by_cell_df)

    size_by_obj_df = _sort_by_stage(
        df[["cell_id", "obj_id", "surface_area", "volume", "surface_to_volume_ratio", "stage"]]
    )
    morphology_by_obj_df = _sort_by_stage(
        df[["cell_id", "obj_id", "sphericity", "aspect_ratio", "solidity", "stage"]]
    )
    return number_by_cell_df, size_by_cell_df, morphology_by_obj_df, size_by_obj_df


def box_plot(
    df: pd.DataFrame,
    measurement_to_plot: str,
    y_axis_label: str,
    show_mean: bool = False,
    add_title: str = None,
):
    """Box plot of ``measurement_to_plot`` across larval stages L1–L4."""
    stages = ["L1", "L2", "L3", "L4"]
    # Keep canonical L1–L4 order for tick labels even if some stages are empty
    sorted_stages = [s for s in stages if s in set(df.stage.dropna().unique())]
    if not sorted_stages:
        sorted_stages = stages
    vals = [df.loc[df["stage"] == s, measurement_to_plot] for s in sorted_stages]

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 4))
    ax.boxplot(
        vals, notch=None, whis=(5, 95), showmeans=True, showfliers=False,
        meanprops={"marker": "^", "markersize": 10, "markerfacecolor": "white", "markeredgecolor": "b"},
        medianprops={"linestyle": "-", "color": "red", "linewidth": 2},
    )
    if show_mean:
        mean_marker = plt.Line2D([0], [0], marker="^", color="b", markersize=10, label="Mean")
        median_line = plt.Line2D([0], [0], linestyle="-", color="red", linewidth=2, label="Median")
        ax.legend(handles=[mean_marker, median_line], loc="best")

    ax.set_xticks(range(1, len(vals) + 1), sorted_stages)
    for i, lst in enumerate(vals):
        xs = np.random.normal(i + 1, 0.04, lst.shape[0])
        ax.scatter(xs, lst, color="k", alpha=0.5)

    ax.set_xlabel("Larval stages post synchronization", fontsize=12)
    ax.set_ylabel(y_axis_label, fontsize=12)
    if add_title is not None:
        ax.set_title(add_title, fontsize=12)
    return fig


# ---------------------------------------------------------------------------
# intensity-based analysis
# ---------------------------------------------------------------------------

def dilated_mask(
    mask: np.ndarray,
    radius: int,
    dilated_3d: bool = False,
    dilate_slice_by_slice: bool = False,
) -> np.ndarray:
    """Dilate ``mask`` in 3D or slice-by-slice; zeros top/bottom Z slices."""
    if dilated_3d:
        out = dilation(mask, footprint=ball(radius))
    elif dilate_slice_by_slice:
        out = np.zeros_like(mask)
        for z in range(mask.shape[0]):
            if np.count_nonzero(mask[z]) > 0:
                out[z] = dilation(mask[z], footprint=disk(radius))
    else:
        raise ValueError("Set dilated_3d=True or dilate_slice_by_slice=True")

    out[0] = 0
    out[-1] = 0
    return out


def coefficient_of_variances(
    gc_mask: np.ndarray,
    img: np.ndarray,
    cell_id: str,
    measured_parameters: List,
) -> pd.DataFrame:
    """Compute CV and QCD of each channel inside ``gc_mask``."""
    channels = [img[..., c][gc_mask > 0] for c in range(3)]
    metrics = [_cv_qcd(ch) for ch in channels]
    values = {
        "cell_id": cell_id,
        "cv_r": metrics[0][0], "cv_g": metrics[1][0], "cv_b": metrics[2][0],
        "qcd_r": metrics[0][1], "qcd_g": metrics[1][1], "qcd_b": metrics[2][1],
    }
    df = _empty_row_df(measured_parameters)
    _assign_row(df, 0, {k: values[k] for k in measured_parameters if k in values})
    return df


def relative_intensity(raw_img: np.ndarray, dilated_mask: np.ndarray, upper_limit: float) -> np.ndarray:
    """Keep pixels at or above ``upper_limit * channel_max`` inside the mask."""
    raw_in_mask = _masked_channels(raw_img, dilated_mask)
    top = np.zeros_like(raw_in_mask)
    for i in range(raw_in_mask.shape[-1]):
        channel = raw_in_mask[..., i]
        top[..., i] = np.where(channel >= upper_limit * channel.max(), channel, 0)
    return _to_uint8_binary(top)


def relative_number_of_pixels(
    raw_img: np.ndarray, dilated_mask: np.ndarray, upper_limit: float
) -> np.ndarray:
    """Keep the top ``upper_limit`` fraction of positive masked pixels per channel."""
    raw_in_mask = _masked_channels(raw_img, dilated_mask)
    top = np.zeros_like(raw_in_mask)

    for i in range(raw_in_mask.shape[-1]):
        channel = raw_in_mask[..., i]
        positive = channel > 0
        n_pos = int(np.count_nonzero(positive))
        if n_pos == 0:
            continue
        n_keep = max(1, int(n_pos * upper_limit))
        # Indices of the n_keep brightest positive pixels
        flat = channel.ravel()
        # Only consider positive pixels: set non-positive to -inf for ranking
        ranked = flat.copy()
        ranked[~positive.ravel()] = -np.inf
        top_idx = np.argpartition(ranked, -n_keep)[-n_keep:]
        coords = np.unravel_index(top_idx, channel.shape)
        top[coords + (i,)] = channel[coords]

    return _to_uint8_binary(top)


def overlap_3channel(top_img: np.ndarray, cell_id: str) -> pd.DataFrame:
    """Compute pairwise and triple-channel overlap ratios from a 3-channel binary image."""
    img_r, img_g, img_b = top_img[..., 0], top_img[..., 1], top_img[..., 2]
    overlap_rg = np.logical_and(img_r, img_g)
    overlap_rb = np.logical_and(img_r, img_b)
    overlap_gb = np.logical_and(img_g, img_b)
    overlap_rgb = np.logical_and.reduce([overlap_rg, overlap_rb, overlap_gb])

    n_r, n_g, n_b = map(np.count_nonzero, (img_r, img_g, img_b))
    n_rg, n_rb, n_gb, n_rgb = map(np.count_nonzero, (overlap_rg, overlap_rb, overlap_gb, overlap_rgb))

    colocal_df = pd.DataFrame([{
        "cell_id": cell_id,
        "rg_over_r": _safe_ratio(n_rg, n_r),
        "rg_over_g": _safe_ratio(n_rg, n_g),
        "rb_over_r": _safe_ratio(n_rb, n_r),
        "rb_over_b": _safe_ratio(n_rb, n_b),
        "gb_over_g": _safe_ratio(n_gb, n_g),
        "gb_over_b": _safe_ratio(n_gb, n_b),
        "rgb_over_r": _safe_ratio(n_rgb, n_r),
        "rgb_over_g": _safe_ratio(n_rgb, n_g),
        "rgb_over_b": _safe_ratio(n_rgb, n_b),
        "stage": extract_stage(cell_id),
    }])
    return colocal_df


def overlap_heatmap(mean_df: pd.DataFrame, stage: str):
    """Plot a 3x3 channel-overlap heatmap for one larval stage."""
    channels = ["Red", "Green", "Blue"]
    overlap = np.array([
        [1.0, mean_df.loc[0, "rg_over_r"], mean_df.loc[0, "rb_over_r"]],
        [mean_df.loc[0, "rg_over_g"], 1.0, mean_df.loc[0, "gb_over_g"]],
        [mean_df.loc[0, "rb_over_b"], mean_df.loc[0, "gb_over_b"], 1.0],
    ], dtype=float)

    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    axs.imshow(overlap, cmap="viridis")
    axs.set_xticks(np.arange(len(channels)), labels=channels, fontsize=15)
    axs.set_yticks(np.arange(len(channels)), labels=channels, fontsize=15)
    for i in range(len(channels)):
        for j in range(len(channels)):
            axs.text(j, i, f"{overlap[i, j]:.3f}", ha="center", va="center", color="red", fontsize=20)
    axs.set_title(f"Ratio of overlapped pixels at {stage} (top 10%)", fontsize=15)
    return fig


# ---------------------------------------------------------------------------
# concentration measurement
# ---------------------------------------------------------------------------

def get_largetest_slice(seg_hole_filled_mask: np.ndarray) -> int:
    """Return the Z index of the largest slice in a 3D binary mask."""
    return int(np.count_nonzero(seg_hole_filled_mask, axis=(1, 2)).argmax())


def find_box_in_binary_region(
    fluorescent_image: np.ndarray,
    nucleus_mask: np.ndarray,
    seg_hole_filled_mask: np.ndarray,
    box_size: int,
) -> np.ndarray:
    """Find the lowest-mean fully-contained box in the nucleoplasm region."""
    binary_mask = (nucleus_mask - seg_hole_filled_mask) > 0
    binary_mask = binary_mask.astype(np.uint8)
    positive_regions = np.where(binary_mask > 0)

    min_mean_intensity = float("inf")
    top_left_coord = None
    box_radius = box_size // 2
    h, w = binary_mask.shape

    for y, x in zip(positive_regions[0], positive_regions[1]):
        if not (box_radius <= y < h - box_radius and box_radius <= x < w - box_radius):
            continue
        y0, y1 = y - box_radius, y + box_radius + 1
        x0, x1 = x - box_radius, x + box_radius + 1
        if np.any(binary_mask[y0:y1, x0:x1] == 0):
            continue
        mean_intensity = float(np.mean(fluorescent_image[y0:y1, x0:x1]))
        if mean_intensity < min_mean_intensity:
            min_mean_intensity = mean_intensity
            top_left_coord = (y, x)

    if top_left_coord is None:
        raise ValueError(f"No valid box of size {box_size} found within the binary region.")

    box_mask = np.zeros_like(binary_mask)
    cy, cx = top_left_coord
    # Preserve original half-open extent (matches prior behaviour)
    box_mask[cy - box_radius:cy + box_radius, cx - box_radius:cx + box_radius] = 1
    return box_mask


def concentration_gc(
    raw_img: np.ndarray,
    raw_bg_subt: np.ndarray,
    background_mask: np.ndarray,
    nucleoplasm_mask: np.ndarray,
    seg_mask: np.ndarray,
    seg_hole_filled_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    cell_id: str,
    measured_parameters: List,
) -> pd.DataFrame:
    """Measure background, dilute, dense, total intensity and partition coefficient."""
    del seg_hole_filled_mask  # retained for API compatibility with callers/notebooks

    def _mean_in(img: np.ndarray, mask: np.ndarray) -> float:
        pixels = img[mask > 0]
        return float(np.mean(pixels)) if pixels.size > 0 else float("nan")

    bg_value = round(_mean_in(raw_bg_subt, background_mask), 1)
    dilute_value = round(_mean_in(raw_bg_subt, nucleoplasm_mask), 1)
    gc_value = round(_mean_in(raw_bg_subt, seg_mask), 1)
    total_value = round(_mean_in(raw_bg_subt, nucleus_mask), 1)

    gc_raw = _mean_in(raw_img, seg_mask)
    dilute_raw = _mean_in(raw_img, nucleoplasm_mask)
    if np.isnan(dilute_raw) or dilute_raw == 0 or np.isnan(gc_raw):
        pc_value = float("nan")
    else:
        pc_value = round(gc_raw / dilute_raw, 3)

    # Map semantic names so column order in measured_parameters cannot swap values
    semantic = {
        "cell_id": cell_id,
        "C_bg": bg_value,
        "C_dilute": dilute_value,
        "C_dense": gc_value,
        "C_total": total_value,
        "pc": pc_value,
        "total": total_value,
    }
    df = _empty_row_df(measured_parameters)
    for col in measured_parameters:
        if col in semantic:
            df.loc[0, col] = semantic[col]
    df["stage"] = extract_stage(cell_id)
    return df


def group_sort_larval_df(df_list: List) -> pd.DataFrame:
    """Concatenate measurement DataFrames and sort by larval stage."""
    grouped_df = pd.concat(df_list, axis=0, ignore_index=True)
    return _sort_by_stage(grouped_df)


# ---------------------------------------------------------------------------
# visualization helpers
# ---------------------------------------------------------------------------

def replace_at_index(input_string: str, character: str, new_value: str) -> str:
    """Replace the first occurrence of ``character`` in ``input_string``."""
    return input_string.replace(character, new_value, 1)


def seg_hole_filled_raw(master_seg_dir: str, master_raw_dir: str, cell_id: str, channel: int):
    """Load hole-filled mask + raw stack and return max-projection + contours."""
    seg_hole_filled = import_imgs(os.path.join(master_seg_dir, cell_id), "hole_filled.tif")
    raw_img = import_imgs(os.path.join(master_raw_dir, cell_id), "Composite_stack.tif")
    max_proj_mask = np.max(seg_hole_filled, axis=0)
    contours_seg = find_contours(max_proj_mask, 0.1)
    max_raw = np.max(raw_img[..., channel], axis=0)
    return max_raw, contours_seg


def plot_raw_contour(
    raw: np.ndarray,
    contours,
    cv: float,
    cell_id: str,
    channel: int,
    min_or_max_cv: str,
    save_dir=None,
    save_fig: bool = False,
):
    """Plot raw max-projection with GC contours overlaid."""
    new_cell_id = replace_at_index(cell_id, os.sep, "_")
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    axs.imshow(raw, cmap="gray")
    for contour in contours:
        axs.plot(contour[:, 1], contour[:, 0], linewidth=2, color="red", linestyle="dashed")
    axs.set_title(f"{new_cell_id} channel {channel} cv = {cv:.2f}", fontsize=10)
    axs.axis("off")
    plt.tight_layout()
    if save_dir is not None:
        if save_fig:
            plt.savefig(
                os.path.join(save_dir, f"{new_cell_id}_channel_{channel}_{min_or_max_cv}_cv_{cv:.2f}.svg"),
                bbox_inches="tight",
            )
        plt.close()
    else:
        plt.show()
