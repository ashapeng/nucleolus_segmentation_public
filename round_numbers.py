"""Significant-digit rounding helpers used by segmentation thresholds."""

import math
import re

from skimage import measure


def decimal_num(n, two_digit: bool = False) -> int:
    """Return the fractional-digit count for rounding to significant digits.

    When ``two_digit`` is True, keeps two significant digits in the fractional
    part (biology-by-numbers convention).
    """
    frac = math.modf(abs(n))[0]
    # Scientific notation of the fractional part: e.g. 3.2e-03 → exponent 3
    exponent = int(("%e" % frac).partition("-")[2])
    return exponent + 1 if two_digit else exponent


def round_up(n, decimals):
    """Round ``n`` up (away from -inf) to ``decimals`` places."""
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def round_down(n, decimals):
    """Round ``n`` down (toward -inf) to ``decimals`` places."""
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


_FLUOR_COLORS = {
    "rfp": "red",
    "cfp": "blue",
    "gfp": "green",
}


def plot_volume(volume, voxel, ax, fluorescence, alp):
    """Plot a marching-cubes surface of ``volume`` coloured by fluorescence channel."""
    level = 0.5 * (volume.max() + volume.min())
    verts, faces, _, _ = measure.marching_cubes(
        volume, spacing=voxel, level=level, allow_degenerate=True, method="lewiner", step_size=1
    )
    z, y, x = verts.T
    color = "gray"
    for key, col in _FLUOR_COLORS.items():
        if re.search(key, fluorescence, re.IGNORECASE):
            color = col
            break
    ax.plot_trisurf(x, y, faces, z, linewidth=0, antialiased=True, color=color, alpha=alp, edgecolor="k")
    ax.dist = 0.5
