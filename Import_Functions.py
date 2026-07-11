"""Image I/O helpers for nucleolus analysis pipelines."""

import os
from typing import Optional

import numpy as np
from numpy import load
from skimage import io


def import_npy(path: str) -> Optional[np.ndarray]:
    """Load a ``.npy`` file; return ``None`` if the path does not exist."""
    if not os.path.exists(path):
        return None
    return load(path)


def import_imgs(input_dir: str, image_name: str, is_mask: bool = False) -> np.ndarray:
    """Read an image from ``input_dir/image_name``.

    Masks are returned as ``uint8``; intensity images as ``float32``.
    """
    img = io.imread(os.path.join(input_dir, image_name))
    return img.astype(np.uint8 if is_mask else np.float32)
