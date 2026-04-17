import os
from typing import Dict, Any

_DEFAULT_CONFIG: Dict[str, Any] = {
    "microscope": {"z_um": 0.2, "xy_um": 0.08},
    "segmentation": {
        "gc_segment": {
            "sigma": 1.0,
            "local_adjust": 1.08,
            "mini_size_otsu": 1000,
            "log_sigma_min": 2.5,
            "log_sigma_max": 4.0,
            "log_sigma_step": 0.25,
            "mini_size_spot": 30,
            "percentile_clip": 99.9,
        }
    },
    "measurement": {
        "resolution_3d": [0.2, 0.08, 0.08],
        "resolution_2d": [0.08, 0.08],
    },
    "ml": {
        "default_backend": "classical",
        "nnunet": {
            "dataset_id": 1,
            "configuration": "2d",
            "folds": "all",
        },
        "cellpose": {
            "model_path": None,
            "diameter": None,
            "do_3d": True,
            "anisotropy": 2.5,
        },
        "augmentation": {
            "n_augmentations": 10,
            "rotation_range": [-180, 180],
            "intensity_scale_range": [0.7, 1.3],
            "elastic_sigma": 5.0,
            "elastic_alpha": 80.0,
        },
    },
}


def load_config(path: str = None) -> Dict[str, Any]:
    """Load config.yaml; fall back to hardcoded defaults if the file is absent or PyYAML is not installed."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except (ImportError, FileNotFoundError):
        return _DEFAULT_CONFIG
