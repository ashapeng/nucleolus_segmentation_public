"""
Preference-Based Bayesian Optimisation for GC segmentation parameters.

Each round:
  1. ask()  → generates K candidate parameter sets
  2. run()  → runs gc_segment() on a representative cell for each candidate
  3. show() → web_viewer serves max-projection overlays to the user's phone
  4. tell() → user taps the best result; GP surrogate is updated with that preference
  5. Repeat until converged or max_rounds reached
  6. write_best() → saves optimal params back to config.yaml

Usage:
    from ml.active_learning import ActiveLearner
    learner = ActiveLearner(sample)
    learner.run_rounds(n_rounds=10, n_candidates=4)
    learner.write_best("config.yaml")
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

# Each entry: (name, low, high, dtype)  — all floats for GP compatibility
_SPACE_DEFS: List[Tuple[str, float, float, str]] = [
    ("sigma",          0.25, 2.0,  "float"),
    ("local_adjust",   0.90, 1.20, "float"),
    ("log_sigma_min",  1.5,  4.0,  "float"),
    ("log_sigma_max",  3.0,  6.0,  "float"),
    ("mini_size_spot", 5.0,  80.0, "float"),   # stored as float, cast to int when used
]

PARAM_NAMES = [d[0] for d in _SPACE_DEFS]


def _skopt_space():
    """Return scikit-optimize Space object."""
    from skopt.space import Real
    return [Real(low, high, name=name) for name, low, high, _ in _SPACE_DEFS]


def _params_to_config(params: List[float]) -> Dict[str, Any]:
    """Convert a flat parameter vector to a config dict accepted by gc_segment()."""
    p = dict(zip(PARAM_NAMES, params))
    return {
        "segmentation": {
            "gc_segment": {
                "sigma":           p["sigma"],
                "local_adjust":    p["local_adjust"],
                "log_sigma_min":   p["log_sigma_min"],
                "log_sigma_max":   p["log_sigma_max"],
                "log_sigma_step":  0.25,
                "mini_size_spot":  int(round(p["mini_size_spot"])),
                "mini_size_otsu":  1000,
                "percentile_clip": 99.9,
            }
        }
    }


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _max_project_overlay(raw_image: np.ndarray, gc_mask: np.ndarray) -> np.ndarray:
    """
    Return an (H, W, 3) uint8 RGB image:
      - greyscale max-projection of LPD-7 channel (ch2) as base
      - GC mask overlay in red
    """
    lpd7 = raw_image[..., 2].astype(np.float32)
    proj = lpd7.max(axis=0)                         # (Y, X)
    lo, hi = proj.min(), proj.max()
    if hi > lo:
        proj = (proj - lo) / (hi - lo)
    else:
        proj = np.zeros_like(proj)
    grey = (proj * 255).astype(np.uint8)

    mask_proj = (gc_mask.max(axis=0) > 0)           # (Y, X) bool

    rgb = np.stack([grey, grey, grey], axis=-1)
    rgb[mask_proj, 0] = 255
    rgb[mask_proj, 1] = 0
    rgb[mask_proj, 2] = 0

    return rgb


def _encode_png_b64(rgb: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 array as a base-64 PNG string."""
    import base64
    import io
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Fallback: write raw PPM wrapped in base64 (viewable in most browsers)
        h, w = rgb.shape[:2]
        header = f"P6\n{w} {h}\n255\n".encode()
        buf = header + rgb.astype(np.uint8).tobytes()
        return base64.b64encode(buf).decode()


# ---------------------------------------------------------------------------
# Core learner
# ---------------------------------------------------------------------------

class ActiveLearner:
    """
    Human-in-the-loop Bayesian optimiser.

    Parameters
    ----------
    sample : CellSample
        Representative labelled cell used for candidate evaluation.
    n_initial_random : int
        Number of random evaluations before GP kicks in (warm-up).
    random_state : int
        RNG seed for reproducibility.
    """

    def __init__(self, sample, n_initial_random: int = 5, random_state: int = 42):
        self.sample = sample
        self.random_state = random_state
        self.n_initial_random = n_initial_random

        self._xs: List[List[float]] = []    # observed parameter vectors
        self._ys: List[float] = []          # observed scores (1 = best, 0 = others)

        try:
            from skopt import Optimizer
            self._opt = Optimizer(
                dimensions=_skopt_space(),
                base_estimator="GP",
                n_initial_points=n_initial_random,
                random_state=random_state,
                acq_func="EI",
            )
        except ImportError:
            raise ImportError(
                "scikit-optimize is required for active learning: "
                "pip install scikit-optimize"
            )

        self._best_params: Optional[List[float]] = None
        self._round = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, n_candidates: int = 4) -> List[List[float]]:
        """Generate n_candidates parameter sets to evaluate."""
        return self._opt.ask(n_points=n_candidates)

    def tell(self, candidates: List[List[float]], winner_index: int) -> None:
        """
        Record user preference.

        All candidates receive score=0 except the winner (score=1 → -1 for
        minimisation).  The GP is updated after every round.
        """
        for i, params in enumerate(candidates):
            score = -1.0 if i == winner_index else 0.0
            self._opt.tell(params, score)
            self._xs.append(params)
            self._ys.append(-score)         # store 1.0 for winner

        winner = candidates[winner_index]
        if self._best_params is None or self._ys[-len(candidates) + winner_index] >= max(self._ys[:-len(candidates)] or [0]):
            self._best_params = winner

        self._round += 1
        logger.info(
            "Round %d complete. Winner params: %s",
            self._round,
            dict(zip(PARAM_NAMES, [round(v, 4) for v in winner])),
        )

    def evaluate_candidates(
        self, candidates: List[List[float]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run gc_segment() for each candidate config.

        Returns list of (gc_mask, overlay_rgb) tuples.
        """
        import seg_util as su
        results = []
        for i, params in enumerate(candidates):
            config = _params_to_config(params)
            logger.info("Evaluating candidate %d/%d: %s", i + 1, len(candidates),
                        {k: round(v, 3) for k, v in zip(PARAM_NAMES, params)})
            try:
                gc_mask, _, _ = su.gc_segment(
                    self.sample.raw,
                    self.sample.nucleus_mask,
                    config=config,
                    backend="classical",
                )
                overlay = _max_project_overlay(self.sample.raw, gc_mask)
            except Exception as exc:
                logger.warning("Candidate %d failed: %s", i + 1, exc)
                gc_mask = np.zeros_like(self.sample.nucleus_mask)
                overlay = _max_project_overlay(self.sample.raw, gc_mask)
            results.append((gc_mask, overlay))
        return results

    def best_config(self) -> Optional[Dict[str, Any]]:
        """Return the best config dict found so far, or None."""
        if self._best_params is None:
            return None
        return _params_to_config(self._best_params)

    def write_best(self, config_path: str = "config.yaml") -> None:
        """Write the best-found parameters back to config.yaml."""
        if self._best_params is None:
            logger.warning("No rounds completed; nothing to write.")
            return
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required: pip install pyyaml")

        best = _params_to_config(self._best_params)["segmentation"]["gc_segment"]

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        cfg.setdefault("segmentation", {}).setdefault("gc_segment", {}).update(best)

        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        logger.info("Best parameters written to %s", config_path)
        logger.info("  %s", best)

    # ------------------------------------------------------------------
    # Convenience: run full loop programmatically (non-interactive)
    # ------------------------------------------------------------------

    def run_rounds(
        self,
        n_rounds: int = 10,
        n_candidates: int = 4,
        viewer=None,
    ) -> None:
        """
        Run n_rounds of active learning.

        Parameters
        ----------
        viewer : web_viewer.Viewer, optional
            If provided, each round's overlays are served via the Flask viewer
            and the loop blocks until the user submits a choice.
            If None, falls back to automatic selection using ground-truth Dice
            (useful for offline testing without a browser).
        """
        from eval_metrics import dice_coefficient

        for round_idx in range(n_rounds):
            logger.info("=== Active learning round %d / %d ===", round_idx + 1, n_rounds)
            candidates = self.ask(n_candidates)
            eval_results = self.evaluate_candidates(candidates)

            overlays_b64 = [_encode_png_b64(overlay) for _, overlay in eval_results]
            param_labels = [
                {k: round(v, 3) for k, v in zip(PARAM_NAMES, c)}
                for c in candidates
            ]

            if viewer is not None:
                winner_index = viewer.present_round(
                    round_idx + 1, n_rounds, overlays_b64, param_labels
                )
            else:
                # Offline fallback: pick candidate with best Dice vs ground truth
                dice_scores = [
                    dice_coefficient(gc_mask, self.sample.gc_mask)
                    for gc_mask, _ in eval_results
                ]
                winner_index = int(np.argmax(dice_scores))
                logger.info(
                    "Offline mode — Dice scores: %s → winner index %d",
                    [round(d, 3) for d in dice_scores], winner_index,
                )

            self.tell(candidates, winner_index)

            if viewer is not None and viewer.stop_requested:
                logger.info("User requested stop after round %d.", round_idx + 1)
                break

        logger.info("Active learning complete after %d rounds.", self._round)
        if self._best_params is not None:
            logger.info(
                "Best parameters: %s",
                {k: round(v, 4) for k, v in zip(PARAM_NAMES, self._best_params)},
            )
