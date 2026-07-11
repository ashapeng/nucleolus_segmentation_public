"""
Human-in-the-loop active learning for GC segmentation parameters.

Launches a mobile-friendly web gallery on your local network.  Open the
printed URL on any phone on the same WiFi and pick the best segmentation
image each round.  The Bayesian optimiser learns from your choices and
converges to the optimal parameter set, which is written back to config.yaml.

Usage:
    # Basic: pick a representative cell, run 10 rounds of 4 candidates each
    python run_active_learning.py --data test_image/

    # More rounds for finer convergence:
    python run_active_learning.py --data test_image/ --rounds 15 --candidates 6

    # Offline mode (uses Dice vs ground truth instead of human picks):
    python run_active_learning.py --data test_image/ --offline

    # Custom port:
    python run_active_learning.py --data test_image/ --port 8080

Requirements:
    pip install scikit-optimize flask pillow pyyaml
    (pillow is optional — falls back to raw PPM encoding)
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(__file__))


def _print_qr(url: str) -> None:
    """Print a QR code for the URL to the terminal (requires qrcode[terminal])."""
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        print(f"\n  Scan the QR code or open:  {url}\n")
    except ImportError:
        print(f"\n  Open this URL on your phone:  {url}")
        print("  (Install qrcode[terminal] to get a QR code: pip install qrcode[terminal])\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human-in-the-loop active learning for GC segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data", default="test_image",
        help="Path to master_folder containing labelled cells (default: test_image/)",
    )
    parser.add_argument(
        "--cell", default=None,
        help="Specific cell_id to use as the representative cell.  "
             "Defaults to the first loaded cell.",
    )
    parser.add_argument(
        "--rounds", type=int, default=10,
        help="Number of active learning rounds (default: 10)",
    )
    parser.add_argument(
        "--candidates", type=int, default=4,
        help="Number of parameter candidates per round (default: 4)",
    )
    parser.add_argument(
        "--port", type=int, default=5050,
        help="Port for the web viewer (default: 5050)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml that will be updated with best params (default: config.yaml)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Offline mode: use Dice vs ground truth instead of human picks "
             "(useful for testing without a phone/browser)",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Load samples
    # ---------------------------------------------------------------
    from ml.data_loader import load_all_samples

    logger.info("Loading samples from: %s", args.data)
    samples = load_all_samples(args.data)
    if not samples:
        logger.error("No samples found in %s", args.data)
        sys.exit(1)

    logger.info("Loaded %d cells", len(samples))

    if args.cell:
        matching = [s for s in samples if args.cell in s.cell_id]
        if not matching:
            logger.error(
                "Cell %r not found. Available: %s",
                args.cell, [s.cell_id for s in samples],
            )
            sys.exit(1)
        representative = matching[0]
    else:
        representative = samples[0]

    logger.info("Representative cell: %s (stage %s)", representative.cell_id, representative.stage)

    # ---------------------------------------------------------------
    # Set up active learner
    # ---------------------------------------------------------------
    from ml.active_learning import ActiveLearner

    learner = ActiveLearner(
        sample=representative,
        n_initial_random=min(5, args.rounds),
    )

    # ---------------------------------------------------------------
    # Set up viewer (unless offline mode)
    # ---------------------------------------------------------------
    viewer = None
    if not args.offline:
        from ml.web_viewer import Viewer

        viewer = Viewer(host="0.0.0.0", port=args.port)
        viewer.start()

        print("\n" + "=" * 60)
        print("  Active Learning — Nucleolus GC Segmentation")
        print("=" * 60)
        print(f"  Rounds:     {args.rounds}")
        print(f"  Candidates: {args.candidates} per round")
        print(f"  Cell:       {representative.cell_id}")
        print()
        _print_qr(viewer.lan_url)
        print("  Keep this terminal open while evaluating on your phone.")
        print("=" * 60 + "\n")
    else:
        logger.info("Offline mode: Dice-based winner selection (no browser needed).")

    # ---------------------------------------------------------------
    # Run rounds
    # ---------------------------------------------------------------
    try:
        learner.run_rounds(
            n_rounds=args.rounds,
            n_candidates=args.candidates,
            viewer=viewer,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    # ---------------------------------------------------------------
    # Save best parameters
    # ---------------------------------------------------------------
    best = learner.best_config()
    if best is None:
        logger.warning("No rounds completed. Config not updated.")
        sys.exit(0)

    gc_params = best["segmentation"]["gc_segment"]
    print("\n" + "=" * 60)
    print("  Optimal parameters found:")
    print("=" * 60)
    for k, v in gc_params.items():
        print(f"  {k:<20} {v}")
    print("=" * 60 + "\n")

    learner.write_best(args.config)
    print(f"Config updated: {os.path.abspath(args.config)}")

    if viewer is not None:
        viewer.stop()
        print("Web viewer stopped. You can close the browser tab.")


if __name__ == "__main__":
    main()
