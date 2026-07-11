"""CLI entry points for the nucleolus pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pipeline", description="Nucleolus GC pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory", help="List and validate experiment cell folders")
    inv.add_argument("--root", required=True, help="Experiment root directory")
    inv.add_argument("--json", action="store_true", help="Print manifest JSON")

    run = sub.add_parser("run", help="Run inventory → segment → QC → measure → report")
    run.add_argument("--root", required=True, help="Experiment root directory")
    run.add_argument("--goal", default="", help="Natural-language goal (used by LLM mode)")
    run.add_argument(
        "--no-llm",
        action="store_true",
        default=True,
        help="Deterministic plan (default)",
    )
    run.add_argument(
        "--llm",
        action="store_true",
        help="Enable optional LLM orchestrator (requires OPENAI_API_KEY)",
    )
    run.add_argument("--stage", default=None, help="Filter to larval stage L1–L4")
    run.add_argument("--max-cells", type=int, default=None, help="Limit number of cells")
    run.add_argument("--with-intensity", action="store_true", help="Also run intensity metrics")
    run.add_argument(
        "--with-easy-adopt",
        action="store_true",
        help="Attach Easy-adopt trust reports for each cell if installed",
    )
    run.add_argument(
        "--allow-ml-backend",
        action="store_true",
        help=(
            "Allow ml.default_backend nnunet|cellpose only after a non-RED Easy-adopt "
            "trust probe (never silent swap from classical)"
        ),
    )
    run.add_argument("--runs-base", default="runs", help="Directory for run artifacts")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "inventory":
        from pipeline.inventory import inventory_experiment

        manifest = inventory_experiment(args.root)
        if args.json:
            print(json.dumps(manifest.to_dict(), indent=2))
        else:
            print(f"Root: {manifest.root}")
            print(f"Included: {len(manifest.cells_included)}")
            print(f"Excluded: {len(manifest.cells_excluded)}")
            for cell in manifest.cells_included:
                print(f"  + {cell.cell_id} stage={cell.stage}")
            for cell in manifest.cells_excluded:
                print(f"  - {cell.cell_id} missing={cell.missing_files}")
        return 0

    if args.command == "run":
        use_llm = bool(args.llm)
        if use_llm:
            from agent.orchestrator import run_goal

            run_dir = run_goal(
                goal=args.goal or "Segment all cells, QC, measure shapes, write a report",
                root=args.root,
                no_llm=False,
                stage=args.stage,
                max_cells=args.max_cells,
                with_intensity=args.with_intensity,
                with_easy_adopt=args.with_easy_adopt,
                allow_ml_backend=args.allow_ml_backend,
                runs_base=args.runs_base,
            )
        else:
            from pipeline.runner import run_deterministic

            run_dir = run_deterministic(
                args.root,
                stage=args.stage,
                max_cells=args.max_cells,
                with_intensity=args.with_intensity,
                with_easy_adopt=args.with_easy_adopt,
                allow_ml_backend=args.allow_ml_backend,
                runs_base=args.runs_base,
                goal=args.goal,
            )
        print(str(run_dir))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
