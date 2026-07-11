import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.tools import RunContext, TOOL_SPECS, execute_tool


def test_execute_tool_inventory(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    cell.mkdir(parents=True)
    for name in ("Composite_stack.tif", "nuclei_mask.tif", "background_mask.tif"):
        (cell / name).write_bytes(b"x")
    ctx = RunContext(root=str(tmp_path), runs_base=str(tmp_path / "runs"))
    result = execute_tool("inventory_experiment", {"root": str(tmp_path)}, ctx)
    assert result["included"] == 1
    assert ctx.manifest is not None
    assert ctx.run_dir is not None
    assert (ctx.run_dir / "manifest.json").is_file()


def test_compare_runs_tool(tmp_path):
    import pandas as pd

    from pipeline.run_store import save_manifest, save_qc_reports
    from pipeline.types import CellRecord, Manifest, QCReport

    def seed(name, vol):
        run_dir = tmp_path / "artifact_runs" / name
        run_dir.mkdir(parents=True)
        save_manifest(
            run_dir,
            Manifest(
                root=str(tmp_path),
                resolution_3d=[0.2, 0.08, 0.08],
                cells_included=[CellRecord("c/1", str(tmp_path), "L1", True)],
                cells_excluded=[],
                config_snapshot={},
            ),
        )
        save_qc_reports(run_dir, [QCReport("c/1", "GREEN", 0.1, False, 0.0, [])])
        pd.DataFrame({"cell_id": ["c/1"], "stage": ["L1"], "volume": [vol]}).to_csv(
            run_dir / "shapes.csv", index=False
        )
        return run_dir

    run_a = seed("a", 10.0)
    run_b = seed("b", 11.0)
    names = {spec["function"]["name"] for spec in TOOL_SPECS}
    assert "compare_runs" in names
    ctx = RunContext(root=str(tmp_path), runs_base=str(tmp_path / "agent_runs"))
    result = execute_tool(
        "compare_runs",
        {"run_a": str(run_a), "run_b": str(run_b)},
        ctx,
    )
    assert "report" in result
    assert result["summary"]["delta_mean_volume"] == 1.0
