import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.tools import RunContext, execute_tool


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
