# Agentic AI Nucleolus Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the nucleolus GC pipeline agent-operable end-to-end—inventory → segment → QC → measure → report—without replacing classical CV, then add a thin tool-calling orchestrator.

**Architecture:** Approach A from `docs/superpowers/specs/2026-07-10-agentic-ai-pipeline-design.md`. New `pipeline/` package wraps existing `seg_util` / `measure_util` / `eval_metrics`. New `agent/` package adds optional LLM orchestration over the same tools. CLI is the primary interface; MCP is deferred.

**Tech stack:** Python 3.9+, existing numpy/skimage/pandas, stdlib `argparse` CLI, pytest. Agent layer: OpenAI-compatible tool calling via env (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`); deterministic `--no-llm` path always available.

## Locked decisions (from design §9)

| Decision | Choice |
|----------|--------|
| Interface | CLI + Python API first; MCP later (out of v1 plan) |
| LLM | Optional API provider; every run also works with `--no-llm` fixed plan |
| Collection | Offline experiment folders only |
| QC fail | Bounded auto-tune (max 3 retries), then **skip cell** and continue; log RED |
| Artifacts | Git-ignored `runs/<run_id>/` |

## Target layout

```text
pipeline/
  __init__.py
  types.py           # dataclasses: Manifest, CellRecord, QCReport, ...
  inventory.py       # inventory_experiment, validate_cell
  segment.py         # run_gc_segment, batch_gc_segment
  qc.py              # qc_masks, bounded param suggestions
  measure.py         # measure_shapes wrapper, batch_intensity
  report.py          # write_run_report, plot_and_export
  run_store.py       # create_run_dir, save_manifest, save_trace
  cli.py             # python -m pipeline ...
  __main__.py
agent/
  __init__.py
  tools.py           # tool schemas + dispatch to pipeline.*
  orchestrator.py    # plan + tool loop (or deterministic planner)
  prompts.py         # system prompt with biology/config bounds
tests/
  test_inventory.py
  test_pipeline_segment.py
  test_qc.py
  test_pipeline_measure.py
  test_report.py
  test_cli.py
  test_agent_tools.py
runs/                # gitignored
docs/superpowers/specs/2026-07-10-agentic-ai-pipeline-design.md
docs/superpowers/plans/2026-07-10-agentic-ai-pipeline.md
```

---

## Phase 0 — Agent-ready API (no LLM)

### Task 1: Scaffold package + ignore `runs/`

**Files:**
- Create: `pipeline/__init__.py`, `pipeline/types.py`, `pipeline/__main__.py`
- Create: `agent/__init__.py`
- Modify: `.gitignore`
- Modify: `installer.txt` (add `pyyaml` if not implied; keep napari optional for CLI)

- [ ] **Step 1:** Append to `.gitignore`:

```gitignore
runs/
.env
*.egg-info/
.venv/
```

- [ ] **Step 2:** Create `pipeline/types.py` with dataclasses:

```python
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

@dataclass
class CellRecord:
    cell_id: str          # e.g. "20220304_L1/10_2"
    path: str
    stage: Optional[str]
    valid: bool
    missing_files: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

@dataclass
class Manifest:
    root: str
    resolution_3d: List[float]
    cells_included: List[CellRecord]
    cells_excluded: List[CellRecord]
    config_snapshot: Dict[str, Any]

@dataclass
class QCReport:
    cell_id: str
    status: str           # GREEN | AMBER | RED
    gc_fraction: float
    empty_gc: bool
    outside_nucleus_fraction: float
    messages: List[str] = field(default_factory=list)

@dataclass
class MaskPaths:
    gc: str
    holes: str
    hole_filled: str
```

- [ ] **Step 3:** Create empty `pipeline/__init__.py` exporting version `__version__ = "0.1.0"`.
- [ ] **Step 4:** Create `pipeline/__main__.py` that calls `pipeline.cli.main` (stub `cli.py` with `def main(): raise SystemExit("not implemented")` temporarily, or skip until Task 7).
- [ ] **Step 5:** Commit:

```bash
git add .gitignore pipeline agent
git commit -m "Scaffold pipeline and agent packages for agentic runs"
```

---

### Task 2: Inventory and validate cell folders

**Files:**
- Create: `pipeline/inventory.py`
- Create: `tests/test_inventory.py`
- Use: `measure_util.extract_stage`, `config_loader.load_config`

**Required files per cell:** `Composite_stack.tif`, `nuclei_mask.tif`, `background_mask.tif`

- [ ] **Step 1: Write failing tests** in `tests/test_inventory.py`:

```python
def test_validate_cell_missing_files(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    cell.mkdir(parents=True)
    (cell / "Composite_stack.tif").write_bytes(b"x")  # incomplete
    from pipeline.inventory import validate_cell
    result = validate_cell(str(cell), experiment_set="20220304_L1")
    assert result.valid is False
    assert "nuclei_mask.tif" in result.missing_files

def test_inventory_includes_valid_cells(tmp_path):
    # create one valid cell with three empty placeholder tiffs
    ...
    from pipeline.inventory import inventory_experiment
    manifest = inventory_experiment(str(tmp_path))
    assert len(manifest.cells_included) == 1
    assert manifest.cells_included[0].stage == "L1"
```

- [ ] **Step 2: Run tests — expect fail**

```bash
python -m pytest tests/test_inventory.py -v
```

- [ ] **Step 3: Implement** `inventory_experiment(root) -> Manifest` and `validate_cell(cell_dir, experiment_set=None) -> CellRecord`:
  - Walk one level of experiment sets, one level of cells (same as notebooks).
  - `cell_id = f"{experiment_set}/{cell_name}"` using `/` (not `os.sep`) for portability in reports.
  - Call `extract_stage(experiment_set)` (or cell_id); on failure set `stage=None` and add note.
  - Snapshot microscope + segmentation config via `load_config()`.

- [ ] **Step 4: Run tests — expect pass**

```bash
python -m pytest tests/test_inventory.py -v
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/inventory.py pipeline/types.py tests/test_inventory.py
git commit -m "Add experiment inventory and cell folder validation"
```

---

### Task 3: Batch GC segmentation API

**Files:**
- Create: `pipeline/segment.py`
- Create: `tests/test_pipeline_segment.py`
- Use: `seg_util.gc_segment`, `Import_Functions.import_imgs`, `imageio` or `skimage.io`

**Note:** Notebook currently calls `gc_segment(raw, nucleus, background_mask, ...)` but library signature is `(raw_image, nucleus_mask, sigma=..., local_adjust_for_GC=..., config=...)`. New API must match the **library**, not the notebook bug.

- [ ] **Step 1: Write failing tests** using synthetic arrays (reuse `_sphere_mask` pattern from `tests/test_seg_util.py`):
  - `test_run_gc_segment_writes_three_masks(tmp_path)` — write fake multi-channel stack + nucleus mask; call `run_gc_segment`; assert `gc.tif`, `holes.tif`, `hole_filled.tif` exist and are uint8.
  - `test_batch_gc_segment_skips_invalid(tmp_path)` — one valid, one missing files; batch returns results only for valid.

- [ ] **Step 2: Run tests — expect fail**

```bash
python -m pytest tests/test_pipeline_segment.py -v
```

- [ ] **Step 3: Implement**

```python
def run_gc_segment(cell_dir: str, config_overrides: dict | None = None, config=None) -> MaskPaths:
    # load Composite_stack.tif + nuclei_mask.tif
    # merge overrides into config["segmentation"]["gc_segment"]
    # gc, holes, hole_filled = gc_segment(...)
    # write with skimage.io.imsave / imageio.volwrite
    # return MaskPaths

def batch_gc_segment(manifest: Manifest, config_overrides=None) -> list[tuple[CellRecord, MaskPaths | None, str | None]]:
    # for each included cell: try run_gc_segment; on exception record error string
```

- [ ] **Step 4: Run tests — expect pass**
- [ ] **Step 5: Commit**

```bash
git add pipeline/segment.py tests/test_pipeline_segment.py
git commit -m "Add batch GC segmentation API wrapping gc_segment"
```

---

### Task 4: QC heuristics for masks

**Files:**
- Create: `pipeline/qc.py`
- Create: `tests/test_qc.py`
- Use: `eval_metrics` only if a reference mask exists (optional); primary checks are absolute heuristics from Easy-adopt `nucleolus_gc` contract

**Heuristics (v1):**
- `empty_gc`: foreground pixels == 0 → RED
- `gc_fraction`: `sum(gc>0) / sum(nucleus>0)`; GREEN if in `[0.05, 0.40]`, AMBER if in `[0.02, 0.05)` or `(0.40, 0.60]`, else RED
- `outside_nucleus_fraction`: `sum((gc>0) & (nucleus==0)) / max(sum(gc>0),1)`; RED if > 0.05, AMBER if > 0.01
- Overall status = worst of component statuses

- [ ] **Step 1: Write failing tests** for empty → RED, good fraction → GREEN, outside nucleus → RED
- [ ] **Step 2: Run — expect fail**
- [ ] **Step 3: Implement** `qc_masks(cell_dir) -> QCReport` loading `gc.tif` + `nuclei_mask.tif`
- [ ] **Step 4: Implement** `suggest_param_overrides(report: QCReport, attempt: int) -> dict | None`:
  - attempt 0→1: if fraction low, decrease `local_adjust` by 0.05 (floor 0.9); if high, increase by 0.05 (ceil 1.3)
  - attempt ≥3 or no suggestion → `None` (caller skips)
- [ ] **Step 5: Run — expect pass**
- [ ] **Step 6: Commit**

```bash
git add pipeline/qc.py tests/test_qc.py
git commit -m "Add GC mask QC heuristics and bounded param suggestions"
```

---

### Task 5: Measurement wrappers + intensity batch

**Files:**
- Create: `pipeline/measure.py`
- Create: `tests/test_pipeline_measure.py`
- Use: `measure_util.batch_measure_shape`, `group_gc_measure_df`, `concentration_gc`, `coefficient_of_variances`, `overlap_3channel`

- [ ] **Step 1: Write failing tests**
  - `measure_shapes` on tmp tree with a simple mask returns non-empty list of DataFrames / concatenated DataFrame.
  - Prefer testing `measure_shapes(master_folder, mask_name="gc.tif") -> pd.DataFrame` that concatenates `batch_measure_shape` results and adds a `stage` column via `extract_stage`.

- [ ] **Step 2: Implement thin wrappers** (do not rewrite morphometry math):

```python
DEFAULT_SHAPE_PARAMS = ["cell_id", "obj_id", "surface_area", "volume",
    "surface_to_volume_ratio", "sphericity", "aspect_ratio", "solidity"]

def measure_shapes(master_folder: str, mask_name: str = "gc.tif", ...) -> pd.DataFrame: ...

def measure_intensity_cell(cell_dir: str, cell_id: str) -> pd.DataFrame:
    # load raw, masks, bg-subtract, concentration_gc + coefficient_of_variances
    # keep logic faithful to intensity notebook; extract carefully

def measure_intensity_batch(manifest: Manifest) -> pd.DataFrame: ...
```

- [ ] **Step 3:** If intensity extraction is large, split: ship `measure_shapes` in this task; intensity in Task 5b same PR/commit series.
- [ ] **Step 4: Tests pass + commit**

```bash
git commit -m "Add measurement wrappers for shapes and intensity batch"
```

---

### Task 6: Run store + markdown report (template)

**Files:**
- Create: `pipeline/run_store.py`, `pipeline/report.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Implement** `create_run_dir(base="runs") -> Path` using UTC timestamp id `YYYYMMDDTHHMMSSZ`.
- [ ] **Step 2: Save** `manifest.json`, `qc.jsonl`, `shapes.csv`, `intensity.csv`, `trace.jsonl`, figures under `figures/`.
- [ ] **Step 3: `write_run_report(...)`** writes `report.md` with sections: Goal/Mode, Config snapshot, Inventory counts, QC summary (GREEN/AMBER/RED counts + RED cell list), Shape summary by stage (mean volume), Intensity highlights if present, Artifacts paths.
- [ ] **Step 4: No LLM in this task** — deterministic templates only (`str.format` / f-strings).
- [ ] **Step 5: Tests** assert report file exists and contains `## QC summary`.
- [ ] **Step 6: Commit**

```bash
git commit -m "Add run artifact store and deterministic markdown reports"
```

---

### Task 7: Deterministic CLI end-to-end

**Files:**
- Create: `pipeline/cli.py`
- Modify: `pipeline/__main__.py`
- Create: `tests/test_cli.py`
- Modify: `README.md` (short “Programmatic / agentic runs” section)

**CLI:**

```bash
python -m pipeline inventory --root test_image
python -m pipeline run --root test_image --no-llm [--stage L2] [--max-cells N]
```

`run --no-llm` fixed plan:
1. inventory → filter by `--stage` if set
2. for each cell: segment → qc → optional retry with `suggest_param_overrides` (max 3) → on final RED, skip measurement for that cell
3. measure_shapes on cells that have GREEN/AMBER masks (or measure from disk for successful writes)
4. optional intensity if `--with-intensity`
5. write report under `runs/<id>/`

- [ ] **Step 1: Write CLI test** using `tmp_path` and `subprocess` or `CliRunner`-style `main(argv)` that returns exit code 0 on empty-but-valid tiny fixture.
- [ ] **Step 2: Implement `cli.main(argv: list[str] | None = None) -> int`**
- [ ] **Step 3: Document in README**
- [ ] **Step 4: Manual smoke** (if `test_image` present unzipped):

```bash
python -m pipeline run --root test_image --no-llm --max-cells 1
```

- [ ] **Step 5: Commit**

```bash
git commit -m "Add deterministic CLI for inventory and full pipeline runs"
```

**Phase 0 exit criteria:** `python -m pipeline run --root <data> --no-llm` produces `runs/<id>/report.md` without notebooks. Existing `tests/test_seg_util.py` and `tests/test_measure_util.py` still pass.

---

## Phase 1 — Single orchestrator agent

### Task 8: Tool registry wrapping pipeline APIs

**Files:**
- Create: `agent/tools.py`
- Create: `tests/test_agent_tools.py`

- [ ] **Step 1: Define** `TOOL_SPECS: list[dict]` (OpenAI function-calling JSON schema) for:
  - `inventory_experiment`
  - `run_gc_segment`
  - `qc_masks`
  - `measure_shapes`
  - `measure_intensity_batch`
  - `write_run_report` (or `finalize_run`)
- [ ] **Step 2: Implement** `execute_tool(name: str, arguments: dict, context: RunContext) -> dict` returning JSON-serializable results (paths, status, summaries—not giant arrays).
- [ ] **Step 3: Unit test** dispatch for `inventory_experiment` on tmp fixture.
- [ ] **Step 4: Commit**

```bash
git commit -m "Expose pipeline functions as agent tool schemas"
```

---

### Task 9: Orchestrator with optional LLM

**Files:**
- Create: `agent/prompts.py`, `agent/orchestrator.py`
- Modify: `pipeline/cli.py` to add `run` without `--no-llm`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: `prompts.py`** — system prompt stating: use only provided tools; do not invent filters; param bounds from config; prefer classical `run_gc_segment`; skip RED cells after retries.
- [ ] **Step 2: `orchestrator.run_goal(goal: str, root: str, *, no_llm: bool, ...) -> Path`**
  - If `no_llm`: call the same fixed plan as CLI Task 7.
  - If LLM: loop tool calls until `finalize_run` or max steps (e.g. 40); write each call to `trace.jsonl`.
- [ ] **Step 3: Provider adapter** — start with one: OpenAI-compatible `chat.completions` with `tools=`. If no API key, exit with clear error telling user to pass `--no-llm`.
- [ ] **Step 4: Test** deterministic path only in CI (no network). Optional `@pytest.mark.integration` for live LLM.
- [ ] **Step 5: Commit**

```bash
git commit -m "Add optional LLM orchestrator over pipeline tools"
```

**Phase 1 exit criteria:**  
`python -m pipeline run --root <data> --goal "Segment all cells and write a report"` works with API key; `--no-llm` remains default for CI.

---

## Phase 2 — QC retries + Easy-adopt gate

### Task 10: Wire auto-tune loop into segment+QC

**Files:**
- Modify: `pipeline/segment.py` or add `pipeline/segment_with_qc.py`
- Modify: `agent/tools.py` (expose `segment_with_qc`)
- Tests: extend `tests/test_qc.py`

- [ ] **Step 1:** `segment_with_qc(cell_dir, max_attempts=3) -> tuple[MaskPaths | None, list[QCReport]]`
- [ ] **Step 2:** Ensure CLI and orchestrator use this instead of bare `run_gc_segment`.
- [ ] **Step 3: Commit**

```bash
git commit -m "Add bounded segment-QC retry loop before skipping cells"
```

---

### Task 11: Easy-adopt optional tool

**Files:**
- Create: `pipeline/trust.py`
- Modify: `agent/tools.py`, `installer.txt` (note optional `easy-adopt`)
- Create: `tests/test_trust.py`

- [ ] **Step 1:** `run_easy_adopt(cell_dir, tool: str, structure: str = "nucleolus_gc") -> dict` subprocess or Python API if available; if package missing, return `{"status": "SKIPPED", "reason": "easy-adopt not installed"}`.
- [ ] **Step 2:** Policy: orchestrator may only suggest alternate segmenters if trust status is not RED; **never** auto-replace `gc_segment` in v1—only attach trust report to `report.md`.
- [ ] **Step 3: Commit**

```bash
git commit -m "Integrate optional Easy-adopt trust reports into pipeline runs"
```

**Phase 2 exit criteria:** Failed QC triggers ≤3 param retries; Easy-adopt results appear in report when installed; RED trust never swaps segmenter silently.

---

## Phase 3 — Stretch (separate follow-up)

Not required to close the first implementation PR series:

- LLM-written narrative paragraphs grounded only in CSV stats (cite numbers).
- Cross-run comparison (`runs/a` vs `runs/b`).
- Optional VLM mid-Z QC (assistive).
- MCP server wrapping `agent/tools.py`.

---

## Testing strategy

| Layer | Command |
|-------|---------|
| Unit | `python -m pytest tests/ -v --ignore=aicssegmentation` |
| Existing | Keep `tests/test_seg_util.py`, `tests/test_measure_util.py` green |
| Smoke | `python -m pipeline run --root test_image --no-llm --max-cells 1` |
| Agent | Deterministic tests only in default CI |

Prefer synthetic TIFF fixtures in `tmp_path` over committing large microscopy data.

---

## Implementation order (checklist)

- [ ] Task 1 — Scaffold
- [ ] Task 2 — Inventory
- [ ] Task 3 — Batch segment
- [ ] Task 4 — QC
- [ ] Task 5 — Measure wrappers
- [ ] Task 6 — Run store + report
- [ ] Task 7 — Deterministic CLI (**Phase 0 done**)
- [ ] Task 8 — Agent tools
- [ ] Task 9 — Orchestrator (**Phase 1 done**)
- [ ] Task 10 — QC retry loop
- [ ] Task 11 — Easy-adopt (**Phase 2 done**)

---

## Self-review

1. **Spec coverage:** Collection, processing, QC, analysis, results, Easy-adopt gate, CLI, optional LLM — mapped to Tasks 2–11. MCP/VLM explicitly deferred to Phase 3.
2. **Placeholder scan:** No TBDs; open decisions locked at top.
3. **Type consistency:** `Manifest` / `QCReport` / `MaskPaths` defined in Task 1 and reused.
4. **Notebook debt:** Task 3 documents signature mismatch and standardizes on library API.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-10-agentic-ai-pipeline.md`.

**Two execution options:**

1. **Subagent-driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline** — execute tasks in this session with checkpoints  

Which do you prefer?
