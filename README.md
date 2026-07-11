# Nucleolus Segmentation

A specialized image segmentation tool for nucleolus analysis, developed during my time at Weber Lab at McGill University. This repository is built on the basis of AllenCell segmenter (Classic Image Segmentation).

# Goals:
1. Segments the nucleolus with markers of granular component (GC).

2. Characterizes features of nucleoli: size, number, and shape.

3. Analyzes the correlation between different nucleolar proteins within the nucleolar mask.

# Installation

## Prerequisites
- Python 3.9
- Anaconda (recommended)

## Step 1: Install aics-segmentation
1. Visit the [aics-segmentation repository](https://github.com/AllenCell/aics-segmentation/tree/main)
2. Follow the [official installation guide](https://github.com/AllenCell/aics-segmentation/blob/main/README.md)

## Step 2: Install nucleolus_segmentation_public
1. Clone the repository:
   ```bash
   cd C:\Projects
   git clone https://github.com/ashapeng/nucleolus_segmentation_public.git
   ```
   
   Alternatively, download the ZIP file and extract it to your project folder.

2. Install required packages:
   ```bash
   cd C:\Projects\nucleolus_segmentation_public
   pip install -r installer.txt
   ```

## Visualization Tools
For visualization, you have two options:
1. **itkwidgets**: Follow the [AllenCell's help guide](https://github.com/AllenCell/aics-segmentation/blob/main/README.md)
2. **napari** (recommended): More stable alternative. Installation instructions available at [napari.org](https://napari.org/stable/tutorials/fundamentals/installation.html#napari-installation)

## 3.Run this repository:
- 3.1: Unzip test_image folder: this will be the data used for running this repository.

- 3.2. Run nucleolus_seg.ipynb: to test segmentation with one example image, then run batch mode to process folders

- 3.3. Run nucleolus_feature_descriptor.ipynb or intensity based analysis, after save nucleolar mask in 3.2

# Validating other segmentation tools

The tool-adoption validator that used to live in `validation/` is now a standalone
package, **[Easy-adopt](https://github.com/ashapeng/Easy-adopt)**. It judges whether a
new, published, or third-party segmentation tool is a trustworthy drop-in on your own
data, emitting a Trust Report of flags instead of a single metric (e.g. it returns RED
for an off-the-shelf nuclei model applied to the nucleolar granular component, despite a
high IoU). The `nucleolus_gc` structure contract is shipped there.

```bash
pip install easy-adopt
easy-adopt --tool stardist --structure nucleolus_gc --cell <your_cell_folder>
```

Optional ML backends (`ml/`: nnU-Net, Cellpose) are for research evaluation and
notebooks. The production `pipeline/` path stays on classical GC segmentation.
Flipping `ml.default_backend` alone does **not** change `python -m pipeline run`;
use `--allow-ml-backend` only after a non-RED Easy-adopt trust probe. Final QC RED
cells automatically attach an informational Easy-adopt escalation (`tool=cellpose`)
when the package is installed.

# Programmatic / agentic runs

The `pipeline/` package wraps the classical GC workflow so it can run without notebooks
(and optionally under an LLM orchestrator in `agent/`).

```bash
# Inventory cell folders
python -m pipeline inventory --root test_image

# Deterministic end-to-end run (default): segment → QC → measure → report
python -m pipeline run --root test_image --no-llm --max-cells 1

# Optional: per-cell Easy-adopt trust reports (+ automatic QC-RED escalation)
python -m pipeline run --root test_image --no-llm --with-easy-adopt --max-cells 1

# Optional: allow ML backend from config only after non-RED Easy-adopt trust
python -m pipeline run --root test_image --no-llm --allow-ml-backend

# Optional LLM orchestrator (requires OPENAI_API_KEY and: pip install openai)
python -m pipeline run --root test_image --llm --goal "Segment L2 cells and write a report" --stage L2
```

Artifacts land in git-ignored `runs/<timestamp>/` (`report.md`, `manifest.json`, `qc.jsonl`, CSVs, optional `easy_adopt.json`).

Design and task plan:
- `docs/superpowers/specs/2026-07-10-agentic-ai-pipeline-design.md`
- `docs/superpowers/plans/2026-07-10-agentic-ai-pipeline.md`
