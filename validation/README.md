# Validation Skill-Contracts

A small, contract-driven engine that judges whether a candidate segmentation tool is
**trustworthy** for a target structure — instead of trusting a single metric.

## Why

A high IoU is not validation. In the experiment that motivated this:

| Tool | Bare metric | Reality |
|------|-------------|---------|
| the repo's own classic pipeline | Dice **1.000** vs `gc.tif` | reproducing its own output — zero independent validation |
| Cellpose | never ran | runtime weight fetch blocked; even if run, it's a nuclei detector |
| StarDist | IoU **0.955** vs `gc.tif` | a **nuclei** detector that matched the GC by channel luck on a single-nucleus crop |

The metric never told the truth; the **flags** would have. This engine encodes the
domain knowledge that distinguishes "it ran" from "you can trust it" as reusable,
machine-readable contracts (the SKILL.md idea applied to bioimage adoption).

See [`SPEC.md`](SPEC.md) for the full design.

## Layout

```
contracts/structures/nucleolus_gc.contract.yaml   biology-aware: what a correct GC mask must do
contracts/tools/{stardist,cellpose}.contract.yaml  what each tool was built for + deploy quirks
validation_engine.py                               the engine (§4 of SPEC.md)
examples/stardist/{stardist_mask,stardist_labels}.tif   bundled candidate output, so the demo runs
```

## Run

```bash
pip install numpy scipy scikit-image tifffile pyyaml
# StarDist contract runs LIVE probes, so also: pip install stardist tensorflow csbdeep
python validation/validation_engine.py --tool stardist     # full report incl. live probes
python validation/validation_engine.py --tool cellpose     # pre-check only (no candidate mask)
```

Point it at any cell folder / candidate output:

```bash
python validation/validation_engine.py --tool stardist \
  --cell test_image/20220305_L2/17_1 \
  --artifacts validation/examples
```

## What the engine checks

- **Pre-check (no install needed):** tool `trained_on` vs the structure's class
  (nuclei detector ≠ sub-nuclear target), and `offline_supported` (predicts firewalled-cluster failure).
- **Multi-reference cross-check:** scores against the *disqualifier* (`nuclei_mask`) too,
  not just the target (`gc.tif`).
- **Failure signatures:** metric-too-perfect, segmenting-the-nucleus, single-object coincidence.
- **Field probe:** wrong-channel invariance — and honestly reports **INCONCLUSIVE** when the
  data can't exercise it, rather than a false pass.
- **Capability stress-test:** a synthetic holed object proves the tool's star-convex limit,
  graded **LATENT** (this corpus is hole-free) vs **ACTIVE** (corpus has concave/holed GC).

Output is a **Trust Report** of 🔴/🟡/🟢 flags + human-judgment questions, not a number.

## Honest boundary

The engine *surfaces* the traps a naive adopter would miss; it does **not** certify
correctness. It is a validation co-pilot. The dangerous failure in this field is not a broken
install — it is a clean install that scores 0.955 and is wrong.

## Adding a new structure or tool

Drop a YAML in `contracts/structures/` or `contracts/tools/`. Tool contracts can be
auto-drafted by introspecting a repo (deps, weights, I/O) and confirmed by a human.
