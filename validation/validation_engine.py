#!/usr/bin/env python3
"""
Validation Skill-Contract engine (v0.1) — §4 of SKILL_CONTRACT_SPEC.md

Consumes a STRUCTURE contract + a TOOL contract and emits a TRUST REPORT of
GREEN/AMBER/RED flags instead of a bare metric. Demonstrated on the StarDist
artifacts already produced by the Wizard-of-Oz experiment.

Usage:
  python validation_engine.py --tool stardist
"""
import argparse, os, sys, glob
import numpy as np
import tifffile, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONTRACTS = os.path.join(HERE, "contracts")
TEST_IMAGE = os.path.join(REPO, "test_image")
DEFAULT_CELL = os.path.join(TEST_IMAGE, "20220305_L2", "17_1")
DEFAULT_ARTIFACTS = os.path.join(HERE, "examples")

# ---------- helpers ----------
def load_yaml(p):
    with open(p) as f: return yaml.safe_load(f)

def binmax(arr):
    """Binarize and max-project to 2D (handles 2D or 3D ZYX)."""
    a = np.asarray(arr)
    if a.ndim == 3: a = a.max(axis=0)
    return a > 0

def iou(a, b):
    a, b = binmax(a), binmax(b)
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0

def dice(a, b):
    a, b = binmax(a), binmax(b)
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else 0.0

class Report:
    def __init__(self): self.flags = []; self.qs = []
    def add(self, level, code, msg): self.flags.append((level, code, msg))
    def ask(self, q): self.qs.append(q)
    def worst(self):
        order = {"RED": 3, "AMBER": 2, "GREEN": 1}
        return max((order[l] for l, _, _ in self.flags), default=1)

# ---------- live probe (tool-specific) ----------
def stardist_on(image2d):
    """Run pretrained StarDist on a single 2D image, return binary mask + n_objects."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    m = StarDist2D.from_pretrained("2D_versatile_fluo")
    labels, _ = m.predict_instances(normalize(image2d.astype(np.float32), 1, 99.8))
    return labels > 0, int(labels.max())

PROBE_RUNNERS = {"stardist": stardist_on}

def synthetic_hole_fill_fraction(runner):
    """Crafted donut: a star-convex tool fills the central hole. Returns fraction filled (0=ideal)."""
    Y, X = np.mgrid[0:80, 0:80]
    r = np.sqrt((Y - 40) ** 2 + (X - 40) ** 2)
    donut = ((r < 25) & (r > 10)).astype(np.float32) * 5000 + np.random.RandomState(0).rand(80, 80) * 50
    mask, _ = runner(donut)
    true_hole = r < 10
    return float((binmax(mask) & true_hole).sum() / max(true_hole.sum(), 1))

def corpus_morphology():
    """What the real GC actually looks like across the dataset: median solidity & hole fraction."""
    from skimage.measure import label, regionprops
    from scipy.ndimage import binary_fill_holes
    sols, holes = [], []
    for f in glob.glob(os.path.join(TEST_IMAGE, "2022*", "*", "gc.tif")):
        g = binmax(tifffile.imread(f))
        if g.sum() == 0:
            continue
        p = max(regionprops(label(g)), key=lambda r: r.area)
        sols.append(p.solidity)
        holes.append((binary_fill_holes(g).sum() - g.sum()) / g.sum())
    if not sols:
        return 1.0, 0.0, 0
    return float(np.median(sols)), float(np.median(holes)), len(sols)

# ---------- engine ----------
def run(tool_id, cell=DEFAULT_CELL, artifacts=DEFAULT_ARTIFACTS):
    structure = load_yaml(os.path.join(CONTRACTS, "structures", "nucleolus_gc.contract.yaml"))
    tool = load_yaml(os.path.join(CONTRACTS, "tools", f"{tool_id}.contract.yaml"))
    rep = Report()

    # load references + candidate
    gc = tifffile.imread(os.path.join(cell, "gc.tif"))
    nuc = tifffile.imread(os.path.join(cell, "nuclei_mask.tif"))
    comp = tifffile.imread(os.path.join(cell, "Composite_stack.tif"))  # ZCYX
    cand_path = os.path.join(artifacts, tool_id, f"{tool_id}_mask.tif")
    lab_path = os.path.join(artifacts, tool_id, f"{tool_id}_labels.tif")

    # ---- STEP 3: pre-check (before trusting any metric) ----
    target_class = structure["target_class"]                     # sub_nuclear_condensate
    trained = tool["trained_on"]["structures"]
    if not any(t in ("nucleolus", "granular_component", "condensate") for t in trained):
        rep.add("RED", "IDENTITY_MISMATCH",
                f"Tool trained on {trained}; target is '{target_class}'. Wrong structural scale.")
    if tool["weights"].get("offline_supported") is False:
        rep.add("RED", "DEPLOYMENT",
                f"Weights fetched at runtime from {tool['weights'].get('weight_host','remote')} "
                "with no offline mode — predicts failure on firewalled/offline clusters.")

    if not os.path.exists(cand_path):
        rep.add("AMBER", "NOT_RUN", "Candidate never produced a mask (blocked upstream). "
                "Metric checks skipped; pre-check flags still stand.")
        return emit(tool_id, rep, {}, artifacts)

    cand = tifffile.imread(cand_path)
    n_obj = int(tifffile.imread(lab_path).max()) if os.path.exists(lab_path) else None

    # ---- STEP 6: multi-reference cross-check ----
    iou_gc, dice_gc, iou_nuc = iou(cand, gc), dice(cand, gc), iou(cand, nuc)
    area_frac = binmax(cand).sum() / max(binmax(nuc).sum(), 1)
    metrics = {"iou_gc": iou_gc, "dice_gc": dice_gc, "iou_nuclei": iou_nuc,
               "area_fraction_of_nucleus": float(area_frac), "n_objects": n_obj}

    tgt_min = structure["reference_masks"]["target"]["expect_iou_min"]
    dq_max = structure["reference_masks"]["disqualifier"]["expect_iou_max"]
    if iou_gc >= tgt_min and "nuclei" in trained:
        rep.add("AMBER", "COINCIDENCE_RISK",
                f"Matches gc (IoU {iou_gc:.3f}) but tool is a nuclei detector "
                f"(IoU vs nuclei {iou_nuc:.3f}). High score may be channel/crop luck — run probes.")

    # ---- STEP 7: failure signatures ----
    fs = structure["failure_signatures"]
    if dice_gc >= fs["metric_too_perfect"]["dice_vs_target_gte"]:
        rep.add("RED", "METRIC_TOO_PERFECT",
                f"Dice {dice_gc:.3f} vs reference — likely reproducing the reference's own output.")
    if area_frac > fs["segmenting_the_nucleus"]["area_fraction_gt"] or iou_nuc > iou_gc:
        rep.add("RED", "SEGMENTING_THE_NUCLEUS",
                f"area={area_frac:.2f} of nucleus / iou_nuclei {iou_nuc:.3f} vs iou_gc {iou_gc:.3f} "
                "— found the nucleus, not the GC.")
    if n_obj == fs["single_object_coincidence"]["n_objects_eq"]:
        rep.add("AMBER", "SINGLE_OBJECT",
                "Single object on a single-nucleus crop — high score may not transfer to multi-nucleus fields.")

    runner = PROBE_RUNNERS.get(tool_id)

    # ---- STEP 8a: field probe — wrong-channel invariance ----
    probe = structure["generalization_probes"]["wrong_channel_invariance"]
    thr = probe["fail_if_any_channel_area_fraction_gt"]
    if runner:
        per_ch, nuc2d_area = {}, binmax(nuc).sum()
        for c in range(comp.shape[1]):
            try:
                mask, _ = runner(comp[:, c, :, :].max(axis=0))
                per_ch[c] = float(binmax(mask).sum() / max(nuc2d_area, 1))
            except Exception:
                per_ch[c] = None
        metrics["probe_area_fraction_per_channel"] = per_ch
        expanded = [c for c, af in per_ch.items() if af is not None and af > thr]
        if expanded:
            rep.add("RED", "PROBE_WRONG_CHANNEL_FAILED",
                    f"On channel(s) {expanded} the tool expands to ~whole nucleus — "
                    "output is driven by input, not GC awareness; the gc match was channel luck.")
        else:
            # honest: not exercised, NOT a pass (no nuclear-fill channel / single-nucleus crop)
            rep.add("AMBER", "PROBE_WRONG_CHANNEL_INCONCLUSIVE",
                    "No channel drove whole-nucleus expansion, but this corpus has no nuclear-fill "
                    "channel and one nucleus per crop — channel-invariance was NOT exercised. "
                    "Re-run on a multi-nucleus field with a nuclear stain before trusting.")

    # ---- STEP 8b: capability stress-test — star-convexity (synthetic, always exercised) ----
    cap = structure["capability_probes"]["star_convexity"]
    if runner:
        fill = synthetic_hole_fill_fraction(runner)
        metrics["synthetic_hole_fill_fraction"] = round(fill, 3)
        sol, holef, n = corpus_morphology()                 # what the real data actually looks like
        metrics["corpus_median_solidity"] = round(sol, 3)
        metrics["corpus_median_hole_fraction"] = round(holef, 3)
        active = holef > 0.02 or sol < 0.85                  # does the corpus contain holed/concave GC?
        if fill > cap["fail_if_hole_fill_fraction_gt"]:
            if active:
                rep.add("RED", "STAR_CONVEXITY_ACTIVE",
                        f"Tool fills {fill:.0%} of a hole it cannot represent, AND this corpus "
                        f"contains concave/holed GC (median solidity {sol:.2f}). Systematic over-segmentation.")
            else:
                rep.add("AMBER", "STAR_CONVEXITY_LATENT",
                        f"Tool is star-convex (fills {fill:.0%} of a synthetic hole) and CANNOT represent "
                        f"holed/concave GC. This corpus happens to be compact/hole-free "
                        f"(median solidity {sol:.2f}) so it is a LATENT risk — fails on holed nucleoli.")

    # human-judgment questions
    rep.ask("Is the auto-picked marker channel truly your GC marker?")
    rep.ask("Do you need per-GC substructure, or is one-blob-per-nucleus acceptable?")
    return emit(tool_id, rep, metrics, artifacts)

def emit(tool_id, rep, metrics, artifacts):
    lvl = {3: "RED", 2: "AMBER", 1: "GREEN"}[rep.worst()]
    lines = [f"TRUST REPORT — tool={tool_id}  target=nucleolus_gc  crop=20220305_L2/17_1",
             "=" * 70]
    if metrics:
        lines.append("METRICS (context, not verdict):")
        for k, v in metrics.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    icon = {"RED": "🔴", "AMBER": "🟡", "GREEN": "🟢"}
    lines.append("FLAGS:")
    for level, code, msg in rep.flags:
        lines.append(f"  {icon[level]} {code}: {msg}")
    if rep.qs:
        lines.append("\nHUMAN-JUDGMENT QUESTIONS:")
        for q in rep.qs: lines.append(f"  • {q}")
    verdict = ("NOT a valid drop-in for nucleolus_gc — see RED flags."
               if lvl == "RED" else
               "Proceed with caution — review AMBER flags." if lvl == "AMBER"
               else "No blocking flags; still confirm on held-out data.")
    lines += ["", f"OVERALL: {lvl} — {verdict}"]
    out = "\n".join(lines)
    print(out)
    outdir = os.path.join(artifacts, tool_id)
    if os.path.isdir(outdir):
        with open(os.path.join(outdir, "trust_report.txt"), "w") as f:
            f.write(out)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="stardist")
    ap.add_argument("--cell", default=DEFAULT_CELL, help="path to a cell folder with gc.tif/nuclei_mask.tif/Composite_stack.tif")
    ap.add_argument("--artifacts", default=DEFAULT_ARTIFACTS, help="dir containing <tool>/<tool>_mask.tif")
    a = ap.parse_args()
    run(a.tool, a.cell, a.artifacts)
