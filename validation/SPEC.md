# Validation Skill-Contract Spec (v0.1)

> An adoption agent that compresses **plumbing** from weeks to an hour is selling a commodity.
> The defensible layer is **validation**: catching the traps a naive adopter ships as wrong science.
> This spec encodes that validation knowledge as machine-readable, agent-actionable contracts.

Grounded in the Wizard-of-Oz experiment (3 cold tool adoptions on `test_image/20220305_L2/17_1`):
- Local repo → Dice **1.000** (reproduced its own output = zero independent validation).
- Cellpose → never ran (runtime weight-fetch blocked).
- StarDist → IoU **0.955** that **looks** like a win but is a nuclei detector matching the GC by channel luck on a single-nucleus crop.

The metric never told the truth. The **flags** would have.

---

## 1. The three contracts

```
STRUCTURE CONTRACT   what the biological target IS, and how a correct mask must behave
        +
TOOL CONTRACT        what a candidate tool/model WAS BUILT FOR, and its deployment quirks
        ↓  (agent matches them)
TRUST REPORT         not pass/fail — a scorecard of GREEN/AMBER/RED flags + human-judgment questions
```

The agent **auto-drafts** the Tool Contract by introspecting the repo (README, deps, weights, I/O),
**looks up** the Structure Contract for the user's target, then runs the validation algorithm (§4).

---

## 2. Structure Contract

A reusable, biology-aware description of a target structure. Authored once per structure, reused across every tool.

```yaml
# structures/nucleolus_gc.contract.yaml
id: nucleolus_gc
aliases: [nucleolus, granular_component, GC]
description: >
  The granular component of the nucleolus — a sub-nuclear, roughly spherical
  membraneless condensate. ONE-to-FEW per nucleus. NOT the nucleus itself.

spatial:
  topology: roughly_spherical          # star-convex-OK, but see failure_signatures
  count_per_nucleus: [1, 4]
  # the discriminating fact the StarDist run proved:
  area_fraction_of_nucleus: [0.05, 0.40]   # GC is a SUBSET of the nucleus, never ≈ it
  may_contain_holes: true               # user pipeline uses holes.tif / hole-filling

containment:
  must_be_inside: nuclei_mask
  must_largely_exclude: nucleoplasm_mask

# how to find the correct input channel (don't trust channel order)
marker_channel:
  method: enrichment_ratio_in_reference   # mean(inside gc)/mean(outside gc), pick max
  min_enrichment: 1.5

# the heart of validation: match the RIGHT reference, and DISqualify on the wrong ones
reference_masks:
  target:        { file: gc.tif,           expect_iou: ">=0.6"  }   # should match
  disqualifier:  { file: nuclei_mask.tif,  expect_iou: "<=0.4"  }   # must NOT match (else it's a nucleus detector)
  context:       { file: nucleoplasm_mask.tif }

# telltale signs of wrong-structure segmentation — each maps to a RED flag
failure_signatures:
  - id: segmenting_the_nucleus
    when: "candidate_area_fraction_of_nucleus > 0.6 OR iou(candidate, nuclei_mask) > iou(candidate, gc)"
    means: "Tool found the whole nucleus, not the GC."
  - id: metric_too_perfect
    when: "dice(candidate, gc) >= 0.98"
    means: "Likely reproducing the reference's own output, not independently validating."
  - id: single_object_coincidence
    when: "n_objects == 1 AND field_has_multiple_nuclei == false"
    means: "High score may be a single-blob crop coincidence; not transferable."

# probes that go BEYOND the convenient test crop (un-skippable)
generalization_probes:
  - id: wrong_channel_invariance
    do: "re-run the tool on the NUCLEAR-stain channel instead of the marker channel"
    pass_if: "output does NOT expand to ~whole-nucleus"
    fails_means: "Tool is a channel-agnostic nucleus/blob detector; the GC match was channel luck."
  - id: multi_nucleus_field
    do: "run on a field with >1 nucleus"
    pass_if: "GC substructure is preserved, not collapsed to one-blob-per-nucleus"

# the user's CURRENT method, so the agent can flag semantic divergence
incumbent_method:
  approach: "intensity threshold + hole-filling + nucleoplasm subtraction (substructure-aware)"
  note: "Generic nuclei/cell DL models are NOT substructure-aware and cannot replicate this."
```

---

## 3. Tool Contract (agent auto-drafts, human confirms)

```yaml
# tools/stardist.contract.yaml
id: stardist
version: "0.9.2"
install: { method: pip, pkgs: [stardist, tensorflow, csbdeep], weight_mb: ~600, deps: heavy_tf }

trained_on: { structures: [nuclei], modality: fluorescence_2d }   # ← the decisive field
output: instance_labels
io:
  ndim: 2
  needs_projection: true              # 3D z-stack → 2D max-proj
  channels: single
  normalize: "percentile 1–99.8 (csbdeep default)"
weights:
  source: bundled_pretrained          # 2D_versatile_fluo
  fetch_at_runtime: true
  offline_supported: false            # ← the Cellpose-class deployment trap
known_mismatches:
  - "sub-nuclear structures (nucleolus, condensates) — model targets whole nuclei"
deployment_flags:
  - runtime_weight_fetch              # fails on firewalled / offline clusters
```

The Cellpose contract would carry `offline_supported: false` + `weight_host: huggingface.co`, which alone
predicts the 403 wall **before** wasting the install.

---

## 4. The validation algorithm (what the agent runs)

```
1. RESOLVE   structure contract for the user's target (nucleolus_gc).
2. DRAFT     tool contract by introspecting the repo; confirm key fields with user.
3. PRE-CHECK compatibility BEFORE installing:
             tool.trained_on ∩ structure  →  identity-mismatch flag
             tool.weights.offline_supported==false & target env offline → deployment flag
   (StarDist & Cellpose both fire here, saving the wasted install.)
4. MAP       pick input channel via structure.marker_channel  → interface flag if ambiguous.
5. RUN       execute on the test crop.
6. CROSS-CHECK metrics against ALL reference_masks, not just the target:
             iou(out, gc)=0.955  AND  iou(out, nuclei)=0.295  AND tool.trained_on=nuclei
             → AMBER "high score may be coincidental; run probes".
7. SIGNATURES apply failure_signatures (too-perfect, segmenting-nucleus, single-object).
8. PROBES     run generalization_probes (wrong-channel-invariance, multi-nucleus) — these are
              what turn the StarDist AMBER into a definitive RED or GREEN.
9. EMIT       Trust Report (§5). Never a bare metric.
```

---

## 5. Trust Report (the actual deliverable to the scientist)

```
TOOL: StarDist 2D_versatile_fluo   TARGET: nucleolus_gc   CROP: 20220305_L2/17_1

PLUMBING ......... GREEN   installed 74s, ran in min
METRIC ........... IoU 0.955 vs gc.tif   ← DO NOT TRUST ALONE, see flags

🔴 IDENTITY MISMATCH   tool trained on whole NUCLEI; target is a SUB-nuclear structure.
🟡 COINCIDENCE RISK    matches gc (0.955) but barely matches nuclei (0.295) — only because
                       the marker channel was fed on a single-nucleus crop.
🔴 PROBE: wrong-channel-invariance  FAILED  → on the nuclear channel it segments the whole
                       nucleus. The GC match was channel luck, not GC awareness. NOT transferable.
🟡 SEMANTIC DIVERGENCE your pipeline uses hole-filling + nucleoplasm subtraction; this model
                       has no substructure awareness and cannot replicate it.

HUMAN-JUDGMENT QUESTIONS:
  • Is channel 2 truly your GC marker? (auto-picked by enrichment ratio 2.72)
  • Do you need per-GC substructure, or is one-blob-per-nucleus acceptable?

VERDICT: Off-the-shelf StarDist is NOT a valid drop-in for nucleolus GC. Use as a nucleus
         pre-segmenter only, or fine-tune. A naive adopter would have shipped IoU 0.955 as success.
```

---

## 6. Why this is the moat (and the honest limit)

- **Plumbing** (steps 4–5) is the commodity every agent automates.
- **Steps 3, 6, 7, 8** — the structure-aware pre-check, multi-reference cross-check, failure signatures,
  and generalization probes — are the defensible layer. They are **authored knowledge**, reusable across
  tools, and exactly the SKILL.md skill-contract pattern.
- **Honest limit:** the agent *surfaces* traps with heuristics; it does **not** certify correctness.
  The product is a validation **co-pilot** ("here's what a naive adopter would miss"), not "validated in
  an hour." That is also the more valuable promise: in this field the dangerous failure is not
  "it didn't install" — it's "it installed, scored 0.955, and was wrong."

---

## 7. Next steps (suggested)
- [ ] Author 3–5 Structure Contracts for the condensate niche (nucleolus GC/DFC/FC, stress granule, generic nuclear focus).
- [ ] Build the tool-contract auto-drafter (repo introspection → YAML).
- [ ] Implement the §4 engine on top of the existing WoZ run scripts in `woz_experiment/`.
- [ ] Validate the engine fires the right flags on a held-out tool the author didn't tune it against.
```
