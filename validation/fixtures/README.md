# Synthetic positive-control fixtures

These are **not real data**. They are crafted unit-test inputs that deliberately contain
the conditions the real C. elegans corpus lacks, so the probe layer can be proven to fire.

The real corpus is compact, hole-free, single-nucleus, and has no nuclear-fill channel — so on
it the runtime probes correctly report `LATENT` / `INCONCLUSIVE`. These fixtures supply the
missing conditions on purpose, as positive controls.

| Fixture | Condition injected | Probe it proves |
|---------|--------------------|-----------------|
| `holed_gc` | a GC with a central hole (annulus) | 🔴 `STAR_CONVEXITY_ACTIVE` — StarDist fills the hole (IoU vs gc drops to ~0.53) and the corpus is graded holed (solidity ~0.78) |
| `multi_nucleus` | 2 nuclei + a nuclear-fill channel (ch0) | 🔴 `PROBE_WRONG_CHANNEL_FAILED` — on ch0 StarDist expands to whole nuclei (area fraction ~1.03), proving output is input-driven, not GC-aware |

## Regenerate

```bash
python validation/fixtures/make_fixtures.py        # needs stardist + tensorflow + csbdeep
```

## Run the engine against them

```bash
python validation/validation_engine.py --tool stardist \
  --cell validation/fixtures/holed_gc --artifacts validation/fixtures/holed_gc \
  --corpus validation/fixtures/holed_gc

python validation/validation_engine.py --tool stardist \
  --cell validation/fixtures/multi_nucleus --artifacts validation/fixtures/multi_nucleus \
  --corpus validation/fixtures/multi_nucleus
```
