#!/usr/bin/env python3
"""
Generate SYNTHETIC positive-control fixtures for the validation engine.

These are NOT real data. They are crafted unit-test inputs that deliberately
contain the morphology / channel structure the real corpus lacks, so that the
ACTIVE star-convexity path and the wrong-channel field probe are exercised and
proven to fire. The real C. elegans corpus is compact, hole-free, single-nucleus
and has no nuclear-fill channel; these fixtures supply those conditions on purpose.

Each fixture is written as a "cell folder" the engine understands:
  gc.tif (ZYX), nuclei_mask.tif (ZYX), Composite_stack.tif (ZCYX),
  stardist/stardist_mask.tif + stardist_labels.tif (candidate output)

Usage:  python validation/fixtures/make_fixtures.py
"""
import os, numpy as np, tifffile
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
Z = 8

def disk(cy, cx, r, shape):
    Y, X = np.mgrid[0:shape[0], 0:shape[1]]
    return (Y - cy) ** 2 + (X - cx) ** 2 <= r * r

def annulus(cy, cx, r_in, r_out, shape):
    Y, X = np.mgrid[0:shape[0], 0:shape[1]]
    d2 = (Y - cy) ** 2 + (X - cx) ** 2
    return (d2 <= r_out * r_out) & (d2 >= r_in * r_in)

def stack3d(mask2d, z0, z1, zsize=Z):
    s = np.zeros((zsize,) + mask2d.shape, np.uint8)
    s[z0:z1] = (mask2d * 255).astype(np.uint8)
    return s

def composite(channels2d, zsize=Z):
    """channels2d: list of 2D float arrays -> ZCYX uint16."""
    yx = channels2d[0].shape
    comp = np.zeros((zsize, len(channels2d), *yx), np.uint16)
    for c, ch in enumerate(channels2d):
        comp[2:6, c] = ch.astype(np.uint16)
    return comp

def stardist_mask2d(image2d):
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    m = StarDist2D.from_pretrained("2D_versatile_fluo")
    labels, _ = m.predict_instances(normalize(image2d.astype(np.float32), 1, 99.8))
    return (labels > 0).astype(np.uint8) * 255, labels.astype(np.uint16)

def write_fixture(name, gc2d, nuc2d, channels2d, marker_channel):
    d = os.path.join(HERE, name)
    os.makedirs(os.path.join(d, "stardist"), exist_ok=True)
    tifffile.imwrite(os.path.join(d, "gc.tif"), stack3d(gc2d, 2, 6))
    tifffile.imwrite(os.path.join(d, "nuclei_mask.tif"), stack3d(nuc2d, 1, 7))
    tifffile.imwrite(os.path.join(d, "Composite_stack.tif"), composite(channels2d))
    rng = np.random.RandomState(0)
    img = channels2d[marker_channel] + rng.rand(*gc2d.shape) * 30
    mask, labels = stardist_mask2d(img)
    tifffile.imwrite(os.path.join(d, "stardist", "stardist_mask.tif"), mask)
    tifffile.imwrite(os.path.join(d, "stardist", "stardist_labels.tif"), labels)
    print(f"wrote {name}: gc={int(gc2d.sum())}px nuc={int(nuc2d.sum())}px "
          f"stardist={int((mask>0).sum())}px")

# ---- Fixture 1: holed GC (exercises STAR_CONVEXITY ACTIVE) ----
def holed_gc():
    shape = (64, 64)
    gc = annulus(32, 32, 8, 18, shape)          # a hole the star-convex model cannot represent
    nuc = disk(32, 32, 26, shape)
    ch_marker = gc.astype(float) * 6000          # GC bright in marker channel
    ch_a = gc.astype(float) * 2000
    ch_b = gc.astype(float) * 1500
    write_fixture("holed_gc", gc, nuc, [ch_a, ch_b, ch_marker], marker_channel=2)

# ---- Fixture 2: multi-nucleus + nuclear-fill channel (exercises wrong-channel probe) ----
def multi_nucleus():
    shape = (64, 128)
    centers = [(32, 32), (32, 96)]
    gc = np.zeros(shape, bool); nuc = np.zeros(shape, bool)
    for cy, cx in centers:
        gc |= disk(cy, cx, 8, shape)             # small GC per nucleus
        nuc |= disk(cy, cx, 20, shape)           # whole nucleus
    ch_marker = gc.astype(float) * 6000          # ch2: GC marker (small blobs)
    ch_nuclear = nuc.astype(float) * 6000        # ch0: NUCLEAR-FILL (whole nuclei) -> probe trap
    ch_dim = gc.astype(float) * 1500
    write_fixture("multi_nucleus", gc, nuc, [ch_nuclear, ch_dim, ch_marker], marker_channel=2)

if __name__ == "__main__":
    holed_gc()
    multi_nucleus()
