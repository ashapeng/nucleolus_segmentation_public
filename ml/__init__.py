"""
ML/AI segmentation backends for nucleolus GC segmentation.

Primary backend : nnU-Net v2  (self-configuring, handles anisotropy automatically)
Secondary backend: Cellpose 3 (instance segmentation, easy fine-tuning)

Usage in notebooks (drop-in for classical pipeline):
    import seg_util as su
    final_gc, gc_dark_spot, hole_filled = su.gc_segment(raw, nucleus_mask, backend="nnunet")
"""
