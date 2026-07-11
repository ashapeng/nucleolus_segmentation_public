"""
ML/AI segmentation backends for nucleolus GC segmentation (research / eval).

Production pipeline default remains classical CV (`seg_util.gc_segment` /
`pipeline` with ``ml.default_backend: classical``). nnU-Net and Cellpose are
optional drop-ins for notebooks, ``ml.evaluate``, and active-learning research —
not silent replacements.

To use an ML backend from ``python -m pipeline run``, you must pass
``--allow-ml-backend`` **and** obtain a non-RED Easy-adopt trust report on a
probe cell. See ``pipeline.trust.resolve_pipeline_backend``.

Notebook / eval usage (explicit backend argument):
    import seg_util as su
    final_gc, gc_dark_spot, hole_filled = su.gc_segment(
        raw, nucleus_mask, backend="nnunet"
    )
"""
