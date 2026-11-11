"""Foveated-sensor geometry for the foveated and square patchers.

Ported on 2026-11-12 from the sibling repo ``fovi`` at 27b1bde (its planar visual-field
geometry), keeping only what canvit's patchers reach:

    coords.py    visual-field sampling coordinates (SamplingCoords) and frame helpers
    manifold.py  visual field -> cortical sensor manifold (planar)
    knn.py       KNN layers on the manifold; KNNPartitioningPatchEmbedding
    samplers.py  grid / KNN-pooling samplers that read the image at the sample positions
    retina.py    RetinalTransform: foveated sampling around a fixation
    square.py    square (axis-aligned) approximations of the foveated patches

Method bodies are verbatim, so patches, KNN routing, weights and state_dict keys are
bit-identical to fovi 27b1bde. Runs pinned to an older ``FOVI_COMMIT`` keep fovi's
geometry of that commit through their pinned ``TRAIN_COMMIT``, which still imports fovi.
"""
