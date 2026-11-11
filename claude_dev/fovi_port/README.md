# fovi port (2026-11-12)

canvit's use of the sibling repo `fovi` was ported into `canvit/core/foveation/`, from fovi
27b1bde (its HEAD; planar visual-field geometry), keeping only what the foveated and square
patchers reach. What was kept, and why it is bit-identical, is in each module's docstring.

**Gate** (`gate.py`): the patchers built and run in the pre-port tree (merge commit c33ef61,
importing fovi) and in the ported tree, compared tensor by tensor. 32/32 cases bit-identical:
25 foveated configurations (every sampler, `rf_interp`, `arch_flag`, `ref_frame_side_length`,
`per_ring_kernel`, the three paddings, ring pruning, `sample_cortex` False / True /
`"geodesic"`, `max_coord_val`, `auto_match_cart_resources`, `force_patches_less_than_matched`,
styles `isotropic_fixn` / `logpolar` (raises identically in both), MLP embed, FiLM,
scale-FiLM, CoordConv, CoordConv+FiLM, multi-retina), 6 square-patcher cases, CUDA
forwards, and the exp22 backbone checkpoint (`step-155648.pt`: strict load, identical
patcher weights and outputs). Run on ggpu01 (V100 node), 2026-11-12.

Found on the way: fovi's `isotropic_fixn` style draws its geometry from the global, unseeded
numpy RNG, so two builds of it differ unless `np.random.seed` is set first (the gate seeds it).

Not ported: fovi's spherical/legacy field geometries, its mesh sampling (trimesh), colour
decay, pre/post transforms, and everything outside the patchers' reach (models, trainer,
augmentations). Pre-port launchers keep fovi's geometry of their `FOVI_COMMIT` through their
pinned `TRAIN_COMMIT`, which still imports fovi.
