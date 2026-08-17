# Third-party code vendored into this directory

This pipeline is self-contained, which means it carries copies of code written
by other people. They are listed here with their origin and licence; the
licence files travel with the code.

## `devitofwi/` — Devito-fwi

**Deep Imaging Group (DIG), King Abdullah University of Science and Technology
(KAUST)**

- Source: https://github.com/DIG-Kaust/Devito-fwi
- Lead author: Matteo Ravasi (`matteo.ravasi@kaust.edu.sa`)
- Contributors (per this repo's history): mrava87, malfarhan7, Matteo Ravasi,
  Gustavo Coelho, FuqiangChen
- Licence: MIT — Copyright (c) 2024 Deep imaging group. Full text in
  [`devitofwi/LICENSE`](devitofwi/LICENSE).

This is the acoustic FWI engine: the Devito-based wave propagator and the
hand-written discrete adjoint-state gradient that the physics Tesseract exposes
as its `vector_jacobian_product`. **The Tesseract wraps it; it does not
reimplement it.** `AcousticWave2D._loss_grad` is called verbatim.

Vendored rather than pip-installed because Devito-fwi is not on PyPI, and in
this workspace it was an editable install pointing back at the parent repo,
which would have made this folder non-portable.

### Local modification

One change relative to upstream, in
[`devitofwi/postproc/acoustic.py`](devitofwi/postproc/acoustic.py):

```python
# upstream
grad = - grad / (vp ** 3)
# here
grad = - 2. * grad / (vp ** 3)
```

Devito returns `dL/dm` for the squared slowness `m = 1/vp²`, so the chain rule
to velocity is `dL/dm · dm/dvp = dL/dm · (-2/vp³)` — the factor of 2 belongs
there. The gradient check in `test_pipeline.py` depends on it: with the
upstream expression the adjoint is off by exactly 2x. This fix originated in
the parent workspace, not here.

## `siren/` — SIREN network

Sinusoidal-representation network (Sitzmann et al., 2020), JAX implementation
adapted from https://github.com/nguyenvanhaibk92/KronosAI_solutions
(`Task1/utils/siren_network.py`). See [`siren/README.md`](siren/README.md) for
what was kept and what was dropped.

## `fwireport.py`

Figure and timing-report toolkit from `script_siren_Nhat/` in the parent
workspace, copied so a run here produces the same figure set as the reference
script `AcousticVel_L2_SIREN_random.py`.

## `data/Marm.bin`

The Marmousi velocity model, from the parent workspace's `data/`.
