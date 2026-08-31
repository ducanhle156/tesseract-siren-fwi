# SIREN-FWI as two composed Tesseracts

> 📄 **[See the project page](https://nhatminhtrange.github.io/tesseract-siren-fwi/)** — results, figures and the inversion movie.

Seismic full-waveform inversion (FWI) recovers a subsurface velocity model from
recorded waveforms. This project runs one, split across **two independent
components** that are differentiated end to end:

```
theta --[ siren-model ]--> Vp --[ devito-fwi ]--> loss
      <--   JAX VJP    <--    <-- adjoint VJP <--
```

A small neural network turns its weights into a velocity model; a wave-equation
solver turns that model into a data misfit. Gradients flow back through both,
even though each half computes its own gradient in a completely different way
and neither knows the other exists.

## The result

**Marmousi is recovered from a featureless random start** — no starting model,
no multiscale continuation. 6000 optimiser steps, about 14 hours.

| | initial | final |
|---|---|---|
| data misfit | 1.38e+04 | 4.06e+01 — **0.29 % of initial** |
| velocity RMSE | 1.0014 km/s | **0.3600 km/s** — 64 % reduction |

The data misfit and the model error fall together, which is what makes this a
real result rather than a curve-fitting artefact.

## Why split it in two

The two halves genuinely cannot share a dependency stack: `devito` and the
newer `jax` need different environments, and Devito's adjoint is not
JAX-traceable, so `jax.grad` cannot cross the boundary on its own.

[Tesseract](https://github.com/pasteurlabs/tesseract-core) makes that boundary
explicit and typed. Each side keeps its own libraries and its own way of
computing a gradient — JAX autodiff on one, a hand-written adjoint-state solve
on the other — and the chain rule joins them across a process or container
boundary.

| | `siren_model/` | `devito_fwi/` |
|---|---|---|
| does | weights → velocity model | velocity model → data misfit |
| gradient | JAX autodiff | discrete adjoint-state |
| stack | jax, equinox | devito, pylops, a C compiler |

## Layout

```
workflow.py         composes both halves and runs the inversion
problem_config.py   the shared grid, geometry and SIREN settings
test_pipeline.py    16 checks, including the composed gradient
Makefile            run / bench / check-grad / test / build / serve

siren_model/        MODEL TESSERACT   — weights -> velocity
devito_fwi/         PHYSICS TESSERACT — velocity -> misfit

siren/              the SIREN network (adapted, see THIRD_PARTY.md)
devitofwi/          VENDORED — Devito-fwi by DIG-KAUST, MIT licensed
data/Marm.bin       the Marmousi velocity model
docs/               the project page
```

## Running it

```bash
git clone https://github.com/nhatminhtrange/tesseract-siren-fwi.git
cd tesseract-siren-fwi

conda create -n fwi python=3.11 && conda activate fwi
pip install "devito==4.8.23" pylops scipy numpy matplotlib tqdm \
            jax optax equinox "tesseract-core[runtime]" tesseract-jax
```

Everything else it needs — the Marmousi model and the vendored FWI engine — is
already in the repo, and every path resolves relative to the file rather than
the working directory, so it runs from anywhere.

```bash
make bench        # time one loss+gradient (~1 min)
make check-grad   # finite-difference the composed gradient
make test         # the 16 checks
make run          # the full inversion — hours, start it detached
```

Both halves also build as containers (`make build`), after which the same
`workflow.py` talks to them over HTTP instead of in-process — the point of the
abstraction is that the workflow code does not change.

Figures land in `results/` and refresh as the run proceeds.

## Credits

The FWI engine is **not mine**. `devitofwi/` is
[Devito-fwi](https://github.com/DIG-Kaust/Devito-fwi) by the **Deep Imaging
Group at KAUST** (lead author Matteo Ravasi), MIT licensed and vendored here
with its `LICENSE`. Its wave propagator and hand-written adjoint-state gradient
*are* the physics half — the Tesseract only puts a typed boundary around them.

`siren/` is adapted from
[KronosAI_solutions](https://github.com/nguyenvanhaibk92/KronosAI_solutions)
(SIREN, Sitzmann et al. 2020).

Full attribution, licences and the one local fix to Devito-fwi are in
[THIRD_PARTY.md](THIRD_PARTY.md).
