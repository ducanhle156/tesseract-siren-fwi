# SIREN-FWI as two composed Tesseracts

> 📄 **[See the project page](https://ducanhle156.github.io/tesseract-siren-fwi/)** — results, figures and the inversion movie.
>
> 🏁 Built for the [Tesseract Hackathon 2026](https://pasteurlabs.ai/tesseract-hackathon-2026/),
> **Track 3 — Hybrid ML + Mechanistic Models**: a learned component (a SIREN
> network) trained by backpropagating through a physics solver wrapped as a
> Tesseract. Team: **Anh Le** and **Nhat Tran**. This repo is a fork of
> [Nhat Tran's original repo](https://github.com/nhatminhtrange/tesseract-siren-fwi).

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

## What is new here — and what is not

Parameterising FWI with an implicit neural network is **published prior art**
([Zhu et al. 2022](https://doi.org/10.1190/geo2020-0933.1),
[Sun et al. 2023](https://doi.org/10.1029/2022JB025964),
[Zhang et al. 2022](https://doi.org/10.1190/image2022-3746334.1),
[Nguyen et al. 2024](https://doi.org/10.1190/image2024-4089099.1)); this
project claims no methodological novelty. The contribution is the
engineering: a reproducible, open pipeline where the two halves live in
incompatible stacks, each computes its gradient its own way, and the bridge
between them is a typed Tesseract boundary whose **composed gradient is
verified by finite differences** — the part that is usually a fragile private
lash-up is here the explicit, tested object.

## The experiment

| | |
|---|---|
| Model | Marmousi, 601 × 221 at 15 m — 9.0 × 3.3 km |
| Acquisition | 64 shots, 300 receivers, full surface coverage |
| Source | 8 Hz Ricker, 6 s record at 2 ms |
| SIREN | 256 wide × 4 hidden, ω₀ = 20 — 198,401 weights |
| Velocity bounds | 1.4 – 5.8 km/s (sigmoid output; brackets Marmousi's 1.484 – 5.695) |
| Optimiser | Adam, lr 1.5e-4, 50-step warmup + cosine decay, 6000 epochs |
| Wall clock | 13 h 53 m; 93.7 % of it inside the physics Tesseract |

Both error metrics are computed against the true Marmousi over the full grid,
water included (pinned to a nominal 1.5 km/s, near-zero error); over the
119,361 unpinned cells alone the final RMSE is 0.380 km/s.

## Verifying the composed gradient

The claim that the chain rule survives the boundary is tested, not assumed:

- `make check-grad` perturbs `theta` along the gradient direction, re-runs the
  **whole** pipeline either side, and compares the finite-difference slope
  against the composed adjoint: rel. err **1.07e-2** against a 3 % tolerance.
  It passes only if both VJPs are right *and* consistent with each other.
- `test_pipeline.py` also checks each half alone: the SIREN VJP against finite
  differences to 1 %, and the Devito adjoint against single-cell finite
  differences to 5e-3.
- The Devito gradient needs one deliberate fix to be correct — a factor of 2
  from the squared-slowness chain rule missing upstream (see
  [THIRD_PARTY.md](THIRD_PARTY.md)). The finite-difference check is what
  catches it: with the upstream expression the adjoint is off by exactly 2×.

## Limitations

- **No grid-parameterised baseline** under identical settings, so the benefit
  of the SIREN parameterisation rests on the prior art above, not on a
  controlled comparison here.
- The random start is a **calibrated** random SIREN (output gain bisected to
  σ = 0.3 km/s), not a raw draw — a raw Sitzmann init collapses to a
  near-constant half-space.
- The run stopped at its **epoch budget**, not a convergence criterion, and
  had largely plateaued over the final 500 epochs.
- The correctness checks run at the **full 601 × 221, 64-shot problem**, so
  each Devito-side check costs a complete wave solve — the suite takes tens of
  minutes and, like the inversion, wants a large-memory machine (see below).

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
git clone https://github.com/ducanhle156/tesseract-siren-fwi.git
cd tesseract-siren-fwi

conda create -n fwi python=3.11 && conda activate fwi
pip install "devito==4.8.23" pylops scipy numpy matplotlib pytest tqdm \
            "jax>=0.7,<0.8" optax equinox "tesseract-core[runtime]" tesseract-jax
```

Two pins that matter, both learned by running this in a fresh environment:
`matplotlib` and `pytest` are import-time dependencies of devito 4.8.23's
bundled seismic examples — without them the forked workers die on first use.
And `jax` must stay below 0.8 for **in-process** mode: newer jax's threading
makes the fork()ed Devito workers die silently. Served/container mode has no
such constraint — the two stacks never share a process, which is the point of
the split.

Everything else it needs — the Marmousi model and the vendored FWI engine — is
already in the repo, and every path resolves relative to the file rather than
the working directory, so it runs from anywhere.

```bash
make bench        # time one loss+gradient (~1 min)
make check-grad   # finite-difference the composed gradient
make test         # the 16 checks
make run          # the full inversion — hours, start it detached
```

By default the physics Tesseract forks **one worker per shot** — 64 processes.
Budget memory accordingly: at the full grid each worker's adjoint solve costs
roughly 8 GB (the saved wavefield lives on the absorbing-boundary-padded
grid), so the stock 64 workers want a ~0.5 TB machine. On anything smaller
set `FWI_NPROCS` to something modest, e.g. `FWI_NPROCS=8 make bench`; it only
changes the wall clock, and leave headroom — worker memory grows somewhat
over repeated solves.

Both halves also build as containers (`make build`), after which the same
`workflow.py` talks to them over HTTP instead of in-process — the point of the
abstraction is that the workflow code does not change.

Figures land in `results/` and refresh as the run proceeds.

## Credits

The FWI engine is **not ours**. `devitofwi/` is
[Devito-fwi](https://github.com/DIG-Kaust/Devito-fwi) by the **Deep Imaging
Group at KAUST** (lead author Matteo Ravasi), MIT licensed and vendored here
with its `LICENSE`. Its wave propagator and hand-written adjoint-state gradient
*are* the physics half — the Tesseract only puts a typed boundary around them.

`siren/` is adapted from
[KronosAI_solutions](https://github.com/nguyenvanhaibk92/KronosAI_solutions)
(SIREN, Sitzmann et al. 2020).

Full attribution, licences and the one local fix to Devito-fwi are in
[THIRD_PARTY.md](THIRD_PARTY.md).

## License

[Apache 2.0](LICENSE). Vendored third-party code keeps its original licences
(MIT), listed in [THIRD_PARTY.md](THIRD_PARTY.md).
