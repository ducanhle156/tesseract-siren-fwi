# SIREN-FWI as two composed Tesseracts

> 📄 **Project page with figures: [nhatminhtrange.github.io/tesseract-siren-fwi](https://nhatminhtrange.github.io/tesseract-siren-fwi/)**
> — the same material as this README, with the run's figures inline. Source in [`docs/`](docs/).

A minimal version of `script_siren_Nhat/AcousticVel_L2_SIREN_random.py`, split
at the boundary where the automatic-differentiation strategy changes:

```
theta --[ siren-model ]--> Vp --[ devito-fwi ]--> loss
      <--    JAX VJP    <--    <-- adjoint VJP <--
```

| | `siren_model/` | `devito_fwi/` |
|---|---|---|
| forward | `vp = vmin + (vmax-vmin)·sigmoid(SIREN([x,z]))` | acoustic FD solve, L2 misfit over all shots |
| gradient | JAX autodiff (`jax_recipes`) | hand-written discrete adjoint-state (`devitofwi`) |
| stack | jax, equinox | devito, pylops, a C compiler |

`workflow.py` composes them and runs `jax.value_and_grad` + Optax Adam over
`theta`. JAX never traces the wave solve — it calls the Devito Tesseract's
`vector_jacobian_product` endpoint, which is `AcousticWave2D._loss_grad` used
verbatim. The chain rule

```
dL/dtheta = (dvp/dtheta)ᵀ @ (dL/dvp)
```

is therefore assembled from two independently implemented gradients on opposite
sides of a process (or container) boundary.

## Why this split is load-bearing

It is not decoration. The two halves genuinely cannot share a dependency stack
in this repo — `devito` and the newer `jax` live in different conda envs — and
Devito's adjoint is not JAX-traceable, so `jax.grad` cannot cross the boundary
without a VJP endpoint. Tesseract makes that boundary explicit and typed
instead of implicit in a `chain_grad` call.

## Credits

The FWI engine is **not mine**. `devitofwi/` is
[Devito-fwi](https://github.com/DIG-Kaust/Devito-fwi) by the **Deep Imaging
Group at KAUST** (lead author Matteo Ravasi), MIT licensed, vendored here with
its `LICENSE`. Its Devito propagator and hand-written adjoint-state gradient
*are* the physics Tesseract — `AcousticWave2D._loss_grad` is called verbatim as
the `vector_jacobian_product`; the Tesseract only puts a typed boundary around
it.

`siren/` is adapted from
[KronosAI_solutions](https://github.com/nguyenvanhaibk92/KronosAI_solutions)
(SIREN, Sitzmann et al. 2020). Full attribution, licences and the one local
modification to Devito-fwi are in [THIRD_PARTY.md](THIRD_PARTY.md).

## Repository layout

```
tesseract_siren_fwi/
├── workflow.py              compose both Tesseracts, jax.value_and_grad + Adam
├── problem_config.py        the shared grid, geometry, SIREN config, water mask
├── fwireport.py             figures and the timing report (from the reference script)
├── test_pipeline.py         16 checks: mask, both VJPs, the composed gradient
├── Makefile                 run / bench / check-grad / test / build / serve
│
├── siren_model/             MODEL TESSERACT — theta -> Vp
│   ├── tesseract_api.py       jax_recipes wrappers over the SIREN forward map
│   ├── tesseract_config.yaml  package_data pulls in siren/, problem_config, Marm.bin
│   └── tesseract_requirements.txt
│
├── devito_fwi/              PHYSICS TESSERACT — Vp -> L2 misfit
│   ├── tesseract_api.py       wraps AcousticWave2D._loss_grad as apply + vjp
│   ├── tesseract_config.yaml  needs gcc/g++ — devito JIT-compiles its stencils
│   └── tesseract_requirements.txt
│
├── siren/                   SIREN network (adapted, see THIRD_PARTY.md)
│   ├── siren_network.py        the MLP with sinusoidal activations
│   └── velocity_field.py       coords -> sigmoid -> [vmin, vmax], water pinning
│
├── devitofwi/               VENDORED — Devito-fwi by DIG-KAUST, MIT (LICENSE inside)
│   ├── waveengine/acoustic.py    the propagator and the adjoint-state gradient
│   ├── postproc/acoustic.py      PostProcessVP — carries the one local fix
│   ├── loss/                     L2, xcorr, dtw
│   └── ...
│
├── data/Marm.bin            Marmousi, 601 x 221 float32
├── docs/                    the GitHub Pages site (index.html + assets/)
└── results/                 run outputs — gitignored
```

The two Tesseracts are the only things with `tesseract_api.py`; everything else
is either shared problem definition or the libraries they wrap.

## What you need

The folder is fully self-contained: `devitofwi/`, `siren/`, `fwireport.py`,
`problem_config.py` and `data/Marm.bin` all live here, so nothing outside it is read
and it can be moved anywhere. From the **environment** it needs only PyPI
packages:

```
devito==4.8.23  pylops  scipy  numpy  matplotlib  tqdm
jax  optax  equinox
tesseract-core[runtime]  tesseract-jax
```

On this machine that is `geo_jxli`, into which `tesseract-core[runtime]`,
`tesseract-jax` and `equinox` were pip-installed — they coexist with devito
fine. Note the vendored `devitofwi/` shadows any installed copy, because the
pipeline directory goes on `sys.path` first.

### Moving the folder somewhere else

Copy it and run — **no path editing needed**. Every path is resolved from
`__file__`, not from the working directory, so all of these work:

```bash
cp -r tesseract_siren_fwi ~/anywhere/
cd ~/anywhere/tesseract_siren_fwi && $PY workflow.py      # from inside
cd /somewhere/else && $PY ~/anywhere/tesseract_siren_fwi/workflow.py   # from outside
```

The one thing that is machine-specific is the interpreter in the `Makefile`.
It falls back to whatever `python` / `tesseract` is on `PATH` when this
workspace's conda envs are absent, and both are overridable:

```bash
make run PYTHON=python
make build TESSERACT=tesseract
```

On a new machine, create an env with the packages above, then verify with
`make bench` (one loss+gradient, ~1 min) or `make test` (16 checks, each a wave
solve over 64 shots — budget ~20 min). Both exit non-zero if anything is wrong.

## Run it

```bash
# geo_jxli is the env with devito; tesseract-core was installed into it
PY=/home/mtran/miniconda3/envs/geo_jxli/bin/python

$PY workflow.py --benchmark     # time one loss+gradient and exit (~1 min)
$PY workflow.py --check-grad    # verify the composed gradient (~7 min)
$PY workflow.py                 # the full inversion
```

The problem is the reference script's, unchanged: Marmousi on **601×221 at
15 m, 64 shots, 300 receivers, 8 Hz**, SIREN **256×4** at omega_0 20, Adam at
**1.5e-4 for 3000 epochs** with 50 warmup steps. `make run` is exactly that.

**It is a long run.** One loss+gradient averaged 7.8 s with one worker per shot
over the completed run, so 6000 epochs took ~14 hours; on a busy box it is
several times that. Start it detached:

```bash
nohup $PY workflow.py > run.out 2>&1 &
```

and watch `results/*tmp.png` — they refresh every `--fig-every` (25) epochs.
`--maxiter` cuts it short if you only want to see it move.

### Outputs

The workflow reuses the reference script's `fwireport.py`, so a run produces the
same file set, the same figures and the same timing report as
`AcousticVel_L2_SIREN_random.py` — only the gradient path differs, which makes
the two directly comparable side by side.

| file | what it is |
|---|---|
| `SirenInit.png` | true model + random initial model |
| `Data.png` | observed shot gathers |
| `Gradient.png` | first gradient w.r.t. Vp |
| `InvertedVP.png` | true / random-initial / inverted, one shared colorbar |
| `Profiles.png` | vertical Vp profiles at three distances |
| `Waveform.png` | observed vs modelled waveforms |
| `Loss.png` | convergence (data misfit + model error) |
| `GradTheta.png` | dL/dθ per layer at the final model, and its history |
| `*tmp.png` | live diagnostics, refreshed every `--fig-every` |
| `Waveformtmp.png` | live observed vs estimated wiggles + residual |
| `snapshots.npy` | movie frames — render with `script_siren_Nhat/make_movie.py` |
| `result.npz` | models, weights, histories, timings |
| `timing.txt` | the timing report printed at the end |
| `code/` | verbatim copy of the sources that produced the run |

The console log matches too — per-iteration loss / `|dL/dθ|` / RMSE, the
per-layer `eval N:` gradient lines, and the final per-layer table.

Throttles: `--fig-every` (live figures), `--wiggle-every` / `--wiggle-shot` /
`--wiggle-tmax` (live waveforms), `--snap-every` (movie frames),
`--no-figures` to skip all of it.

### Results — a completed run

**Marmousi is recovered from a featureless random start**, with no multiscale
continuation and no starting-model information. 6000 Adam epochs, 13 h 53 m
wall clock:

| | initial | final |
|---|---|---|
| data misfit Φ | 1.3799e+04 | 4.0559e+01 — **0.29 % of initial** |
| model error ‖(m−m_true)/m_true‖₂ | 177.43 | 33.00 — **81.4 % reduction** |
| velocity RMSE | 1.0014 km/s | **0.3600 km/s** — 64 % reduction |

Both curves fall together, which is the point — the misfit is not being reduced
at the model's expense. See `Loss.png` and `InvertedVP.png`; the faults, the
dipping high-velocity wedge and the layer geometry are all recovered.

Timing: 93.7 % of wall clock is the objective (loss + gradient + VJP), 7.80 s
mean over 6000 calls with 50 workers. The Tesseract boundary itself — `theta`
out and `dL/dθ` back, 6000 times each way — does not show up in the budget.

**Caveats.** This run used **50 shots and 6000 epochs**, not the 64 / 3000 that
`make run` and the reference script use, and started from a calibrated random
SIREN (gain 8.775, σ 0.300 km/s). It had largely plateaued — over the final 500
epochs the misfit moved 7 % and the model error 0.58 % — and was stopped at its
epoch budget, not at a convergence criterion.

**Correctness checks were run on a cheaper grid** (a decimated Marmousi used
during development): all 16 checks in `test_pipeline.py`, including the composed
gradient against finite differences to ~1%, the Devito adjoint to 2e-5–4e-4 per
cell, and the fully containerised `--served` path returning an identical
gradient. Those are properties of the code, not of the grid, but they have not
been re-run at full size — `make test` there costs a wave solve per check.

## Verifying the gradient

`--check-grad` finite-differences the *composed* forward map and compares
against the composed adjoint. It only passes if both VJPs are correct and
consistent with each other:

```
    -grad: adjoint +3.081590e-05  finite-diff +3.115019e-05  rel.err 1.07e-02  OK
```

The check asserts on the gradient direction — the one Adam steps along, and the
best conditioned. Random directions in 8577-dimensional space are nearly
orthogonal to the gradient (they see ~1% of its norm), so their finite
differences are dominated by second-order terms of the other components; they
are printed for information but not asserted on. Verified separately,
single-cell perturbations of the Devito VJP agree to 2e-5–4e-4.

## Running as containers

The in-process mode above (`Tesseract.from_tesseract_api`) keeps the demo to one
command, but it is the container mode that actually delivers the isolation the
split is for:

```bash
make build                     # or: tesseract build siren_model/ && ... devito_fwi/
tesseract serve siren-model    # one image per invocation; note the port
tesseract serve devito-fwi     # note this port too
$PY workflow.py --served http://127.0.0.1:PORT1 http://127.0.0.1:PORT2
tesseract teardown <name>      # names are printed by serve, or `tesseract ps`
```

`workflow.py` is unchanged between the two modes — only how the clients are
constructed differs, which is the point of the abstraction.

This box has rootless **podman and no `docker` binary**, but the Tesseract CLI
shells out to `docker`, so builds need a shim:

```bash
printf '#!/bin/sh\nexec podman "$@"\n' > ~/.local/bin/docker && chmod +x ~/.local/bin/docker
```

The `Makefile` puts `~/.local/bin` on `PATH` for the build targets.

## Notes

- **`JAX_PLATFORMS=cpu`** is set inside `siren_model/tesseract_api.py`. The
  coordinate network is a few thousand weights; a GPU buys nothing, costs a
  transfer per call, and on a shared box competes with whatever else is
  resident.
- **`jax_enable_x64`** is enabled in the workflow: the FWI Tesseract returns a
  float64 loss.
- **The pool is warmed once.** `devito_fwi` forks one worker per shot on the
  first call and reuses them, so Devito's JIT compilation and the observed-data
  modelling are paid once per served Tesseract rather than once per gradient.
- **`apply` computes the gradient anyway.** `_loss_grad(computegrad=False)` has
  a bug upstream in Devito-fwi — it crops an unbound `grad` at
  `devitofwi/waveengine/acoustic.py:616` — so the loss-only shortcut is
  unusable. The forward solve dominates, so the wasted adjoint is a modest cost
  rather than a doubling. Worth reporting upstream.
- **`PostProcessVP` carries one local fix** relative to Devito-fwi upstream
  (`-grad/vp³` → `-2·grad/vp³`, the correct chain rule for `m = 1/vp²`). The
  gradient check depends on it — with the upstream expression the adjoint is
  off by exactly 2x. Documented in [THIRD_PARTY.md](THIRD_PARTY.md), and also
  worth sending upstream.
- **Two implicit devito dependencies** had to be named in
  `devito_fwi/tesseract_requirements.txt`, because the conda env supplies them
  and a clean image does not: `matplotlib` (imported at module level by
  `devitofwi/waveengine/acoustic.py`) and `pytest` (devito 4.8.23's bundled
  `examples/seismic/model.py` carries a module-level
  `@pytest.mark.parametrize`, so it is an *import-time* dependency).
  A missing one kills the forked workers, which the pool now reports as a
  clear error instead of a bare `ConnectionResetError`.
