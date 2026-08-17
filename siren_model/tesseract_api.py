"""Model Tesseract: SIREN weights theta -> velocity model Vp.

    vp(x, z) = vmin + (vmax - vmin) * sigmoid(SIREN([x_norm, z_norm]))

The forward map is pure JAX, so the gradient endpoints are the stock
``jax_recipes`` wrappers: JAX differentiates the network for us, and
``vector_jacobian_product`` is exactly the

    dL/dtheta = (dvp/dtheta)^T @ (dL/dvp)

chain rule that ``SirenVelocity.chain_grad`` performs in the monolithic script.
The difference is that here it is an endpoint the physics Tesseract's gradient
can be handed to across a process boundary, rather than a method call inside
one interpreter.

theta crosses the boundary as a flat float32 vector (the ravelled pytree) so the
schema is a single array rather than a nested list of per-layer matrices. The
unravel function is rebuilt from the architecture on each call, which is cheap
and keeps the Tesseract stateless.
"""

import os
import sys
from pathlib import Path

# The coordinate network is tiny (a few thousand weights over ~8k grid nodes),
# so a GPU buys nothing and costs a transfer per call - and on a shared box it
# competes with whatever else is resident. Must be set before jax is imported.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import equinox as eqx
import jax
import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.jax_recipes import (
    jax_abstract_eval,
    jax_apply,
    jax_jacobian,
    jax_jvp,
    jax_vjp,
)

# ``problem_config.py`` and the ``siren`` package sit one level up, in the pipeline
# directory. Inside a built container both are copied in via ``package_data``
# in tesseract_config.yaml and land next to this file, so both layouts work.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import problem_config as problem  # noqa: E402
from siren.siren_network import init_mlp_params  # noqa: E402
from siren.velocity_field import SirenVelocity  # noqa: E402


#
# Static configuration
#
# The network architecture and grid are fixed at build time rather than passed
# per call: they define what this Tesseract *is*, and making them inputs would
# force a retrace on every call. Overridable by env var so the same module
# serves the small and full problems.
#

# The water mask has to be identical on both sides of the boundary, or the
# chain rule is inconsistent: the model side would pin a cell the physics side
# still reports a gradient for. Both take it from ``problem.water_mask``.
# The sigmoid bounds are stated constants, so those at least need no knowledge
# of the answer.
_MASK = problem.water_mask()
VMIN, VMAX = problem.VP_MIN, problem.VP_MAX

HIDDEN = problem.SIREN_HIDDEN
OMEGA0 = problem.SIREN_OMEGA0

# One SirenVelocity instance provides the coordinate grid, the masked forward
# map and the flat<->pytree ravel. Its own ``params`` are only used as the
# template for the unravel and as the initial theta served by ``init_theta``.
_SIREN = SirenVelocity(
    problem.SHAPE, problem.SPACING, VMIN, VMAX,
    hidden=HIDDEN, omega_0=OMEGA0, seed=problem.SIREN_SEED,
    origin=problem.ORIGIN, mask=_MASK, mask_value=problem.MASK_VALUE,
    isotropic=True,
)

NPARAMS = _SIREN.nparams
NX, NZ = problem.SHAPE


#
# Schemata
#


class InputSchema(BaseModel):
    theta: Differentiable[Array[(NPARAMS,), Float32]] = Field(
        description='Flat SIREN weight vector (ravelled parameter pytree).'
    )


class OutputSchema(BaseModel):
    vp: Differentiable[Array[(NX, NZ), Float32]] = Field(
        description='Velocity model in km/s on the (nx, nz) grid.'
    )


#
# Required endpoints
#


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    params = _SIREN._unravel(inputs['theta'])
    return {'vp': _SIREN._vp_from_params(params)}


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**jax_apply(apply_jit, inputs))


#
# Jax-handled gradient endpoints
#


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return jax_jacobian(apply_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict,
):
    return jax_jvp(apply_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    """dL/dtheta from dL/dvp - the chain rule of the SIREN parameterisation."""
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)


def abstract_eval(abstract_inputs):
    return jax_abstract_eval(apply_jit, abstract_inputs)


#
# Helper for the workflow (not a Tesseract endpoint)
#


def init_theta(seed=problem.SIREN_SEED, init_seed=0, gain=None, target_std=0.2):
    """A random starting theta, as in the reference script's random start.

    The stock Sitzmann initialisation puts every bias at zero and scales the
    hidden weights by 1/omega_0, so the network output sits near 0 and vp
    collapses to a near-constant half-space. Redrawing the sine-layer biases
    over a full phase turn, and rescaling the output layer until the model has
    a target standard deviation, gives a genuinely random starting model.
    """
    params = init_mlp_params([2, *HIDDEN, 1], jax.random.PRNGKey(seed),
                             omega_0=OMEGA0)

    def randomize(p, g):
        key = jax.random.PRNGKey(init_seed)
        nl = len(p)
        out = []
        for i, (w, b) in enumerate(p):
            if i < nl - 1:
                key, kb = jax.random.split(key)
                b = jax.random.uniform(kb, b.shape,
                                       minval=-np.pi / OMEGA0,
                                       maxval=np.pi / OMEGA0)
            else:
                w, b = w * g, b * g
            out.append([w, b])
        return out

    if gain is None:
        # Bisection in log-gain: the std below the seabed grows monotonically
        # with the gain and saturates at (vmax - vmin)/2.
        sub = np.asarray(_MASK, dtype=bool).ravel()

        def std_at(g):
            vp = _SIREN._vp_from_params(randomize(params, g))
            return float(np.asarray(vp).ravel()[sub].std())

        lo, hi = 1e-2, 1e4
        if std_at(hi) < target_std:
            gain = hi
        else:
            for _ in range(40):
                mid = np.sqrt(lo * hi)
                if std_at(mid) < target_std:
                    lo = mid
                else:
                    hi = mid
            gain = float(np.sqrt(lo * hi))

    flat, _ = jax.flatten_util.ravel_pytree(randomize(params, gain))
    return np.asarray(flat, dtype=np.float32)
