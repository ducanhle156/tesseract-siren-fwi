# siren/siren_network.py
"""SIREN network (Sitzmann et al. 2020), in JAX.

Adapted from KronosAI_solutions/Task1/utils/siren_network.py
(https://github.com/nguyenvanhaibk92/KronosAI_solutions).

The only change w.r.t. the original is that ``omega_0`` is passed explicitly
instead of being read from that project's global ``CONFIG`` (which carries the
electromagnetic-PINN settings we do not want here). The maths -- the
``first_layer_sine_init`` / ``sine_init`` scheme and the sine forward pass --
is unchanged.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap
from functools import partial

DEFAULT_OMEGA_0 = 30.0


def init_mlp_params(layer_widths, rng_key, omega_0=DEFAULT_OMEGA_0):
    """Initialize parameters for a SIREN network (PyTorch-style init).

    Args:
        layer_widths: [input_dim, hidden1, ..., output_dim]
        rng_key: JAX PRNG key
        omega_0: frequency scaling of the sine activations

    Returns:
        List of [weight, bias] pairs, one per layer.
    """
    params = []
    keys = random.split(rng_key, len(layer_widths) - 1)

    for i, (n_in, n_out) in enumerate(zip(layer_widths[:-1], layer_widths[1:])):
        weight_key, _bias_key = random.split(keys[i])

        if i == 0:  # first_layer_sine_init: U(-1/n_in, 1/n_in)
            weights = jax.random.uniform(
                weight_key, shape=(n_in, n_out),
                minval=-1 / n_in, maxval=1 / n_in,
            )
        else:  # sine_init: U(-sqrt(6/n_in)/omega_0, +...)
            bound = jnp.sqrt(6 / n_in) / omega_0
            weights = jax.random.uniform(
                weight_key, shape=(n_in, n_out),
                minval=-bound, maxval=bound,
            )

        biases = jnp.zeros((n_out,))
        params.append([weights, biases])

    return params


def SIREN_neural_one_sample(x_input, params, omega_0=DEFAULT_OMEGA_0):
    """Forward pass for a single sample."""
    # First layer with sine activation
    x = jnp.sin(omega_0 * (x_input @ params[0][0] + params[0][1]))

    # Hidden layers with sine activation
    for i in range(1, len(params) - 1):
        x = jnp.sin(omega_0 * (x @ params[i][0] + params[i][1]))

    # Output layer: linear, no omega_0 scaling
    return (x @ params[-1][0]) + params[-1][1]


@partial(jax.jit, static_argnums=(2,))
def SIREN_neural(x_input, params, omega_0=DEFAULT_OMEGA_0):
    """Batched forward pass. x_input: (N, input_dim) -> (N, output_dim)."""
    return vmap(SIREN_neural_one_sample, in_axes=(0, None, None))(
        x_input, params, omega_0)
