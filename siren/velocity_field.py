# siren/velocity_field.py
"""SIREN as a velocity-model parameterisation for FWI.

The FWI engine (devitofwi/Devito) hands back dL/dvp on the grid. This module
turns the grid into a coordinate -> velocity network so the unknowns become the
network weights instead of the 601x221 grid values:

    vp(x, z) = vmin + (vmax - vmin) * sigmoid(SIREN([x_norm, z_norm]))

and chains the FWI gradient into a parameter gradient with a single VJP:

    dL/dtheta = (dvp/dtheta)^T @ (dL/dvp)

The FWI side never needs to know a network is involved -- it still sees a
(nx, nz) velocity array in and a (nx, nz) gradient out.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .siren_network import init_mlp_params, SIREN_neural, DEFAULT_OMEGA_0


def make_grid(shape, spacing, origin=(0.0, 0.0), isotropic=False):
    """Coordinates of every grid node, normalised for the network input.

    Args:
        isotropic: when False (default) each axis is scaled independently onto
            [-1, 1]. That distorts the aspect ratio -- on a 9.0 x 3.3 km model
            one normalised unit is 4.5 km along x but 1.65 km along z, so a
            single ``omega_0`` produces spatial wavelengths 2.7x shorter in z
            than in x. When True both axes are divided by the *same* factor
            (half the longer side), so x spans [-1, 1] and z spans the shorter
            [-0.37, 0.37]: one normalised unit is the same distance either way
            and ``omega_0`` maps to a single physical wavelength.

    Returns an (nx*nz, 2) float32 array ordered to match ``vp.ravel()`` for a
    (nx, nz) C-ordered array.
    """
    nx, nz = shape
    x = np.arange(nx) * spacing[0] + origin[0]
    z = np.arange(nz) * spacing[1] + origin[1]
    if isotropic:
        # Shared scale factor: centre each axis, then divide both by the same
        # half-extent so the physical aspect ratio survives.
        half = 0.5 * max(x.max() - x.min(), z.max() - z.min())
        xn = (x - 0.5 * (x.min() + x.max())) / half
        zn = (z - 0.5 * (z.min() + z.max())) / half
    else:
        xn = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        zn = 2.0 * (z - z.min()) / (z.max() - z.min()) - 1.0
    XN, ZN = np.meshgrid(xn, zn, indexing='ij')  # (nx, nz), C-order == vp.ravel()
    return np.stack([XN.ravel(), ZN.ravel()], axis=-1).astype(np.float32)


class SirenVelocity:
    """SIREN parameterisation of a velocity model.

    Args:
        shape: (nx, nz)
        spacing: (dx, dz)
        vmin, vmax: velocity bounds; the sigmoid output is mapped into this range
        hidden: hidden layer widths
        omega_0: SIREN frequency scaling
        seed: PRNG seed
        origin: grid origin
        mask: optional (nx, nz) mask; where it is 0 the velocity is pinned to
            ``mask_value`` instead of being produced by the network. Masking the
            grid gradient alone is not enough here: every weight of a SIREN
            affects every grid node, so a zero gradient in the water layer does
            not stop an update to the weights from changing the velocity there.
            Pinning the value in the forward map is what actually freezes it,
            and it also makes the chain rule consistent (dvp/dtheta is zero
            exactly where the masked gradient is).
        mask_value: velocity used where ``mask`` is 0
        isotropic: normalise both coordinate axes by the same factor instead of
            stretching each onto [-1, 1]; see ``make_grid``. Defaults to False
            so existing runs are unchanged.
    """

    def __init__(self, shape, spacing, vmin, vmax,
                 hidden=(256, 256, 256, 256, 256),
                 omega_0=DEFAULT_OMEGA_0, seed=0, origin=(0.0, 0.0),
                 mask=None, mask_value=1.5, isotropic=False):
        self.shape = tuple(shape)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.omega_0 = float(omega_0)
        self.mask = (None if mask is None else
                     jnp.asarray(np.asarray(mask, dtype=np.float32).reshape(self.shape)))
        self.mask_value = float(mask_value)

        self.isotropic = bool(isotropic)
        self.coords = jnp.asarray(make_grid(shape, spacing, origin,
                                            isotropic=self.isotropic))
        layer_widths = [2] + list(hidden) + [1]
        self.params = init_mlp_params(layer_widths, jax.random.PRNGKey(seed),
                                      omega_0=self.omega_0)

        # flat <-> pytree conversion, so scipy optimisers can be used too
        self._flat0, self._unravel = jax.flatten_util.ravel_pytree(self.params)
        self.nparams = int(self._flat0.size)

    # ---------------------------------------------------------------- forward

    def _vp_from_params(self, params):
        """(nx, nz) velocity from network parameters (jax array)."""
        out = SIREN_neural(self.coords, params, self.omega_0)  # (N, 1)
        vp = self.vmin + (self.vmax - self.vmin) * jax.nn.sigmoid(out[:, 0])
        vp = vp.reshape(self.shape)
        if self.mask is not None:
            vp = self.mask * vp + (1.0 - self.mask) * self.mask_value
        return vp

    def vp(self, params=None):
        """Velocity model as a float32 numpy array, ready for the FWI engine."""
        params = self.params if params is None else params
        return np.asarray(self._vp_from_params(params), dtype=np.float32)

    # --------------------------------------------------------------- backward

    def chain_grad(self, grad_vp, params=None):
        """Chain a grid gradient dL/dvp into a parameter gradient dL/dtheta.

        Args:
            grad_vp: (nx, nz) or (nx*nz,) array from the FWI engine
            params: parameters the gradient was evaluated at (defaults to current)

        Returns:
            Parameter gradient, same pytree structure as ``self.params``.
        """
        params = self.params if params is None else params
        g = jnp.asarray(np.asarray(grad_vp, dtype=np.float32).reshape(self.shape))
        _, vjp_fn = jax.vjp(self._vp_from_params, params)
        return vjp_fn(g)[0]

    # ------------------------------------------------------------- flat views

    def flatten(self, params=None):
        params = self.params if params is None else params
        return np.asarray(jax.flatten_util.ravel_pytree(params)[0], dtype=np.float64)

    def unflatten(self, flat):
        return self._unravel(jnp.asarray(flat, dtype=jnp.float32))

    # ------------------------------------------------------------- pretraining

    def fit(self, vp_target, n_epochs=2000, learning_rate=1e-4,
            warmup_steps=200, max_grad_norm=1.0, verbose_every=200):
        """Fit the network to a given velocity model (initial-model warm start).

        Without this, the randomly-initialised network starts from a
        meaningless velocity field and FWI has nowhere sensible to descend from.
        Uses the same optax schedule/clipping recipe as the reference repo.
        """
        target = jnp.asarray(np.asarray(vp_target, dtype=np.float32))

        def loss_fn(params):
            return jnp.mean((self._vp_from_params(params) - target) ** 2)

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=max(n_epochs - warmup_steps, 1),
            end_value=learning_rate * 0.1,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule),
        )
        opt_state = optimizer.init(self.params)
        value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        params = self.params
        history = []
        for t in range(n_epochs):
            val, grads = value_and_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            if verbose_every and t % verbose_every == 0:
                rmse = float(jnp.sqrt(val))
                print(f'  fit epoch {t}: MSE = {float(val):.6e}  RMSE = {rmse:.4f} km/s',
                      flush=True)
                history.append((t, float(val)))

        self.params = params
        return history
