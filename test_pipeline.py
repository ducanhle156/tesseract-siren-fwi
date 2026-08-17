"""Checks for the two-Tesseract pipeline.

Run with the env that has devito:

    /home/mtran/miniconda3/envs/geo_jxli/bin/python test_pipeline.py

The gradient check is the substantive one - it is what catches a sign error or
a transposed grid in either VJP. Each check is a wave solve over all 64 shots,
so budget a while: one loss+gradient is ~3 minutes on 64 cores.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import problem_config as problem  # noqa: E402

FAILURES = []


def check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    print(f'  [{status}] {name}' + (f'  {detail}' if detail else ''), flush=True)
    if not condition:
        FAILURES.append(name)


def load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_problem():
    print('\nproblem_config.py')
    vp_true = problem.load_vp_true()
    check('grid shape', vp_true.shape == problem.SHAPE,
          f'{vp_true.shape}')
    # Both Tesseracts must freeze exactly the same cells, or the chain rule is
    # inconsistent - one side pins a cell the other still reports a gradient
    # for. They both call water_mask, so this checks it is deterministic
    # whether or not it is handed the model.
    check('water_mask agrees with itself',
          np.array_equal(problem.water_mask(), problem.water_mask(vp_true)))
    # Tolerance because the model is float32: Marmousi's 5.5 km/s peak reads
    # back as 5.5000004768, which is not a bounds violation worth failing on.
    tol = 1e-5
    check('declared bounds contain the model',
          problem.VP_MIN - tol <= vp_true.min()
          and vp_true.max() <= problem.VP_MAX + tol,
          f'model {vp_true.min():.4f}-{vp_true.max():.4f}, '
          f'declared {problem.VP_MIN}-{problem.VP_MAX}')


def test_siren(siren):
    print('\nsiren-model Tesseract')
    theta = siren.init_theta()
    check('theta is flat float32', theta.ndim == 1 and theta.dtype == np.float32,
          f'{theta.shape}')

    vp = np.asarray(siren.apply(siren.InputSchema(theta=theta)).vp)
    check('vp shape', vp.shape == problem.SHAPE, f'{vp.shape}')
    check('vp within the sigmoid bounds',
          vp.min() >= problem.VP_MIN - 1e-4 and vp.max() <= problem.VP_MAX + 1e-4,
          f'{vp.min():.3f}-{vp.max():.3f}')
    mask = problem.water_mask()
    check('water layer pinned',
          np.allclose(vp[mask == 0], problem.MASK_VALUE),
          f'{np.unique(vp[mask == 0])}')

    # VJP against a finite difference of a randomly scalarised output. This is
    # pure JAX, so it should be tight.
    rng = np.random.default_rng(0)
    w = rng.standard_normal(problem.SHAPE).astype(np.float32)
    w /= np.linalg.norm(w)

    def scalar(th):
        return float((np.asarray(siren.apply(siren.InputSchema(theta=th)).vp) * w).sum())

    g = np.asarray(siren.vector_jacobian_product(
        siren.InputSchema(theta=theta), {'theta'}, {'vp'}, {'vp': w})['theta'])
    v = rng.standard_normal(theta.shape).astype(np.float32)
    v /= np.linalg.norm(v)
    eps = 1e-3
    fd = (scalar(theta + eps * v) - scalar(theta - eps * v)) / (2 * eps)
    ad = float(g @ v)
    rel = abs(fd - ad) / max(abs(ad), 1e-30)
    check('JAX VJP matches finite difference', rel < 1e-2,
          f'adjoint {ad:+.6e} vs fd {fd:+.6e}, rel.err {rel:.2e}')


def test_devito(fwi):
    print('\ndevito-fwi Tesseract (each check is a wave solve)')
    vp_true = problem.load_vp_true()
    mask = problem.water_mask(vp_true)

    loss_true = fwi.apply(fwi.InputSchema(vp=vp_true.astype(np.float32))).loss
    check('loss vanishes at the true model', abs(loss_true) < 1e-12,
          f'{loss_true:.3e}')

    from scipy.ndimage import gaussian_filter
    vp0 = gaussian_filter(vp_true, sigma=[15, 10]) * mask
    vp0[vp0 == 0] = problem.MASK_VALUE
    vp0 = vp0.astype(np.float32)

    loss0 = fwi.apply(fwi.InputSchema(vp=vp0)).loss
    check('loss positive away from the true model', loss0 > 0, f'{loss0:.6e}')

    g = np.asarray(fwi.vector_jacobian_product(
        fwi.InputSchema(vp=vp0), {'vp'}, {'loss'},
        {'loss': np.float64(1.0)})['vp']).astype(np.float64)
    check('gradient shape', g.shape == problem.SHAPE, f'{g.shape}')
    check('gradient masked in the water', np.all(g[mask == 0] == 0.))

    g2 = np.asarray(fwi.vector_jacobian_product(
        fwi.InputSchema(vp=vp0), {'vp'}, {'loss'},
        {'loss': np.float64(2.5)})['vp']).astype(np.float64)
    # The gradient crosses the schema as float32, so exact equality is not
    # available: compare at float32 resolution rather than float64.
    lin = np.abs(g2 - 2.5 * g).max() / max(np.abs(2.5 * g).max(), 1e-300)
    check('VJP is linear in the cotangent', lin < 1e-6, f'rel {lin:.2e}')

    # Single-cell finite differences: the cleanest test of the adjoint, since
    # one cell at a time keeps the other 8455 second-order terms out of it.
    def L(v):
        return fwi.apply(fwi.InputSchema(vp=v.astype(np.float32))).loss

    idx = np.dstack(np.unravel_index(np.argsort(-np.abs(g).ravel()),
                                     g.shape))[0][:3]
    worst = 0.
    for (i, j) in idx:
        eps = 1e-2
        vp_p = vp0.copy(); vp_p[i, j] += eps
        vp_m = vp0.copy(); vp_m[i, j] -= eps
        fd = (L(vp_p) - L(vp_m)) / (2 * eps)
        worst = max(worst, abs(fd - g[i, j]) / max(abs(g[i, j]), 1e-30))
    check('adjoint matches single-cell finite differences', worst < 5e-3,
          f'worst rel.err {worst:.2e} over {len(idx)} cells')


def test_composed():
    print('\ncomposed pipeline (theta -> Vp -> loss)')
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from tesseract_core import Tesseract
    from tesseract_jax import apply_tesseract

    s = Tesseract.from_tesseract_api(HERE / 'siren_model' / 'tesseract_api.py')
    f = Tesseract.from_tesseract_api(HERE / 'devito_fwi' / 'tesseract_api.py')

    def objective(th):
        return apply_tesseract(f, {'vp': apply_tesseract(s, {'theta': th})['vp']})['loss']

    siren = load('_s', HERE / 'siren_model' / 'tesseract_api.py')
    theta0 = jnp.asarray(siren.init_theta())

    loss, grad = jax.value_and_grad(objective)(theta0)
    grad = np.asarray(grad)
    check('composed gradient is finite and non-zero',
          np.all(np.isfinite(grad)) and np.linalg.norm(grad) > 0,
          f'|g| {np.linalg.norm(grad):.6e}')

    # Along the gradient direction - the one Adam steps along, and the best
    # conditioned. A random direction in 8577 dimensions is nearly orthogonal
    # to the gradient, so its directional derivative is tiny and its finite
    # difference is dominated by the other components' curvature.
    v = grad / np.linalg.norm(grad)
    ad = float(grad @ v)
    best = min(
        (abs((float(objective(theta0 + e * jnp.asarray(v)))
              - float(objective(theta0 - e * jnp.asarray(v)))) / (2 * e) - ad)
         / abs(ad), e)
        for e in (1e-3, 3e-4)
    )
    check('composed dL/dtheta matches finite difference', best[0] < 3e-2,
          f'rel.err {best[0]:.2e} (eps {best[1]:.0e})')


if __name__ == '__main__':
    print(f'Grid {problem.SHAPE}, {problem.PAR["ns"]} shots')
    test_problem()
    test_siren(load('_siren', HERE / 'siren_model' / 'tesseract_api.py'))
    test_devito(load('_fwi', HERE / 'devito_fwi' / 'tesseract_api.py'))
    test_composed()

    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES))
        sys.exit(1)
    print('all checks passed')
