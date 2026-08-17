"""Physics Tesseract: velocity model Vp -> acoustic FWI loss.

This is the half JAX cannot differentiate. The forward map is a Devito-compiled
finite-difference solve of the acoustic wave equation over ``ns`` shots, and the
gradient comes from the discrete adjoint-state method that ``devitofwi``
implements by hand:

    apply                        L(vp) = 0.5 * sum_s ||d_mod(vp) - d_obs||^2
    vector_jacobian_product      dL/dvp = adjoint-state gradient

Because ``L`` is scalar, the VJP is just the gradient scaled by the incoming
cotangent - which is why ``AcousticWave2D._loss_grad`` can be used verbatim as
the VJP endpoint. No JAX tracing ever enters this module; the boundary is
exactly where the AD strategy changes from autodiff to a hand-written adjoint.

Shots are split across a persistent pool of forked workers, each owning its own
propagator, as in the reference script. The pool is built on the first call and
reused, so the Devito JIT compilation and the observed-data modelling are paid
once per served Tesseract rather than once per gradient.
"""

import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

# Devito's threading must be configured before it is imported anywhere.
os.environ.setdefault('DEVITO_LANGUAGE', 'C')
os.environ.setdefault('DEVITO_LOGGING', 'ERROR')
os.environ.setdefault('OMP_NUM_THREADS', '1')
# devitofwi wraps its shot loop in tqdm; with one worker per shot that is a
# progress bar per process per call, all interleaved on the same stderr.
os.environ.setdefault('TQDM_DISABLE', '1')

# pylops imports cupyx, which warns about its experimental jit on every forked
# worker - one line per process per run, drowning the actual log.
import warnings  # noqa: E402
warnings.filterwarnings('ignore', message='.*cupyx.jit.rawkernel.*')

from tesseract_core.runtime import Array, Differentiable, Float32, Float64  # noqa: E402

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import problem_config as problem  # noqa: E402

NX, NZ = problem.SHAPE

# Workers. One per shot is the right default: the objective is limited by the
# slowest worker, so any worker holding two shots doubles the epoch time no
# matter how idle the rest are. Capped at the physical core count so a machine
# smaller than the survey does not thrash, and overridable with FWI_NPROCS.
NPROCS = int(os.environ.get(
    'FWI_NPROCS', min(problem.PAR['ns'], os.cpu_count() or 8)))


#
# Schemata
#


class InputSchema(BaseModel):
    vp: Differentiable[Array[(NX, NZ), Float32]] = Field(
        description='Velocity model in km/s on the (nx, nz) grid.'
    )


class OutputSchema(BaseModel):
    loss: Differentiable[Float64] = Field(
        description='L2 data misfit summed over all shots.'
    )


#
# Shot-parallel Devito engine
#


def _worker(conn, isrcs):
    """Persistent worker owning the propagators for shots ``isrcs``.

    Protocol: ('mod',) once to model the observed data and build the inversion
    engine, then repeated ('grad', vp) messages, then ('stop',).
    """
    from pylops.basicoperators import Identity

    from devitofwi.waveengine.acoustic import AcousticWave2D
    from devitofwi.loss.l2 import L2
    from devitofwi.postproc.acoustic import PostProcessVP

    from devito import configuration
    configuration['log-level'] = 'ERROR'

    vp_true = problem.load_vp_true()
    msk = problem.water_mask(vp_true)
    x_s, x_r = problem.source_receiver_positions()

    common = dict(src_type='Ricker', f0=problem.PAR['freq'],
                  space_order=problem.SPACE_ORDER, nbl=problem.NBL,
                  clearcache=False)

    amod = AcousticWave2D(problem.SHAPE, problem.ORIGIN, problem.SPACING,
                          x_s[isrcs, 0], x_s[isrcs, 1], x_r[:, 0], x_r[:, 1],
                          0., problem.TMAX, vp=vp_true, **common)
    ainv = None
    postproc = PostProcessVP(scaling=1., mask=msk)

    while True:
        msg = conn.recv()
        if msg[0] == 'stop':
            conn.close()
            return

        if msg[0] == 'mod':
            dobs, dtobs = amod.mod_allshots()
            l2loss = L2(Identity(int(np.prod(dobs.shape[1:]))),
                        dobs.reshape(len(isrcs), -1))
            ainv = AcousticWave2D(problem.SHAPE, problem.ORIGIN, problem.SPACING,
                                  x_s[isrcs, 0], x_s[isrcs, 1],
                                  x_r[:, 0], x_r[:, 1],
                                  0., problem.TMAX,
                                  vprange=(vp_true.min(), vp_true.max()),
                                  loss=l2loss, **common)
            conn.send((dobs, dtobs))

        elif msg[0] == 'modvp':
            # model this chunk's shots through an arbitrary vp, resampled onto
            # the time axis of the observed data so the two are comparable
            _, vp, dt = msg
            vp_keep = amod.vp
            amod.vp = np.ascontiguousarray(vp, dtype=np.float32)
            try:
                conn.send(amod.mod_allshots(dt=dt))
            finally:
                amod.vp = vp_keep

        elif msg[0] == 'modshot':
            # one shot only, through an arbitrary vp: the live waveform figure
            # needs a single gather per iteration, not the whole survey
            _, ishot, vp, dt = msg
            if ishot not in isrcs:
                continue                       # another worker owns this shot
            vp_keep = amod.vp
            amod.vp = np.ascontiguousarray(vp, dtype=np.float32)
            try:
                model = amod._create_model(problem.SHAPE, problem.ORIGIN,
                                           problem.SPACING, amod.vp,
                                           problem.SPACE_ORDER, problem.NBL,
                                           amod.fs)
                d, dtm = amod._mod_oneshot(
                    model, int(np.where(isrcs == ishot)[0][0]), dt=dt)
                conn.send((d, dtm))
            finally:
                amod.vp = vp_keep

        elif msg[0] == 'grad':
            # NOTE: always computed with the gradient. ``_loss_grad`` has an
            # upstream bug on the ``computegrad=False`` path - it crops an
            # unbound ``grad`` at acoustic.py:616 - so the loss-only shortcut
            # is not usable. The forward solve dominates anyway, so an unused
            # adjoint is a modest waste rather than a doubling.
            _, vp, want_grad = msg
            loss, grad = ainv._loss_grad(vp, postprocess=postproc.apply)
            conn.send((loss, grad if want_grad else None))


class ShotPool:
    """Persistent pool of shot-parallel propagators."""

    def __init__(self, nprocs, nsrc):
        self.chunks = [c for c in np.array_split(np.arange(nsrc), nprocs)
                       if len(c) > 0]
        ctx = mp.get_context('fork')
        self.conns, self.procs = [], []
        self.closed = False
        for chunk in self.chunks:
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child_conn, chunk),
                            daemon=True)
            p.start()
            child_conn.close()
            self.conns.append(parent_conn)
            self.procs.append(p)

    def _gather_data(self, msg):
        """Send ``msg`` to every worker and concatenate the gathers in order."""
        for c in self.conns:
            c.send(msg)
        out = [c.recv() for c in self.conns]
        nt_max = min(d.shape[1] for d, _ in out)
        return (np.concatenate([d[:, :nt_max] for d, _ in out], axis=0),
                out[0][1])

    def model_vp(self, vp, dt):
        """Model all shots through ``vp``, resampled at ``dt`` (s)."""
        return self._gather_data(('modvp', np.ascontiguousarray(
            vp, dtype=np.float32), dt))

    def model_shot(self, ishot, vp, dt):
        """Model the single shot ``ishot`` through ``vp``, resampled at ``dt``.

        Only the worker owning that shot answers; the others skip the message.
        One gather costs ~1/ns of a full modelling pass, which is what makes
        the live waveform figure affordable every few iterations.
        """
        iw = next(k for k, c in enumerate(self.chunks) if ishot in c)
        self.conns[iw].send(('modshot', int(ishot),
                             np.ascontiguousarray(vp, dtype=np.float32), dt))
        return self.conns[iw].recv()

    def model_observed(self):
        for c in self.conns:
            c.send(('mod',))
        out = []
        for c in self.conns:
            try:
                out.append(c.recv())
            except (EOFError, ConnectionResetError):
                # A worker died before answering. Its traceback went to the
                # process's stderr, which inside a served container is not the
                # HTTP response - so say plainly what happened rather than let
                # a bare ConnectionResetError surface as a 500.
                raise RuntimeError(
                    'a Devito worker died while modelling the observed data; '
                    'check the container/process stderr for its traceback '
                    '(a missing import in the image is the usual cause)'
                ) from None
        return out

    def loss_grad_vp(self, vp, computegrad=True):
        """Loss and, if asked, the adjoint-state gradient dL/dvp."""
        vp = np.ascontiguousarray(vp, dtype=np.float32)
        for c in self.conns:
            c.send(('grad', vp, computegrad))
        loss = 0.
        grad = np.zeros(problem.SHAPE, dtype=np.float64)
        for c in self.conns:
            try:
                l, g = c.recv()
            except EOFError:
                # A worker died mid-solve. Without this the parent blocks on
                # recv() forever and the whole inversion hangs with no message.
                raise RuntimeError(
                    'a Devito worker died during the solve; its traceback is '
                    'above (a wave-equation instability or an out-of-memory '
                    'kill are the usual causes)'
                ) from None
            loss += l
            if computegrad and g is not None:
                grad += g
        return float(loss), grad

    def close(self):
        if self.closed:
            return
        self.closed = True
        for c, p in zip(self.conns, self.procs):
            try:
                c.send(('stop',))
                c.close()
            except (BrokenPipeError, OSError):
                pass
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


_POOL = None
_DOBS = None
_LAST_GRAD = None


def _pool():
    """The shot pool, built and warmed on first use.

    Both endpoints go through here, so whichever is called first pays the
    Devito compilation and the observed-data modelling and every later call is
    just the solve.
    """
    global _POOL, _DOBS
    if _POOL is None:
        import atexit
        pool = ShotPool(NPROCS, problem.PAR['ns'])
        out = pool.model_observed()
        # Keep the observed gathers: the workflow's figures want them, and
        # re-modelling them would be a whole extra forward pass.
        nt_max = min(d.shape[1] for d, _ in out)
        _DOBS = (np.concatenate([d[:, :nt_max] for d, _ in out], axis=0),
                 out[0][1])
        atexit.register(pool.close)
        _POOL = pool
    return _POOL


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    loss, _ = _pool().loss_grad_vp(inputs.vp, computegrad=False)
    return OutputSchema(loss=loss)


#
# Gradient endpoint: the hand-written adjoint, exposed as a VJP
#


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
) -> dict:
    """dL/dvp from the adjoint-state method, scaled by the cotangent.

    The output is the scalar loss, so the Jacobian is a single row and the VJP
    reduces to ``cotangent * dL/dvp`` - no transpose solve beyond the adjoint
    the FWI engine already runs.
    """
    global _LAST_GRAD
    if vjp_inputs != {'vp'} or vjp_outputs != {'loss'}:
        raise ValueError(
            f'only d(loss)/d(vp) is available, got {vjp_outputs}/{vjp_inputs}'
        )
    _, grad = _pool().loss_grad_vp(inputs.vp, computegrad=True)
    # Keep the unscaled gradient so a caller that wants to *look* at dL/dvp -
    # the workflow's live figure does - can read it back instead of paying
    # another 64-shot adjoint solve for a picture.
    _LAST_GRAD = grad
    w = float(np.asarray(cotangent_vector['loss']))
    return {'vp': (w * grad).astype(np.float32)}


def abstract_eval(abstract_inputs):
    return {'loss': {'shape': (), 'dtype': 'float64'}}


#
# Helper for the workflow (not a Tesseract endpoint)
#


def loss_and_grad(vp):
    """Loss and dL/dvp in one solve - used to report the reference gradient."""
    return _pool().loss_grad_vp(vp, computegrad=True)


def last_gradient():
    """The dL/dvp of the most recent ``vector_jacobian_product`` call.

    The live gradient figure wants the grid gradient that the optimiser just
    used. Recomputing it costs a full 64-shot adjoint solve - as much as the
    iteration itself - so it is cached on the way through the VJP instead.
    Returns None if no VJP has run yet.
    """
    return _LAST_GRAD


def observed_data():
    """The observed gathers and their time step, ``(dobs, dt)``.

    Modelled once when the pool is warmed and cached, so the workflow's figures
    do not pay a second forward pass for them.
    """
    _pool()
    return _DOBS


def model_all(vp, dt):
    """Model every shot through ``vp``, resampled at ``dt`` (s)."""
    return _pool().model_vp(vp, dt)


def model_shot(ishot, vp, dt):
    """Model the single shot ``ishot`` through ``vp``, resampled at ``dt``."""
    return _pool().model_shot(ishot, vp, dt)
