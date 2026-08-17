"""Compose the two Tesseracts into one differentiable pipeline and invert.

    theta --[siren-model]--> Vp --[devito-fwi]--> loss

Both arrows are Tesseract calls. ``jax.value_and_grad`` differentiates straight
through the composition: JAX asks each Tesseract for its ``vector_jacobian_
product``, so the chain rule

    dL/dtheta = (dvp/dtheta)^T @ (dL/dvp)

is assembled from two independently-implemented gradients - JAX autodiff on the
SIREN side, a hand-written adjoint-state solve on the Devito side. Neither
Tesseract knows the other exists, and the workflow never sees a wavefield or a
network weight matrix.

The reporting is deliberately the same as the monolithic
``AcousticVel_L2_SIREN_random.py``: the same ``FigureKit`` figures, the same
live tmp*.png diagnostics, the same movie snapshots, the same timing report and
the same ``result.npz``. Only the gradient path differs, so the outputs should
be comparable side by side.

Outputs (in ``--outdir``)
    SirenInit.png      true model, random initial model
    Data.png           observed shot gathers
    Gradient.png       first gradient w.r.t. VP
    InvertedVP.png     true / random-initial / inverted models
    Profiles.png       vertical VP profiles at three distances
    Waveform.png       observed vs modelled waveforms
    Loss.png           convergence (data misfit + model error)
    GradTheta.png      dL/dtheta per layer at the final model, and its history
    InvertedVPtmp.png / Gradienttmp.png / GradThetatmp.png / Losstmp.png
                       live diagnostics
    Waveformtmp.png    live observed vs estimated wiggles
    snapshots.npy      movie frames (render with make_movie.py)
    result.npz         models, weights, histories, timings
    timing.txt         the timing report printed at the end
    code/              verbatim copy of the sources that produced the run

The problem is the reference script's: Marmousi on 601 x 221 at 15 m, 64 shots,
300 receivers, 8 Hz, SIREN 256x4 at omega_0 20, Adam at 1.5e-4 for 3000 epochs.
One loss+gradient is ~3 minutes on 64 cores, so a full run takes days - start
it detached.

Run:
    python workflow.py                    # the full inversion, defaults as above
    python workflow.py --benchmark        # time one loss+gradient and exit
    python workflow.py --check-grad       # finite-difference the composed gradient
    python workflow.py --served URL1 URL2 # against containers (see README)
"""

import argparse
import os
import time
import traceback
from pathlib import Path

T_START = time.time()

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
# devitofwi wraps its shot loop in tqdm; with one worker per shot that is a
# progress bar per process, all interleaved on the same stderr.
os.environ.setdefault('TQDM_DISABLE', '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent

# ``siren/`` and ``fwireport.py`` are vendored into this directory (copies of
# the ones next to the reference script) so the folder is self-contained: it
# needs only an environment, not the rest of the repo.
import sys
sys.path.insert(0, str(HERE))

import problem_config as problem  # noqa: E402
from fwireport import (FigureKit, SnapshotRecorder, Timings, apply_style,  # noqa: E402
                       format_report, hms, iteration_axis, panel_label,
                       snapshot_code, write_report, C_INV)

apply_style()

T_IMPORT = time.time()


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--maxiter', type=int, default=3000, help='Adam epochs')
    p.add_argument('--lr', type=float, default=1.5e-4,
                   help='Adam peak learning rate in weight space')
    p.add_argument('--warmup', type=int, default=50, help='Adam warmup steps')
    p.add_argument('--end-lr-frac', type=float, default=0.1,
                   help='final LR as a fraction of --lr (cosine decay)')
    p.add_argument('--check-grad', action='store_true',
                   help='finite-difference check of the composed gradient, '
                        'then exit')
    p.add_argument('--benchmark', action='store_true',
                   help='time a single loss/gradient evaluation and exit')
    p.add_argument('--served', nargs=2, metavar=('SIREN_URL', 'FWI_URL'),
                   help='connect to Tesseracts already served at these URLs '
                        'instead of running the API modules in-process')
    p.add_argument('--outdir', type=str, default=str(HERE / 'results'))
    p.add_argument('--fig-every', type=int, default=25,
                   help='iterations between the live tmp*.png figures')
    p.add_argument('--wiggle-every', type=int, default=10,
                   help='iterations between live waveform figures; <=0 off')
    p.add_argument('--wiggle-shot', type=int, default=-1,
                   help='shot index for that figure; <0 -> middle of the line')
    p.add_argument('--wiggle-tmax', type=float, default=0.,
                   help='last time (s) plotted there; <=0 -> whole record')
    p.add_argument('--snap-every', type=int, default=5,
                   help='iterations between recorded movie frames; <=0 off')
    p.add_argument('--no-figures', action='store_true',
                   help='skip every figure (still writes result.npz)')
    return p


ARGS = build_parser().parse_args()
ITER_LABEL = 'Adam epoch'


##################################################################
# Weight-space gradient diagnostics (as in the reference script)
##################################################################

def theta_norms(grad_theta, params):
    """Per-layer L2 norms of ``dL/dtheta`` and of ``theta`` itself.

    A single number for a large gradient hides what matters: whether the update
    actually reaches the hidden layers or only the last one.
    """
    gn, tn = [], []
    for (gw, gb), (w, b) in zip(grad_theta, params):
        gw, gb = np.asarray(gw, np.float64), np.asarray(gb, np.float64)
        w, b = np.asarray(w, np.float64), np.asarray(b, np.float64)
        gn.append(np.sqrt((gw ** 2).sum() + (gb ** 2).sum()))
        tn.append(np.sqrt((w ** 2).sum() + (b ** 2).sum()))
    return np.array(gn), np.array(tn)


def layer_labels(widths):
    """``['W1: 2->64', ...]`` from ``[2, 64, ..., 1]``."""
    return [f'W{k + 1}: {a}->{b}'
            for k, (a, b) in enumerate(zip(widths[:-1], widths[1:]))]


def format_theta_grad(gn, tn, widths, prefix='  '):
    """One-line-per-layer table of the weight-space gradient."""
    tot_g, tot_t = np.sqrt((gn ** 2).sum()), np.sqrt((tn ** 2).sum())
    L = [f'{prefix}dL/dtheta by layer '
         f'(total |g| = {tot_g:.4e}, |g|/|theta| = {tot_g / max(tot_t, 1e-30):.4e})',
         f'{prefix}  {"matrix":<16}{"|g|":>12}{"|theta|":>12}'
         f'{"|g|/|theta|":>14}{"share":>9}']
    for g, t, lab in zip(gn, tn, layer_labels(widths)):
        L.append(f'{prefix}  {lab:<16}{g:>12.4e}{t:>12.4e}'
                 f'{g / max(t, 1e-30):>14.4e}{100. * g / max(tot_g, 1e-30):>8.1f}%')
    return '\n'.join(L)


def plot_theta_grad(figpath, gh, th, widths, name='GradThetatmp.png',
                    final=False, g=None, t=None):
    """Weight-space gradient: per-layer magnitude now, and its history."""
    gh, th = np.atleast_2d(gh), np.atleast_2d(th)
    g = gh[-1] if g is None else np.asarray(g)
    t = th[-1] if t is None else np.asarray(t)
    nl = len(widths) - 1
    ramp = plt.cm.Blues(np.linspace(0.42, 0.95, nl))
    ticks = np.arange(nl)
    short = [f'W{k + 1}' for k in range(nl)]
    arch = '-'.join(str(w) for w in widths)

    def _bars(ax, vals, letter, title, ylabel):
        ax.bar(ticks, vals, color=ramp, width=0.68, edgecolor='none')
        ax.set_yscale('log')
        ax.set_xticks(ticks)
        ax.set_xticklabels(short)
        ax.set_xlabel('Weight matrix')
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y', which='major', ls=':', lw=0.5)
        ax.set_axisbelow(True)
        panel_label(ax, letter, title)

    ncol = 3 if final else 2
    fig, axs = plt.subplots(1, ncol, figsize=(7.2, 2.9 if final else 5.4 / 2))
    _bars(axs[0], g, 'a',
          'Final gradient' if final else f'Gradient (eval {gh.shape[0]})',
          '$\\|\\partial L/\\partial\\theta_\\ell\\|_2$')
    if final:
        _bars(axs[1], g / np.maximum(t, 1e-30), 'b', 'Relative to the weights',
              '$\\|\\partial L/\\partial\\theta_\\ell\\|/\\|\\theta_\\ell\\|$')

    ax = axs[-1]
    n = np.arange(1, gh.shape[0] + 1)
    for k in range(nl):
        ax.semilogy(n, np.maximum(gh[:, k], 1e-300), color=ramp[k], lw=1.0,
                    label=short[k])
    ax.semilogy(n, np.maximum(np.sqrt((gh ** 2).sum(axis=1)), 1e-300),
                color=C_INV, lw=1.4, label='all')
    ax.set_xlabel('Function evaluation')
    ax.set_ylabel('$\\|\\partial L/\\partial\\theta\\|_2$')
    ax.grid(True, which='major', ls=':', lw=0.5)
    ax.set_xlim(1, max(gh.shape[0], 2))
    panel_label(ax, 'c' if final else 'b', f'History ({arch})')
    ax.legend(loc='best', ncol=2, fontsize=6.5, handlelength=1.2,
              columnspacing=0.8, labelspacing=0.25, borderpad=0.3)

    fig.tight_layout(w_pad=1.6)
    iteration_axis(ax, max(gh.shape[0], 2))
    fig.savefig(os.path.join(figpath, name), **({} if final else {'dpi': 110}))
    plt.close(fig)


##################################################################
# Tesseract clients
##################################################################

def open_tesseracts(args):
    """The two Tesseract clients.

    ``from_tesseract_api`` runs the API module in this interpreter, which keeps
    the demo to one command and one env. ``from_url`` talks to containers
    served by ``tesseract serve`` - the same client interface, and the only
    mode where the two halves really do have separate dependency stacks.
    """
    from tesseract_core import Tesseract
    if args.served:
        return (Tesseract.from_url(args.served[0]),
                Tesseract.from_url(args.served[1]))
    return (Tesseract.from_tesseract_api(HERE / 'siren_model' / 'tesseract_api.py'),
            Tesseract.from_tesseract_api(HERE / 'devito_fwi' / 'tesseract_api.py'))


def build_report(tm, args, obj_times, cb_times, conv=None, init_info='',
                 nworkers=0, siren_desc=''):
    """End-of-run timing report (also written to ``timing.txt``)."""
    mode = ('served containers' if args.served else 'in-process API modules')
    config = [('composition', 'siren-model -> devito-fwi (Tesseract)'),
              ('client mode', mode),
              ('workers x OMP threads', f'{nworkers} x 1'),
              ('model (nx x nz)',
               f'{problem.PAR["nx"]} x {problem.PAR["nz"]}, '
               f'{problem.PAR["ns"]} src, {problem.PAR["nr"]} rec, '
               f'f0 {problem.PAR["freq"]:g} Hz'),
              ('SIREN', siren_desc),
              ('starting model', init_info or 'random (phase)'),
              ('optimiser', f'adam, maxiter {args.maxiter}, lr {args.lr:g}'),
              ('live figures', f'every {args.fig_every} iterations'
                               if args.fig_every > 1 else 'every iteration')]
    breakdown = [('python + jax + tesseract imports', 'import', False),
                 ('setup (velocity model, acquisition)', 'setup', False),
                 ('Tesseract clients open', 'tesseract_open', False),
                 ('SIREN random start', 'siren_init', False),
                 ('observed data + engine build-up', 'model_obs', False),
                 ('first loss + gradient', 'grad0', False),
                 ('inversion (adam)', 'fwi', False),
                 ('objective (loss + gradient + both VJPs)', 'objective', True),
                 ('callback (live figures)', 'callback', True),
                 ('optimiser overhead',
                  None if 'fwi' not in tm else
                  tm['fwi'] - tm.get('objective', 0.) - tm.get('callback', 0.),
                  True),
                 ('gradient w.r.t. theta at the final model', 'grad_final', False),
                 ('final forward modelling', 'model_final', False),
                 ('figures + I/O', 'figures', False)]
    per_call = [('objective', obj_times), ('callback', cb_times)]
    conv_rows = []
    if conv:
        conv_rows = [
            ('iterations / function evaluations',
             f'{conv["nit"]} / {conv["nfev"]}'),
            ('time per iteration',
             f'{tm.get("fwi", 0.) / max(conv["nit"], 1):.1f} s'),
            ('loss  initial -> final',
             f'{conv["loss0"]:.4e} -> {conv["loss1"]:.4e} '
             f'({100. * conv["loss1"] / conv["loss0"]:.2f} % of initial)'),
            ('model error  initial -> final',
             f'{conv["err0"]:.4f} -> {conv["err1"]:.4f} '
             f'({100. * (1 - conv["err1"] / conv["err0"]):.1f} % reduction)'),
            ('exit status', conv['message'])]
    return format_report('TIMING REPORT - Tesseract SIREN-FWI',
                         tm, config, breakdown, per_call, conv_rows)


def main():
    """Run the inversion and always report the timings, even if it dies."""
    tm = Timings()
    tm['import'] = T_IMPORT - T_START
    obj_times, cb_times = [], []
    run = {'conv': None, 'init_info': '', 'nworkers': 0, 'siren_desc': ''}

    try:
        _run(tm, obj_times, cb_times, run)
    except KeyboardInterrupt:
        print('\nInterrupted - reporting the timings collected so far.',
              flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        tm['total'] = time.time() - T_START
        if not ARGS.check_grad:
            write_report(build_report(tm, ARGS, obj_times, cb_times,
                                      run['conv'], run['init_info'],
                                      run['nworkers'], run['siren_desc']),
                         ARGS.outdir)


def _run(tm, obj_times, cb_times, run):
    args = ARGS
    import jax
    # The FWI Tesseract returns a float64 loss (the misfit is a sum over
    # millions of samples and the gradient check differences it), so JAX has to
    # be able to hold one. Must be set before any array is created.
    jax.config.update('jax_enable_x64', True)

    import jax.numpy as jnp
    import optax
    from tesseract_jax import apply_tesseract

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # ------------------------------------------------------------- geometry
    tic = time.time()
    vp_true = problem.load_vp_true()
    msk = problem.water_mask(vp_true)
    x_s, x_r = problem.source_receiver_positions()
    par = problem.PAR
    x = np.arange(par['nx']) * par['dx'] + par['ox']
    z = np.arange(par['nz']) * par['dz'] + par['oz']

    # The smoothed model is NOT the starting model - the inversion starts from
    # the random network. It is kept as the usual reference in the figures.
    from scipy.ndimage import gaussian_filter
    vp_ref = gaussian_filter(vp_true, sigma=[15, 10]) * msk
    vp_ref[vp_ref == 0] = problem.MASK_VALUE
    tm['setup'] = time.time() - tic

    print(f'Grid {problem.SHAPE}, {par["ns"]} shots, {par["nr"]} receivers, '
          f'{par["freq"]:g} Hz', flush=True)

    # A results directory that holds only figures is not reproducible: copy the
    # sources that produced it next to the results, as the reference script does.
    if not args.no_figures:
        with tm.timed('figures'):
            snapshot_code(outdir, __file__,
                          extra=[str(HERE / 'problem_config.py'),
                                 str(HERE / 'siren_model'),
                                 str(HERE / 'devito_fwi'),
                                 str(HERE / 'siren'),
                                 str(HERE / 'fwireport.py')])

    # ---------------------------------------------------- Tesseract clients
    with tm.timed('tesseract_open'):
        siren_tx, fwi_tx = open_tesseracts(args)
    print(f'Tesseracts open ({tm["tesseract_open"]:.1f}s, '
          f'{"served" if args.served else "in-process"})', flush=True)
    print(f'  siren-model  endpoints: {sorted(siren_tx.available_endpoints)}')
    print(f'  devito-fwi   endpoints: {sorted(fwi_tx.available_endpoints)}',
          flush=True)

    def theta_to_vp(theta):
        return apply_tesseract(siren_tx, {'theta': theta})['vp']

    def objective_fn(theta):
        """theta -> SIREN -> Vp -> Devito FWI -> loss, end to end."""
        return apply_tesseract(fwi_tx, {'vp': theta_to_vp(theta)})['loss']

    value_and_grad = jax.value_and_grad(objective_fn)

    # ------------------------------------------------------- starting model
    # The random start is the SIREN Tesseract's business, so ask its module for
    # one rather than reimplementing the initialisation here. The module also
    # gives the pytree structure the per-layer diagnostics need.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_siren_api', HERE / 'siren_model' / 'tesseract_api.py')
    siren_api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(siren_api)
    siren = siren_api._SIREN

    with tm.timed('siren_init'):
        theta0 = jnp.asarray(siren_api.init_theta())
        vp_start = np.asarray(theta_to_vp(theta0))

    hidden = list(problem.SIREN_HIDDEN)
    widths = [2, *hidden, 1]
    siren_desc = (f'{"-".join(str(w) for w in hidden)}, '
                  f'omega_0 {problem.SIREN_OMEGA0:g}, no pre-fit')
    run['siren_desc'] = siren_desc
    print(f'SIREN: {widths}, omega_0 {problem.SIREN_OMEGA0:g}, '
          f'{theta0.size} weights vs {int(np.prod(problem.SHAPE))} grid '
          f'unknowns ({theta0.size / np.prod(problem.SHAPE):.2f}x), '
          f'isotropic coords', flush=True)

    sub = vp_start[msk == 1]
    rmse0 = float(np.sqrt(((vp_start - vp_true) ** 2).mean()))
    run['init_info'] = (f'random phase, std {sub.std():.3f} km/s, '
                        f'RMSE {rmse0:.4f} km/s')
    print(f'  starting model below the seabed: {sub.min():.3f}-{sub.max():.3f} '
          f'km/s, mean {sub.mean():.3f}, std {sub.std():.3f}', flush=True)
    print(f'  RMSE to the true model {rmse0:.4f} km/s, to the smoothed '
          f'reference {np.sqrt(((vp_start - vp_ref) ** 2).mean()):.4f} km/s',
          flush=True)

    # --------------------------------------------------------- figure kit
    fk = FigureKit(par, x, z, x_s, x_r, vp_true, vp_ref, outdir)
    if not args.no_figures:
        with tm.timed('figures'):
            fk.models(vp_ini=vp_start,
                      labels=('True model', 'Random initial model'),
                      name='SirenInit.png')

    # --------------------------------------------------- gradient check only
    if args.check_grad:
        check_gradient(objective_fn, value_and_grad, theta0, jnp)
        return

    # ------------------------------------------------------- observed data
    # Only reachable in-process: the served client has no such helper, and the
    # data figure is a nicety, not part of the composition.
    fwi_api = dobs = dtobs = None
    if not args.served:
        spec2 = importlib.util.spec_from_file_location(
            '_fwi_api', HERE / 'devito_fwi' / 'tesseract_api.py')
        fwi_api = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(fwi_api)
        print('Model data...', flush=True)
        with tm.timed('model_obs'):
            dobs, dtobs = fwi_api.observed_data()
        run['nworkers'] = len(fwi_api._pool().chunks)
        print(f'  {par["ns"]} shots over {run["nworkers"]} workers: '
              f'{tm["model_obs"]:.1f}s (dobs {dobs.shape}, '
              f'dt {dtobs * 1e3:.4f} ms)', flush=True)
        if not args.no_figures:
            with tm.timed('figures'):
                fk.data(dobs, dtobs)

    # --------------------------------------------- first loss and gradient
    print('Compute gradient...', flush=True)
    with tm.timed('grad0'):
        loss0, grad_theta0 = value_and_grad(theta0)
        loss0 = float(loss0)
    # dL/dvp at the starting model, for the Gradient.png figure: the grid
    # gradient the physics Tesseract produced on the way through the VJP above,
    # read back rather than recomputed.
    grad_vp0 = None if fwi_api is None else fwi_api.last_gradient()
    print(f'  one loss+gradient: {tm["grad0"]:.1f}s (loss {loss0:.6e}, '
          f'|dL/dtheta| {float(jnp.linalg.norm(grad_theta0)):.6e})', flush=True)

    if grad_vp0 is not None and not args.no_figures:
        with tm.timed('figures'):
            fk.gradient(grad_vp0, np.abs(grad_vp0).max(),
                        title='Initial gradient w.r.t. $V_P$ (normalised)')

    if args.benchmark:
        print(f'BENCH workers={run["nworkers"]} obs={tm.get("model_obs", 0):.1f}s '
              f'grad={tm["grad0"]:.1f}s loss={loss0:.6e}', flush=True)
        return

    # ---------------------------------------------------- live wiggle setup
    iwig = par['ns'] // 2 if args.wiggle_shot < 0 else int(args.wiggle_shot)
    iwig = int(np.clip(iwig, 0, par['ns'] - 1))
    dobs_wig = eobs_wig = None
    if dobs is not None and args.wiggle_every > 0:
        dobs_wig = dobs[iwig]
        eobs_wig = float(np.sqrt((dobs_wig ** 2).mean())) or 1.
        print(f'  live waveform comparison: shot {iwig} '
              f'($x_s$ = {x_s[iwig, 0]:.2f} km) every {args.wiggle_every} '
              f'iterations -> Waveformtmp.png', flush=True)

    # ------------------------------------------------------------- histories
    vp_flat = vp_true.ravel()
    losshistory, loss_iter = [], []
    vp_error = [float(np.linalg.norm((vp_start.ravel() - vp_flat) / vp_flat))]
    gtheta_hist, theta_hist = [], []

    snaps = SnapshotRecorder(
        os.path.join(outdir, 'snapshots.npy'), vp_true.shape,
        max_frames=args.maxiter // max(args.snap_every, 1) + 2,
        every=args.snap_every,
        meta=dict(extent=np.asarray(fk.extent), vmin=fk.vmin, vmax=fk.vmax,
                  vp_true=vp_true, vp_start=vp_start, x=x, z=z))
    snaps(vp_start, 0)

    # ----------------------------------------------------------------- Adam
    # optax's ``decay_steps`` is the TOTAL length of the schedule, warmup
    # included - it subtracts the warmup internally - so pass --maxiter, not
    # --maxiter minus the warmup.
    warmup = max(min(args.warmup, args.maxiter - 1), 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr, warmup_steps=warmup,
        decay_steps=max(args.maxiter, warmup + 1),
        end_value=args.lr * args.end_lr_frac)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(theta0)

    print(f'\nRun FWI from the random start (SIREN weights as unknowns via '
          f'Tesseract, optimiser=adam)...', flush=True)
    print(f'  Adam: lr {args.lr:g} (warmup {warmup}, cosine to '
          f'{args.lr * args.end_lr_frac:g}), {args.maxiter} epochs', flush=True)
    print(f'{"iter":>5} {"loss":>14} {"|dL/dtheta|":>13} {"RMSE(vp)":>10} '
          f'{"err":>9} {"s":>6}', flush=True)

    theta = theta0
    best_loss, best_theta = np.inf, theta0

    with tm.timed('fwi'):
        for it in range(args.maxiter):
            tic = time.time()
            loss, grad_theta = value_and_grad(theta)
            loss = float(loss)
            # The SIREN forward pass is wanted again to report the model error;
            # it is milliseconds against a multi-second wave solve.
            vp = np.asarray(theta_to_vp(theta))
            obj_times.append(time.time() - tic)

            losshistory.append(loss)
            if loss < best_loss:
                best_loss, best_theta = loss, theta

            # per-layer diagnostics, on the pytree the network actually uses
            gn, tn = theta_norms(siren.unflatten(np.asarray(grad_theta)),
                                 siren.unflatten(np.asarray(theta)))
            gtheta_hist.append(gn)
            theta_hist.append(tn)
            tot_g, tot_t = np.sqrt((gn ** 2).sum()), np.sqrt((tn ** 2).sum())
            print(f'  eval {it + 1}: |dL/dtheta| = {tot_g:.6e}, '
                  f'|theta| = {tot_t:.6e}, '
                  f'ratio = {tot_g / max(tot_t, 1e-30):.4e}, per layer ['
                  + ', '.join(f'{v:.2e}' for v in gn) + ']', flush=True)

            # ------------------------------------------------ callback
            tic_cb = time.time()
            err = float(np.linalg.norm((vp.ravel() - vp_flat) / vp_flat))
            vp_error.append(err)
            if not loss_iter:
                loss_iter.append(loss)      # iteration 0 = starting model
            loss_iter.append(loss)
            rmse = float(np.sqrt(((vp - vp_true) ** 2).mean()))
            snaps(vp, it + 1)

            if not args.no_figures and (
                    it <= 1 or it >= args.maxiter - 1 or args.fig_every <= 1
                    or it % args.fig_every == 0):
                fk.live_model(vp, it, vp_ini=vp_start,
                              labels=('True model', 'Random initial model',
                                      'Inverted model (SIREN)'))
                # The grid gradient the VJP just produced, read back from the
                # physics Tesseract rather than recomputed: another
                # loss_and_grad here would be a second 64-shot adjoint solve
                # per figure, i.e. it would double the cost of the iteration.
                if fwi_api is not None:
                    gvp = fwi_api.last_gradient()
                    if gvp is not None:
                        fk.live_gradient(gvp, it)
                plot_theta_grad(outdir, gtheta_hist, theta_hist, widths)
                fk.live_convergence(loss_iter, vp_error, xlabel=ITER_LABEL,
                                    errlabel=ITER_LABEL)

            if (dobs_wig is not None and args.wiggle_every > 0
                    and (it % args.wiggle_every == 0 or it == 1)):
                dest, dt_e = fwi_api.model_shot(iwig, vp, dtobs)
                n = min(len(dobs_wig), len(dest))
                do_w, de_w = dobs_wig[:n], dest[:n]
                rel = float(np.sqrt(((de_w - do_w) ** 2).mean())) / eobs_wig
                fk.live_waveforms(do_w, de_w, dt_e, it, ishot=iwig, rms=rel,
                                  tmax=(args.wiggle_tmax
                                        if args.wiggle_tmax > 0 else None),
                                  labels=('Observed', 'Estimated'))
            cb_times.append(time.time() - tic_cb)

            print(f'{it:>5} {loss:>14.6e} '
                  f'{float(jnp.linalg.norm(grad_theta)):>13.4e} '
                  f'{rmse:>10.4f} {err:>9.4f} {obj_times[-1]:>6.1f}', flush=True)

            updates, opt_state = optimizer.update(grad_theta, opt_state, theta)
            theta = optax.apply_updates(theta, updates)

    tm['objective'] = float(np.sum(obj_times))
    tm['callback'] = float(np.sum(cb_times))
    snaps.close()

    nfev = nit = args.maxiter
    print(f'\nTotal FWI time (s) = {tm["fwi"]:.2f} over {nit} Adam epochs '
          f'({nfev} function evaluations, {tm["fwi"] / max(nfev, 1):.1f}s each) '
          f'- {hms(tm["fwi"])}')
    print('---------------------------------------------------------\n')
    print(f'  best loss {best_loss:.6e}', flush=True)

    # ------------------------------ weight-space gradient at the final model
    theta_final = best_theta
    vp_inv = np.asarray(theta_to_vp(theta_final))
    print('Gradient w.r.t. the network weights at the final model...',
          flush=True)
    with tm.timed('grad_final'):
        loss_final, gtheta_final = value_and_grad(theta_final)
        loss_final = float(loss_final)
    gn_f, tn_f = theta_norms(siren.unflatten(np.asarray(gtheta_final)),
                             siren.unflatten(np.asarray(theta_final)))
    print(f'  loss {loss_final:.6e}')
    print(format_theta_grad(gn_f, tn_f, widths), flush=True)

    rmse1 = float(np.sqrt(((vp_inv - vp_true) ** 2).mean()))
    run['conv'] = dict(nit=nit, nfev=nfev,
                       loss0=float(losshistory[0]), loss1=float(best_loss),
                       err0=float(vp_error[0]), err1=float(vp_error[-1]),
                       message=f'{nit} Adam epochs completed')

    # ------------------------------------------------- final forward modelling
    dini = dinv = None
    if fwi_api is not None and not args.no_figures:
        print('Model data through the random start and inverted models...',
              flush=True)
        with tm.timed('model_final'):
            dini, _ = fwi_api.model_all(vp_start, dtobs)
            dinv, _ = fwi_api.model_all(vp_inv, dtobs)
        print(f'  done in {tm["model_final"]:.1f}s', flush=True)

    # ------------------------------------------------------------- figures
    if not args.no_figures:
        with tm.timed('figures'):
            plot_theta_grad(outdir, gtheta_hist, theta_hist, widths,
                            name='GradTheta.png', final=True, g=gn_f, t=tn_f)
            fk.convergence(loss_iter, vp_error, xlabel=ITER_LABEL,
                           errlabel=ITER_LABEL)
            fk.models(vp_inv, vp_ini=vp_start,
                      labels=('True model', 'Random initial model',
                              'Inverted model (SIREN)'))
            fk.profiles(vp_inv, vp_ini=vp_start,
                        labels=('True', 'Random initial', 'Inverted (SIREN)'))
            if dinv is not None:
                nt_c = min(dobs.shape[1], dini.shape[1], dinv.shape[1])
                fk.waveforms(dobs[:, :nt_c], dinv[:, :nt_c], dini[:, :nt_c],
                             dt=dtobs,
                             labels=('Observed', 'Random initial',
                                     'Inverted (SIREN)'))

    np.savez(os.path.join(outdir, 'result.npz'),
             vp_inv=vp_inv, vp_ref=vp_ref, vp_start=vp_start, vp_true=vp_true,
             losshistory=np.array(losshistory), loss_iter=np.array(loss_iter),
             vp_error=np.array(vp_error),
             obj_times=np.array(obj_times), cb_times=np.array(cb_times),
             timing_keys=np.array(list(tm.keys())),
             timing_values=np.array(list(tm.values())),
             t_fwi=tm['fwi'],
             theta=np.asarray(theta_final), theta0=np.asarray(theta0),
             optimizer='adam', lr=args.lr,
             hidden=np.array(hidden), layers=len(hidden),
             omega0=problem.SIREN_OMEGA0,
             gtheta_hist=np.array(gtheta_hist),
             theta_hist=np.array(theta_hist),
             gtheta_final=np.asarray(gtheta_final),
             gtheta_final_bylayer=gn_f, theta_final_bylayer=tn_f,
             layer_labels=np.array(layer_labels(widths)))

    print(f'\n  loss  {losshistory[0]:.6e} -> {best_loss:.6e} '
          f'({100 * best_loss / losshistory[0]:.1f}% of initial)')
    print(f'  RMSE  {rmse0:.4f} -> {rmse1:.4f} km/s '
          f'({100 * (1 - rmse1 / rmse0):.1f}% reduction)')
    print(f'  wrote {outdir}', flush=True)


def check_gradient(objective, value_and_grad, theta0, jnp):
    """Finite-difference the composed gradient against the composed forward map.

    This is the load-bearing test of the whole pipeline: it only passes if the
    JAX VJP of the SIREN Tesseract and the adjoint-state VJP of the Devito
    Tesseract are correct and consistent with each other.
    """
    print('\nFinite-difference check of dL/dtheta '
          '(each direction costs 3 FWI solves)...', flush=True)
    loss0, grad = value_and_grad(theta0)
    grad = np.asarray(grad)
    gnorm = np.linalg.norm(grad)
    print(f'  loss {float(loss0):.8e}, |grad| {gnorm:.6e}', flush=True)

    # The gradient direction is the one that matters - it is what Adam steps
    # along - and it is also the best conditioned. A random direction is nearly
    # orthogonal to the gradient in this many dimensions, which makes ``g @ v``
    # small and its finite difference correspondingly noisy; those are reported
    # too, but only the gradient direction is asserted on.
    rng = np.random.default_rng(0)
    dirs = [('-grad', grad / gnorm)]
    for k in range(2):
        v = rng.standard_normal(theta0.shape).astype(np.float32)
        dirs.append((f'random {k}', v / np.linalg.norm(v)))

    ok = True
    for name, v in dirs:
        ad = float(grad @ v)
        best = None
        for eps in (3e-3, 1e-3, 3e-4):
            lp = float(objective(theta0 + eps * jnp.asarray(v)))
            lm = float(objective(theta0 - eps * jnp.asarray(v)))
            fd = (lp - lm) / (2 * eps)
            rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-300)
            if best is None or rel < best[0]:
                best = (rel, fd, eps)
        rel, fd, eps = best
        if name == '-grad':
            flag = 'OK' if rel < 3e-2 else 'MISMATCH'
            ok &= rel < 3e-2
        else:
            flag = f'(sees {abs(ad) / gnorm:.1e} of |g|)'
        print(f'  {name:>9}: adjoint {ad:+.6e}  finite-diff {fd:+.6e}  '
              f'(eps {eps:.0e})  rel.err {rel:.2e}  {flag}', flush=True)

    print(f'\n  {"GRADIENT CHECK PASSED" if ok else "GRADIENT CHECK FAILED"} '
          f'- the composed dL/dtheta matches the finite difference of the '
          f'composed forward map,\n  i.e. the JAX VJP of the SIREN Tesseract '
          f'and the adjoint-state VJP of the Devito Tesseract agree.',
          flush=True)


if __name__ == '__main__':
    main()
