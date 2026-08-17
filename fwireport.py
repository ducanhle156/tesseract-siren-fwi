r"""Timing report and research-quality figures shared by the acoustic FWI scripts.

Two things live here, both used by ``AcousticVel_L2_1stage_parallel.py`` and
``AcousticVel_L2_SIREN.py``:

``Timings`` / ``format_report``
    wall-clock accumulators and the end-of-run report, so a run reports where
    its time actually went (imports, engine build-up, objective, figures, ...)
    instead of only the total inversion time.

``FigureKit``
    the figures, in one consistent journal style: shot gathers, gradient,
    model panels, vertical VP profiles, observed-vs-modelled wiggles (final,
    and a live observed-vs-estimated version with the residual beside it) and
    the convergence curves.

``SnapshotRecorder``
    the raw per-iteration models, appended to a ``.npy`` stack during the run
    and rendered to MP4/GIF afterwards by ``make_movie.py``. Recording is
    array writes only - no matplotlib - so it stays cheap enough to run every
    iteration, and the rendering choices (colormap, fps, clip) are not baked
    in at run time.

Import this module *after* ``matplotlib.use('Agg')`` in the calling script.

Edited by Minh Nhat--Tran, University of Houston
"""

import os
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

# One colour per entity, used identically in every figure of every script.
C_TRUE, C_INIT, C_INV = '#1b3b5f', '#8c95a0', '#e07b39'
C_RES = '#8a1c1c'   # data residual (estimated - observed)

STYLE = {
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 1.0,
    'legend.frameon': True,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '0.8',
    'legend.fancybox': False,
    'grid.color': '0.88',
    'grid.linewidth': 0.5,
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
}


def apply_style():
    """Switch matplotlib to the journal-figure defaults used here."""
    plt.rcParams.update(STYLE)


def _thousands(v, _pos=None):
    """Integer tick label with no thousands separator: 10000."""
    # No separator at all: at figure tick sizes a thin space still reads as a
    # gap between two numbers, and a comma only widens the label.
    return f'{int(round(v))}'


def iteration_axis(ax, n, group=True):
    """Integer x ticks on an iteration axis, spaced so the labels never touch.

    ``MaxNLocator(integer=True)`` alone picks ~9 intervals whatever the axis
    is worth: at 10 000 Adam epochs that is ten five-digit labels across a 3 in
    panel, which overlap into a solid bar. The number of ticks that actually
    fits depends on the width of a label, so derive it from both - the drawn
    width of the panel and the digit count of the largest tick - instead of
    leaving it at the default.

    ``group`` adds a thin-space thousands separator, which costs a little width
    but makes 10000 readable at a glance.

    Parameters
    ----------
    ax : Axes
        Axis to set up; its x limits should already be final.
    n : int
        Largest value on the axis, i.e. the last iteration.
    """
    # Panel width in inches, at draw time, so a tight_layout or a colorbar that
    # shrank the axes is accounted for.
    fig = ax.get_figure()
    w_in = ax.get_position().width * fig.get_size_inches()[0]
    # Label width: digits (plus separators) at the tick font size, ~0.62 em per
    # digit for a serif face, plus a half-label gap so neighbours are clearly
    # separated without thinning the axis to two or three ticks. A full-label
    # gap reads as safer but is not: MaxNLocator only offers 1/2/2.5/5 x 10^k
    # steps, so asking for slightly too few bins drops it a whole step - at
    # 10 000 epochs that is ticks every 4000, which leaves the axis unlabelled
    # at its own endpoint.
    digits = len(_thousands(n)) if group else len(str(int(n)))
    em = plt.rcParams['xtick.labelsize'] / 72.
    lab_in = 0.62 * em * digits
    nbins = max(2, int(w_in / (1.5 * lab_in)))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, integer=True))
    if group:
        ax.xaxis.set_major_formatter(FuncFormatter(_thousands))


##################################################################
# Timing
##################################################################

class Timings(dict):
    """Named wall-clock accumulators.

    ``with tm.timed('fwi'): ...`` adds the elapsed time to the ``fwi`` entry,
    so a key used several times (e.g. ``figures``) accumulates.
    """

    def add(self, key, dt):
        self[key] = self.get(key, 0.) + dt

    def timed(self, key):
        return _Timed(self, key)


class _Timed:
    def __init__(self, timings, key):
        self.timings, self.key = timings, key

    def __enter__(self):
        self.tic = time.time()
        return self

    def __exit__(self, *exc):
        self.timings.add(self.key, time.time() - self.tic)
        return False


def hms(s):
    """Seconds as ``1h 02m 03s``."""
    h, rem = divmod(int(round(s)), 3600)
    m, sec = divmod(rem, 60)
    return f'{h:d}h {m:02d}m {sec:02d}s' if h else f'{m:d}m {sec:02d}s'


def call_stats(v):
    """One-line mean/min/max/total for a list of per-call durations."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return 'not called'
    return (f'n={v.size:<5d} mean {v.mean():7.2f}s  min {v.min():7.2f}s  '
            f'max {v.max():7.2f}s  total {v.sum():8.1f}s')


_W = 44          # label column width
_RULE = 78       # total report width


def format_report(title, tm, config_rows, breakdown, per_call=(), conv_rows=()):
    """Assemble the end-of-run timing report.

    Parameters
    ----------
    title : str
        Report heading.
    tm : Timings
        Must contain ``total``; every other key is optional.
    config_rows : list of (label, value)
        Run configuration, printed verbatim.
    breakdown : list of (label, key, is_sub)
        Wall-clock rows. ``key`` is either a key of ``tm`` or a float; rows
        whose key is missing from ``tm`` are skipped. Sub-rows are indented and
        excluded from the "unaccounted" remainder.
    per_call : list of (label, durations)
        Per-call statistics (objective, callback, ...).
    conv_rows : list of (label, value)
        Convergence summary, printed verbatim.
    """
    total = tm.get('total', 0.) or 1e-12
    L = ['', '=' * _RULE, f' {title}', '=' * _RULE, ' configuration']
    for label, value in config_rows:
        L.append(f'   {label:<{_W}} {value}')

    L += ['', ' wall-clock breakdown'.ljust(_W + 3) + '     time      share',
          ' ' + '-' * (_RULE - 2)]
    accounted = 0.
    for label, key, is_sub in breakdown:
        val = tm.get(key) if isinstance(key, str) else key
        if val is None:
            continue
        if not is_sub:
            accounted += val
        pad = '     ' if is_sub else '   '
        name = ('- ' + label) if is_sub else label
        L.append(f'{pad}{name:<{_W - (len(pad) - 3)}} {val:>8.1f} s   '
                 f'{100. * val / total:>5.1f} %')
    rest = total - accounted
    if abs(rest) > 0.05:
        L.append(f'   {"unaccounted":<{_W}} {rest:>8.1f} s   '
                 f'{100. * rest / total:>5.1f} %')
    L += [' ' + '-' * (_RULE - 2),
          f'   {"TOTAL (wall clock)":<{_W}} {total:>8.1f} s   {hms(total)}']

    if per_call:
        L += ['', ' per-call statistics']
        for label, durations in per_call:
            L.append(f'   {label:<12} {call_stats(durations)}')
    if conv_rows:
        L += ['', ' convergence']
        for label, value in conv_rows:
            L.append(f'   {label:<{_W}} {value}')
    L += ['=' * _RULE, '']
    return '\n'.join(L)


def write_report(report, figpath, name='timing.txt'):
    """Print the report and drop it next to the figures."""
    print(report, flush=True)
    try:
        with open(os.path.join(figpath, name), 'w') as f:
            f.write(report)
    except OSError as e:
        print(f'(could not write {name}: {e})', flush=True)


def snapshot_code(figpath, script, extra=(), name='code'):
    """Copy the running script into ``figpath/code/`` so a run reproduces itself.

    A results directory that holds only figures is not reproducible: the
    defaults at the top of the script, the network geometry and the physics all
    move between runs, and the command line in ``command.txt`` only records what
    was overridden. Copying the sources verbatim next to the figures pins the
    other half.

    ``script`` is the entry point (pass ``__file__``); ``extra`` names the
    modules it imports that live beside it - a file is copied, a directory is
    copied whole minus its ``__pycache__``. Also written is ``command.txt``:
    the interpreter, the argv and the working directory that produced the run,
    plus the git commit if the tree is a repository.
    """
    dest = os.path.join(figpath, name)
    try:
        os.makedirs(dest, exist_ok=True)
        for src in [script] + [s for s in extra if s]:
            src = os.path.abspath(src)
            target = os.path.join(dest, os.path.basename(src))
            if os.path.isdir(src):
                shutil.copytree(src, target, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('__pycache__'))
            elif os.path.isfile(src):
                shutil.copy2(src, target)

        lines = [f'date     : {time.strftime("%Y-%m-%d %H:%M:%S")}',
                 f'host     : {socket.gethostname()}',
                 f'cwd      : {os.getcwd()}',
                 f'python   : {sys.executable}',
                 f'command  : {" ".join(sys.argv)}']
        commit = _git_commit(os.path.dirname(os.path.abspath(script)))
        if commit:
            lines.append(f'git      : {commit}')
        lines.append('')
        lines.append('Rerun with:')
        lines.append(f'    {sys.executable} {" ".join(sys.argv)}')
        lines.append('')
        with open(os.path.join(dest, 'command.txt'), 'w') as f:
            f.write('\n'.join(lines))
    except OSError as e:
        print(f'(could not snapshot the code: {e})', flush=True)


def _git_commit(cwd):
    """``<sha> (dirty)`` for the repository holding the script, or ``''``."""
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=cwd,
                             capture_output=True, text=True, timeout=5)
        if sha.returncode != 0:
            return ''
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=cwd,
                               capture_output=True, text=True, timeout=5)
        flag = ' (dirty)' if dirty.stdout.strip() else ''
        return sha.stdout.strip() + flag
    except (OSError, subprocess.SubprocessError):
        return ''


##################################################################
# Figures
##################################################################

def panel_label(ax, letter, title):
    """``(a) title``, left-aligned above the axes - the journal convention."""
    ax.set_title(f'({letter}) {title}', loc='left', pad=4)


_panel = panel_label   # short alias used throughout this module


class FigureKit:
    """All figures for one run, sharing geometry, colour scale and style.

    Parameters
    ----------
    par : dict
        The acquisition/model parameter dictionary of the calling script.
    x, z : ndarray
        Model axes in km.
    x_s, x_r : ndarray
        Source and receiver positions, ``(n, 2)`` in km.
    vp_true, vp_init : ndarray
        Reference models, ``(nx, nz)`` in km/s.
    figpath : str
        Output directory (created by the caller).
    profile_x : tuple
        Distances (km) at which the vertical profiles are extracted.
    """

    def __init__(self, par, x, z, x_s, x_r, vp_true, vp_init, figpath,
                 profile_x=(2.25, 4.5, 6.75), wiggle_every=12,
                 trace_offsets=(0.5, 2.0, 4.0)):
        apply_style()
        self.par, self.x, self.z = par, x, z
        self.x_s, self.x_r = x_s, x_r
        self.vp_true, self.vp_init = vp_true, vp_init
        self.figpath = figpath
        self.profile_x = profile_x
        self.wiggle_every = wiggle_every
        self.trace_offsets = trace_offsets
        self.vmin, self.vmax = np.percentile(vp_true, [2, 98])
        self.extent = (x[0], x[-1], z[-1], z[0])

    def _save(self, fig, name, dpi=None):
        fig.savefig(os.path.join(self.figpath, name),
                    **({'dpi': dpi} if dpi else {}))
        plt.close(fig)

    # ---------------------------------------------------------------- data

    def data(self, dobs, dt, name='Data.png'):
        """Observed shot gathers at both ends and the middle of the line."""
        clip = np.percentile(np.abs(dobs), 98)
        nt = dobs.shape[1]
        ishots = [0, self.par['ns'] // 2, self.par['ns'] - 1]
        fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)
        for k, (ax, ishot) in enumerate(zip(axs, ishots)):
            ax.imshow(dobs[ishot], aspect='auto', cmap='gray',
                      vmin=-clip, vmax=clip,
                      extent=(self.x_r[0, 0], self.x_r[-1, 0], (nt - 1) * dt, 0.))
            _panel(ax, 'abc'[k],
                   f'shot {ishot} ($x_s$ = {self.x_s[ishot, 0]:.2f} km)')
            ax.set_xlabel('Receiver position (km)')
        axs[0].set_ylabel('Time (s)')
        fig.tight_layout(w_pad=1.0)
        self._save(fig, name)

    # ------------------------------------------------------------ gradient

    def gradient(self, grad, scaling=None, name='Gradient.png',
                 title='Gradient (normalised)', clip=1e-1):
        g = np.asarray(grad, dtype=float)
        s = scaling if scaling else (np.abs(g).max() or 1.)
        fig, ax = plt.subplots(figsize=(7.2, 2.9))
        im = ax.imshow(g.T / s, cmap='seismic', vmin=-clip, vmax=clip,
                       extent=self.extent, aspect='equal')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Depth (km)')
        _panel(ax, 'a', title)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.026)
        cb.outline.set_linewidth(0.6)
        fig.tight_layout()
        self._save(fig, name)

    # --------------------------------------------------------- acquisition

    def acquisition(self, ax, rec_every=4, legend=True, strip=False):
        """Draw the acquisition: receivers as triangles, sources as crosses.

        Seismic convention: receivers are triangles on the surface, sources are
        crosses just above them. Both sit at z = 0 in the data, so they are
        separated vertically here to keep the two readable; the legend carries
        the true counts since the receivers are decimated.

        With ``strip=True`` the axes is a dedicated band above the model rather
        than the model panel itself: it spans the model's x range, carries no
        frame, and uses its own y range so the markers never overlap the image
        and never steal height from the model panel (which is aspect-locked).
        """
        if strip:
            # x is shared with the model panels: do not touch set_xlim here or
            # the markers stop lining up with the images below.
            # Hide the frame and ticks but not the axes itself - set_axis_off()
            # would also hide the panel title this strip carries.
            ax.set_ylim(-1., 1.)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            ax.patch.set_visible(False)
            # Receivers hug the bottom of the strip, right above the model top.
            z_rec, z_src = -0.72, 0.35
        else:
            z_rec, z_src = self.z[0], self.z[0] - 0.11
        # clip_on only off in-panel, where the sources sit above the image top.
        ax.plot(self.x_r[::rec_every, 0],
                np.full(self.x_r[::rec_every].shape[0], z_rec),
                marker='v', ls='none', ms=2.6, mfc='w', mec='k', mew=0.5,
                clip_on=strip, label=f'receivers ({len(self.x_r)})')
        ax.plot(self.x_s[:, 0], np.full(len(self.x_s), z_src), marker='x',
                ls='none', ms=4.0, mec='#cc2222', mew=1.1, clip_on=strip,
                label=f'sources ({len(self.x_s)})')
        if legend:
            # above the strip, opposite the left-aligned title: covers no model
            ax.legend(loc='lower right', bbox_to_anchor=(1., 1.0), ncol=2,
                      frameon=False, columnspacing=1.2, handletextpad=0.4,
                      borderpad=0.2)

    # -------------------------------------------------------------- models

    def models(self, vp_inv=None, vp_ini=None, labels=('True model',
                                                       'Initial model',
                                                       'Inverted model'),
               name='InvertedVP.png', dpi=None):
        """True / initial / inverted VP stacked, with one shared colorbar.

        The model panels share one x range and one aspect ratio, so they must
        render at exactly the same size. The acquisition therefore lives in its
        own thin strip above panel (a) instead of inside it: drawn in the panel,
        its out-of-axes markers and legend shrink that panel alone and the
        models stop matching.

        Panels are whichever of ``vp_ini`` / ``vp_inv`` are given, always after
        the true model: pass ``vp_inv=None`` for a two-panel true/initial figure
        (a starting model has no inversion to show yet). ``labels`` is trimmed
        to the panels actually drawn.
        """
        vp_ini = self.vp_init if vp_ini is None else vp_ini
        mods = [self.vp_true, vp_ini] + ([] if vp_inv is None else [vp_inv])
        npan = len(mods)
        labels = list(labels)[:npan]
        x0, x1, zbot, ztop = self.extent
        # A panel is (x1 - x0) by (zbot - ztop) in data units and aspect='equal'
        # fixes its shape, so the figure has to be sized to that ratio or the
        # panels leave gaps. Derive the axes width from the figure width and the
        # left/right margins below, then give each panel the matching height.
        fig_w, left, right = 7.2, 0.085, 0.86
        hspace, strip_h = 0.16, 0.20          # hspace in fractions of panel_h
        title_h = 0.20                        # inches reserved for the (a) title
        top_pad, bot_pad = 0.10, 0.55         # inches: legend / xlabel + ticks
        panel_w = fig_w * (right - left)      # inches of axes width, colorbar aside
        panel_h = panel_w * (zbot - ztop) / (x1 - x0)
        # Total axes height the grid must supply, so the cells already match the
        # aspect-locked panels and set_adjustable('box') has nothing to trim.
        grid_h = (npan * panel_h + (npan - 1) * hspace * panel_h
                  + strip_h + title_h)
        fig_h = grid_h + top_pad + bot_pad
        fig = plt.figure(figsize=(fig_w, fig_h))
        # Outer rows: the acquisition strip sits in the same cell as panel (a)
        # so it can be pushed right down onto it, while the panels keep their
        # even hspace. Row 0 is empty space that holds the (a) title clear of
        # the strip below it.
        gs = fig.add_gridspec(1 + npan, 1,
                              height_ratios=([title_h, strip_h + panel_h]
                                             + [panel_h] * (npan - 1)),
                              hspace=hspace, left=left, right=right,
                              bottom=bot_pad / fig_h, top=1 - top_pad / fig_h)
        # Inside the (a) cell: strip hard against the panel, hspace = 0.
        gs_a = gs[1].subgridspec(2, 1, height_ratios=[strip_h, panel_h],
                                 hspace=0.)
        # No sharex: panel (a) sits in a nested subgridspec, and a shared x
        # group anchored there propagates its hidden tick labels to the bottom
        # panel. Every axes gets the same xlim explicitly instead, which lines
        # the acquisition markers up with the images just as reliably.
        axs = [fig.add_subplot(gs_a[1])]
        axs += [fig.add_subplot(gs[k]) for k in range(2, 1 + npan)]
        ax_acq = fig.add_subplot(gs_a[0])
        for ax in axs + [ax_acq]:
            ax.set_xlim(x0, x1)
        for k, (ax, m, ttl) in enumerate(zip(axs, mods, labels)):
            im = ax.imshow(m.T, vmin=self.vmin, vmax=self.vmax, cmap='jet',
                           extent=self.extent, aspect='equal')
            # 'box' honours aspect='equal' by shrinking the axes box to the data
            # (no padded ylim, unlike 'datalim'). The grid cells above are
            # already cut to that exact ratio, so nothing actually shrinks and
            # the three panels come out identical.
            ax.set_adjustable('box')
            ax.set_ylim(zbot, ztop)
            # Panel (a)'s title goes on the strip above it, not on the image:
            # the strip sits flush on the panel, so a title on the panel itself
            # would land underneath the markers.
            _panel(ax_acq if k == 0 else ax, 'abc'[k], ttl)
            ax.set_ylabel('Depth (km)')
        # Only the bottom panel keeps x tick labels. With no shared x group
        # these are independent axes, so this sticks.
        for ax in axs[:-1]:
            ax.tick_params(labelbottom=False)
        self.acquisition(ax_acq, strip=True)
        axs[-1].set_xlabel('Distance (km)')
        cb = fig.colorbar(im, ax=axs, pad=0.015, fraction=0.02, aspect=45)
        cb.set_label('$V_P$ (km/s)')
        cb.outline.set_linewidth(0.6)
        # aspect='equal' + adjustable='box' shrinks the model panels inside
        # their grid cells, and the colorbar shifts them again. The strip is not
        # aspect-locked, so it keeps the full cell width and its x scale drifts
        # from the panels' - drawing a receiver at 9 km past the image edge.
        # Match the strip to panel (a)'s final box once the layout has settled.
        fig.canvas.draw()
        pos = axs[0].get_position()
        sp = ax_acq.get_position()
        ax_acq.set_position([pos.x0, sp.y0, pos.width, sp.height])
        self._save(fig, name, dpi=dpi)

    def live_model(self, vp, iteration, vp_ini=None, name='InvertedVPtmp.png',
                   labels=('True model', 'Initial model', 'Inverted model')):
        """Live model snapshot: the same three panels as the final figure.

        Same layout as ``models`` so the live view and the final figure are read
        the same way, at a lower dpi because it is redrawn throughout the run.
        The third panel's label carries the iteration number.
        """
        labels = list(labels)
        labels[2] = f'{labels[2]} - iteration {iteration}'
        self.models(np.asarray(vp).reshape(self.vp_true.shape),
                    vp_ini=vp_ini, labels=labels, name=name, dpi=110)

    def live_gradient(self, grad, iteration, name='Gradienttmp.png'):
        """Current dL/dvp, normalised by its own peak."""
        g = np.asarray(grad, dtype=float)
        gmax = np.abs(g).max() or 1.
        fig, ax = plt.subplots(figsize=(7.2, 2.9))
        im = ax.imshow(g.T / gmax, cmap='seismic', vmin=-1e-1, vmax=1e-1,
                       extent=self.extent, aspect='equal')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Depth (km)')
        _panel(ax, 'a', f'$\\partial L/\\partial V_P$ - iteration {iteration} '
                        f'(peak {gmax:.3e})')
        fig.colorbar(im, ax=ax, pad=0.02, fraction=0.026)
        fig.tight_layout()
        self._save(fig, name, dpi=110)

    # ------------------------------------------------------------ profiles

    def profiles(self, vp_inv, vp_ini=None, name='Profiles.png',
                 labels=('True', 'Initial', 'Inverted')):
        """Vertical VP profiles at ``profile_x`` - true / initial / inverted."""
        vp_ini = self.vp_init if vp_ini is None else vp_ini
        dx, ox = self.par['dx'], self.par['ox']
        ixs = [int(round((xp - ox) / dx)) for xp in self.profile_x]
        fig, axs = plt.subplots(1, len(ixs), figsize=(7.2, 4.8), sharey=True)
        axs = np.atleast_1d(axs)
        lo = 0.95e3 * min(self.vp_true.min(), vp_ini.min(), vp_inv.min())
        hi = 1.04e3 * max(self.vp_true.max(), vp_ini.max(), vp_inv.max())
        for k, (ax, ix) in enumerate(zip(axs, ixs)):
            ax.plot(self.vp_true[ix] * 1e3, self.z, color=C_TRUE, lw=1.1,
                    label=labels[0])
            ax.plot(vp_ini[ix] * 1e3, self.z, color=C_INIT, lw=1.0,
                    ls=(0, (4, 2.5)), label=labels[1])
            ax.plot(vp_inv[ix] * 1e3, self.z, color=C_INV, lw=1.1, label=labels[2])
            ax.set_xlabel('$V_P$ (m/s)')
            _panel(ax, 'abcdef'[k], f'$x$ = {self.x[ix]:.2f} km')
            ax.grid(True, ls=':', lw=0.5)
            ax.set_ylim(self.z[-1], self.z[0])
            ax.set_xlim(lo, hi)
        axs[0].set_ylabel('Depth (km)')
        axs[0].legend(loc='lower left')
        fig.tight_layout(w_pad=0.8)
        self._save(fig, name)

    # ----------------------------------------------------------- waveforms

    @staticmethod
    def _wiggle(ax, d, tt, itr, xtr, color, lw, norm, fill=None, label=None,
                clip=2.2):
        """Wiggle traces of ``d`` (nt x nr), drawn at their receiver position.

        ``norm`` is one amplitude per drawn trace (the observed peak), so the
        near-offset direct wave does not swamp the far offsets while the
        observed/modelled amplitude ratio is still visible. Excursions beyond
        ``clip`` trace spacings are clipped, as is usual in wiggle displays.
        """
        step = (xtr[1] - xtr[0]) if len(xtr) > 1 else 1.
        for k, ir in enumerate(itr):
            tr = np.clip(0.9 * d[:, ir] / norm[k], -clip, clip) * step
            ax.plot(xtr[k] + tr, tt, color=color, lw=lw,
                    label=label if k == 0 else None,
                    zorder=2 if fill is not None else 3)
            if fill is not None:
                ax.fill_betweenx(tt, xtr[k], xtr[k] + tr, where=tr > 0,
                                 color=fill, lw=0, zorder=1)

    def waveforms(self, dobs, dinv, dini=None, dt=None, ishot=None,
                  name='Waveform.png', tmax=None,
                  labels=('Observed', 'Initial', 'Inverted')):
        """Observed vs modelled waveforms for one shot.

        Panel (a) overlays the observed (filled, dark) and inverted (line,
        orange) wiggles over the decimated receiver line; the right column
        compares single traces at a few offsets from the source.
        All three data sets must share the time axis (sampling ``dt``).
        """
        ishot = self.par['ns'] // 2 if ishot is None else ishot
        do, dv = dobs[ishot], dinv[ishot]
        di = None if dini is None else dini[ishot]
        tt = np.arange(do.shape[0]) * dt
        it = slice(None) if tmax is None else slice(0, int(tmax / dt) + 1)
        tt, do, dv = tt[it], do[it], dv[it]
        di = None if di is None else di[it]

        every = self.wiggle_every
        itr = np.arange(0, self.par['nr'], every)
        xtr = self.x_r[itr, 0]
        # trace-by-trace normalisation by the observed peak
        norm = np.abs(do[:, itr]).max(axis=0)
        norm[norm == 0] = 1.

        fig = plt.figure(figsize=(7.2, 5.2))
        gs = fig.add_gridspec(len(self.trace_offsets), 2, width_ratios=[1.5, 1.],
                              wspace=0.24, hspace=0.30)

        ax = fig.add_subplot(gs[:, 0])
        self._wiggle(ax, do, tt, itr, xtr, C_TRUE, 0.6, norm, fill=C_TRUE,
                     label=labels[0])
        self._wiggle(ax, dv, tt, itr, xtr, C_INV, 0.7, norm, label=labels[2])
        ax.set_xlabel('Receiver position (km)')
        ax.set_ylabel('Time (s)')
        ax.set_ylim(tt[-1], tt[0])
        half = 1.2 * every * self.par['dr']
        ax.set_xlim(xtr[0] - half, xtr[-1] + half)
        _panel(ax, 'a', f'shot {ishot} ($x_s$ = {self.x_s[ishot, 0]:.2f} km), '
                        f'every {every}th trace')

        for k, off in enumerate(self.trace_offsets):
            ir = int(np.argmin(np.abs(self.x_r[:, 0]
                                      - (self.x_s[ishot, 0] + off))))
            axt = fig.add_subplot(gs[k, 1])
            a = np.abs(do[:, ir]).max() or 1.   # normalise by the observed peak
            axt.plot(tt, do[:, ir] / a, color=C_TRUE, lw=0.9, label=labels[0])
            if di is not None:
                axt.plot(tt, di[:, ir] / a, color=C_INIT, lw=0.8,
                         ls=(0, (4, 2.5)), label=labels[1])
            axt.plot(tt, dv[:, ir] / a, color=C_INV, lw=0.9, label=labels[2])
            axt.set_xlim(tt[0], tt[-1])
            amax = max(np.abs(d[:, ir]).max() for d in (do, di, dv)
                       if d is not None) / a
            axt.set_ylim(-1.15 * amax, 1.15 * amax)
            axt.grid(True, ls=':', lw=0.5)
            _panel(axt, 'bcdefg'[k],
                   f'offset {off:.1f} km ($x_r$ = {self.x_r[ir, 0]:.2f} km)')
            axt.set_ylabel('Norm. amp.')
            if k == len(self.trace_offsets) - 1:
                axt.set_xlabel('Time (s)')
            else:
                axt.set_xticklabels([])
            if k == 0:
                handles, hlabels = axt.get_legend_handles_labels()

        # one legend for the whole figure, under both columns: an in-axes
        # legend covers the wiggles on the left and the traces on the right
        fig.legend(handles, hlabels, loc='lower center', ncol=len(handles),
                   bbox_to_anchor=(0.5, -0.015), frameon=False,
                   columnspacing=1.6, handlelength=1.8)
        self._save(fig, name)

    def live_waveforms(self, dobs, dest, dt, iteration, ishot=None,
                       name='Waveformtmp.png', tmax=None, every=None,
                       labels=('Observed', 'Estimated'), rms=None, dpi=220):
        """Live observed-vs-estimated wiggle comparison, one shot.

        The counterpart of ``waveforms`` for the running job: one wiggle panel
        (observed filled dark, estimated over it in orange) plus the residual
        panel next to it, so what the misfit curve reports as a number can be
        read off the waveforms themselves - which arrivals are matched in phase,
        which are still cycle-skipped, and where the residual energy sits.

        ``dobs``/``dest`` are ``(nt, nr)`` for the single shot ``ishot`` and
        must share the time axis ``dt``. The residual is drawn on the same
        per-trace normalisation as the wiggles, so its size is directly
        comparable with the traces on the left.

        ``tmax`` is the last time plotted; ``None`` (the default) keeps the
        whole record, a number cuts it there, and ``'auto'`` trims to just
        past the last arrival.
        """
        ishot = self.par['ns'] // 2 if ishot is None else ishot
        do, de = np.asarray(dobs, dtype=float), np.asarray(dest, dtype=float)
        nt_c = min(do.shape[0], de.shape[0])
        do, de = do[:nt_c], de[:nt_c]
        tt = np.arange(nt_c) * dt
        if tmax == 'auto':
            # Crop to where the data actually is. Useful when the record runs
            # well past the last arrival: a panel that is a third flat traces
            # wastes the height the waveforms need. Off by default, because
            # cropping also hides late energy the inversion has not explained
            # yet - which on this figure is exactly what you are looking for.
            env = np.abs(do).max(axis=1)
            live = np.nonzero(env > 0.01 * (env.max() or 1.))[0]
            tmax = min(tt[-1], tt[live[-1]] * 1.08) if live.size else None
        if tmax is not None:
            it = slice(0, int(tmax / dt) + 1)
            tt, do, de = tt[it], do[it], de[it]

        # Denser than the wiggle panel of ``waveforms``: that figure shares its
        # width with a column of single traces, this one has half a page for
        # the gather, and phase matching is easier to judge when neighbouring
        # traces are close enough to show the moveout as a continuous event.
        # ~60 traces is about the limit before the wiggles merge at this width.
        if every is None:
            every = max(int(np.ceil(do.shape[1] / 60.)), 1)
        itr = np.arange(0, do.shape[1], every)
        xtr = self.x_r[itr, 0]
        norm = np.abs(do[:, itr]).max(axis=0)
        norm[norm == 0] = 1.

        fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)

        # Tighter clip than the default: at this trace density an excursion of
        # two spacings buries the neighbouring trace, and the point of the
        # panel is to compare two wavefields trace by trace.
        ax = axs[0]
        self._wiggle(ax, do, tt, itr, xtr, C_TRUE, 0.75, norm, fill=C_TRUE,
                     clip=1.3)
        self._wiggle(ax, de, tt, itr, xtr, C_INV, 0.85, norm, clip=1.3)
        _panel(ax, 'a', f'shot {ishot} ($x_s$ = {self.x_s[ishot, 0]:.2f} km), '
                        f'iteration {iteration}')

        # Residual on the same normalisation as (a), so its size is read
        # directly against the traces on the left: a flat panel (b) means the
        # waveforms match, without having to unpick the overlay.
        ax = axs[1]
        self._wiggle(ax, de - do, tt, itr, xtr, C_RES, 0.75, norm, fill=C_RES,
                     clip=1.3)
        r0 = (f', $E/E_\\mathrm{{obs}}$ = {rms:.3g}') if rms is not None else ''
        _panel(ax, 'b', f'Residual (estimated $-$ observed){r0}')

        for ax in axs:
            ax.set_xlabel('Receiver position (km)')
            ax.set_ylim(tt[-1], tt[0])
            half = 1.0 * every * self.par['dr']
            ax.set_xlim(xtr[0] - half, xtr[-1] + half)
        axs[0].set_ylabel('Time (s)')

        # Proxy handles: the real artists are one polyline per trace, so a
        # legend built from them shows a vertical baseline rather than a
        # wiggle. Draw the key by hand instead.
        keys = [plt.Line2D([], [], color=C_TRUE, lw=1.2, label=labels[0]),
                plt.Line2D([], [], color=C_INV, lw=1.2, label=labels[1])]
        # y is inverted, so 'upper left' is early time: the quiet band above the
        # first arrival, the one place on a shot gather reliably free of traces.
        axs[0].legend(handles=keys, loc='upper left', ncol=2, fontsize=7.5,
                      handlelength=1.5, columnspacing=1.0, borderpad=0.35,
                      handletextpad=0.5)
        fig.tight_layout(w_pad=1.0)
        # Not the 110 dpi of the other live figures. Those are single imshow
        # panels, which resample cleanly; this one is ~60 hairline wiggles side
        # by side, and below ~200 dpi a 0.6 pt stroke lands on less than a pixel
        # and the traces break up into dashes.
        self._save(fig, name, dpi=dpi)

    # --------------------------------------------------------- convergence

    def live_convergence(self, losshistory, vp_error, name='Losstmp.png', **kw):
        """Convergence figure refreshed every iteration (cheap, low dpi).

        Same panels as ``convergence`` so the running job can be watched
        without waiting for the run to finish.
        """
        if len(losshistory) < 2 or len(vp_error) < 2:
            return                       # nothing to draw a curve through yet
        self.convergence(losshistory, vp_error, name=name, dpi=110, **kw)

    def convergence(self, losshistory, vp_error, name='Loss.png',
                    xlabel='Iteration', errlabel='Iteration', dpi=None):
        """Normalised data misfit and model-error histories, side by side.

        Both histories are indexed by iteration and both start at index 0 (the
        starting model), so the two panels line up. Feed the misfit recorded in
        the callback, not the one recorded in the objective: L-BFGS-B evaluates
        the objective several times per iteration during its line search.
        """
        loss = np.asarray(losshistory, dtype=float)
        err = np.asarray(vp_error, dtype=float)
        if loss.size == 0 or err.size == 0:
            return
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))

        ax = axs[0]
        n = np.arange(loss.size)
        ax.semilogy(n, loss / loss[0], color=C_INV, lw=1.2,
                    marker='o' if loss.size <= 40 else None, ms=2.6,
                    mfc='w', mew=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Normalised misfit  $\\Phi/\\Phi_0$')
        ax.grid(True, which='both', ls=':', lw=0.5)
        ax.set_xlim(0, max(loss.size - 1, 1))
        _panel(ax, 'a', 'Data misfit')
        ax.annotate(f'$\\Phi/\\Phi_0$ = {loss[-1] / loss[0]:.3g}',
                    xy=(loss.size - 1, loss[-1] / loss[0]),
                    xytext=(-6, 12), textcoords='offset points', ha='right',
                    fontsize=8, color=C_INV,
                    bbox=dict(fc='w', ec='none', alpha=0.75, pad=1.5))

        ax = axs[1]
        ax.plot(np.arange(err.size), err, color=C_TRUE, lw=1.2,
                marker='o' if err.size <= 40 else None, ms=2.6, mfc='w', mew=0.7)
        ax.axhline(err[0], color=C_INIT, lw=0.9, ls=(0, (4, 2.5)))
        ax.text(0.99, err[0], 'initial ', color=C_INIT, fontsize=8, va='bottom',
                ha='right', transform=ax.get_yaxis_transform())
        ax.set_xlabel(errlabel)
        ax.set_ylabel('$\\|(m-m_{\\mathrm{true}})/m_{\\mathrm{true}}\\|_2$')
        ax.grid(True, ls=':', lw=0.5)
        ax.set_xlim(0, max(err.size - 1, 1))
        _panel(ax, 'b', f'Model error ({100. * (1 - err[-1] / err[0]):.1f} % '
                        f'reduction)')

        fig.tight_layout(w_pad=1.4)
        # After tight_layout: the tick spacing is chosen from the drawn width of
        # each panel, which is only final once the layout has settled.
        iteration_axis(axs[0], max(loss.size - 1, 1))
        iteration_axis(axs[1], max(err.size - 1, 1))
        self._save(fig, name, dpi=dpi)


##################################################################
# Snapshots -> movie
##################################################################

class SnapshotRecorder:
    """Per-iteration models, appended to a ``.npy`` stack on disk.

    The point of writing raw arrays rather than PNG frames is that recording
    then costs no matplotlib call at all - one ``float32`` write per iteration,
    ~0.5 MB for a 601x221 model - so it can run every iteration without
    competing with the objective for wall clock. Rendering is deferred to
    ``make_movie.py``, which means the colormap, frame rate and clipping of the
    movie can be changed afterwards without repeating the inversion.

    The file is a plain ``.npy`` written incrementally: a standard header
    reserving space for ``max_frames``, then the frames appended in place and
    the header's shape rewritten on ``close``. A run killed halfway therefore
    still leaves a readable stack of everything recorded up to that point,
    which is the case that matters for a multi-hour FWI.

    Parameters
    ----------
    path : str
        Output ``.npy`` file.
    shape : tuple
        Model shape ``(nx, nz)`` of a single frame.
    max_frames : int
        Upper bound on the number of frames, used to size the header. Writes
        past it are dropped (with a warning), never silently lost.
    every : int
        Record one frame every ``every`` iterations. ``0`` disables recording.
    meta : dict, optional
        Extra arrays/scalars saved alongside as ``<path stem>_meta.npz`` -
        axes, colour limits, the true model - so the movie can be rendered
        with the same geometry as the figures without the driving script.
    """

    def __init__(self, path, shape, max_frames, every=1, meta=None):
        self.path, self.shape, self.every = path, tuple(shape), int(every)
        self.max_frames, self.n = int(max_frames), 0
        self.iters, self.fp, self._closed = [], None, False
        if self.every <= 0:
            self._closed = True
            return
        # open_memmap gives a real .npy (header included) that can be grown
        # into without holding every frame in RAM
        from numpy.lib.format import open_memmap
        self.fp = open_memmap(path, mode='w+', dtype=np.float32,
                              shape=(self.max_frames, *self.shape))
        if meta:
            np.savez(os.path.splitext(path)[0] + '_meta.npz', **meta)

    def __call__(self, vp, iteration):
        """Record ``vp`` if this iteration is due. Returns True if written."""
        if self._closed or iteration % self.every:
            return False
        if self.n >= self.max_frames:
            print(f'  [snapshots] stack full at {self.max_frames} frames, '
                  f'iteration {iteration} not recorded', flush=True)
            return False
        self.fp[self.n] = np.asarray(vp, dtype=np.float32).reshape(self.shape)
        self.iters.append(int(iteration))
        self.n += 1
        return True

    def close(self):
        """Trim the stack to the frames actually written and flush it."""
        if self._closed:
            return
        self._closed = True
        self.fp.flush()
        del self.fp                       # release the memmap before rewriting
        self.fp = None
        # Rewrite the header for the real frame count and drop the unused tail.
        # Copy frame by frame into a second memmap and rename over the
        # original: saving straight back onto the path being read truncates the
        # file under the reader, and holding the whole stack in RAM to avoid
        # that defeats the point of memory-mapping it.
        from numpy.lib.format import open_memmap
        tmp = self.path + '.tmp'
        src = np.load(self.path, mmap_mode='r')
        dst = open_memmap(tmp, mode='w+', dtype=src.dtype,
                          shape=(self.n, *self.shape))
        for k in range(self.n):
            dst[k] = src[k]
        dst.flush()
        del dst, src
        os.replace(tmp, self.path)
        np.save(os.path.splitext(self.path)[0] + '_iters.npy',
                np.asarray(self.iters, dtype=int))
        print(f'  [snapshots] {self.n} frames -> {self.path}', flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

