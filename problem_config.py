"""Shared problem definition for the two-Tesseract SIREN-FWI pipeline.

Both Tesseracts need to agree on the same grid, and the workflow needs the true
model to generate observed data. Rather than duplicate the numbers in three
places, they live here.

The parameters are exactly those of
``script_siren_Nhat/AcousticVel_L2_SIREN_random.py``: Marmousi on a 601 x 221
grid at 15 m, 64 shots and 300 receivers spread over the whole surface, a 6 s
record at 2 ms, and an 8 Hz Ricker source.

Nothing here imports devito or jax - it is pure numpy geometry, so both
Tesseracts can import it regardless of which stack they carry.
"""

import os

import numpy as np

# ``data/Marm.bin`` is vendored into this directory so the folder is
# self-contained. Overridable for a different model or a container layout.
DATA_DIR = os.environ.get(
    'FWI_DATA_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
VELOCITY_FILE = os.path.join(DATA_DIR, 'Marm.bin')

# Marmousi as stored on disk.
_FILE_NX, _FILE_NZ = 601, 221

# Sources and receivers span the whole surface, as in the reference script:
# with the line confined to part of the model, the rest is only ever lit by
# wide-angle arrivals and inverts poorly. SRC_EDGE keeps a margin at each end
# so no source sits on top of the absorbing boundary (nbl = 20 cells = 0.30 km).
SRC_EDGE = 0.30

PAR = {'nx': 601,  'dx': 0.015,  'ox': 0.,
       'nz': 221,  'dz': 0.015,  'oz': 0.,
       'ns': 64,   'ds': 0.,     'os': 0.,  'sz': 0.,
       'nr': 300,  'dr': 0.,     'or': 0.,  'rz': 0.,
       'nt': 3000, 'dt': 0.002,  'ot': 0.,
       'freq': 8.,
       }

# Spread both lines over the full surface. Derived from the grid rather than
# hard-coded so they stay consistent if nx/dx/ns/nr change.
_xmax = PAR['ox'] + (PAR['nx'] - 1) * PAR['dx']
PAR['os'] = PAR['ox'] + SRC_EDGE
PAR['ds'] = ((_xmax - 2 * SRC_EDGE - PAR['ox']) / (PAR['ns'] - 1)
             if PAR['ns'] > 1 else 0.)
PAR['or'] = PAR['ox']
PAR['dr'] = (_xmax - PAR['ox']) / (PAR['nr'] - 1) if PAR['nr'] > 1 else 0.

SHAPE = (PAR['nx'], PAR['nz'])
SPACING = (PAR['dx'], PAR['dz'])
ORIGIN = (PAR['ox'], PAR['oz'])
SPACE_ORDER = 4
NBL = 20
TMAX = (PAR['nt'] - 1) * PAR['dt'] + PAR['ot']

# Water layer: the reference script masks on vp > 1.52 and pins the masked
# cells to 1.5 km/s.
WATER_VP = 1.52
MASK_VALUE = 1.5

# Velocity bounds of the SIREN sigmoid: the range the network can represent.
#
# The reference script takes these from vp_true.min()/max(). That is knowledge
# of the answer, but the bounds only have to *contain* the model - a real
# inversion would set them from the expected geology, wider than the truth.
# Stating them as constants keeps the model Tesseract from loading vp_true at
# all, while still bracketing Marmousi's 1.484-5.695 km/s. They must not be
# tighter than the truth: a sigmoid saturating inside the true range could
# never reach the extremes.
VP_MIN, VP_MAX = 1.4, 5.8


def load_vp_true():
    """True velocity model, (nx, nz) float32 in km/s."""
    vp = np.fromfile(VELOCITY_FILE, np.float32).reshape(_FILE_NZ, _FILE_NX).T
    return np.ascontiguousarray(vp, dtype=np.float32)


def source_receiver_positions():
    """``(x_s, x_r)``, each (n, 2) arrays of (x, z) coordinates in km."""
    x_s = np.zeros((PAR['ns'], 2))
    x_s[:, 0] = np.arange(PAR['ns']) * PAR['ds'] + PAR['os']
    x_s[:, 1] = PAR['sz']

    x_r = np.zeros((PAR['nr'], 2))
    x_r[:, 0] = np.arange(PAR['nr']) * PAR['dr'] + PAR['or']
    x_r[:, 1] = PAR['rz']
    return x_s, x_r


def water_mask(vp_true=None):
    """1 below the seabed, 0 in the water. Same rule as the reference script.

    Both Tesseracts must freeze exactly the same cells, or the chain rule is
    inconsistent - one side would pin a cell the other still reports a gradient
    for. So both call this, and the model Tesseract does read the true model to
    get it. Picking the seabed off the data instead would be the honest thing
    in a real survey; a flat cut at a nominal depth is not good enough here,
    because the Marmousi seabed sits at cell 22 or 23 depending on the trace.
    """
    vp_true = load_vp_true() if vp_true is None else vp_true
    return (vp_true > WATER_VP).astype(np.float32)


# --- SIREN configuration -------------------------------------------------
# The reference script's defaults: 256 wide, 4 hidden layers, omega_0 20.
SIREN_HIDDEN = (256, 256, 256, 256)
SIREN_OMEGA0 = 20.0
SIREN_SEED = 0
