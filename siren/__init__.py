# siren/__init__.py
"""SIREN network + velocity-model parameterisation for FWI.

The network code is taken from
https://github.com/nguyenvanhaibk92/KronosAI_solutions (Task1/utils), keeping
only what SIREN itself needs -- the electromagnetic PINN, PML physics, level-set
permittivity and plotting utilities of that repo are not used here.
"""

from .siren_network import (
    init_mlp_params,
    SIREN_neural_one_sample,
    SIREN_neural,
    DEFAULT_OMEGA_0,
)
from .velocity_field import SirenVelocity, make_grid

__all__ = [
    'init_mlp_params',
    'SIREN_neural_one_sample',
    'SIREN_neural',
    'DEFAULT_OMEGA_0',
    'SirenVelocity',
    'make_grid',
]
