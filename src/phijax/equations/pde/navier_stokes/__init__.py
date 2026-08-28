from phijax.equations.pde.navier_stokes.cartesian import (
    cartesian_2d_navier_stokes,
    cartesian_3d_navier_stokes,
)
from phijax.equations.pde.navier_stokes.polar import polar_navier_stokes
from phijax.equations.pde.navier_stokes.spherical import spherical_navier_stokes

__all__ = [
    "cartesian_2d_navier_stokes",
    "cartesian_3d_navier_stokes",
    "polar_navier_stokes",
    "spherical_navier_stokes",
]
