from phijax.equations.boundary import base_boundary_residual, free_slip_boundary, free_slip_residual, no_slip_residual
from phijax.equations.data_fidelity import (
    base_data_fidelity,
    base_data_fidelity_residual,
    phase_wrapped_fidelity,
    phase_wrapped_residuals,
)
from phijax.equations.metadata import get_default_ntk_stream, get_residual_names, residual_equation
from phijax.equations.pde import (
    burgers_1d,
    cartesian_2d_navier_stokes,
    cartesian_3d_navier_stokes,
    polar_navier_stokes,
    spherical_navier_stokes,
)

__all__ = [
    "base_boundary_residual",
    "base_data_fidelity",
    "base_data_fidelity_residual",
    "burgers_1d",
    "cartesian_2d_navier_stokes",
    "cartesian_3d_navier_stokes",
    "free_slip_boundary",
    "free_slip_residual",
    "get_default_ntk_stream",
    "get_residual_names",
    "no_slip_residual",
    "phase_wrapped_fidelity",
    "phase_wrapped_residuals",
    "polar_navier_stokes",
    "residual_equation",
    "spherical_navier_stokes",
]
