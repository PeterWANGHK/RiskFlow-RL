"""
DRIFT PDE Solver
================
Re-exports PDE solver from root level for package organization.
"""

import sys
import os

# Add parent directory to path to import root-level pde_solver
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import and re-export from root pde_solver
from pde_solver import (
    PDESolver,
    create_vehicle,
    move_vehicle,
    compute_Q_vehicle,
    compute_Q_occlusion,
    compute_Q_merge,
    compute_total_Q,
    compute_velocity_field,
    compute_diffusion_field,
)

__all__ = [
    'PDESolver',
    'create_vehicle',
    'move_vehicle',
    'compute_Q_vehicle',
    'compute_Q_occlusion',
    'compute_Q_merge',
    'compute_total_Q',
    'compute_velocity_field',
    'compute_diffusion_field',
]
