from dataclasses import dataclass
from typing import Callable


@dataclass
class ModelParameters:
    Lx: float  # Domain dimensions
    Ly = float  # Domain dimensions
    Nx: int  # Number of elements
    Ny = int  # Number of elements
    dt = float  # Time step
    T = float  # Total simulation time
    U0 = float  # characteristic velocity of the macroscopic flow
    L0 = float  # characteristic length of the macroscopic flow
    lambd = float  # characteristic relaxation time of a dumbbell
    viscosity_s = float
    viscosity_p = float
    regularization = bool
    b = float # ...
    delta = float # ...
