from dataclasses import dataclass
from typing import Callable


@dataclass
class ModelParameters:
    Lx: float = 1.  # Domain dimensions
    Ly: float = 1.  # Domain dimensions
    Nx: int = 50  # Number of elements
    Ny: int = 50  # Number of elements
    T: float = 100  # Total simulation time
    num_time_steps: int = 1000  # time steps
    U0: float = 1.  # characteristic velocity of the macroscopic flow
    L0: float = 1.  # characteristic length of the macroscopic flow
    lambd: float = 1.  # characteristic relaxation time of a dumbbell
    viscosity_s: float = 1.
    viscosity_p: float = 1.
    b: float = 1.  # ...
    eps: float = 0.5  # ...
