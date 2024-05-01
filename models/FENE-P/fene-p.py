import dolfinx
import ufl
import numpy as np
from mpi4py import MPI

# Define parameters
Lx, Ly = 1.0, 1.0  # Domain dimensions
Nx, Ny = 32, 32  # Number of elements
dt = 0.01  # Time step
T = 1.0  # Total simulation time

comm = MPI.COMM_WORLD

# Create mesh
mesh = dolfinx.mesh.create_rectangle(comm, [[0., 0.], [Lx, Ly]], [Nx, Ny])

# Define function space
V = dolfinx.fem.VectorFunctionSpace(mesh, dolfinx.fem.function.ElementMetaData("CG", 1))

# Define trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define parameters
K, b = 1.0, 1.0


def y0_init(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = 0.0
    return values


def y1_init(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = 0.0
    return values


# Define initial condition
u_init = dolfinx.fem.Function(V)
u_init.sub(0).interpolate(y0_init)
u_init.sub(1).interpolate(y1_init)

# Define boundary conditions
bc = dolfinx.fem.dirichletbc(u_init, dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.linalg.norm(x) < 1.0e-10))

# Define FENE-P model
a = ufl.dot(u, u)
inv = ufl.inv(ufl.Identity(2))
outer = ufl.outer(u, u)
denom = (1 - ufl.dot(u, u) / K ** 2)
print(inv) #2x2
print(outer) #2x2
print(denom)
print(u)
ba = ufl.dot(inv + outer / denom, u)
from IPython import embed
embed()


# Create time stepper
problem = dolfinx.fem.LinearProblem(F, u, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
t = 0
while t < T:
    # Update time
    t += dt

    # Solve FENE-P equation
    problem.solve()

    # Output
    if t % 0.1 == 0:
        u.rename("velocity", "velocity")
        dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"u_{t:.1f}.xdmf", "w").write_mesh(mesh)
        dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"u_{t:.1f}.xdmf", "w").write_function(u)

# Plot solution
import dolfinx.plot

dolfinx.plot.create_vtk_topology(mesh, 2)
dolfinx.plot.plot(u)

# Hold plot
import matplotlib.pyplot as plt

plt.show()
