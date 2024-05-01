import fenics as fe

import numpy as np

# Define parameters
Lx, Ly = 1.0, 1.0  # Domain dimensions
Nx, Ny = 32, 32    # Number of elements
dt = 0.01          # Time step
T = 1.0            # Total simulation time

# Define mesh
mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(Lx, Ly), Nx, Ny)

# Define function space
P1 = fe.FiniteElement('P', mesh.ufl_cell(), 1)
element = fe.MixedElement([P1, P1, P1])
V = fe.FunctionSpace(mesh, element)

# Define trial and test functions
u, v, p = fe.TrialFunctions(V)
w, q, r = fe.TestFunctions(V)

# Define parameters
K, b, dt = 1.0, 1.0, 0.01

# Define initial condition
u_init = fe.Expression(('0', '0'), degree=1)
p_init = fe.Expression('0', degree=1)
u_n = fe.interpolate(u_init, V.sub(0).collapse())
p_n = fe.interpolate(p_init, V.sub(2).collapse())

# Define boundary conditions
bcu = fe.DirichletBC(V.sub(0), fe.Constant((0, 0)), 'on_boundary')
bcp = fe.DirichletBC(V.sub(2), fe.Constant(0), 'on_boundary')

# Define FENE-P model
U = 0.5 * (fe.dot(u, u) + b * b * fe.inner(fe.inv(fe.Identity(2)) + fe.outer(u, u) / (1 - fe.dot(u, u) / K**2), u))
F = -fe.inner(fe.grad(p), u) + fe.inner(fe.grad(u) * u, v) * fe.dx + 2 * fe.inner(fe.sym(fe.grad(u)), fe.sym(fe.grad(v))) * fe.dx - fe.div(v) * p * fe.dx

# Time-stepping
t = 0
while t < T:
    # Update time
    t += dt

    # Solve FENE-P equation
    fe.solve(fe.inner(fe.grad(u_n) * u_n, w) * fe.dx + 2 * fe.inner(fe.sym(fe.grad(u_n)), fe.sym(fe.grad(w))) * fe.dx - fe.div(w) * p_n * fe.dx == 0, u_n)

    # Apply boundary conditions
    bcu.apply(u_n.vector())
    bcp.apply(p_n.vector())

    # Output
    if t % 0.1 == 0:
        fe.File(f'u_{t:.1f}.pvd') << u_n

# Plot solution
fe.plot(u_n, title='Velocity field')
fe.plot(p_n, title='Pressure field')

# Hold plot
fe.interactive()
