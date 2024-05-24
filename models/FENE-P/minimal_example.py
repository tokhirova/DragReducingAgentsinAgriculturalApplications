import dolfinx
import scipy.linalg
import scipy
import ufl
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
import basix
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import petsc4py
import fene_p_parameters
import dolfinx.fem.petsc
from dolfinx.nls.petsc import NewtonSolver

# Constants
Lx = 1.0
Ly = 1.0
Nx = 50
Ny = 50
b = 10
Wi = 1

# 1 dim test
# mesh
comm_t = MPI.COMM_WORLD
domain = dolfinx.mesh.create_rectangle(comm_t, [np.array([0., 0.]), np.array([Lx, Ly])],
                                       [Nx, Ny], dolfinx.mesh.CellType.triangle)

x = ufl.SpatialCoordinate(domain)

# function space
s_el_dim = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
V = dolfinx.fem.functionspace(domain, s_el_dim)

# boundary conditions
u_ufl = (1 + x[0] + 2 * x[1])


def u_exact(x): return eval(str(u_ufl))


u_D = dolfinx.fem.Function(V)
u_D.interpolate(u_exact)
fdim = domain.topology.dim - 1
boundary_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = dolfinx.fem.dirichletbc(u_D, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))

# Test function definition
sigma = dolfinx.fem.Function(V)
phi = ufl.TestFunction(V)

# vector field definition u(x,y)=(-y,x)
vector_field1 = -x[1]
vector_field2 = x[0]

# minimal examples, der Doppelpunkt ist definiert als A:B = tr(A*B^T)

# ausmultiplizierte variante von dot(grad(u),grad(v)) -> funktioniert
F_geht1 = (sigma.dx(0) * phi.dx(0) + sigma.dx(1) * phi.dx(1)) * ufl.dx

# ausmultiplizierte variante von vector_field1*dot(grad(u),grad(v)) -> funktioniert
F_geht2 = (vector_field1 * sigma.dx(0) * phi.dx(0) + vector_field1 * sigma.dx(1) * phi.dx(1)) * ufl.dx

# ausmultiplizierte variante von F_geht2 nur mit vector_field2 als Vorfaktor in der 2. Komponente -> functioniert nicht
F_geht_nicht1 = (vector_field1 * sigma.dx(0) * phi.dx(0) + vector_field2 * sigma.dx(1) * phi.dx(1)) * ufl.dx

# zweiter Term der Fokker Plank Gleichung ((vector_field * grad)*sigma):phi -> Funktioniert nicht
F_geht_nicht2 = (vector_field1 * sigma.dx(0) * phi + vector_field2 * sigma.dx(1) * phi) * ufl.dx

# dritter Term der Fokker Plank Gleichung ((grad(vector_field)*sigma):phi)  -> Funktioniert nicht
F_geht_nicht3 = (vector_field1.dx(0) * sigma * phi + vector_field2.dx(1) * sigma * phi) * ufl.dx

# fünfter Term (nichtlinearität) der Fokker Plank Gleichung (A(sigma)/Wi*sigma):phi)  -> Funktioniert ohne die -1
F_geht3 = (((sigma*phi)/(1-sigma/b))) * ufl.dx
# Funktioniert nicht mit die -1
F_geht_nicht4 = (((sigma*phi)/(1-sigma/b)-phi)) * ufl.dx

problem = dolfinx.fem.petsc.NonlinearProblem(F_geht_nicht2-F_geht_nicht3-F_geht_nicht3+F_geht_nicht4, sigma, bcs=[bc])

solver = NewtonSolver(MPI.COMM_WORLD, problem)
# solver.convergence_criterion = "incremental"
# solver.rtol = 1e-6
solver.report = True

# ksp = solver.krylov_solver
# opts = PETSc.Options()
# option_prefix = ksp.getOptionsPrefix()
# opts[f"{option_prefix}ksp_type"] = "cg"
# opts[f"{option_prefix}pc_type"] = "gamg"
# opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# ksp.setFromOptions()
#
# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
n, converged = solver.solve(sigma)
assert (converged)
print(f"Number of interations: {n:d}")
