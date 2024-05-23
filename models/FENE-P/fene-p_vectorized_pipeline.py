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
L0 = 1.0
U0 = 1.0
lamb = 1.0
vis_s = 1.0
vis_p = 1.0
Re = L0 * U0 / (vis_s + vis_p)
Wi = lamb * U0 / L0
eps = 0.5
b=1

# mesh
comm = MPI.COMM_WORLD
mesh = dolfinx.mesh.create_rectangle(comm, [np.array([0., 0.]), np.array([Lx, Ly])],
                                     [Nx, Ny], dolfinx.mesh.CellType.triangle)

# init function spaces and Trial/Test functions
s_el = basix.ufl.element("Lagrange", mesh.topology.cell_name(),1 ,shape=(2,2))
S_h_0 = dolfinx.fem.functionspace(mesh, s_el)

sigma = dolfinx.fem.Function(S_h_0)
phi = ufl.TestFunction(S_h_0)

# vector field
v_el = basix.ufl.element("Lagrange", mesh.topology.cell_name(),1 ,shape=(2,))
V = dolfinx.fem.functionspace(mesh, v_el)
u = dolfinx.fem.Function(V)
u.sub(0).interpolate(lambda x: -x[1])
u.sub(1).interpolate(lambda x: x[0])

# boundary conditions
sigmaD = dolfinx.fem.Function(S_h_0)
sigmaD.sub(0).interpolate(lambda x: 0*x[0])
sigmaD.sub(1).interpolate(lambda x: 0*x[0])
sigmaD.sub(2).interpolate(lambda x: 0*x[0])
sigmaD.sub(3).interpolate(lambda x: 0*x[0])

tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(S_h_0, fdim, boundary_facets)
bc1 = dolfinx.fem.dirichletbc(sigmaD, boundary_dofs)

# defining the problem

a = 1/(1-ufl.tr(sigma)/b)
dets = ufl.det(sigma)

term2 = ((u[0]*sigma[0,0].dx(0)+u[1]*sigma[0,0].dx(1))*phi[0,0] + (u[0]*sigma[0,1].dx(0)+u[1]*sigma[0,1].dx(1))*phi[0,1]
         + (u[0]*sigma[1,0].dx(0)+u[1]*sigma[1,0].dx(1))*phi[1,0] + (u[0]*sigma[1,1].dx(0)+u[1]*sigma[1,1].dx(1))*phi[1,1] )*dx

term3 = ((u[0].dx(0)*sigma[0,0]+u[1].dx(1)*sigma[1,0])*phi[0,0] + (u[0].dx(0)*sigma[0,1]+u[1].dx(1)*sigma[1,1])*phi[0,1]
         + (u[1].dx(0)*sigma[1,0]+u[1].dx(1)*sigma[1,0])*phi[1,0] + (u[1].dx(0)*sigma[0,1]+u[1].dx(1)*sigma[1,1])*phi[1,1] )*dx

term4 = ((sigma[0,0]*u[0].dx(0) + sigma[0,1]*u[0].dx(1))*phi[0,0] + (sigma[1,0]*u[1].dx(0) + sigma[0,1]*u[1].dx(1))*phi[0,1]
         + (sigma[1,0]*u[0].dx(0) + sigma[1,1]*u[0].dx(1))*phi[1,0] + (sigma[1,0]*u[1].dx(1) + sigma[1,1]*u[1].dx(0))*phi[1,1])*dx

term5 = (((a-sigma[1,1]/dets)*sigma[0,0]-sigma[0,1]/dets*sigma[1,0])*phi[0,0] + ((a-sigma[1,0]/dets)*sigma[0,1]-sigma[0,1]/dets*sigma[1,1])*phi[0,1]
         + ((a-sigma[0,0]/dets)*sigma[1,0]-sigma[1,0]/dets*sigma[0,0])*phi[1,0] + (((a-sigma[0,0]/dets)*sigma[1,1]-sigma[1,0]/dets*sigma[0,1])*phi[1,1])) *dx

F = term2 - term3 - term4 + term5
problem = dolfinx.fem.petsc.NonlinearProblem(F, sigma, bcs=[bc1])
from IPython import embed
embed()

solver = NewtonSolver(MPI.COMM_WORLD, problem)
#solver.convergence_criterion = "incremental"
#solver.rtol = 1e-6
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