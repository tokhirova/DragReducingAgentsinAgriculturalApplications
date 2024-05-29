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
# comm = MPI.COMM_WORLD
# mesh = dolfinx.mesh.create_rectangle(comm, [np.array([0., 0.]), np.array([Lx, Ly])],
#                                      [Nx, Ny], dolfinx.mesh.CellType.triangle)
#
# # init function spaces and Trial/Test functions
# s_el = basix.ufl.element("Lagrange", mesh.topology.cell_name(),1 ,shape=(2,2))
# S_h_0 = dolfinx.fem.functionspace(mesh, s_el)
#
# sigma = dolfinx.fem.Function(S_h_0)
# phi = ufl.TestFunction(S_h_0)
#
# # vector field
# v_el = basix.ufl.element("Lagrange", mesh.topology.cell_name(),1 ,shape=(2,))
# V = dolfinx.fem.functionspace(mesh, v_el)
# u = dolfinx.fem.Function(V)
# u.interpolate(lambda x: [-x[1],x[0]])
#
# # boundary conditions
# sigmaD = dolfinx.fem.Function(S_h_0)
# sigmaD.sub(0).interpolate(lambda x: [0,0])
# sigmaD.sub(1).interpolate(lambda x: [0,0])
# sigmaD.sub(2).interpolate(lambda x: [0,0])
# sigmaD.sub(3).interpolate(lambda x: [0,0])
#
# tdim = mesh.topology.dim
# fdim = tdim - 1
# mesh.topology.create_connectivity(fdim, tdim)
# boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
# boundary_dofs = dolfinx.fem.locate_dofs_topological(S_h_0, fdim, boundary_facets)
# bc1 = dolfinx.fem.dirichletbc(sigmaD, boundary_dofs)
#
# # defining the problem
#
# a = 1/(1-ufl.tr(sigma)/b)
# dets = ufl.det(sigma)
#
# term2 = ((u[0]*sigma[0,0].dx(0)+u[1]*sigma[0,0].dx(1))*phi[0,0] + (u[0]*sigma[0,1].dx(0)+u[1]*sigma[0,1].dx(1))*phi[0,1]
#          + (u[0]*sigma[1,0].dx(0)+u[1]*sigma[1,0].dx(1))*phi[1,0] + (u[0]*sigma[1,1].dx(0)+u[1]*sigma[1,1].dx(1))*phi[1,1] )*dx
#
# term3 = ((u[0].dx(0)*sigma[0,0]+u[1].dx(1)*sigma[1,0])*phi[0,0] + (u[0].dx(0)*sigma[0,1]+u[1].dx(1)*sigma[1,1])*phi[0,1]
#          + (u[1].dx(0)*sigma[1,0]+u[1].dx(1)*sigma[1,0])*phi[1,0] + (u[1].dx(0)*sigma[0,1]+u[1].dx(1)*sigma[1,1])*phi[1,1] )*dx
#
# term4 = ((sigma[0,0]*u[0].dx(0) + sigma[0,1]*u[0].dx(1))*phi[0,0] + (sigma[1,0]*u[1].dx(0) + sigma[0,1]*u[1].dx(1))*phi[0,1]
#          + (sigma[1,0]*u[0].dx(0) + sigma[1,1]*u[0].dx(1))*phi[1,0] + (sigma[1,0]*u[1].dx(1) + sigma[1,1]*u[1].dx(0))*phi[1,1])*dx
#
# term5 = (((a-sigma[1,1]/dets)*sigma[0,0]-sigma[0,1]/dets*sigma[1,0])*phi[0,0] + ((a-sigma[1,0]/dets)*sigma[0,1]-sigma[0,1]/dets*sigma[1,1])*phi[0,1]
#          + ((a-sigma[0,0]/dets)*sigma[1,0]-sigma[1,0]/dets*sigma[0,0])*phi[1,0] + (((a-sigma[0,0]/dets)*sigma[1,1]-sigma[1,0]/dets*sigma[0,1])*phi[1,1])) *dx
#
# F = term2 - term3 - term4 + term5


# 1 dim test
comm_t = MPI.COMM_WORLD
mesh_t = dolfinx.mesh.create_rectangle(comm_t, [np.array([0., 0.]), np.array([Lx, Ly])],
                                     [Nx, Ny], dolfinx.mesh.CellType.triangle)
s_el_dim = basix.ufl.element("Lagrange", mesh_t.topology.cell_name(),1)
S_h_0_dim = dolfinx.fem.functionspace(mesh_t, s_el_dim)

sigma_t = dolfinx.fem.Function(S_h_0_dim)
phi_t = ufl.TestFunction(S_h_0_dim)

# vector field
v_el_t = basix.ufl.element("Lagrange", mesh_t.topology.cell_name(),1)
V_t = dolfinx.fem.functionspace(mesh_t, v_el_t)
# = dolfinx.fem.Function(V_t)
#u_t.sub(0).interpolate(lambda x: [0])#[-x[1],x[0]])
#u_t.sub(1).interpolate(lambda x: [0])#[-x[1],x[0]])
# def vector_field(x):
#     values = np.zeros((2, x.shape[1]))
#     values[0] = 0#-x[1]
#     values[1] = 0#x[0]
#     return values
#
# u_t = dolfinx.fem.Function(V_t)
# u_t.interpolate(vector_field)

# boundary conditions
def q(u):
    return 1 + u**2


domain = dolfinx.mesh.create_rectangle(comm_t, [np.array([0., 0.]), np.array([Lx, Ly])],
                                     [Nx, Ny], dolfinx.mesh.CellType.triangle)
x = ufl.SpatialCoordinate(domain)
u_ufl = (1 + x[0] + 2 * x[1])
s_el_dim = basix.ufl.element("Lagrange", mesh_t.topology.cell_name(),1)
V = dolfinx.fem.functionspace(domain, s_el_dim)
def u_exact(x): return eval(str(u_ufl))
u_D = dolfinx.fem.Function(V)
u_D.interpolate(u_exact)
fdim = domain.topology.dim - 1
boundary_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = dolfinx.fem.dirichletbc(u_D, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))
uh = dolfinx.fem.Function(V)
v = ufl.TestFunction(V)

vector_field1 = -x[1]
vector_field2 = x[0]

F_geht1 = (uh.dx(0)* v.dx(0) + uh.dx(1)* v.dx(1)  )*ufl.dx
F_geht2 = (vector_field1*uh.dx(0)* v.dx(0) + vector_field1*uh.dx(1)* v.dx(1)  )*ufl.dx
F_geht_nicht1 = (vector_field1*uh.dx(0)* v.dx(0) + vector_field2*uh.dx(1)* v.dx(1)  )*ufl.dx
F_geht_nicht2 = ((uh.dx(0)+ uh.dx(1))* v)*ufl.dx
F_geht_nicht3 = ((vector_field1*uh.dx(0)+ vector_field2*uh.dx(1))* v)*ufl.dx

#F = (ufl.div(uh)*v)*ufl.dx
#F = (uh* v.dx(0) + uh* v.dx(1)  )*ufl.dx
# tdim_t = mesh_t.topology.dim
# fdim_t = tdim_t - 1
# mesh_t.topology.create_connectivity(fdim_t, tdim_t)
# boundary_facets_t = dolfinx.mesh.exterior_facet_indices(mesh_t.topology)
# boundary_dofs_t = dolfinx.fem.locate_dofs_topological(S_h_0_dim, fdim_t, boundary_facets_t)
# bc1_t = dolfinx.fem.dirichletbc(sigmaD_dim, boundary_dofs_t)

#F1 = (ufl.dot(ufl.grad(sigma_t),ufl.grad(phi_t)))*dx


problem = dolfinx.fem.petsc.NonlinearProblem(F_geht3, uh, bcs=[bc])

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
n, converged = solver.solve(uh)
assert (converged)
print(f"Number of interations: {n:d}")