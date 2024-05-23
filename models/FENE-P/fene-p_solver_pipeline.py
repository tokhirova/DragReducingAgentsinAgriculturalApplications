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


# helper functions
def petsc_matrix_to_numpy(mat):
    dim1, dim2 = mat.size
    return np.array([[mat.getValue(i, j) for i in range(dim1)] for j in range(dim2)])


def petsc_vector_to_numpy(vec):
    dim = vec.size
    return np.array([vec.getValue(i) for i in range(dim)])


def array2petsc4py(g):
    Xpt = PETSc.Mat().createAIJ(g.shape)
    Xpt.setUp()
    Xpt.setValues(range(0, g.shape[0]), range(0, g.shape[1]), g)
    Xpt.assemble()
    return Xpt


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

# init mesh
comm = MPI.COMM_WORLD
mesh = dolfinx.mesh.create_rectangle(comm, [np.array([0., 0.]), np.array([Lx, Ly])],
                                     [Nx, Ny], dolfinx.mesh.CellType.triangle)

# init function spaces and Trial/Test functions
S_h_0 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
sigma_11 = dolfinx.fem.Function(S_h_0)
sigma_12 = dolfinx.fem.Function(S_h_0)
sigma_21 = dolfinx.fem.Function(S_h_0)
sigma_22 = dolfinx.fem.Function(S_h_0)

phi_11 = ufl.TestFunction(S_h_0)
phi_12 = ufl.TestFunction(S_h_0)
phi_21 = ufl.TestFunction(S_h_0)
phi_22 = ufl.TestFunction(S_h_0)

# vector field
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u1 = dolfinx.fem.Function(V)
u1.interpolate(lambda x: -x[1])
u2 = dolfinx.fem.Function(V)
u2.interpolate(lambda x: x[0])
u = [u1,u2]

# boundary conditions
sigmaD = dolfinx.fem.Function(S_h_0)
sigmaD.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)

tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(S_h_0, fdim, boundary_facets)
bc1 = dolfinx.fem.dirichletbc(sigmaD, boundary_dofs)

f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(-6))

# defining the problem
# rhs
L = dolfinx.fem.form(f * phi_11 * dx)

# helper variables
dets = sigma_11 * sigma_22 - sigma_12 * sigma_21
trs = sigma_11 + sigma_22

# lhs 1
#lhs1 = ((sigma_11 - sigma_init[0, 0]) * phi_11 + (sigma_12 - sigma_init[0, 1]) * phi_21) * dx
#from IPython import embed
#embed()
lhs1 = -(-((u[0] * sigma_11.dx(0) + u[1] * sigma_12.dx(1)) * phi_11 + (
            u[0] * sigma_12.dx(0) + u[1] * sigma_12.dx(1)) * phi_21)) * dx
lhs1 -= (u[0].dx(0) * (sigma_11 * phi_11 + sigma_12 * phi_21) + u[0].dx(1) * (
            sigma_21 * phi_11 + sigma_22 * phi_21)) * dx
lhs1 -= (sigma_11 * (u[0].dx(0) * phi_11 + u[1].dx(1) * phi_21) + sigma_12 * (
            u[1].dx(0) * phi_11 + u[1].dx(1) * phi_21)) * dx
lhs1 += ((((sigma_11 * (1/(1 - trs) - sigma_21 / dets) + sigma_21 * sigma_12) / dets) * phi_11 +
          ((sigma_12 * (1/(1 - trs) - sigma_22 / dets) + sigma_22 * sigma_12) / dets) * phi_21)) / Wi * dx
a1 = dolfinx.fem.form(lhs1)   # ufl.dot(ufl.grad(sigma_12), ufl.grad(phi_11)) * dx)

# A1 = assemble_matrix(a1, bcs=[bc1])
# A1.assemble()
# A1_numpy = petsc_matrix_to_numpy(A1)
# b1 = create_vector(L)
# b1.assemble()
# set_bc(b1, bcs=[bc1])
# b1_numpy = petsc_vector_to_numpy(b1)

# lhs 2
#lhs2 = ((sigma_11-sigma_init[0,0])*phi_12 + (sigma_12 - sigma_init[0,1])*phi_22)*dx
lhs2 = -(-((u[0]*sigma_11.dx(0) + u[1]*sigma_12.dx(1))*phi_12+(u[0]*sigma_12.dx(0)+u[1]*sigma_12.dx(1))*phi_22))*dx
lhs2 -= (u[0].dx(0)*(sigma_11*phi_12 + sigma_12*phi_22) + u[0].dx(1)*(sigma_21*phi_12 + sigma_22*phi_22))*dx
lhs2 -= (sigma_11*(u[0].dx(0)*phi_12+u[1].dx(1)*phi_22)+ sigma_12*(u[1].dx(0)*phi_12+u[1].dx(1)*phi_22))*dx
lhs2 += (((sigma_11 * (ufl.inv(1 - trs) - sigma_21 / dets) + sigma_21 * sigma_12 / dets) * phi_12 +
          (sigma_12 * (ufl.inv(1 - trs) - sigma_22 / dets) + sigma_22 * sigma_12 / dets) * phi_22)/ Wi) * dx
a2 = dolfinx.fem.form(lhs2)

# A2 = assemble_matrix(a2, bcs=[bc1])
# A2.assemble()
# A2_numpy = petsc_matrix_to_numpy(A2)
# b2 = create_vector(L)
# b2.assemble()
# set_bc(b2, bcs=[bc1])
# b2_numpy = petsc_vector_to_numpy(b2)

#lhs3 = ((sigma_21-sigma_init[1,0])*phi_11 + (sigma_22 - sigma_init[1,1])*phi_21)*dx
lhs3 = -(-((u[0]*sigma_21.dx(0) + u[1]*sigma_21.dx(1))*phi_11+(u[0]*sigma_22.dx(0)+u[1]*sigma_22.dx(1))*phi_21))*dx
lhs3 -= (u[1].dx(0)*(sigma_11*phi_11 + sigma_12*phi_21) + u[1].dx(1)*(sigma_21*phi_22 + sigma_22*phi_21))*dx
lhs3 -= (sigma_21*(u[0].dx(0)*phi_11+u[1].dx(1)*phi_21)+ sigma_22*(u[1].dx(0)*phi_11+u[1].dx(1)*phi_21))*dx
lhs3 += (((sigma_21 * sigma_11 / dets + sigma_11 * (ufl.inv(1 - trs) - sigma_11 / dets)) * phi_11 +
         (sigma_21 * sigma_12 / dets + sigma_22 * (ufl.inv(1 - trs) - sigma_22 / dets)) * phi_21)/ Wi) * dx
a3 = dolfinx.fem.form(lhs3)

# A3 = assemble_matrix(a1, bcs=[bc1])
# A3.assemble()
# A3_numpy = petsc_matrix_to_numpy(A3)
# b3 = create_vector(L)
# b3.assemble()
# set_bc(b3, bcs=[bc1])
# b3_numpy = petsc_vector_to_numpy(b3)

#lhs4 = ((sigma_21 - sigma_init[1, 0]) * phi_12 + (sigma_22 - sigma_init[1, 1]) * phi_22) * dx
lhs4 = -(-((u[0] * sigma_21.dx(0) + u[1] * sigma_21.dx(1)) * phi_12 + (
        u[0] * sigma_22.dx(0) + u[1] * sigma_22.dx(1)) * phi_22)) * dx
lhs4 -= (u[1].dx(0) * (sigma_11 * phi_12 + sigma_12 * phi_22) + u[1].dx(1) * (
        sigma_21 * phi_21 + sigma_22 * phi_22)) * dx
lhs4 -= (sigma_21 * (u[0].dx(0) * phi_12 + u[1].dx(1) * phi_22) + sigma_22 * (
        u[1].dx(0) * phi_12 + u[1].dx(1) * phi_22)) * dx
lhs4 += (((sigma_21 * sigma_11 / dets + sigma_21 * (ufl.inv(1 - trs) - sigma_11 / dets)) * phi_12 +
          (sigma_21 * sigma_12 / dets + sigma_22 * (ufl.inv(1 - trs) - sigma_22 / dets)) * phi_22) / Wi) * dx

a4 = dolfinx.fem.form(lhs4)

# A4 = assemble_matrix(a2, bcs=[bc1])
# A4.assemble()
# A4_numpy = petsc_matrix_to_numpy(A4)
# b4 = create_vector(L)
# b4.assemble()
# set_bc(b4, bcs=[bc1])
# b4_numpy = petsc_vector_to_numpy(b4)

# assemble matrix A and vector b
# A1_row, A1_col = A1_numpy.shape[0], A1_numpy.shape[1]
# A2_row, A2_col = A2_numpy.shape[0], A2_numpy.shape[1]
# A3_row, A3_col = A3_numpy.shape[0], A3_numpy.shape[1]
# A4_row, A4_col = A4_numpy.shape[0], A4_numpy.shape[1]
# A = np.block([[A1_numpy, np.zeros((A1_row, A2_col)), np.zeros((A1_row, A3_col)), np.zeros((A1_row, A4_col))],
#               [np.zeros((A2_row, A1_col)), A2_numpy, np.zeros((A2_row, A3_col)), np.zeros((A2_row, A4_col))],
#               [np.zeros((A3_row, A1_col)), np.zeros((A3_row, A2_col)), A3_numpy, np.zeros((A3_row, A4_col))],
#               [np.zeros((A4_row, A1_col)), np.zeros((A4_row, A2_col)), np.zeros((A4_row, A3_col)), A4_numpy]])
# b = np.concatenate([b1_numpy, b2_numpy, b3_numpy, b4_numpy], axis=0)

# solving
#sigma = scipy.linalg.solve(A, b)

list_problem = dolfinx.fem.form([lhs1, lhs2, lhs3, lhs4])
from IPython import embed
embed()
problem = dolfinx.fem.petsc.NonlinearProblem(list_problem, sigma_11, bcs=[bc1])

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
n, converged = solver.solve(sigma_11)
assert (converged)
print(f"Number of interations: {n:d}")

from IPython import embed
embed()
# plotting the mesh #conda activate fenicsx-env
plot = False
if plot:
    import pyvista

    print(pyvista.global_theme.jupyter_backend)
    # pyvista.start_xvfb()
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    grid.point_data["sigma11"] = sigma11_h.x.array.real
    # sigma11_h.set_active_scalars("sigma11")

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        figure = plotter.screenshot("fundamentals_mesh.png")
