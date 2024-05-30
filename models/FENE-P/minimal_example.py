import dolfinx
import scipy
import ufl
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, ds, dx, inner, lhs, nabla_grad, rhs, sym)
import basix
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import petsc4py
import dolfinx.fem.petsc
from dolfinx.nls.petsc import NewtonSolver
import pyvista
from IPython import embed

# Constants
Lx = 1.0
Ly = 1.0
Nx = 50
Ny = 50
b = 10
Wi = 2
Re = 1

# mesh
comm_t = MPI.COMM_WORLD
domain = dolfinx.mesh.create_rectangle(comm_t, [np.array([0., 0.]), np.array([Lx, Ly])],
                                       [Nx, Ny], dolfinx.mesh.CellType.triangle)

# function space
s_el_dim = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(2, 2))
V = dolfinx.fem.functionspace(domain, s_el_dim)

# boundary conditions
x = ufl.SpatialCoordinate(domain)

s_D = dolfinx.fem.Function(V)
s_D.vector.set(0.0)
fdim = domain.topology.dim - 1
boundary = lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[1], 1.0)),
                                   np.logical_or(np.isclose(x[0], 1.0), np.isclose(x[1], 0.0)))
boundary_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, boundary)
bc = dolfinx.fem.dirichletbc(s_D, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))

# Test function definition
sigma = dolfinx.fem.Function(V)
phi = ufl.TestFunction(V)

# vector field definition u(x,y)=(-y,x)
vector_field1 = x[1]-x[1]**2
vector_field2 = 0#x[0]


# Problem definition


def problem_definition(sigma, sigma_n, dt):
    def A(sigma, b):
        return 1 / (1 - ufl.tr(sigma) / b)

    t1 = (Re*ufl.tr((sigma - sigma_n) / dt * ufl.transpose(phi))) * dx
    # t2 = ufl.tr(ufl.nabla_div(ufl.as_vector([vector_field1, vector_field2]))*sigma * ufl.transpose(phi)) * dx
    nabla_term = vector_field1 * sigma.dx(0) + vector_field2 * sigma.dx(1)
    t2 = (Re*ufl.tr(nabla_term * ufl.transpose(phi))) * dx
    t3 = (ufl.tr(ufl.grad(ufl.as_vector([vector_field1, vector_field2])) * sigma * ufl.transpose(phi))) * dx
    t4 = (ufl.tr(
        sigma * ufl.transpose(ufl.grad(ufl.as_vector([vector_field1, vector_field2]))) * ufl.transpose(phi))) * dx
    t5 = (A(sigma, b) / Wi * ufl.tr(sigma * ufl.transpose(phi)) - ufl.tr(phi)) * dx
    F_new = t1 + t2 - t3 - t4 + t5
    return F_new


steps = 100
T = 1
t = t0 = 0
num_it = int((T - t0) / steps)
dt = T / steps

# solver.convergence_criterion = "incremental"
# solver.rtol = 1e-6
# ksp = solver.krylov_solver
# opts = PETSc.Options()
# option_prefix = ksp.getOptionsPrefix()
# opts[f"{option_prefix}ksp_type"] = "cg"
# opts[f"{option_prefix}pc_type"] = "gamg"
# opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# ksp.setFromOptions()
#
# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

sigma_n = dolfinx.fem.Function(V)
sigma_n.vector.set(0.0)

sigma_11_solution_data = []
sigma_12_solution_data = []
sigma_21_solution_data = []
sigma_22_solution_data = []
sigma_n_array = sigma_n.x.array
n = int(sigma_n_array.shape[0] / 4)
sigma_11_solution_data.append([sigma_n_array[4 * k] for k in range(n)])
sigma_12_solution_data.append([sigma_n_array[4 * k + 1] for k in range(n)])
sigma_21_solution_data.append([sigma_n_array[4 * k + 2] for k in range(n)])
sigma_22_solution_data.append([sigma_n_array[4 * k + 3] for k in range(n)])
time_values_data = np.zeros(steps + 1)

# plotter = pyvista.Plotter()
# plotter.open_gif("deformation.gif", fps=3)
# topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
# x = np.concatenate([x[:,0:2],sigma_11.reshape(-1,1)],axis=1)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# pts = grid.points.copy()

for k in range(steps):
    t += dt
    # navier stokes u= ...
    F = problem_definition(sigma, sigma_n, dt)
    problem = dolfinx.fem.petsc.NonlinearProblem(F, sigma,
                                                 bcs=[bc])

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.report = True
    n, converged = solver.solve(sigma)
    assert (converged)
    print(f"Number of interations: {n:d}")

    sigma_array = sigma.x.array
    n = int(sigma_array.shape[0] / 4)
    sigma_11_solution_data.append([sigma_array[4 * k] for k in range(n)])
    sigma_12_solution_data.append([sigma_array[4 * k + 1] for k in range(n)])
    sigma_21_solution_data.append([sigma_array[4 * k + 2] for k in range(n)])
    sigma_22_solution_data.append([sigma_array[4 * k + 3] for k in range(n)])
    time_values_data[k + 1] = t

    sigma.x.scatter_forward()
    sigma_n.x.array[:] = sigma.x.array
    t_n = t
    # embed()
    # grid.point_data["sigma"] = sigma_11_solution_data[k]

    # plotter.write_frame()


# plotter.close()


# sigma_array = sigma.x.array
# n = int(sigma_array.shape[0] / 4)
# sigma_11 = np.array([sigma_array[4 * k] for k in range(n)])
# sigma_12 = np.array([sigma_array[4 * k + 1] for k in range(n)])
# sigma_21 = np.array([sigma_array[4 * k + 2] for k in range(n)])
# sigma_22 = np.array([sigma_array[4 * k + 3] for k in range(n)])


def plotting(sigma_sol):
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    # x = np.concatenate([x[:,0:2],sigma_11.reshape(-1,1)],axis=1)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["sigma"] = sigma_sol
    grid.set_active_scalars("sigma")
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(grid, show_edges=True)
    u_plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        u_plotter.show()


def plotting_gif(sigma_list):
    plotter = pyvista.Plotter()
    plotter.open_gif("sigma_11_1_1.gif", fps=30)
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    # x = np.concatenate([x[:,0:2],sigma_11.reshape(-1,1)],axis=1)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["sigma"] = sigma_list[0]
    warped = grid.warp_by_scalar("sigma", factor=0.5)
    plotter.add_mesh(warped, show_edges=True, clim=[np.min(sigma_list), np.max(sigma_list)])
    # grid.set_active_scalars("sigma")
    for sigma_sol in sigma_list:
        new_warped = grid.warp_by_scalar("sigma", factor=0.1)
        warped.points[:, :] = new_warped.points
        warped.point_data["sigma"][:] = sigma_sol
        plotter.write_frame()
    # plotter.view_xy()
    plotter.close()


# plotting
# plotting(sigma_11_solution_data[-1])
plotting_gif(sigma_11_solution_data)
# embed()
# plotting(sigma_12)
# plotting(sigma_21)
# plotting(sigma_22_solution_data[1])
