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

# conda activate fenicsx-env


def function_space(domain):
    # function space
    s_el_dim = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(2, 2))
    V = dolfinx.fem.functionspace(domain, s_el_dim)

    # Test function definition
    sigma = dolfinx.fem.Function(V)
    phi = ufl.TestFunction(V)
    return V, sigma, phi


def boundary_conditions(domain, V, x):
    # boundary conditions
    s_D = dolfinx.fem.Function(V)
    s_D.vector.set(0.0)
    fdim = domain.topology.dim - 1
    boundary = lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[1], 1.0)),
                                       np.logical_or(np.isclose(x[0], 1.0), np.isclose(x[1], 0.0)))
    boundary_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, boundary)
    bc = dolfinx.fem.dirichletbc(s_D, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))
    return bc


def vector_field(x,mesh):
    # vector field definition u(x,y)=(y-y²,0)
    v_cg2 = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
    V = dolfinx.fem.functionspace(mesh, v_cg2)
    f = dolfinx.fem.Function(V)
    f.vector.set(0.0)
    u1 = -x[1]#x[1] - x[1] ** 2
    u2 = x[0] # x[0]
    return [u1,u2]


def A(sigma, b):
    return 1 / (1 - ufl.tr(sigma) / b)


def problem_definition(sigma, sigma_n, dt, vector_field1, vector_field2, phi, b, Wi, alpha):
    # Problem definition
    t1 = (ufl.tr((sigma - sigma_n) / dt * ufl.transpose(phi))) * dx
    # t2 = ufl.tr(ufl.nabla_div(ufl.as_vector([vector_field1, vector_field2]))*sigma * ufl.transpose(phi)) * dx
    nabla_term = vector_field1 * sigma.dx(0) + vector_field2 * sigma.dx(1)
    t2 = (ufl.tr(nabla_term * ufl.transpose(phi))) * dx
    t3 = (ufl.tr(ufl.grad(ufl.as_vector([vector_field1, vector_field2])) * sigma * ufl.transpose(phi))) * dx
    t4 = (ufl.tr(
        sigma * ufl.transpose(ufl.grad(ufl.as_vector([vector_field1, vector_field2]))) * ufl.transpose(phi))) * dx
    t5 = (A(sigma, b) / Wi * ufl.tr(sigma * ufl.transpose(phi)) - ufl.tr(phi)) * dx
    triple_dot = ufl.inner(ufl.grad(sigma), ufl.grad(phi))
    extra_diffusion = (alpha * triple_dot) * dx
    F_new = t1 + t2 - t3 - t4 + t5 + extra_diffusion
    return F_new


def solution_initialization(steps, V):
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
    return sigma_n, sigma_11_solution_data, sigma_12_solution_data, sigma_21_solution_data, sigma_22_solution_data, time_values_data


def solve(sigma, sigma_n, dt, vector_field1, vector_field2, bc, phi,b, Wi, alpha):
    F = problem_definition(sigma, sigma_n, dt, vector_field1, vector_field2, phi,b, Wi, alpha)
    problem = dolfinx.fem.petsc.NonlinearProblem(F, sigma)#, bcs=[bc])
    try:
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.report = True
        n, converged = solver.solve(sigma)
        #assert (converged)
        return False
    except:
        return True
    #print(f"Number of iterations: {n:d}")


def save_solutions(sigma, sigma_11_solution_data, sigma_12_solution_data, sigma_21_solution_data,
                   sigma_22_solution_data, time_values_data, k, t):
    sigma_array = sigma.x.array
    n = int(sigma_array.shape[0] / 4)
    sigma_11_solution_data.append([sigma_array[4 * k] for k in range(n)])
    sigma_12_solution_data.append([sigma_array[4 * k + 1] for k in range(n)])
    sigma_21_solution_data.append([sigma_array[4 * k + 2] for k in range(n)])
    sigma_22_solution_data.append([sigma_array[4 * k + 3] for k in range(n)])
    time_values_data[k + 1] = t


def plotting(sigma_sol, V):
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


def plotting_gif(sigma_list, V):
    plotter = pyvista.Plotter()
    plotter.open_gif("sigma_11_with_vector_field.gif", fps=30)
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


def pipeline():
    Lx = 1.0
    Ly = 1.0
    Nx = 50
    Ny = 50
    b, Wi, alpha = 30, 50, 0.1
    # mesh
    comm_t = MPI.COMM_WORLD
    domain = dolfinx.mesh.create_rectangle(comm_t, [np.array([0., 0.]), np.array([Lx, Ly])],
                                           [Nx, Ny], dolfinx.mesh.CellType.triangle)

    V, sigma, phi = function_space(domain)
    x = ufl.SpatialCoordinate(domain)
    bc = boundary_conditions(domain, V, x)
    u = vector_field(x, domain)
    vector_field1 = u[0]
    vector_field2 = u[1]
    steps = 100
    T = 1
    t = t0 = 0
    num_it = int((T - t0) / steps)
    dt = T / steps

    sigma_n, sigma_11_solution_data, sigma_12_solution_data, sigma_21_solution_data, sigma_22_solution_data, time_values_data = solution_initialization(
        steps, V)

    for k in range(steps):
        t += dt
        solve(sigma, sigma_n, dt, vector_field1, vector_field2, bc, phi,b, Wi, alpha)

        save_solutions(sigma, sigma_11_solution_data, sigma_12_solution_data, sigma_21_solution_data,
                       sigma_22_solution_data, time_values_data, k, t)

        sigma.x.scatter_forward()
        sigma_n.x.array[:] = sigma.x.array
        t_n = t

    plotting_gif(sigma_11_solution_data, V)

# plotting
# plotting(sigma_11_solution_data[-1])
#pipeline()
#plotting_gif(sigma_11_solution_data,V)
# embed()
# plotting(sigma_12)
# plotting(sigma_21)
# plotting(sigma_22_solution_data[1])
