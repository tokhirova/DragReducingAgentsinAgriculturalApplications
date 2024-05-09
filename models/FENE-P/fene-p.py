import dolfinx
import ufl
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)
import basix
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import fene_p_parameters


class Fene_P:
    def __init__(self, parameters: fene_p_parameters):
        # constants
        self.t = 0
        self.dim_u = None
        self.dim_sigma = None
        self.Re = None
        self.Wi = None
        self.regularized_model = False

        # functions
        self.colon_operator = None
        self.beta_d = None
        self.A_d = None
        self.f_n = None
        self.phi_lim = None
        self.jump_value = None
        self.problem = None
        self.eps = None

    def init_parameters_and_functions(self, parameters: fene_p_parameters):
        self.dim_u = parameters.Nx
        self.dim_sigma = parameters.Ny
        self.Re = parameters.L0 * parameters.U0 / (parameters.viscosity_s + parameters.viscosity_p)
        self.Wi = parameters.lambd * parameters.U0 / parameters.L0
        self.eps = parameters.eps

        self.colon_operator = lambda X, Y: ufl.tr(ufl.transpose(X) * Y)

        self.A = lambda phi: ufl.inv((1 / (1 - ufl.tr(phi) / parameters.b))) * ufl.Identity(self.dim_sigma) - ufl.inv(
            phi)
        self.f = [0,0]

    def mesh(self, parameters: fene_p_parameters):
        comm = MPI.COMM_WORLD
        # Create mesh
        return dolfinx.mesh.create_rectangle(comm, [np.array([0., 0.]), np.array([parameters.Lx, parameters.Ly])],
                                             [self.dim_u, self.dim_sigma], dolfinx.mesh.CellType.triangle)

    def init_model(self, mesh, parameters: fene_p_parameters):
        # function spaces
        v_el = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
        s_el = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim, mesh.geometry.dim))
        V_h_0 = dolfinx.fem.functionspace(mesh, v_el)  # dolfinx.fem.function.ElementMetaData("CG", 1))
        S_h_0 = dolfinx.fem.functionspace(mesh, s_el)  # dolfinx.fem.function.ElementMetaData("CG", 0))

        # trial and test functions
        u = ufl.TrialFunction(V_h_0)
        sigma = ufl.TrialFunction(S_h_0)
        v = ufl.TestFunction(V_h_0)
        phi = ufl.TestFunction(S_h_0)

        # init conditions
        def y0_init(x):
            values = np.zeros((1, x.shape[1]))
            values[0] = 0.0
            return values

        def y1_init(x):
            values = np.zeros((1, x.shape[1]))
            values[0] = 0.0
            return values

        # Define initial condition
        u_init = dolfinx.fem.Function(V_h_0)
        u_init.sub(0).interpolate(y0_init)
        u_init.sub(1).interpolate(y1_init)

        sigma_init = dolfinx.fem.Function(S_h_0)
        sigma_init.sub(0).interpolate(y0_init)
        sigma_init.sub(1).interpolate(y1_init)

        f_n = dolfinx.fem.Function(V_h_0)
        f_n.sub(0).interpolate(y0_init)
        f_n.sub(1).interpolate(y1_init)

        dt = parameters.T / parameters.num_time_steps

        # Define boundary conditions
        def outflow(x):
            return np.isclose(x[0], 0)
        bc = dolfinx.fem.dirichletbc(u_init,
                                     dolfinx.fem.locate_dofs_geometrical(V_h_0, outflow))

        P1 = self.Re * dot((u - u_init) / dt, v) * dx
        P1 -= self.Re * self.colon_operator(ufl.dot(u, u), ufl.grad(v)) * dx
        P1 += (1 - parameters.eps) * self.colon_operator(ufl.grad(u), ufl.grad(v)) * dx
        P1 += parameters.eps / self.Wi * ((self.colon_operator(
            (1. - ufl.tr(sigma) / parameters.b) * sigma, ufl.grad(v))) * dx)
        P1 -= dot(f_n, v) * dx
        from IPython import embed; embed()

        P2 = dot((sigma - sigma_init) / dt, phi) * dx
        P2 += dot(dot(u, ufl.grad(u)) * sigma, phi) * dx
        P2 -= ufl.grad(u) * sigma * phi * dx
        P2 -= sigma * ufl.transpose(ufl.grad(u)) * phi * dx
        P2 += self.A(sigma) * sigma / self.Wi * phi * dx

        return V_h_0, S_h_0, dolfinx.fem.form(P1), dolfinx.fem.form(P2), bc, u, sigma, u_init

    def solve(self, parameters: fene_p_parameters, mesh, V, S, P1, P2, bc, u, sigma, u_n):
        # problem = dolfinx.fem.LinearProblem(P1, P2, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        t = t_n = 0
        dt = parameters.T / parameters.num_time_steps

        solution_data = [np.zeros((parameters.num_time_steps + 1), V.dim()),
                         np.zeros((parameters.num_time_steps + 1), S.dim())]
        time_values_data = np.zeros(parameters.num_time_steps + 1)

        for time_step in range(parameters.num_time_steps):
            t += dt
            # dolfinx.fem.petsc.assemble_vector(P1, P2)
            dolfinx.solve([P1, P2] == [0, 0], [u, sigma], bc)
            solution_data[0][t + 1, :] = np.array(u.vector())
            solution_data[1][t + 1, :] = np.array(sigma.vector())
            time_values_data[t + 1] = t

            u_n.assign(u)
            t_n = t

        return solution_data, time_values_data

    def run(self,parameters):
        self.init_parameters_and_functions(parameters)
        mesh = self.mesh(parameters)
        V, S, P1, P2, bc, u, sigma, u_init = self.init_model(mesh, parameters)
        return self.solve(parameters, mesh, V, S, P1, P2, bc, u, sigma, u_init)

    # Plot solution
    def plot(self, mesh, P):
        import dolfinx.plot

        dolfinx.plot.create_vtk_topology(mesh, 2)
        dolfinx.plot.plot(P)

        # Hold plot
        import matplotlib.pyplot as plt

        plt.show()


params = fene_p_parameters.ModelParameters()
model = Fene_P(params)
sol, t = model.run(params)

