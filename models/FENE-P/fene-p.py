import dolfinx
import ufl
import numpy as np
from mpi4py import MPI
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

        self.init_parameters_and_functions(parameters)

    def init_parameters_and_functions(self, parameters: fene_p_parameters):
        self.dim_u = parameters.Lx
        self.dim_sigma = parameters.Ly
        self.Re = parameters.L0 * parameters.U0 / (parameters.viscosity_s + parameters.viscosity_p)
        self.Wi = parameters.lambd * parameters.U0 / parameters.L0
        self.regularized_model = parameters.regularization

        self.colon_operator = lambda X, Y: ufl.tr(ufl.transpose(X) * Y)
        if self.regularized_model:
            self.beta_d = lambda s: ufl.max_value(s, parameters.delta)
            self.A_d = lambda phi, n: ((1 / parameters.delta + 1 / (1 - n / parameters.b)) * ufl.Identity(self.dim_sigma)
                                       - (1 / parameters.delta + 1 / phi)) if ((1 - n / parameters.b) <
                                                                                 parameters.delta) \
                else (1 / (1 - n / parameters.b)) * ufl.Identity(self.dim_sigma) - 1 / phi
        else:
            self.beta_d = None
            self.A_d = ...
        self.f_n = ...
        self.phi_lim = ...
        self.jump_value = ...

    def mesh(self, parameters: fene_p_parameters):
        comm = MPI.COMM_WORLD
        # Create mesh
        return dolfinx.mesh.create_rectangle(comm, [[0., 0.], [parameters.Lx, parameters.Ly]], [self.dim_u, self.dim_sigma])

    def init_model(self, mesh):
        # function spaces
        V_h_0 = dolfinx.fem.VectorFunctionSpace(mesh, dolfinx.fem.function.ElementMetaData("CG", 1))
        S_h_0 = ...

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

        # Define boundary conditions
        bc = dolfinx.fem.dirichletbc(u_init, dolfinx.fem.locate_dofs_geometrical(V_h_0, lambda x: np.linalg.norm(x) < 1.0e-10))

        P1 = ...

        P2 = ...

        P = [P1,P2] #TODO: see if this works

        self.problem = dolfinx.fem.LinearProblem(P1, P2, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})


    def solve(self, problem,parameters: fene_p_parameters, mesh,P):
        while self.t < parameters.T:
            self.t += parameters.dt
            problem.solve()
            # Output
            if self.t % 0.1 == 0:
                P.rename("velocity", "velocity")
                dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"u_{self.t:.1f}.xdmf", "w").write_mesh(mesh)
                dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"u_{self.t:.1f}.xdmf", "w").write_function(P)

    # Plot solution
    def plot(self, mesh, P):
        import dolfinx.plot

        dolfinx.plot.create_vtk_topology(mesh, 2)
        dolfinx.plot.plot(P)

        # Hold plot
        import matplotlib.pyplot as plt

        plt.show()
