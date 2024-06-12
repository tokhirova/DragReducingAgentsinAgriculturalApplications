import sys
import dolfinx
import pyvista
import numpy as np
from basix.ufl import element
from dolfinx.fem import functionspace

import mesh_init

gdim = 2

mesh, ft, inlet_marker, wall_marker, outlet_marker, obstacle_marker = mesh_init.create_mesh(gdim)
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
V = functionspace(mesh, v_cg2)

s_el_dim = element("Lagrange", mesh.topology.cell_name(), 1, shape=(2, 2))
S = dolfinx.fem.functionspace(mesh, s_el_dim)


def plotting_gif(list, fs, path, var_name):
    plotter = pyvista.Plotter()
    plotter.open_gif(path, fps=30)
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(fs)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data[var_name] = list[0]
    warped = grid.warp_by_scalar(var_name, factor=0.5)
    plotter.add_mesh(warped, show_edges=True, clim=[np.min(list), np.max(list)])
    for sol in list:
        new_warped = grid.warp_by_scalar(var_name, factor=0.1)
        warped.points[:, :] = new_warped.points
        warped.point_data[var_name][:] = sol
        plotter.write_frame()
    plotter.close()


plot = 3
i = 2
if plot == 1:
    with open(f"results/arrays/experiments/{i}/u1.npy", "rb") as f:
        plotting_gif(np.load(f), V, f"plots/experiments/u1_new.gif", "u1")
elif plot == 2:
    with open(f"results/arrays/experiments/{i}/u2.npy", "rb") as f:
        plotting_gif(np.load(f), V, f"plots/experiments/u2_new.gif", "u2")
elif plot == 3:
    with open(f"results/arrays/experiments/{i}/sigma11.npy", "rb") as f:
        plotting_gif(np.load(f), S, f"plots/experiments/sigma_11_new.gif", "sigma11")
elif plot == 4:
    with open(f"results/arrays/experiments/{i}/sigma12.npy", "rb") as f:
        plotting_gif(np.load(f), S, f"plots/experiments/sigma_12_new.gif", "sigma12")
elif plot == 5:
    with open(f"results/arrays/experiments/{i}/sigma21.npy", "rb") as f:
        plotting_gif(np.load(f), S, f"plots/experiments/sigma_21_new.gif", "sigma21")
elif plot == 6:
    with open(f"results/arrays/experiments/{i}/sigma22.npy", "rb") as f:
        plotting_gif(np.load(f), S, f"plots/experiments/sigma_22_new.gif", "sigma22")
elif plot == 7:
    with open("results/arrays/u1NS.npy", "rb") as f:
        plotting_gif(np.load(f), V, "plots/u1NS_new.gif", "u1NS")
elif plot == 8:
    with open("results/arrays/u2NS.npy", "rb") as f:
        plotting_gif(np.load(f), V, "plots/u2NS_new.gif", "u2NS")
