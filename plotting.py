import sys
import dolfinx
import pyvista
import numpy as np
from basix.ufl import element
from dolfinx.fem import functionspace
import matplotlib.pyplot as plt
import time
# ---------------------------------------------------------------------------------------------------------------------
# pipeline to create videos for displaying (partial) results
NUMBER_X: int = 1
NUMBER_Y: int = 1

CANVAS_WIDTH: int = 10
CANVAS_HEIGHT: int = 10

import mesh_init

gdim = 2

mesh, ft, inlet_marker, wall_marker, outlet_marker, obstacle_marker = mesh_init.create_mesh(gdim)
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
V = functionspace(mesh, v_cg2)

s_el_dim = element("Lagrange", mesh.topology.cell_name(), 1, shape=(2, 2))
S = dolfinx.fem.functionspace(mesh, s_el_dim)


def plotting_gif(list, fs, path, var_name):
    plotter = pyvista.Plotter()
    plotter.open_movie(path)
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


def plotting_2d_gif(list, fs, path, var_name):
    plotter = pyvista.Plotter()
    plotter.open_movie(path)
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(fs)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data[var_name] = list[0]
    grid.set_active_scalars(var_name)
    plotter.add_mesh(grid, show_edges=True, clim=[np.min(list), np.max(list)])
    plotter.view_xy()

    for sol in list:
        grid.point_data[var_name][:] = sol
        plotter.write_frame()
    plotter.close()


def streamlines_animation(data, fluid_mesh):
    line_streamlines = fluid_mesh.streamlines(
        pointa=(0, -5, 0),
        pointb=(0, 5, 0),
        n_points=25,
        max_time=100.0,
        compute_vorticity=False,  # vorticity already exists in dataset
    )

    clim = [0, 20]
    camera_position = [(7, 0, 20.0), (7, 0.0, 0.0), (0.0, 1.0, 0.0)]

    p = pyvista.Plotter()
    for i in range(1, len(mesh)):
        p.add_mesh(mesh[i], color='k')
    p.add_mesh(line_streamlines.tube(radius=0.05), scalars="vorticity_mag", clim=clim)
    p.view_xy()
    p.show(cpos=camera_position)


plot = 1
i = 113
if plot == 1:
    with open(f"results/arrays/experiments/{i}/u1.npy", "rb") as f:
        plotting_gif(np.load(f), V, f"plots/experiments/{i}/u1_new.mp4", "u1")
elif plot == 2:
    with open(f"results/arrays/experiments/{i}/u2.npy", "rb") as f:
        plotting_gif(np.load(f), V, f"plots/experiments/{i}/u2_new.mp4", "u2")
elif plot == 3:
    with open(f"results/arrays/experiments/{i}/sigma11.npy", "rb") as f:
        plotting_gif(np.load(f), S, f"plots/experiments/{i}/sigma_11_new.mp4", "sigma11")
elif plot == 4:
    with open(f"results/arrays/experiments/{i}/sigma12.npy", "rb") as f:
        plotting_gif(np.load(f), S, f"plots/experiments/{i}/sigma_12_new.mp4", "sigma12")
elif plot == 5:
    with open(f"results/arrays/experiments/{i}/sigma21.npy", "rb") as f:
        plotting_gif(np.load(f), S, f"plots/experiments/{i}/sigma_21_new.mp4", "sigma21")
elif plot == 6:
    with open(f"results/arrays/experiments/{i}/sigma22.npy", "rb") as f:
        plotting_gif(np.load(f), S, f"plots/experiments/{i}/sigma_22_new.mp4", "sigma22")
elif plot == 7:
    with open("results/arrays/u1NS.npy", "rb") as f:
        plotting_gif(np.load(f), V, "plots/u1NS_new.mp4", "u1NS")
elif plot == 8:
    with open("results/arrays/u2NS.npy", "rb") as f:
        plotting_gif(np.load(f), V, "plots/u2NS_new.mp4", "u2NS")
elif plot == 9:
    with open(f"results/arrays/experiments/{i}/u1.npy", "rb") as f:
        with open(f"results/arrays/experiments/{i}/u1.npy", "rb") as g:
            #plotting_gif(np.load(f) ** 2 + np.load(g) ** 2, V, f"plots/experiments/{i}/u_magnitude.gif", "||u||")
            plotting_2d_gif(np.load(f) ** 2 + np.load(g) ** 2, V, f"plots/experiments/{i}/u_magnitude.mp4", "||u||")
