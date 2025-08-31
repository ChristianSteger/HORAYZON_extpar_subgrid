# Description: Visualise ICON mesh refinement (to explain method)
#
# Author: Christian R. Steger, August 2025

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import style, tri

from functions import refine_mesh_nc

style.use("classic")

# Paths
path_plot = "/scratch/mch/csteger/HORAYZON_extpar_subgrid/plots/"

###############################################################################
# Load ICON grid data and pre-process
###############################################################################

# vertices: (num_vertices, 3), float64
# vertex_of_cell (3, num_cells), int32, [0, vertices.shape[0] - 1]
# edge_vertices (2, num_edges), int32 [0, vertices.shape[0] - 1]
# edge_of_cell (3, num_cells), int32, [0, edge_vertices.shape[1] - 1 ]

# Define starting triangle on sphere
vlon = np.array([1.0, 5.0, 3.0], dtype=np.float64)
vlat = np.array([1.0, 1.0, 1.0 + np.sqrt(12)], dtype=np.float64)
vertex_of_cell = np.array([[0, 1, 2]], dtype=np.int32).reshape(3, 1)
edge_vertices = np.array([[0, 1, 2],
                          [1, 2, 0]], dtype=np.int32)
edge_of_cell = np.array([[0, 1, 2]], dtype=np.int32).reshape(3, 1)

# Cartesian coordinates (unit sphere)
vx = np.cos(np.deg2rad(vlat)) * np.cos(np.deg2rad(vlon))
vy = np.cos(np.deg2rad(vlat)) * np.sin(np.deg2rad(vlon))
vz = np.sin(np.deg2rad(vlat))
vector_v = np.ascontiguousarray(np.vstack([vx, vy, vz]).T)
n_sel = 10
vertices_child, faces_child = refine_mesh_nc(vector_v, vertex_of_cell,
                                             edge_of_cell, edge_vertices,
                                             n_sel)
vlon_child = np.rad2deg(np.arctan2(vertices_child[:, 1], vertices_child[:, 0]))
vlat_child = np.rad2deg(np.arcsin(vertices_child[:, 2]))

# Test plot
fig = plt.figure(figsize=(8.0, 6.5))
ax = plt.axes()
# -----------------------------------------------------------------------------
triangles = tri.Triangulation(vlon, vlat, vertex_of_cell.transpose())
plt.triplot(triangles, color="grey", lw=5.0, zorder=1)
plt.scatter(vlon, vlat, s=200, marker="o", color="grey", zorder=1)
# -----------------------------------------------------------------------------
patches = [Polygon(np.column_stack((vlon_child[faces_child][i, :],
                                    vlat_child[faces_child][i, :])))
                                    for i in range(faces_child.shape[0])]
collection = PatchCollection(patches, facecolor="skyblue", edgecolor="none",
                             alpha=0.5)
ax.add_collection(collection)
collection = PatchCollection(patches, facecolor="none", edgecolor="royalblue",
                             alpha=1.0)
ax.add_collection(collection)
plt.scatter(vlon_child, vlat_child, s=25, marker="o", color="black", zorder=2)
# -----------------------------------------------------------------------------
grid_color = "black"
grid_lines = np.arange(0.0, 6.0, 0.3)
plt.hlines(y=grid_lines, xmin=0.0, xmax=6.0, color=grid_color,
           linestyle="-", linewidth=1.0, zorder=0)
plt.vlines(x=grid_lines, ymin=0.0, ymax=6.0, color=grid_color,
           linestyle="-", linewidth=1.0, zorder=0)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_color(grid_color)
    spine.set_linewidth(1.0)
# -----------------------------------------------------------------------------
plt.axis((0.6, 5.4, 0.6, 4.8))
# plt.show()
fig.savefig(path_plot + "visualise_mesh_refinement.jpg", dpi=300)
plt.close()
