# Description: Old code that might still be useful at some point...
#
# Author: Christian R. Steger, May 2025

from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("classic")

###############################################################################
# Split triangles in subdomains
###############################################################################

# Splitting axis
num_split_x = 3
x_bound = np.linspace(vlon.min() - 0.1, vlon.max() + 0.1, num_split_x + 1)
num_split_y = 2
y_bound = np.linspace(vlat.min() - 0.1, vlat.max() + 0.1, num_split_y + 1)

# Centroids
clon = vlon[vertex_of_cell].mean(axis=0)
clat = vlat[vertex_of_cell].mean(axis=0)

# Split triangles mesh in subdomain to allow parallel processing
t_beg = perf_counter()
vlon_all = []
vlat_all = []
vertex_of_cell_all = []
for ind_y in range(num_split_y):
    for ind_x in range(num_split_x):

        # Mask for triangles and vertices
        mask_tri_sd = ((clon >= x_bound[ind_x]) & (clon <= x_bound[ind_x + 1]) 
                    & (clat >= y_bound[ind_y]) & (clat <= y_bound[ind_y + 1]))
        ind_vert_sd = np.unique(vertex_of_cell[:, mask_tri_sd].ravel())
        num_vert_sd = ind_vert_sd.size

        # Map indices of selected vertices to [0, num_vert]
        # mapping: ind_vert_sd -> [0, num_vert_sd - 1]
        mapping = np.empty(ind_vert_sd.max() + 1, dtype=np.int32)
        mapping[ind_vert_sd] = np.arange(num_vert_sd)
        vertex_of_cell_sd = vertex_of_cell[:, mask_tri_sd].copy()
        # -> slow part: could be parallelised...
        for i in range(vertex_of_cell_sd.shape[0]):
            for j in range(vertex_of_cell_sd.shape[1]):
                vertex_of_cell_sd[i, j] = mapping[vertex_of_cell_sd[i, j]]

        vlon_all.append(vlon[ind_vert_sd])
        vlat_all.append(vlat[ind_vert_sd])
        vertex_of_cell_all.append(vertex_of_cell_sd)
t_end = perf_counter()
print(f"Elapsed time: {t_end - t_beg:.6f} s")

# Check additional vertices
num_vert_red = sum([i.size for i in vlon_all]) - vlon.size
per_vert_red = (sum([i.size for i in vlon_all]) / vlon.size - 1.0) * 100.0
print(f"Number of redundant vertices: {num_vert_red} ({per_vert_red:.2f} %)")
if vertex_of_cell.shape[1] != sum([i.shape[1] for i in vertex_of_cell_all]):
    raise ValueError("Number of triangles is not equal")

# Plot
ind = 4
plt.figure()
plt.triplot(vlon_all[ind], vlat_all[ind], vertex_of_cell_all[ind].T,
            color="grey", lw=0.5)
plt.scatter(vlon_all[ind], vlat_all[ind], s=15, color="black")
plt.hlines(y_bound, x_bound[0], x_bound[-1], color="red", lw=1.5)
plt.vlines(x_bound, y_bound[0], y_bound[-1], color="red", lw=1.5)
plt.show()

# Ideas to remove redundant vertices in merged mesh (if necessary):
# val_uniq, val_count = np.unique(vertex_of_cell_all[ind].ravel(),
#                                 return_counts=True)
# mask_edge = (val_count != 6)
# -> in case detection of boundary vertices is too slow for increased 
#    resolution mesh -> label_tri parent vertices and inherit to child vertices
#   -> small issue: some edges are erroneously detected as boundary vertices

# Merge parts of triangle mesh
vlon_merged = np.hstack(vlon_all)
vlat_merged = np.hstack(vlat_all)
num_tri = sum([i.shape[1] for i in vertex_of_cell_all])
vertex_of_cell_merged = np.empty((3, num_tri), dtype=np.int32)
vertex_of_cell_merged.fill(-1)
ind_beg = 0
add = np.append(0, np.cumsum([i.size for i in vlon_all]))
for ind_i, i in enumerate(vertex_of_cell_all):  
    ind_end = i.shape[1]
    vertex_of_cell_merged[:, ind_beg:(ind_beg + ind_end)] = (i + add[ind_i])
    ind_beg = (ind_beg + ind_end)

# Plot
plt.figure()
plt.triplot(vlon_merged, vlat_merged, vertex_of_cell_merged.T,
            color="grey", lw=0.5)
plt.scatter(vlon_merged, vlat_merged, s=15, color="black")
plt.scatter(clon, clat, s=10, color="blue")
plt.hlines(y_bound, x_bound[0], x_bound[-1], color="red", lw=1.5)
plt.vlines(x_bound, y_bound[0], y_bound[-1], color="red", lw=1.5)
plt.show()
