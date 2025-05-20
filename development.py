# Description: Develop algorithm to refine ICON triangle mesh
#
# Author: Christian R. Steger, May 2025

import math

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri
from time import perf_counter
import cartopy.crs as ccrs
import cartopy.feature as feature
import trimesh
from scipy.spatial import KDTree

from functions import refine_mesh_4c_trimesh
from functions import refine_mesh_4c_py, refine_mesh_4c_nb

style.use("classic")

# Paths
path_ige = "/store_new/mch/msopr/csteger/Data/Miscellaneous/" \
    + "ICON_grids_EXTPAR/"
path_plot = "/scratch/mch/csteger/HORAYZON_extpar/plots/"
# path_ige = "/Users/csteger/Dropbox/MeteoSwiss/Data/Miscellaneous/" \
#     + "ICON_grids_EXTPAR/"
# path_plots = "/Users/csteger/Desktop/"

###############################################################################
# Load ICON grid data
###############################################################################

# Select ICON resolution
icon_res = "2km"  # "2km", "1km", "500m"

# -----------------------------------------------------------------------------

# ICON grids
# icon_grids = {"2km": "MeteoSwiss/icon_grid_0002_R19B07_mch.nc",
#               "1km": "MeteoSwiss/icon_grid_0001_R19B08_mch.nc",
#               "500m": "MeteoSwiss/icon_grid_00005_R19B09_DOM02.nc"}
icon_grids = {"2km": "test/icon_grid_DOM01.nc",}

# Resolution of DEM
res_dem = (40075.0 * 1000.0 / 360.0) * (1.0 / 3600.0)  # ~max. resolution [m]
print(f"DEM resolution: {(res_dem):.1f} m")
cell_area_dem = res_dem ** 2  # [m2]

# Load ICON grid with specific resolution
ds = xr.open_dataset(path_ige + icon_grids[icon_res])
vlon = ds["vlon"].values
vlat = ds["vlat"].values
vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
# ordered counter-clockwise
edge_of_cell = ds["edge_of_cell"].values - 1  # (3, num_cell; int32)
# ordered counter-clockwise
edge_vertices = ds["edge_vertices"].values - 1  # (2, num_edge; int32)
k = int(ds.attrs["grid_level"])
n = int(ds.attrs["grid_root"])
ds.close()
res_icon = 5500.0 / (n * 2 ** k) * 1000  # [m]
print(f"ICON resolution: {res_icon:.1f} m")
cell_area_icon = res_icon ** 2  # [m2]

# Cartesian coordinates
vx = np.cos(vlat) * np.cos(vlon)
vy = np.cos(vlat) * np.sin(vlon)
vz = np.sin(vlat)
vector_v = np.ascontiguousarray(np.vstack([vx, vy, vz]).T)

###############################################################################
# Check if edges are also ordered counter-clockwise
###############################################################################

# for ind_cell in range(vertex_of_cell.shape[1]):
#     ind_vert = vertex_of_cell[:, ind_cell]
#     ind_vert_2d = edge_vertices[:, edge_of_cell[:, ind_cell]]
#     for i in range(3):
#         if (ind_vert_2d[0, i] != ind_vert[i]):
#             ind_vert_2d[0, i], ind_vert_2d[1, i] \
#                 = ind_vert_2d[1, i], ind_vert_2d[0, i]
#     flag_a = np.all(ind_vert_2d[0, :] == ind_vert)
#     flag_b = np.all(np.roll(ind_vert_2d[1, :], 1) == ind_vert)
#     if (not flag_a) or (not flag_b):
#         raise ValueError("Edges are not ordered counter-clockwise")

# -> all above ICON grid meshes checked -> ok!

###############################################################################
# Refine mesh by iteratively splitting triangles into 4 child triangles
###############################################################################

# Compute refinement level
# 4 ** k = cell_area_icon / cell_area_dem -> solve for k
k = math.log(cell_area_icon / cell_area_dem) / (2.0 * math.log(2.0))
k = round(k) # closest to DEM resolution
# k = math.ceil(k) # first higher resolution than DEM
print(f"Bisection steps (k): {k}")
res_icon_fine = math.sqrt(cell_area_icon / (4 ** k))
print(f"ICON resolution: {res_icon_fine:.1f} m")
num_tri_fine = vertex_of_cell.shape[1] * (4 ** k)
print(f"Number of resulting triangles: {num_tri_fine:,}".replace(",", "'"))

# -----------------------------------------------------------------------------
# Trimesh
# -----------------------------------------------------------------------------

# Refine triangle mesh
faces = np.ascontiguousarray(vertex_of_cell.T)
mesh = trimesh.Trimesh(vertices=vector_v, faces=faces)
t_beg = perf_counter()
mesh_fine = refine_mesh_4c_trimesh(mesh, level=2)
t_end = perf_counter()
print(f"Elapsed time: {t_end - t_beg:.2f} s")
print(mesh_fine.vertices.shape, mesh_fine.faces.shape) # (float64, int64)

# Plot
num_tri_show = 25_000  # 25_000, None
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                  vertex_of_cell.transpose())
plt.triplot(triangles, color="black", lw=1.0)
vlon_fine = np.arctan2(mesh_fine.vertices[:, 1], mesh_fine.vertices[:, 0])
vlat_fine = np.arcsin(mesh_fine.vertices[:, 2])
triangles_fine = tri.Triangulation(np.rad2deg(vlon_fine),
                                   np.rad2deg(vlat_fine),
                                   mesh_fine.faces[:num_tri_show, :])
plt.triplot(triangles_fine, color="red", lw=0.5)
ax.add_feature(feature.BORDERS.with_scale("10m"), # type: ignore
            linestyle="-", linewidth=0.6)
ax.add_feature(feature.COASTLINE.with_scale("10m"), # type: ignore
            linestyle="-", linewidth=0.6)
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, # type: ignore
                  color="black", alpha=0.5, linestyle=":", draw_labels=True)
gl.top_labels = False
gl.right_labels = False
plt.show()

# -----------------------------------------------------------------------------
# Python/Numba implementation
# -----------------------------------------------------------------------------

# Refine triangle mesh
vertices = vector_v.copy()
faces = np.ascontiguousarray(vertex_of_cell.T)
t_beg = perf_counter()
for _ in range(2):
    # vertices, faces = refine_mesh_4c_py(vertices, faces)
    vertices, faces = refine_mesh_4c_nb(vertices, faces.astype(np.uint32))
vlon_fine = np.arctan2(vertices[:, 1], vertices[:, 0])
vlat_fine = np.arcsin(vertices[:, 2])
t_end = perf_counter()
print(f"Elapsed time: {t_end - t_beg:.2f} s")
print(vertices.shape, faces.shape)

# Plot
num_tri_show = 25_000  # 25_000, None
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                  vertex_of_cell.transpose())
plt.triplot(triangles, color="black", lw=1.0)
triangles_fine = tri.Triangulation(np.rad2deg(vlon_fine),
                                   np.rad2deg(vlat_fine),
                                   faces[:num_tri_show, :])
plt.triplot(triangles_fine, color="red", lw=0.5)
ax.add_feature(feature.BORDERS.with_scale("10m"), # type: ignore
            linestyle="-", linewidth=0.6)
ax.add_feature(feature.COASTLINE.with_scale("10m"), # type: ignore
            linestyle="-", linewidth=0.6)
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, # type: ignore
                  color="black", alpha=0.5, linestyle=":", draw_labels=True)
gl.top_labels = False
gl.right_labels = False
plt.show()

###############################################################################
# Refine mesh by splitting into n ** 2 child triangles
###############################################################################

# Compute refinement level
# n ** 2 = cell_area_icon / cell_area_dem -> solve for n
n = math.sqrt(cell_area_icon / cell_area_dem)
n = round(n) # closest to DEM resolution
# n = math.ceil(n) # first higher resolution than DEM
print(f"Division steps (n): {n}")
res_icon_fine = math.sqrt(cell_area_icon / (n ** 2))
print(f"ICON resolution: {res_icon_fine:.1f} m")
num_tri_fine = vertex_of_cell.shape[1] * (n ** 2)
print(f"Number of resulting triangles: {num_tri_fine:,}".replace(",", "'"))

# -----------------------------------------------------------------------------
# Check approximation of spherical triangle by planar triangle
# -----------------------------------------------------------------------------

rad_earth = 6371 # [km]
dist_per_angle = (2.0 * rad_earth * np.pi) / 360.0  # [km / degree]
angle = 0.05 # [degree] (2km: ~ 0.02 deg)
arc_length = dist_per_angle * angle  # [km]
print(f"Arc length:   {arc_length:.8f} km")
chord_length = 2.0 * rad_earth * np.sin(np.deg2rad(angle) / 2.0)  # [km]
print(f"Chord length: {chord_length:.8f} km")
print(f"Difference:   {((arc_length - chord_length) * 1000.0):.5f} m") # [m]

# -----------------------------------------------------------------------------
# 2-dimensional
# -----------------------------------------------------------------------------

# A = np.array([0.0, 0.0, 0.0])
# B = np.array([1.0, 0.0, 0.0])
# C = np.array([0.5, 1.0, 0.0])

# n = 7
# plt.figure()
# plt.scatter(A[0], A[1], s=100, color="grey")
# plt.scatter(B[0], B[1], s=100, color="grey")
# plt.scatter(C[0], C[1], s=100, color="grey")
# # --------------- all vertices ------------------
# for i in range(n + 1):
#     for j in range(n + 1 - i):
#         k = n - i - j
#         P = (i * A + j * B + k * C) / n
#         plt.scatter(P[0], P[1], s=50, color="blue",
#                     alpha=0.5)
# # --------- only interior vertices ---------------
# for i in range(1, n):
#     for j in range(1, n - i):
#         k = n - i - j
#         P = (i * A + j * B + k * C) / n
#         plt.scatter(P[0], P[1], s=25, color="red")
# # ------------------------------------------------
# plt.show()

# -----------------------------------------------------------------------------
# 3-dimensional
# -----------------------------------------------------------------------------

# Settings
ind_cell = 13_343
n = 5

vertices = vector_v[vertex_of_cell[:, ind_cell]]
vlon = np.arctan2(vertices[:, 1], vertices[:, 0])
vlat = np.arcsin(vertices[:, 2])

plt.figure()
ax = plt.axes()
plt.scatter(np.rad2deg(vlon), np.rad2deg(vlat), s=100, color="grey")
# -----------------------------------------------------------------
# All vertices
# -----------------------------------------------------------------

num_vertex_interior = 0
for i in range(n - 1):
    num_vertex_interior += i
vertices_child = np.empty((n * 3 + num_vertex_interior, 3), dtype=np.float64)
index_2d = np.empty((n + 1, n + 1), dtype=np.int32)
index_2d.fill(-999)
vertex_0 = vertices[0, :]
vertex_1 = vertices[1, :]
vertex_2 = vertices[2, :]
ind_vertex = 0
for i in range(n + 1):
    for j in range(n + 1 - i):
        k = n - i - j
        vertex_new = (k * vertex_0 + i * vertex_1 + j * vertex_2) / n
        vertices_child[ind_vertex, :] = vertex_new
        index_2d[i, j] = ind_vertex
        ind_vertex += 1
vertices_child /= np.linalg.norm(vertices_child, axis=1)[:, np.newaxis]
# unit vectors
vlon_child = np.arctan2(vertices_child[:, 1], vertices_child[:, 0])
vlat_child = np.arcsin(vertices_child[:, 2])
plt.scatter(np.rad2deg(vlon_child), np.rad2deg(vlat_child),
            s=50, color="blue", alpha=0.5)
for i in range(vertices_child.shape[0]):
    plt.text(np.rad2deg(vlon_child[i]), np.rad2deg(vlat_child[i]), str(i),
                fontsize=12, color="black")

# Create child triangles
faces = np.empty((n ** 2, 3), dtype=np.int32)
ind_face = 0
for i in range(n):
    for j in range(n - i):
        ind_v0 = index_2d[i, j]
        ind_v1 = index_2d[i + 1, j]
        ind_v2 = index_2d[i, j + 1]
        faces[ind_face, :] = (ind_v0, ind_v1, ind_v2)
        ind_face += 1
        if i + j + 1 < n:
            inv_v3 = index_2d[i + 1, j + 1]
            faces[ind_face, :] = (ind_v1, inv_v3, ind_v2)
            ind_face += 1
# print(faces)

# Plot child triangles
triangles = tri.Triangulation(np.rad2deg(vlon_child), np.rad2deg(vlat_child),
                              faces)
plt.tripcolor(triangles, facecolors=[0.5] * faces.shape[0], cmap="binary",
              vmin=0.0, vmax=1.0, edgecolor="black", lw=1.0, alpha=0.5)

# -----------------------------------------------------------------
# Only interior vertices
# -----------------------------------------------------------------

# ind = 0
# for i in range(1, n):
#     for j in range(1, n - i):
#         k = n - i - j
#         vertex_new = (k * vertex_0 + i * vertex_1 + j * vertex_2) / n
#         vlon_new = np.arctan2(vertex_new[1], vertex_new[0])
#         vlat_new = np.arcsin(vertex_new[2])
#         plt.scatter(np.rad2deg(vlon_new), np.rad2deg(vlat_new),
#                     s=25, color="red")
#         plt.text(np.rad2deg(vlon_new), np.rad2deg(vlat_new), str(ind),
#                     fontsize=12, color="black")
#         ind += 1

# -----------------------------------------------------------------
ax.autoscale_view()
plt.show()

# -----------------------------------------------------------------------------
# Algorithm for grid refinement
# -----------------------------------------------------------------------------

# Settings
n = 4 # number of subdivisions (2: 2 ** 2 = 4)

vertices = vector_v.copy()

# Compute number of new vertices
num_vertex_in = vertices.shape[0]
num_vertex_edge = edge_vertices.shape[1] * (n - 1)
num_vertex_interior_pgc = 0  # number of interior vertices per grid cell
for i in range(n - 1):
    num_vertex_interior_pgc += i
num_vertex_interior = vertex_of_cell.shape[1] * num_vertex_interior_pgc
num_vertex_fine = num_vertex_in + num_vertex_edge + num_vertex_interior

# Mapping of triangle vertex indices
num_vert_per_tri = 3 + 3 * (n - 1) + num_vertex_interior_pgc
print(f"Number of vertices per triangle: {num_vert_per_tri}")
mapping = np.empty(num_vert_per_tri, dtype=np.int32)
ind = 0
mapping[ind] = 0
ind += 1
mapping[ind:(ind + (n - 1))] \
    = np.arange(3 + 2 * (n - 1), 3 + 2 * (n - 1) + (n - 1), dtype=np.int32)
ind += (n - 1)
mapping[ind] = 2
ind += 1
start = n * 3
for i in range(0, n - 1):
    mapping[ind] = 3 + i
    ind += 1
    mapping[ind:(ind + n - 2 - i)] \
        = np.arange(start, start + n - 2 - i, dtype=np.int32)
    start = start + n - 2 - i
    ind += (n - 2 - i)
    mapping[ind] = 3 + (n - 1) + i
    ind += 1
mapping[ind] = 1
ind += 1
if not np.all(np.diff(np.unique(mapping)) == 1) or (num_vert_per_tri != ind):
    raise ValueError("Error while computing the remapping array")

# Compute faces
index_2d = np.empty((n + 1, n + 1), dtype=np.int32)
index_2d.fill(-999)
ind_vertex = 0
for i in range(n + 1):
    for j in range(n + 1 - i):
        index_2d[i, j] = ind_vertex
        ind_vertex += 1
faces = np.empty((n ** 2, 3), dtype=np.int32)
ind_face = 0
for i in range(n):
    for j in range(n - i):
        ind_v0 = index_2d[i, j]
        ind_v1 = index_2d[i + 1, j]
        ind_v2 = index_2d[i, j + 1]
        faces[ind_face, :] = (ind_v0, ind_v1, ind_v2)
        ind_face += 1
        if i + j + 1 < n:
            inv_v3 = index_2d[i + 1, j + 1]
            faces[ind_face, :] = (ind_v1, inv_v3, ind_v2)
            ind_face += 1

# Allocate arrays for refined mesh
vertices_child = np.empty((num_vertex_fine, 3), dtype=np.float64)
vertices_child.fill(np.nan) # temporary
faces_child = np.empty((n ** 2 * vertex_of_cell.shape[1], 3), dtype=np.int32)

# Plot
# -------------------------- plot start --------------------------
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
vlon = np.arctan2(vertices_child[:, 1], vertices_child[:, 0])
vlat = np.arcsin(vertices_child[:, 2])
slice_v = slice(None, num_vertex_in)
triangles = tri.Triangulation(np.rad2deg(vlon[slice_v]),
                              np.rad2deg(vlat[slice_v]),
                              vertex_of_cell.transpose())
plt.triplot(triangles, color="black", lw=0.8)
# --------------------------- plot end ---------------------------
# ----------------------------------------------------------------
# Add vertices from base mesh
# ----------------------------------------------------------------
vertices_child[:num_vertex_in, :] = vertices
# ----------------------------------------------------------------
# Add vertices located on the edge of bash mesh (shared)
# ----------------------------------------------------------------
t = np.linspace(0.0, 1.0, num=(n + 1))[1:-1]
ind_vertex = num_vertex_in
for i in range(edge_vertices.shape[1]):  # loop through all edges
    vertex_0 = vertices[edge_vertices[0, i], :]
    vertex_1 = vertices[edge_vertices[1, i], :]
    for j in range(n - 1):
        vertices_child[ind_vertex, :] = vertex_0 + t[j] * (vertex_1 - vertex_0)
        ind_vertex += 1
# ----------------------------------------------------------------
# Add vertices located in the interior of base mesh triangles
# ----------------------------------------------------------------
for ind_cell in range(vertex_of_cell.shape[1]): # loop through all cells
    vertex_0 = vertices[vertex_of_cell[0, ind_cell]]
    vertex_1 = vertices[vertex_of_cell[1, ind_cell]]
    vertex_2 = vertices[vertex_of_cell[2, ind_cell]]
    for i in range(1, n):
        for j in range(1, n - i):
            k = n - i - j
            vertices_child[ind_vertex, :] \
                = (k * vertex_0 + i * vertex_1 + j * vertex_2) / n
            ind_vertex += 1
vertices_child /= np.linalg.norm(vertices_child, axis=1)[:, np.newaxis]
# unit vectors
# -------------------------- plot start --------------------------
vlon = np.arctan2(vertices_child[:, 1], vertices_child[:, 0])
vlat = np.arcsin(vertices_child[:, 2])
slice_v = slice(0, num_vertex_in)
plt.scatter(np.rad2deg(vlon[slice_v]), np.rad2deg(vlat[slice_v]),
            s=20, color="blue")
slice_v = slice(num_vertex_in, num_vertex_in + num_vertex_edge)
plt.scatter(np.rad2deg(vlon[slice_v]), np.rad2deg(vlat[slice_v]),
            s=20, color="red")
slice_v = slice(num_vertex_in + num_vertex_edge, ind_vertex)
plt.scatter(np.rad2deg(vlon[slice_v]), np.rad2deg(vlat[slice_v]),
            s=20, color="green", alpha=0.5)
# --------------------------- plot end ---------------------------
# ----------------------------------------------------------------
# Connect vertices into child triangle
# ----------------------------------------------------------------
indices = np.empty(num_vert_per_tri, dtype=np.int32)
ind_face = 0
for ind_cell in range(vertex_of_cell.shape[1]):
# for ind_cell in range(500):
    indices[:3] = vertex_of_cell[:, ind_cell]
    # counter-clockwise ordered
    ind_vertex = 0
    for ind in edge_of_cell[:, ind_cell]:
        indices_edge = np.arange(num_vertex_in + ind * (n - 1),
                                 num_vertex_in + (ind + 1) * (n - 1))
        # ordering (clockwise vs. counter-clockwise) not consistent
        if ind_vertex == 0: # order: 0 -> 1
            if vertex_of_cell[ind_vertex, ind_cell] \
                != edge_vertices[0, edge_of_cell[ind_vertex, ind_cell]]:
                indices_edge = indices_edge[::-1]
        else: # order: 2 -> 1, 0 -> 2
            if vertex_of_cell[ind_vertex, ind_cell] \
                == edge_vertices[0, edge_of_cell[ind_vertex, ind_cell]]:
                indices_edge = indices_edge[::-1]
        slice_v = slice(3 + ind_vertex * (n - 1),
                        3 + (ind_vertex + 1) * (n - 1))
        indices[slice_v] = indices_edge
        ind_vertex += 1
    indices[slice_v.stop:] \
        = np.arange(num_vertex_in + num_vertex_edge 
                    + ind_cell * num_vertex_interior_pgc,
                    num_vertex_in + num_vertex_edge 
                    + (ind_cell + 1) * num_vertex_interior_pgc)
    # ------------------------ plot start ------------------------
    # plt.scatter(np.rad2deg(vlon[indices]), np.rad2deg(vlat[indices]),
    #             s=50, facecolor="none", edgecolor="black", linewidth=2.0)
    # shift = 0.0003
    # for i, ind in enumerate(indices):
    #     plt.text(np.rad2deg(vlon[ind]) + shift,
    #              np.rad2deg(vlat[ind]) + shift,
    #              f"{i}", fontsize=12, color="black")
    # ------------------------- plot end -------------------------
    for i in range(n ** 2):
        faces_child[ind_face, :] = indices[mapping[faces[i, :]]]
        ind_face += 1
# -------------------------- plot start --------------------------
triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                              faces_child)
facecolors=[0.5] * faces_child.shape[0]
num_tri = faces_child.shape[0]
num_ls = 11
num_rep = int(np.ceil(num_tri / num_ls))
facecolors = np.tile(np.linspace(0.0, 1.0, num_ls), num_rep)[:num_tri]
plt.tripcolor(triangles, facecolors=facecolors, cmap="Spectral",
              vmin=0.0, vmax=1.0, edgecolor="black", lw=1.0, alpha=0.5)
centroids = vertices_child[faces_child].mean(axis=1)
centroids /= np.linalg.norm(centroids, axis=1)[:, np.newaxis] # unit vectors
clon = np.arctan2(centroids[:, 1], centroids[:, 0])
clat = np.arcsin(centroids[:, 2])
plt.scatter(np.rad2deg(clon), np.rad2deg(clat),
            s=20, color="black", marker="x")
ax.add_feature(feature.BORDERS.with_scale("10m"), # type: ignore
            linestyle="-", linewidth=0.6)
ax.add_feature(feature.COASTLINE.with_scale("10m"), # type: ignore
            linestyle="-", linewidth=0.6)
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, # type: ignore
                  color="black", alpha=0.5, linestyle=":", draw_labels=True)
gl.top_labels = False
gl.right_labels = False
plt.show()
# --------------------------- plot end ---------------------------

# Check that vertices are correctly connected into triangles
if ((faces_child.min() != 0)
    or (vertices_child.shape[0] != faces_child.max() + 1)):
    raise ValueError("Array 'faces_child' is erroneous")
# ----------------------- dot product check ----------------------
dot_prod = (vertices_child[faces_child] 
            * centroids[:, np.newaxis, :]).sum(axis=2)
alpha = np.rad2deg(np.arccos(dot_prod))
rad_earth = 6_378_000.0 # [m]
dist_per_deg = 2.0 * np.pi * rad_earth / 360.0  # [m / deg]
print(f"Minimal chord distance {(alpha.min() * dist_per_deg):.1f} m")
print(f"Maximal chord distance {(alpha.max() * dist_per_deg):.1f} m")
# ------------------- nearest neighbour check --------------------
tree = KDTree(vertices_child)
dist, ind = tree.query(centroids, k=3, workers=4)
if np.any(np.sort(faces_child, axis=1) != np.sort(ind, axis=1)):
    raise ValueError("Array 'faces_child' is erroneous")
# ----------------------------------------------------------------
