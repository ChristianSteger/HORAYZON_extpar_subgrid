# Description: Increase resolution of ICON triangle mesh by splitting one 
#              triangle in 4 child triangles
#
# Author: Christian R. Steger, December 2024

import math

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, rcParams, tri, colors
from matplotlib.patches import Polygon
from time import perf_counter
from numba import jit
from numba import types
import cartopy.crs as ccrs
import cartopy.feature as feature
import trimesh

from functions import refine_mesh_4c_trimesh
from functions import refine_mesh_4c_py, refine_mesh_4c_nb

# Paths
path_ige = "/store_new/mch/msopr/csteger/Data/Miscellaneous/" \
    + "ICON_grids_EXTPAR/"
path_plot = "/scratch/mch/csteger/HORAYZON_extpar/plots/"
# path_ige = "/Users/csteger/Dropbox/MeteoSwiss/Data/Miscellaneous/" \
#     + "ICON_grids_EXTPAR/"
# path_plots = "/Users/csteger/Desktop/"

###############################################################################
# Load ICON grid data and pre-process
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
clon = ds["clon"].values
clat = ds["clat"].values
vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
# ordered counter-clockwise
edge_of_cell = ds["edge_of_cell"].values - 1  # (3, num_cell; int32)
# ordered counter-clockwise
edge_vertices = ds["edge_vertices"].values - 1  # (2, num_edge; int32)
k = int(ds.attrs["grid_level"])
n = int(ds.attrs["grid_root"])
ds.close()
res_icon = 5500.0 / (n * 2 ** k) * 1000  # [metre]
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

###############################################################################
# Refine mesh by iteratively splitting triangles into 4 child triangles
###############################################################################

# Compute refinement level
# 4 ** k = cell_area_icon / cell_area_dem -> solve for k
k = math.log(cell_area_icon / cell_area_dem) / (2.0 * math.log(2.0))
k = round(k) # closest to DEM resolution
# k = math.ceil(k) # first higher resolution than DEM
print(f"Bisection steps (k): {k}")
res_icon_ref = math.sqrt(cell_area_icon / (4 ** k))
print(f"ICON resolution: {res_icon_ref:.1f} m")
num_tri_ref = vertex_of_cell.shape[1] * (4 ** k)
print(f"Number of resulting triangles: {num_tri_ref:,}".replace(",", "'"))

# -----------------------------------------------------------------------------
# Trimesh
# -----------------------------------------------------------------------------

# Refine triangle mesh
faces = np.ascontiguousarray(vertex_of_cell.T)
mesh = trimesh.Trimesh(vertices=vector_v, faces=faces)
t_beg = perf_counter()
mesh_ref = refine_mesh_4c_trimesh(mesh, level=2)
t_end = perf_counter()
print(f"Elapsed time: {t_end - t_beg:.2f} s")
print(mesh_ref.vertices.shape, mesh_ref.faces.shape)

# Plot
num_tri_show = 25_000  # 25_000, None
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                  vertex_of_cell.transpose())
plt.triplot(triangles, color="black", lw=1.0)
vlon_ref = np.arctan2(mesh_ref.vertices[:, 1], mesh_ref.vertices[:, 0])
vlat_ref = np.arcsin(mesh_ref.vertices[:, 2])
triangles_ref = tri.Triangulation(np.rad2deg(vlon_ref), np.rad2deg(vlat_ref),
                                  mesh_ref.faces[:num_tri_show, :])
plt.triplot(triangles_ref, color="red", lw=0.5)
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
    vertices, faces = refine_mesh_4c_nb(vertices, faces)
vlon_ref = np.arctan2(vertices[:, 1], vertices[:, 0])
vlat_ref = np.arcsin(vertices[:, 2])
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
triangles_ref = tri.Triangulation(np.rad2deg(vlon_ref), np.rad2deg(vlat_ref),
                                  faces[:num_tri_show, :])
plt.triplot(triangles_ref, color="red", lw=0.5)
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
res_icon_ref = math.sqrt(cell_area_icon / (n ** 2))
print(f"ICON resolution: {res_icon_ref:.1f} m")
num_tri_ref = vertex_of_cell.shape[1] * (n ** 2)
print(f"Number of resulting triangles: {num_tri_ref:,}".replace(",", "'"))

# -----------------------------------------------------------------------------
# Check approximation of spherical triangle by planar triangle
# -----------------------------------------------------------------------------

rad_earth = 6371 # [km]
dist_per_angle = (2.0 * rad_earth * np.pi) / 360.0  # [km / degree]
angle = 0.05 # [degree] (2km: ~ 0.02 deg)
arc_length = dist_per_angle * angle  # [km]
print(arc_length)
chord_length = 2.0 * rad_earth * np.sin(np.deg2rad(angle) / 2.0)  # [km]
print(chord_length)
print((arc_length - chord_length) * 1000.0) # [m]

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
n = 6 # number of subdivisions (2: 2 ** 2 = 4)

vertices = vector_v.copy()

# Compute number of new vertices
num_vertex_in = vertices.shape[0]
num_vertex_edge = edge_vertices.shape[1] * (n - 1)
num_vertex_interior_pgc = 0  # number of interior vertices per grid cell
for i in range(n - 1):
    num_vertex_interior_pgc += i
num_vertex_interior = vertex_of_cell.shape[1] * num_vertex_interior_pgc
num_vertex_ref = num_vertex_in + num_vertex_edge + num_vertex_interior

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
vertices_child = np.empty((num_vertex_ref, 3), dtype=np.float64)
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
vertices_child[:num_vertex_in, :] = vector_v
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
slice_tri = slice(0, ind_face, None)
triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat), ####################### weitermachen!!!!
                              faces_child[slice_tri, :])
num_tri = faces_child[slice_tri, :].shape[0]
cmap = "Spectral"
facecolors=[0.5] * num_tri
plt.tripcolor(triangles, facecolors=facecolors, cmap=cmap,
              vmin=0.0, vmax=1.0, edgecolor="black", lw=1.0, alpha=0.5)
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

# Check that triangles are correctly connected
if ((faces_child.min() != 0)
    or (vertices_child.shape[0] != faces_child.max() + 1)):
    raise ValueError("Array 'faces_child' is erroneous")
dot_prod = np.empty((faces_child.shape[0], 3), dtype=np.float64)
for i in range(faces_child.shape[0]):
    vertices_tri = vertices_child[faces_child[i, :], :]
    centroid = vertices_tri.mean(axis=0)
    centroid /= np.linalg.norm(centroid) # unit vector
    dot_prod[i, :] = np.dot(vertices_tri, centroid)
print(dot_prod.max() - dot_prod.min())

alpha = np.rad2deg(np.arccos(dot_prod))

alpha_min = np.rad2deg(np.arccos(dot_prod.max()))
print(alpha_min)
alpha_max = np.rad2deg(np.arccos(dot_prod.min()))
print(alpha_max)


########### old stuff below...

def normalize(v):
    return v / np.linalg.norm(v)

def subdivide_spherical_triangle(A, B, C, n):
    """Subdivide a spherical triangle into n^2 sub-triangles on the unit sphere."""
    A, B, C = map(normalize, (A, B, C))
    points = {}
    triangles = []

    # Generate barycentric points and project to sphere
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            P = (i * A + j * B + k * C) / n
            points[(i, j)] = normalize(P)

    # Connect points into triangles
    for i in range(n):
        for j in range(n - i):
            v0 = points[(i, j)]
            v1 = points[(i + 1, j)]
            v2 = points[(i, j + 1)]
            triangles.append((v0, v1, v2))

            if i + j + 1 < n:
                v3 = points[(i + 1, j + 1)]
                triangles.append((v1, v3, v2))

    return triangles  # List of 3-tuples of points on the sphere


# Define triangle on unit sphere (e.g., normalized lat/lon coordinates)
A = vector_v[vertex_of_cell[0, 0]]
B = vector_v[vertex_of_cell[1, 0]]
C = vector_v[vertex_of_cell[2, 0]]

# Subdivide into 5x5 grid (n=5)
triangles = subdivide_spherical_triangle(A, B, C, n=5)

print(f"Generated {len(triangles)} spherical sub-triangles.")


plt.plot()
for i in triangles:
    for j in i:
        x, y, z = j
        lon = np.arctan2(y, x)
        lat = np.arcsin(y)
        plt.scatter(np.rad2deg(lon), np.rad2deg(lat), s=10, color="green")
plt.show()









def vector_interp(u, v, num_points=10):
    """Spherical linear interpolation between unit vectors u and v."""
    angle = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))  # angle between u and v
    vectors = np.empty((num_points, 3), dtype=np.float64)
    for ind, t in enumerate(np.linspace(0, 1, num_points)):
        sin_angle = np.sin(angle)
        vectors[ind] = (np.sin((1 - t) * angle) / sin_angle) * u + \
                           (np.sin(t * angle) / sin_angle) * v
    return vectors


ind = 0
u = vector_v[edge_vertices[0, ind], :]
v = vector_v[edge_vertices[1, ind], :]

n_points = 10
vectors_new = vector_interp(u, v, num_points=10)

lat = np.arcsin(vectors_new[:, 2])
lon = np.arctan2(vectors_new[:, 1], vectors_new[:, 0])



# Test plot
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                  vertex_of_cell.transpose())
plt.triplot(triangles, color="black", lw=0.5)
plt.scatter(np.rad2deg(vlon), np.rad2deg(vlat), s=10, color="green")
# ----------------------------------------------------------------
# Split edge of triangle
# ----------------------------------------------------------------

# ----------------------------------------------------------------
ax.add_feature(feature.BORDERS.with_scale("10m"),
            linestyle="-", linewidth=0.6)
ax.add_feature(feature.COASTLINE.with_scale("10m"),
            linestyle="-", linewidth=0.6)
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color="black",
                alpha=0.5, linestyle=":", draw_labels=True)
gl.top_labels = False
gl.right_labels = False
plt.show()









# @njit
def subdivide_mesh(vector_v, vertex_of_cell):

    # midpoint_cache = Dict.empty(
    #     key_type=types.int64,
    #     value_type=types.int64
    # )
    midpoint_cache = {}

    num_tri = vertex_of_cell.shape[1]
    num_vert = vector_v.shape[0]
    vector_v_new = np.empty((num_vert + num_tri * 3, 3), dtype=np.float64)
    vector_v_new[:num_vert, :] = vector_v
    vertex_of_cell_new = np.empty((3, num_tri * 4), dtype=np.int32)
    vertex_of_cell_new[:, :num_tri] = vertex_of_cell
    num_encode = num_vert

 
    for ind_tri in range(num_tri):
        for ind in range(3):
            ind_0 = np.int64(vertex_of_cell[ind, ind_tri])
            ind_1 = np.int64(vertex_of_cell[(ind + 1) % 3, ind_tri])
            edge_key = np.minimum(ind_0, ind_1) * num_encode + np.maximum(ind_0, ind_1)
            # must be 

            if edge_key in midpoint_cache:
                ind_mp = midpoint_cache[edge_key]
            else:
                vector_v_mp = (vector_v[ind_0, :] + vector_v[ind_1, :]) / 2.0
                vector_v_mp /= np.linalg.norm(vector_v_mp)
                vector_v_new[num_vert, :] = vector_v_mp
                ind_mp = num_vert
                midpoint_cache[edge_key] = ind_mp
                num_vert += 1



            edge = make_sorted_edge(ind_v0, ind_v1)
            if edge in midpoint_cache:
                ind_mp = midpoint_cache[edge]
            else:
                v = (vector_v[ind_v0, :] + vector_v[ind_v1, :]) / 2
                v /= np.linalg.norm(v)

    for tri in faces:
        i0, i1, i2 = tri

        def midpoint(i, j):
            edge = make_sorted_edge(i, j)
            if edge in midpoint_cache:
                return midpoint_cache[edge]
            else:
                v = (vertices[i, :] + vertices[j, :]) / 2
                v /= np.linalg.norm(v)  # project to unit sphere
                idx = len(new_vertices)
                new_vertices.append(v)
                midpoint_cache[edge] = idx
                return idx

        a = midpoint(i0, i1)
        b = midpoint(i1, i2)
        c = midpoint(i2, i0)

        new_faces.append([i0, a, c])
        new_faces.append([i1, b, a])
        new_faces.append([i2, c, b])
        new_faces.append([a, b, c])

    return np.array(new_vertices), np.array(new_faces)












# Create mesh
faces = np.ascontiguousarray(vertex_of_cell.T)
mesh = trimesh.Trimesh(vertices=vector_v, faces=faces)

# def subdivide_and_project(mesh, levels=1):
#     for _ in range(levels):
#         mesh = mesh.subdivide()
#     mesh.vertices = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1)[:, None]
#     return mesh

t_beg = perf_counter()
mesh_ref = subdivide_and_project(mesh, levels=4)
t_end = perf_counter()
print(f"Elapsed time: {t_end - t_beg:.2f} s")

t_beg = perf_counter()
mesh_ref2 = subdivide_and_project(mesh_ref, levels=1)
t_end = perf_counter()
print(f"Elapsed time: {t_end - t_beg:.2f} s")

vlon_ref = np.arctan2(mesh_ref.vertices[:, 1], mesh_ref.vertices[:, 0])
vlat_ref = np.arcsin(mesh_ref.vertices[:, 2])

# Test plot
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                  vertex_of_cell.transpose())
plt.triplot(triangles, color="black", lw=1.0)

triangles_ref = tri.Triangulation(np.rad2deg(vlon_ref), np.rad2deg(vlat_ref),
                                  mesh_ref.faces[:5_000, :])
plt.triplot(triangles_ref, color="red", lw=0.5)

ax.add_feature(feature.BORDERS.with_scale("10m"),
            linestyle="-", linewidth=0.6)
ax.add_feature(feature.COASTLINE.with_scale("10m"),
            linestyle="-", linewidth=0.6)
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color="black",
                alpha=0.5, linestyle=":", draw_labels=True)
gl.top_labels = False
gl.right_labels = False
plt.show()




###############################################################################
# Increase grid resolution: artificial data (small)
###############################################################################

# Cartesian coordinates
vx = np.cos(vlat) * np.cos(vlon)
vy = np.cos(vlat) * np.sin(vlon)
vz = np.sin(vlat)
vector_v = np.ascontiguousarray(np.vstack([vx, vy, vz]).T)

# Test plot
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                  vertex_of_cell.transpose())
plt.triplot(triangles, color="black", lw=0.5)
plt.scatter(np.rad2deg(vlon), np.rad2deg(vlat), s=10, color="green")
# ----------------------------------------------------------------
# Split edge of triangle
# ----------------------------------------------------------------
ind = 3_333
plt.scatter(np.rad2deg(vlon[edge_vertices[:, ind]]),
            np.rad2deg(vlat[edge_vertices[:, ind]]), s=100,
            marker="*", color="red")
v_0 = vector_v[edge_vertices[0, ind], :]
v_1 = vector_v[edge_vertices[1, ind], :]
angle = np.arccos(np.dot(v_0, v_1))
axis_rot = np.cross(v_0, v_1)

temp = rotation.from_rotvec(angle * axis_rot)

# ----------------------------------------------------------------
ax.add_feature(feature.BORDERS.with_scale("10m"),
            linestyle="-", linewidth=0.6)
ax.add_feature(feature.COASTLINE.with_scale("10m"),
            linestyle="-", linewidth=0.6)
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color="black",
                alpha=0.5, linestyle=":", draw_labels=True)
gl.top_labels = False
gl.right_labels = False
plt.show()







# def midpoint(ind_v0, ind_v1, edge_midpoint_cache, vlon_new,
#              vlat_new, num_vert):

#     edge_key = tuple(sorted((ind_v0, ind_v1)))
#     if edge_key not in edge_midpoint_cache:

#         vlon_new[num_vert] = (vlon_new[ind_v0] + vlon_new[ind_v1]) / 2.0
#         vlat_new[num_vert] = (vlat_new[ind_v0] + vlat_new[ind_v1]) / 2.0
#         edge_midpoint_cache[edge_key] = num_vert
#         num_vert += 1

#     return edge_midpoint_cache[edge_key], num_vert

def subdivide_mesh_dict(vlon, vlat, vertex_of_cell, label_tri):
    """Subdivide triangles mesh (1 -> 4) with dictionary implementation"""

    # Allocate new arrays
    num_vert = vlon.size
    num_tri = vertex_of_cell.shape[1]
    vlon_new = np.empty(num_vert + num_tri * 3, dtype=np.float64)
    # maximal required size
    vlon_new[:num_vert] = vlon
    vlat_new = np.empty(num_vert + num_tri * 3, dtype=np.float64)
    # maximal required size
    vlat_new[:num_vert] = vlat
    vertex_of_cell_new = np.empty((3, num_tri * 4), dtype=np.int32)
    # exact required size
    label_tri_new = np.empty(num_tri * 4, dtype=np.int32)

    # Save indices of edge's vertices (sorted) and index to midpoint
    edge_midpoint_ind = {}  # with dictionary

    # Loop through triangles
    ind_out = 0
    for ind_tri in range(vertex_of_cell.shape[1]):

        ind_v0, ind_v1, ind_v2 = vertex_of_cell[:, ind_tri]

        # Get or compute midpoints for each edge
        ind_mp_0, num_vert = midpoint(ind_v0, ind_v1, edge_midpoint_ind, 
                                      vlon_new, vlat_new, num_vert)
        ind_mp_1, num_vert = midpoint(ind_v1, ind_v2, edge_midpoint_ind, 
                                      vlon_new, vlat_new, num_vert)
        ind_mp_2, num_vert = midpoint(ind_v2, ind_v0, edge_midpoint_ind, 
                                      vlon_new, vlat_new, num_vert)
 
        # Create 4 new triangles (counter-clockwise orientated)
        vertex_of_cell_new[:, ind_out] = [ind_v0, ind_mp_0, ind_mp_2]
        ind_out += 1
        vertex_of_cell_new[:, ind_out] = [ind_mp_0, ind_v1, ind_mp_1]
        ind_out += 1
        vertex_of_cell_new[:, ind_out] = [ind_mp_2, ind_mp_1, ind_v2]
        ind_out += 1
        vertex_of_cell_new[:, ind_out] = [ind_mp_0, ind_mp_1, ind_mp_2]
        ind_out += 1
        label_tri_new[(ind_tri * 4):((ind_tri + 1) * 4)] = label_tri[ind_tri]

    return vlon_new[:num_vert], vlat_new[:num_vert], vertex_of_cell_new, \
        label_tri_new

# -----------------------------------------------------------------------------

@jit(nopython=True)  # fastmath=True
def subdivide_mesh_array(vlon, vlat, vertex_of_cell, label_tri):
    """Subdivide triangles mesh (1 -> 4) with array implementation"""

    # Allocate new arrays
    num_vert = vlon.size
    num_tri = vertex_of_cell.shape[1]
    vlon_new = np.empty(num_vert + num_tri * 3, dtype=np.float64)
    # maximal required size
    vlon_new[:num_vert] = vlon
    vlat_new = np.empty(num_vert + num_tri * 3, dtype=np.float64)
    # maximal required size
    vlat_new[:num_vert] = vlat
    vertex_of_cell_new = np.empty((3, num_tri * 4), dtype=np.int32)
    # exact required size
    label_tri_new = np.empty(num_tri * 4, dtype=np.int32)

    # Compute size of hash table related arrays and allocate them
    prime = 17  # 17, 4007, 71993
    range_size = vlon.size * 3  # approximate number of edges
    counts = np.zeros(range_size, dtype=np.int32)
    for ind_tri in range(num_tri):
        for i in range(3):
            ind_v0 = vertex_of_cell[i, ind_tri]
            ind_v1 = vertex_of_cell[(i + 1) % 3, ind_tri]
            ind_v_min, ind_v_max = np.sort([ind_v0, ind_v1])
            # hash_value = (v_min + v_max)
            hash_value = (ind_v_min * prime + ind_v_max) % range_size 
            # can generate large values -> use 64-bit integer
            counts[hash_value] += 1
    indptr = np.append(0, np.cumsum(counts))
    indices = np.empty((indptr[-1], 3), dtype=np.int32)
    indices.fill(-1)
    # print("Hash table statistics (with shared edges counted twice):")
    # print(f"Maximal collisions: {counts.max()}")
    # print(f"Mean coll. per non-empty index: {counts[counts >= 1].mean():.2f}")
    # print(f"Empty indices frac.: {((counts == 0).sum() / counts.size):.2f}")

    # Loop through triangles
    ind_mp = np.empty(3, dtype=np.int32)
    ind_out = 0
    for ind_tri in range(num_tri):

        # Get or compute midpoints for each edge
        for i in range(3):
            ind_v0 = vertex_of_cell[i, ind_tri]
            ind_v1 = vertex_of_cell[(i + 1) % 3, ind_tri]
            ind_v_min, ind_v_max = np.sort([ind_v0, ind_v1])
            hash_value = (ind_v_min * prime + ind_v_max) % range_size
            for j in range(indptr[hash_value], indptr[hash_value + 1]):
                if (indices[j, 0] == -1):
                    indices[j, :] = [ind_v_min, ind_v_max, num_vert]
                    vlon_new[num_vert] = (vlon[ind_v0] + vlon[ind_v1]) / 2.0
                    vlat_new[num_vert] = (vlat[ind_v0] + vlat[ind_v1]) / 2.0
                    ind_mp[i] = num_vert
                    num_vert += 1
                    break
                elif ((ind_v_min == indices[j, 0]) 
                    and (ind_v_max == indices[j, 1])):
                    ind_mp[i] = indices[j, 2]
                    break

        # Create 4 new triangles (counter-clockwise orientated)
        ind_v0, ind_v1, ind_v2 = vertex_of_cell[:, ind_tri]
        ind_mp_0, ind_mp_1, ind_mp_2 = ind_mp
        vertex_of_cell_new[:, ind_out] = [ind_v0, ind_mp_0, ind_mp_2]
        ind_out += 1
        vertex_of_cell_new[:, ind_out] = [ind_mp_0, ind_v1, ind_mp_1]
        ind_out += 1
        vertex_of_cell_new[:, ind_out] = [ind_mp_2, ind_mp_1, ind_v2]
        ind_out += 1
        vertex_of_cell_new[:, ind_out] = [ind_mp_0, ind_mp_1, ind_mp_2]
        ind_out += 1
        label_tri_new[(ind_tri * 4):((ind_tri + 1) * 4)] = label_tri[ind_tri]

    # print("Hash table statistics (with shared edges counted once):")
    # temp_count = []
    # for i in range(indptr.size - 1):
    #     temp = indices[indptr[i]:indptr[i + 1], :]
    #     if temp.size > 0:
    #         temp_count.append((temp[:, 0] != -1).sum())
    # print(f"Maximal collisions: {np.max(temp_count)}")
    # print(f"Mean coll. per non-empty index: {np.mean(temp_count):.2f}")

    return vlon_new[:num_vert], vlat_new[:num_vert], vertex_of_cell_new, \
        label_tri_new

# -----------------------------------------------------------------------------

# Simple sample data
vlon = np.array([4.0, 5.0, 6.0,  9.0, 8.0, 11.0, 12.0, 12.0, 14.0, 18.0, 15.0, 3.0])
vlat = np.array([3.0, 7.0, 11.0, 8.0, 4.0, 2.0,  6.0,  9.0,  3.0,  6.0,  8.0,  10.0])
vertex_of_cell = np.array([
    [0, 4, 1],
    [1, 4, 3],
    [3, 2, 1],
    [3, 6, 7],
    [4, 6, 3],
    [4, 5, 6],
    [5, 8, 6],
    [8, 9, 6],
    [9, 10, 6],
    [1, 2, 11]
], dtype=np.int32).T  # Counter-clockwise orientation
label_tri = np.arange(vertex_of_cell.shape[1], dtype=np.int32)

# Compare implementations
vlon_dict, vlat_dict, vertex_of_cell_dict, label_tri_dict \
        = subdivide_mesh_dict(vlon, vlat, vertex_of_cell, label_tri)
vlon_arr, vlat_arr, vertex_of_cell_arr, label_tri_arr \
        = subdivide_mesh_array(vlon, vlat, vertex_of_cell, label_tri)
if not all((np.allclose(vlon_arr, vlon_dict), 
            np.allclose(vlat_arr, vlat_dict),
            np.all(vertex_of_cell_arr == vertex_of_cell_dict), 
            np.all(label_tri_arr == label_tri_dict))):
    raise ValueError("Results are not equal")

# t_beg = perf_counter()
# vlon, vlat, vertex_of_cell, label_tri \
#         = subdivide_mesh_array(vlon, vlat, vertex_of_cell, label_tri)
# print(f"Number of triangles: {vertex_of_cell.shape[1]}")
# t_end = perf_counter()
# print(f"Elapsed time: {t_end - t_beg:.6f} s")

# -----------------------------------------------------------------------------

# Test plot
num_divide = 2
s = 200
lw = 2.0
plt.figure()
plt.scatter(vlon, vlat, s=s, color="black", zorder=3)
for i in range(vertex_of_cell.shape[1]):
    ind_cyc = np.append(vertex_of_cell[:, i], vertex_of_cell[0, i])
    plt.plot(vlon[ind_cyc], vlat[ind_cyc], lw=lw, color="black")
for j in range(num_divide):
    print("Split triangles to 4 child triangles")
    vlon, vlat, vertex_of_cell, label_tri \
        = subdivide_mesh_array(vlon, vlat, vertex_of_cell, label_tri)
    plt.scatter(vlon, vlat, s=(s / (2 ** (j + 1))), color="black", zorder=3)
    for i in range(vertex_of_cell.shape[1]):
        ind_cyc = np.append(vertex_of_cell[:, i], vertex_of_cell[0, i])
        plt.plot(vlon[ind_cyc], vlat[ind_cyc], lw=(lw / (2 ** (j + 1))),
                 color="black")
plt.tripcolor(vlon, vlat, vertex_of_cell.T, label_tri, cmap="Spectral")
plt.show()

###############################################################################
# Increase grid resolution: ICON data (large)
###############################################################################

# Load ICON grid information
ds = xr.open_dataset(icon_grids["500m"])
vlon = np.rad2deg(ds["vlon"].values)
vlat = np.rad2deg(ds["vlat"].values)
elon = np.rad2deg(ds["elon"].values)
elat = np.rad2deg(ds["elat"].values)
clon = np.rad2deg(ds["clon"].values)
clat = np.rad2deg(ds["clat"].values)
vertex_of_cell = ds["vertex_of_cell"].values - 1  # [3, number of cells], counterclockwise
edge_of_cell = ds["edge_of_cell"].values - 1  # [3, number of cells]
print(f"Number of triangles: {vertex_of_cell.shape[1]}")
ds.close()
print(f"Ratio (edges / vertices): {(elon.size/vlon.size):.5f}")
label_tri = np.arange(vertex_of_cell.shape[1], dtype=np.int32)

# Test plot
ind_tri = 100_000
plt.figure()
# ---------------------------- Triangle vertices ------------------------------
plt.scatter(clon[ind_tri], clat[ind_tri], s=100, marker="*", color="grey")
ind_cyc = np.append(vertex_of_cell[:, ind_tri], vertex_of_cell[0, ind_tri])
plt.plot(vlon[ind_cyc], vlat[ind_cyc], lw=1.5, color="black")
plt.scatter(vlon[ind_cyc], vlat[ind_cyc], s=25, color="black")
for i in range(3):
    plt.text(vlon[vertex_of_cell[i, ind_tri]],
             vlat[vertex_of_cell[i, ind_tri]] + 0.0005,
             str(i), fontsize=15, color="black")
# ---------------------------- Triangle edge ----------------------------------
plt.scatter(elon[edge_of_cell[:, ind_tri]], elat[edge_of_cell[:, ind_tri]],
            s=50, color="red")
for i in range(3):
    plt.text(elon[edge_of_cell[i, ind_tri]],
             elat[edge_of_cell[i, ind_tri]] + 0.0005,
             str(i), fontsize=15, color="red")
# -----------------------------------------------------------------------------
plt.show()

# Use array implementation
t_beg = perf_counter()
vlon, vlat, vertex_of_cell, label_tri \
        = subdivide_mesh_array(vlon, vlat, vertex_of_cell, label_tri)
print(f"Number of triangles: {vertex_of_cell.shape[1]}")
print(f"Number of vertices: {vlon.size}")
t_end = perf_counter()
print(f"Elapsed time: {t_end - t_beg:.6f} s")

# -----------------------------------------------------------------------------
# Split triangles in subdomains
# -----------------------------------------------------------------------------

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
