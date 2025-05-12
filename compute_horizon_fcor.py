# Description: Compute terrain horizon and slope angle/aspect for sub-grid
#              cells, spatially aggregated correction factors (f_cor) and
#              compress f_cor information in discrete values.
#
# Author: Christian R. Steger, May 2025

import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri, colors

from functions import centroid_values

style.use("classic")

# Paths
# path_in = "/scratch/mch/csteger/temp/ICON_refined_mesh/"
path_in = "/Users/csteger/Desktop/"

# Path to Cython/C++ functions
# sys.path.append("/scratch/mch/csteger/HORAYZON_extpar_subgrid/")
sys.path.append("/Users/csteger/Desktop/HORAYZON_extpar_subgrid/")
from horizon_svf import horizon_svf_comp_py

###############################################################################
# Load refined ICON mesh
###############################################################################

# Select mesh file
file_mesh = "ICON_refined_mesh_test_2km.nc"
# file_mesh = "ICON_refined_mesh_mch_500m.nc"
# file_mesh = "ICON_refined_mesh_mch_1km.nc"
# file_mesh = "ICON_refined_mesh_mch_2km.nc"

# Load data
ds = xr.open_dataset(path_in + file_mesh)
vlon = ds["vlon"].values # (num_vertex; float64)
vlat = ds["vlat"].values # (num_vertex; float64)
elevation = ds["elevation"].values # (num_vertex; float32)
faces = ds["faces"].values # (num_cell, 3; int32) (transposed 'vertex_of_cell')
num_cell_parent = int(ds["num_cell_parent"])
num_cell_child_per_parent = int(ds["num_cell_child_per_parent"])
ds.close()

# -----------------------------------------------------------------------------
# Test plot
# -----------------------------------------------------------------------------

num_tri_parent = 13
num_tri_child = num_tri_parent * num_cell_child_per_parent
triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                 faces[:num_tri_child, :])

# Elevation
elevation_centroids = centroid_values(elevation, faces)
plt.figure(figsize=(10, 10))
plt.tripcolor(triangles, elevation_centroids[:num_tri_child], cmap="terrain",
              vmin=elevation_centroids[:num_tri_child].min(),
              vmax=elevation_centroids[:num_tri_child].max(),
              edgecolors="black", linewidth=0.1)
plt.show()

###############################################################################
# Compute spatially aggregated correction factors (f_cor)
###############################################################################

# Settings 
num_hori = 24 # number of azimuth angles
dist_search = 40_000.0 #  horizon search distance [m]
ray_org_elev = 0.2 # 0.1, 0.2 [m]
ind_hori_out = np.array([0, 1, 5, 10, 3_000, 21_000, 101_000, 3], dtype=np.int32)
# output horizon for specific locations

# Compute f_cor
f_cor, horizon_out = horizon_svf_comp_py(
    vlon, vlat,
    elevation.astype(np.float64),
    faces,
    ind_hori_out,
    num_cell_parent, num_cell_child_per_parent,
    num_hori, dist_search,
    ray_org_elev)

# Test plot f_cor
azim = np.arange(0.0, 360.0, 360 // num_hori)
elev = np.linspace(0.0, 90.0, 91)
ind = 3334
plt.figure(figsize=(14, 6))
plt.pcolormesh(azim, elev, f_cor[ind, :, :].transpose(), vmin=0.0, vmax=2.0,
               cmap="RdBu_r")
plt.axis((0.0, 345.0, 0.0, 70.0))
plt.colorbar()
plt.show()

plt.figure()
plt.plot(elev, f_cor[ind, 16, :])
plt.show()

# Test plot horizon
plt.figure()
for i in range(ind_hori_out.size):
        plt.plot(azim, horizon_out[i, :])
plt.show()

# -----------------------------------------------------------------------------
# Temporary stuff to check C++ code...
# -----------------------------------------------------------------------------

# rad_earth = 6371229.0
# points_x = (rad_earth + elevation[0]) * np.cos(vlat[0]) * np.cos(vlon[0])
# points_y = (rad_earth + elevation[0]) * np.cos(vlat[0]) * np.sin(vlon[0])
# points_z = (rad_earth + elevation[0]) * np.sin(vlat[0])

# faces_flat = faces.ravel()
# for i in range(4):
#     triangle = (int(faces_flat[(i * 3) + 0]),
#                 int(faces_flat[(i * 3) + 1]),
#                 int(faces_flat[(i * 3) + 2]))
#     print(f"Triangle {i}: {triangle}")

# rad_earth = 6371229.0
# y = rad_earth * np.cos(np.deg2rad(45.1216))
# z = rad_earth - rad_earth * np.sin(np.deg2rad(45.1216))
# print("North Pole: ", y, z)

# for i in range(num_cell_parent):
#     for j in range(num_cell_child_per_parent):
#         ind_cell = i * num_cell_child_per_parent + j
# print(ind_cell)

elev_num = 91
elev_spac = np.deg2rad(90.0) / float(elev_num - 1)

elev_ang = 0.0
for m in range(elev_num -1):
    elev_ang += elev_spac
    print(np.rad2deg(elev_ang), m)



# -----------------------------------------------------------------------------

###############################################################################
# Compress f_cor information in discrete values
###############################################################################
