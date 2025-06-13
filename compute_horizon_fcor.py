# Description: Compute terrain horizon and slope angle/aspect for sub-grid
#              cells, spatially aggregated correction factors (f_cor) and
#              compress f_cor information in discrete values.
#
# Author: Christian R. Steger, May 2025

import sys
from time import perf_counter

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri, colors
from netCDF4 import Dataset

from functions import centroid_values

style.use("classic")

# Paths
path_in_out = "/scratch/mch/csteger/temp/ICON_refined_mesh/"
# path_in_out = "/Users/csteger/Desktop/"

# Path to Cython/C++ functions
# sys.path.append("/scratch/mch/csteger/HORAYZON_extpar_subgrid/")
sys.path.append("/Users/csteger/Desktop/HORAYZON_extpar_subgrid/")
from horizon_svf import horizon_svf_comp_py

###############################################################################
# Load refined ICON mesh
###############################################################################

# Select mesh file
# icon_res = "2km"
# file_mesh = "ICON_refined_mesh_test_2km.nc"
# file_out = "SW_dir_cor_" + "test_" + icon_res + ".nc"

# icon_res = "2km"
# file_mesh = "ICON_refined_mesh_mch_2km.nc"
# file_out = "SW_dir_cor_" + "mch_" + icon_res + ".nc"

# icon_res = "1km"
# file_mesh = "ICON_refined_mesh_mch_1km.nc"
# file_out = "SW_dir_cor_" + "mch_" + icon_res + ".nc"

icon_res = "500m"
file_mesh = "ICON_refined_mesh_mch_500m.nc"
file_out = "SW_dir_cor_" + "mch_" + icon_res + ".nc"

# Settings
check_plots = False

# Load data
t_beg = perf_counter()
ds = xr.open_dataset(path_in_out + file_mesh)
vlon = ds["vlon"].values # (num_vertex; float64)
vlat = ds["vlat"].values # (num_vertex; float64)
elevation = ds["elevation"].values # (num_vertex; float32)
faces = ds["faces"].values
# (num_cell, 3; uint32) (transposed 'vertex_of_cell')
num_cell_parent = int(ds["num_cell_parent"])
num_cell_child_per_parent = int(ds["num_cell_child_per_parent"])
ds.close()
t_end = perf_counter()
print(f"Open NetCDF file: {t_end - t_beg:.1f} s")

# -----------------------------------------------------------------------------
# Test plot
# -----------------------------------------------------------------------------

if check_plots:

    num_tri_parent = 13
    num_tri_child = num_tri_parent * num_cell_child_per_parent
    triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                    faces[:num_tri_child, :])

    # Elevation
    elevation_centroids = centroid_values(elevation, faces)
    plt.figure(figsize=(10, 10))
    plt.tripcolor(triangles, elevation_centroids[:num_tri_child],
                  cmap="terrain",
                  vmin=elevation_centroids[:num_tri_child].min(),
                  vmax=elevation_centroids[:num_tri_child].max(),
                  edgecolors="black", linewidth=0.1)
    plt.show()
    del triangles, elevation_centroids

###############################################################################
# Compute spatially aggregated correction factors (f_cor)
###############################################################################

# Settings 
num_hori = 24 # number of azimuth angles
dist_search = 40_000.0 #  horizon search distance [m]
ray_org_elev = 0.5 # 0.1, 0.2 [m]
ind_hori_out = np.array([0, 1, 5, 10, 3_000, 3], dtype=np.uint32)
# Indices of 'num_cell_child' to output terrain horizon
num_elev = 91 # number of elevation angles for sw_dir_cor computation
sw_dir_cor_max = 25.0 # maximum value for SW_dir correction factor
cons_area_factor = 1 # use area factor for SW_dir correction factor

# -----------------------------------------------------------------------------
# Select 'ind_hori_out' based on parent cell mesh
# -----------------------------------------------------------------------------

# MeteoSwiss stations
locations = {
     "Vicosoprano": (9.6278,   46.353019),
     "Vals":        (9.188711, 46.627758),
     "Piotta":      (8.688039, 46.514811),
     "Cevio":       (8.603161, 46.320486),
     "south_face":  (9.730859, 46.181635),
     "north_face":  (9.879585, 46.143744),
     }

# Load data
path_ige = "/store_new/mch/msopr/csteger/Data/Miscellaneous/" \
    + "ICON_grids_EXTPAR/"

# icon_grid = "MeteoSwiss/icon_grid_0002_R19B07_mch.nc" # 2km
# icon_grid = "MeteoSwiss/icon_grid_0001_R19B08_mch.nc" # 1km
icon_grid = "MeteoSwiss/icon_grid_00005_R19B09_DOM02.nc" # 500m
ds = xr.open_dataset(path_ige + icon_grid)
vlon_parent = np.rad2deg(ds["vlon"].values)
vlat_parent = np.rad2deg(ds["vlat"].values)
clon_parent = np.rad2deg(ds["clon"].values)
clat_parent = np.rad2deg(ds["clat"].values)
if clon_parent.size != num_cell_parent:
    raise ValueError("Inconsistent data loaded")
vertex_of_cell_parent = ds["vertex_of_cell"].values - 1
triangles = tri.Triangulation(vlon_parent, vlat_parent,
                              vertex_of_cell_parent.transpose())
ds.close()

# Get relevant cell indices
ind_tri_all = {}
tri_finder = triangles.get_trifinder()
for i in locations.keys():
    ind_tri = int(tri_finder(*locations[i]))  # type: ignore
    ind_tri_all[i] = ind_tri
    print(i, ind_tri, clon_parent[ind_tri], clat_parent[ind_tri])

# # Test plot of point in triangle
# for i in locations.keys():
#     fig = plt.figure()
#     v0, v1, v2 = vertex_of_cell_parent[:, ind_tri_all[i]]
#     plt.plot(vlon_parent[v0], vlat_parent[v0], "o", color="red")
#     plt.plot(vlon_parent[v1], vlat_parent[v1], "o", color="red")
#     plt.plot(vlon_parent[v2], vlat_parent[v2], "o", color="red")
#     plt.plot(*locations[i], "o", color="blue")
#     plt.savefig("/scratch/mch/csteger/HORAYZON_extpar_subgrid/plots/"
#                 + f"point_triangle_{icon_res}_{i}.png", dpi=250)
#     plt.close()

ind_hori_out = np.array([], dtype=np.uint32)
for i in locations.keys():
    ind_hori_out = np.append(
         ind_hori_out,
         np.arange(ind_tri_all[i] * num_cell_child_per_parent,
                   (ind_tri_all[i] + 1) * num_cell_child_per_parent,
                   dtype=np.uint32))
print("Size of 'ind_hori_out':", ind_hori_out.size)

# -----------------------------------------------------------------------------

# Compute f_cor
f_cor, horizon_out, slope_out = horizon_svf_comp_py(
    vlon, vlat,
    elevation.astype(np.float64),
    faces,
    ind_hori_out,
    num_cell_parent, num_cell_child_per_parent,
    num_hori, dist_search,
    ray_org_elev, num_elev,
    sw_dir_cor_max, cons_area_factor)

# Save SW_dir correction factors to NetCDF file
t_beg = perf_counter()
ncfile = Dataset(filename=path_in_out + file_out, mode="w", format="NETCDF4")
ncfile.createDimension(dimname="num_cell_parent", size=f_cor.shape[0])
ncfile.createDimension(dimname="num_hori", size=f_cor.shape[1])
ncfile.createDimension(dimname="num_elev", size=f_cor.shape[2])
ncfile.createDimension(dimname="num_hori_out", size=horizon_out.shape[0])
ncfile.createDimension(dimname="vec_comp", size=3)
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="f_cor", datatype="f4",
                                dimensions=("num_cell_parent", "num_hori",
                                            "num_elev"))
nc_data.units = "-"
nc_data.long_name = "SW_dir correction factor"
nc_data[:] = f_cor
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="ind_hori_out", datatype="i4",
                                dimensions=("num_hori_out"))
nc_data.units = "-"
nc_data.long_name = "Indices of num_cell_child to output terrain horizon"
nc_data[:] = ind_hori_out
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="horizon", datatype="f8",
                                dimensions=("num_hori_out", "num_hori"))
nc_data.units = "deg"
nc_data.long_name = "Terrain horizon"
nc_data[:] = horizon_out
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="slope", datatype="f8",
                                dimensions=("num_hori_out", "vec_comp"))
nc_data.units = "-"
nc_data.long_name = "Terrain surface normal vector (local ENU coordinates)"
nc_data[:] = slope_out
# -----------------------------------------------------------------------------
ncfile.close()
t_end = perf_counter()
print(f"Write output NetCDF file: {t_end - t_beg:.1f} s")

# -----------------------------------------------------------------------------
# Simple checks of output
# -----------------------------------------------------------------------------

if check_plots:

    azim = np.arange(0.0, 360.0, 360 // num_hori)
    elev = np.linspace(0.0, 90.0, 91)

    # Plot f_cor
    ind = 3334
    fig = plt.figure(figsize=(14, 6))
    plt.pcolormesh(azim, elev, f_cor[ind, :, :].transpose(),
                   vmin=0.0, vmax=2.0, cmap="RdBu_r")
    plt.axis((0.0, 345.0, 0.0, 70.0))
    plt.colorbar()
    # plt.show()
    plt.savefig("/scratch/mch/csteger/HORAYZON_extpar_subgrid/f_cor_2d.png",
                dpi=250)
    plt.close()

    fig = plt.figure()
    for i in range(0, num_hori, 2):
            plt.plot(elev, f_cor[ind, i, :])
    # plt.show()
    plt.savefig("/scratch/mch/csteger/HORAYZON_extpar_subgrid/f_cor_1d.png",
                dpi=250)
    plt.close()

    # Plot terrain horizon
    fig = plt.figure()
    for i in range(ind_hori_out.size):
            plt.plot(azim, horizon_out[i, :])
    # plt.show()
    plt.savefig("/scratch/mch/csteger/HORAYZON_extpar_subgrid/ter_horizon.png",
                dpi=250)
    plt.close()

###############################################################################
# Temporary - performance scaling with 1 km mesh
###############################################################################

# # 1km mesh
# num_cell_parent = np.array([25_000, 50_000, 100_000, 200_000, 500_000, 1_147_980])
# ray_tracing = np.array([109.66, 321.61, 817.5, 744.40, 2059.60, 2245.14])

# # 2km mesh
# num_cell_parent = np.array([10_000, 25_000, 50_000, 100_000, 283_876])
# ray_tracing = np.array([183.44, 602.71, 698.67, 1359.35, 2393.92])

# # Plot
# plt.figure()
# plt.plot(num_cell_parent, ray_tracing / num_cell_parent, "-", color="blue", lw=1.5)
# plt.scatter(num_cell_parent, ray_tracing / num_cell_parent, color="blue", s=30)
# plt.xlabel("Number of parent cells")
# plt.ylabel("Ray tracing time / number of parent cells [s]")
# plt.show()

###############################################################################
# Temporary stuff to check C++ code...
###############################################################################

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

# num_elev = 91
# elev_spac = np.deg2rad(90.0) / float(num_elev - 1)
# elev_ang = 0.0
# for m in range(num_elev -1):
#     elev_ang += elev_spac
#     print(np.rad2deg(elev_ang), m)
