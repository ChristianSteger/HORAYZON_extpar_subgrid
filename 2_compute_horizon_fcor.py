# Description: Compute terrain horizon and slope angle/aspect for sub-grid
#              cells and derive relevant ICON grid-scale quantities from these
#              parameters. Additionally, it is possible to output terrain
#              horizon and slope angle/aspect for selected sub-grid cells.
#
# Author: Christian R. Steger, May 2025

import sys
from time import perf_counter
import json

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri
from netCDF4 import Dataset

from functions.refine_icon_mesh import centroid_values

style.use("classic")

# Paths
path_in_out = "/scratch/mch/csteger/temp/ICON_refined_mesh/"
# path_in_out = "/Users/csteger/Desktop/"

# Path to Cython/C++ functions
sys.path.append("/scratch/mch/csteger/HORAYZON_extpar_subgrid/")
# sys.path.append("/Users/csteger/Desktop/HORAYZON_extpar_subgrid/")
from horizon_svf import horizon_svf_comp_py

###############################################################################
# Load refined ICON mesh
###############################################################################

# Select ICON domain
# icon_dom = "test_2km"
# icon_dom = "mch_2km"
icon_dom = "mch_1km"
# icon_dom = "mch_500m"

# Settings
check_plots = False

# Load data
t_beg = perf_counter()
file_mesh = f"ICON_refined_mesh_{icon_dom}.nc"
ds = xr.open_dataset(path_in_out + file_mesh)
vlon = ds["vlon"].values # (num_vertex; float64)
vlat = ds["vlat"].values # (num_vertex; float64)
elevation = ds["elevation"].values # (num_vertex; float32)
dem_name = ds["elevation"].source
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
# Settings and optionally compute 'idx_hori_out'
###############################################################################

# Settings
num_hori = 24 # number of azimuth angles
dist_search = 40_000.0 #  horizon search distance [m]
ray_org_elev = 0.5 # 0.1, 0.2 [m]
idx_hori_out = np.array([0, 1, 5, 10, 3_000, 3], dtype=np.uint32)
# Indices of 'num_cell_child' to output terrain horizon
num_elev = 91 # number of elevation angles for sw_dir_cor computation
sw_dir_cor_max = 25.0 # maximum value for SW_dir correction factor
cons_area_factor = 0 # use area factor for SW_dir correction factor
idx_hori_out = np.array([0], dtype=np.uint32) # dummy array
file_out = (f"SW_dir_cor_{icon_dom}", "nc")
loc_name = None

# -----------------------------------------------------------------------------
# Save subgrid parameters (horizon and terrain normal) for certain locations
# -----------------------------------------------------------------------------

# If only the subgrid parameters for the below locations are required, then
# the ray tracing part can be sped up by including the commented out part in
# 'mo_lradtopo_horayzon.cpp' around line 560 (-> recompile code!)

# # Own selection (some MCH stations and other interesting locations)
# locations = [
#      # ------- MeteoSwiss stations -----------
#      ["Vicosoprano",     [9.6278,   46.353019]],
#      ["Vals",            [9.188711, 46.627758]],
#      ["Piotta",          [8.688039, 46.514811]],
#      ["Cevio",           [8.603161, 46.320486]],
#      ["Goeschenen",      [8.595364, 46.692678]],
#      ["Grono",           [9.163758, 46.255075]],
#      ["Glarus",          [9.066961, 47.034586]],
#      # ---------------------------------------
#      ["Veltlin_S_fac",   [9.730859, 46.181635]],
#      ["Veltlin_N_fac",   [9.879585, 46.143744]],
#      ["Kloental",        [8.992863, 47.013992]],
#      ["Limmeren",        [8.995542, 46.858968]],
#      ["Eiger_below",     [8.003756, 46.601543]],
#      ["Gondo",           [8.140656, 46.196015]],
#      ["Gondo_fort",      [8.113924, 46.195547]],
#      ["Calancatal_1",    [9.117380, 46.259365]],
#      ["Calancatal_2",    [9.114330, 46.301508]],
#      ["Calancatal_3",    [9.116637, 46.445454]],
#      ["Calancatal_4",    [9.113043, 46.469920]],
#      ["Schiben_below",   [8.960114, 46.820576]],
#      ["Linthal",         [8.981141, 46.864608]],
#      ["Engelhorn_below", [8.163189, 46.668290]],
#      ["Gr_Scheidegg",    [8.101539, 46.655433]],
#      ["Lauterbrunnen_1", [7.908836, 46.570909]],
#     #  # --------------------------------------- only 1km and 500 m mesh!
#      ["Kandertal_S_fac", [7.733795, 46.457283]],
#      ["Kandertal_val",   [7.737744, 46.445441]],
#      ["Kandertal_N_fac", [7.742341, 46.432649]],
#     #  # ---------------------------------------
#      ["Oberrohrdorf",    [8.312873, 47.420988]],
#      ["Napf",            [7.935345, 47.002321]],
#      ["Zuerich",         [8.533633, 47.375246]],
#      ["Walensee",        [9.234161, 47.121523]],
#      ["Po_valley",       [9.509584, 45.426837]],
#     #  # ---------------------------------------
#      ["Piz_Aul",         [9.124848, 46.622926]],
#      ["Zefreilahorn",    [9.065229, 46.552159]],
#      ["Rheinwaldhorn",   [9.040183, 46.493834]],
#      ["Finsteraarhorn",  [8.126588, 46.537015]],
#      ["Schreckhorn",     [8.119095, 46.589719]],
#      ["Kroenten",        [8.569259, 46.782166]],
#      ["Piz_Cengalo",     [9.602163, 46.294972]],
#      ["Piz_Bernina",     [9.908377, 46.382217]]
#     #  # --------------------------------------- only 500 m mesh!
#     #  ["Gredetsch_E_fac", [7.924869, 46.356297]],
#     #  ["Gredetsch_val",   [7.933481, 46.357383]],
#     #  ["Gredetsch_W_fac", [7.940311, 46.357533]]
#      # ---------------------------------------
#     ]
# loc_name = "loc_own"

# # All MCH stations
# path_obs = "/scratch/mch/csteger/temp/movero_obs_data/"
# file_obs = path_obs + "20241224sfc.atab"
# data = np.genfromtxt(file_obs, skip_header=13, max_rows=4, dtype=str,
#                      autostrip=True)
# station_names = data[0, 1:]
# lat_obs = data[1, 1:].astype(np.float32)
# lon_obs = data[2, 1:].astype(np.float32)
# locations = [[str(name), [float(lon), float(lat)]]
#              for name, lat, lon in zip(station_names, lat_obs, lon_obs)]
# loc_name = "loc_mch"

# # Save selected locations
# file_json = path_in_out + f"{loc_name}.json"
# with open(file_json, "w") as f:
#     json.dump(locations, f, indent=4)

# # Load ICON grid information
# path_grid = "/store_new/mch/msopr/csteger/Data/Miscellaneous/ICON_grids/"
# # icon_grid = "test/icon_grid_DOM01.nc"
# # icon_grid = "MeteoSwiss/icon_grid_0002_R19B07_mch.nc" # 2km
# icon_grid = "MeteoSwiss/icon_grid_0001_R19B08_mch.nc" # 1km
# # icon_grid = "MeteoSwiss/icon_grid_00005_R19B09_DOM02.nc" # 500m
# ds = xr.open_dataset(path_grid + icon_grid)
# vlon_parent = np.rad2deg(ds["vlon"].values)
# vlat_parent = np.rad2deg(ds["vlat"].values)
# clon_parent = np.rad2deg(ds["clon"].values)
# clat_parent = np.rad2deg(ds["clat"].values)
# if clon_parent.size != num_cell_parent:
#     raise ValueError("Inconsistent data loaded")
# vertex_of_cell_parent = ds["vertex_of_cell"].values - 1
# triangles = tri.Triangulation(vlon_parent, vlat_parent,
#                               vertex_of_cell_parent.transpose())
# ds.close()

# # Get relevant cell indices
# idx_tri_all = np.empty(len(locations), dtype=np.uint32) # parent cell indices
# tri_finder = triangles.get_trifinder()
# for idx, loc in enumerate(locations):
#     idx_tri = int(tri_finder(*loc[1]))  # type: ignore
#     idx_tri_all[idx] = idx_tri
#     print(loc[0], idx_tri, clon_parent[idx_tri], clat_parent[idx_tri])
# idx_hori_out = np.array([], dtype=np.uint32)
# for idx in idx_tri_all:
#      idx_hori_out = np.append(
#           idx_hori_out,
#           np.arange(idx * num_cell_child_per_parent,
#                     (idx + 1) * num_cell_child_per_parent, dtype=np.uint32))
# print(f"Size of 'idx_hori_out': {idx_hori_out.size}")

###############################################################################
# Compute subgrid parameters and save to NetcDF file
###############################################################################

# Compute subgrid parameters
f_cor, shadow_angle_idx, terrain_normal, horizon_out, slope_out = \
    horizon_svf_comp_py(
    vlon, vlat,
    elevation.astype(np.float64),
    faces,
    idx_hori_out,
    num_cell_parent, num_cell_child_per_parent,
    num_hori, dist_search,
    ray_org_elev, num_elev,
    sw_dir_cor_max, cons_area_factor)

# Save SW_dir correction factors to NetCDF file
t_beg = perf_counter()
if loc_name is None:
    file_out = ".".join(file_out)
else:
    file_out = file_out[0] + f"_{loc_name}." + file_out[1]
ncfile = Dataset(filename=path_in_out + file_out, mode="w", format="NETCDF4")
ncfile.dem_source = dem_name
ncfile.area_factor_used = str(bool(cons_area_factor))
ncfile.createDimension(dimname="num_cell_parent", size=f_cor.shape[0])
ncfile.createDimension(dimname="num_hori", size=f_cor.shape[1])
ncfile.createDimension(dimname="num_elev", size=f_cor.shape[2])
ncfile.createDimension(dimname="angles", size=3)
ncfile.createDimension(dimname="vec_comp", size=3)
ncfile.createDimension(dimname="num_hori_out", size=horizon_out.shape[0])
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="f_cor", datatype="f4",
                                dimensions=("num_cell_parent", "num_hori",
                                            "num_elev"))
nc_data.units = "-"
nc_data.long_name = "SW_dir correction factor"
nc_data[:] = f_cor
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="shadow_angle_idx", datatype="i4",
                                dimensions=("num_cell_parent", "num_hori",
                                            "angles"))
nc_data.units = "-"
nc_data.long_name = "Relevant angles (indices) for shadow casting"
nc_data[:] = shadow_angle_idx
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="terrain_normal", datatype="f4",
                                dimensions=("num_cell_parent", "vec_comp"))
nc_data.units = "-"
nc_data.long_name = "Sub-grid cell average terrain normal (not normalised)"
nc_data[:] = terrain_normal
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="ind_hori_out", datatype="i4",
                                dimensions=("num_hori_out"))
nc_data.units = "-"
nc_data.long_name = "Indices of num_cell_child to output terrain horizon"
nc_data[:] = idx_hori_out
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
