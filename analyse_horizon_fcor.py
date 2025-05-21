# Description: Analyse and compare computed f_cor and terrain horizon
#
# Author: Christian R. Steger, May 2025

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri, colors
from netCDF4 import Dataset

from functions import centroid_values

style.use("classic")

# Paths
path_in_out = "/scratch/mch/csteger/temp/ICON_refined_mesh/"
path_ige = "/store_new/mch/msopr/csteger/Data/Miscellaneous/" \
    + "ICON_grids_EXTPAR/"

###############################################################################
# Compare terrain horizon
###############################################################################

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------

# 1km
icon_res = "1km"
file_mesh = "ICON_refined_mesh_mch_1km.nc"
file_hori_fcor = "SW_dir_cor_" + "mch_" + icon_res + ".nc"
file_extpar = "MeteoSwiss/extpar_grid_shift_topo/" \
    + "extpar_icon_grid_0001_R19B08_mch.nc"

# # 500m
# icon_res = "500m"
# file_mesh = "ICON_refined_mesh_mch_500m.nc"
# file_hori_fcor = "SW_dir_cor_" + "mch_" + icon_res + ".nc"
# file_extpar = "MeteoSwiss/extpar_grid_shift_topo/" \
#     + "extpar_icon_grid_00005_R19B09_DOM02.nc"

# Get initial information
ds = xr.open_dataset(path_in_out + file_hori_fcor)
ind_hori_out = ds["ind_hori_out"].values # # index_cell_child
ds.close()
ds = xr.open_dataset(path_in_out + file_mesh)
num_cell_child_per_parent = int(ds["num_cell_child_per_parent"])
ds.close()
ind_cell_parent = ind_hori_out[slice(0, None, num_cell_child_per_parent)] \
    // num_cell_child_per_parent

# Load information from SW_dir_cor computation
ds = xr.open_dataset(path_in_out + file_hori_fcor)
horizon_child = ds["horizon"].values # (num_hori_out, num_hori)
f_cor = ds["f_cor"][ind_cell_parent, :, :].values # (num_hori_out, num_hori)
ds.close()

# Load terrain horizon (grid-scale cell)
ds = xr.open_dataset(path_ige + file_extpar)
horizon_gs = ds["HORIZON"].values[:, ind_cell_parent]
ds.close()

ind_loc = 1 # (0, 1, 2, 3) (Vicosoprano, Vals, Piotta, Cevio)

# Plot for location
azim = np.arange(0.0, 360.0, 360 // horizon_child.shape[1])
elev = np.linspace(0.0, 90.0, 91)
plt.figure(figsize=(10, 5))
plt.pcolormesh(azim, elev, f_cor[ind_loc, :, :].transpose(), shading="auto",
               cmap="RdBu_r", vmin=0.0, vmax=2.0)
cbar = plt.colorbar()
cbar.set_label("Subgrid SW_dir correction factor [-]", labelpad=10)
plt.contour(azim, elev, f_cor[ind_loc, :, :].transpose(),
            levels=[0.1, 0.5, 0.9], colors="grey",
            linestyles=["--", "-", "--"], linewidths=1.5)
plt.plot(azim, horizon_gs[:, ind_loc], color="black", linewidth=2.5)
plt.xlabel("Azimuth angle (clockwise from North) [deg]")
plt.ylabel("Elevation angle [deg]")
# plt.show()
plt.savefig("/scratch/mch/csteger/HORAYZON_extpar_subgrid/Vals.png",
            dpi=250)
plt.close()

###############################################################################

# Load mesh data
ds = xr.open_dataset(path_in_out + file_mesh)
vlon = np.rad2deg(ds["vlon"].values)
vlat = np.rad2deg(ds["vlat"].values)
faces = ds["faces"][ind_hori_out, :].values
ds.close()
triangles = tri.Triangulation(vlon, vlat, faces)
tri_finder = triangles.get_trifinder()

ind_loc = 3 # (0, 1, 2, 3) (Vicosoprano, Vals, Piotta, Cevio)

# MeteoSwiss stations
locations = (
     (9.6278,   46.353019), # Vicosoprano
     (9.188711, 46.627758), # Vals
     (8.688039, 46.514811), # Piotta
     (8.603161, 46.320486), # Cevio
)

ind_tri = int(tri_finder(*locations[ind_loc])) # type: ignore

# Plot for location
plt.figure(figsize=(10, 5))
for i in range(num_cell_child_per_parent):
    plt.plot(azim,
             horizon_child[ind_loc * num_cell_child_per_parent + i, :],
             color="grey", alpha=0.5)
plt.plot(azim, horizon_child[ind_tri, :], color="red", alpha=1.0, lw=1.0)
plt.plot(azim, horizon_gs[:, ind_loc], color="black", linewidth=2.5)
plt.xlabel("Azimuth angle (clockwise from North) [deg]")
plt.ylabel("Elevation angle [deg]")
# plt.show()
plt.savefig("/scratch/mch/csteger/HORAYZON_extpar_subgrid/Cevio.png",
            dpi=250)
plt.close()



###############################################################################
########## Old stuff below...
###############################################################################

# Grid information
ds = xr.open_dataset(path_in_out + file_mesh)
# vlon_child = np.rad2deg(ds["vlon"].values)
# vlat_child = np.rad2deg(ds["vlat"].values)
# slice_cells = slice(ind_hori_out[0], ind_hori_out[-1] + 1)
# faces = ds["faces"][slice_cells, :].values
num_cell_child_per_parent = int(ds["num_cell_child_per_parent"])
ds.close()

ind_parent = ind_hori_out[ind_loc * num_cell_child_per_parent] // num_cell_child_per_parent

# Load terrain horizon (subgrid cells)
ds = xr.open_dataset(path_in_out + file_hori_fcor)
horizon = ds["horizon"].values # (num_hori_out, num_hori)
ind_hori_out = ds["ind_hori_out"].values # (num_hori_out)
ind_parent = ind_hori_out[ind * num_cell_child_per_parent] // num_cell_child_per_parent
ds.close()



# Plot
azim = np.arange(0.0, 360.0, 360 // horizon.shape[1])
fig = plt.figure(figsize=(10, 5))
slice_hori = slice(ind * num_cell_child_per_parent,
                   (ind + 1) * num_cell_child_per_parent)
for i in range(num_cell_child_per_parent):
    plt.plot(azim, horizon[slice_hori, :][i, :], color="grey", alpha=0.5)
# plt.plot(azim, horizon.mean(axis=0), color="black", linewidth=2.0)
# plt.plot(azim, horizon[ind_tri, :], color="red", alpha=0.5)
plt.plot(azim, horizon_gs, color="blue", linewidth=2.0)
plt.show()
# plt.savefig(f"/scratch/mch/csteger/HORAYZON_extpar_subgrid/"
#             + f"ter_horizon_{icon_res}.png",
#             dpi=250)
# plt.close()
