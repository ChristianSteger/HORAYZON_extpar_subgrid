# Description: Compress f_cor information.
#
# Author: Christian R. Steger, June 2025

from time import perf_counter

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri, colors
from netCDF4 import Dataset
from numba import njit, float64, int64

from functions import centroid_values

style.use("classic")

# Paths
path_in_out = "/scratch/mch/csteger/temp/ICON_refined_mesh/" \
    + "SW_dir_cor_all_computed/"

###############################################################################
# Function
###############################################################################

@njit((float64[:])(float64, float64, int64, float64))
def spacing_exp(x_start, x_end, num, exp):
    """
    Computes spacing between x_start and x_end with exponentially increasing 
    spacing towards the right. The output array starts/ends exactly 
    with x_start/x_end.
    """
    x_spac = np.empty(num, dtype=np.float64)
    x_spac[0] = x_start
    for i in range(1, num - 1):
         x_spac[i] = x_start + (x_end - x_start) * (i / (num - 1)) ** (exp + 1)
    x_spac[num - 1] = x_end
    return x_spac

@njit
def spacing_exp_opt(elev_start, elev_end, num, exponents, elev, f_cor_loc,
                    sin_elev):
    """
    Computes spacing between x_start and x_end with exponentially increasing 
    spacing towards the right and select the optimal exponent. The output 
    array starts/ends exactly with x_start/x_end.
    """
    dev_abs = np.empty(exponents.size, dtype=np.float64)
    for j in range(exponents.size):
        elev_expo = spacing_exp(elev_start, elev_end, num, exp=exponents[j])
        f_cor_nodes = np.interp(x=elev_expo, xp=elev, fp=f_cor_loc)
        # extrapolation never occurs
        f_cor_rec = np.interp(x=elev, xp=elev_expo, fp=f_cor_nodes)
        # extrapolation can occur but boundary values of [0.0, 1.0] correct
        dev_abs[j] = np.abs((f_cor_rec - f_cor_loc) * sin_elev).mean()
    exp_opt = exponents[np.argmin(dev_abs)]
    elev_expo = spacing_exp(elev_start, elev_end, num, exp=exp_opt)
    return elev_expo, exp_opt

###############################################################################
# Compress f_cor information
###############################################################################

# Input file
# icon_res = "2km"
icon_res = "1km"
# icon_res = "500m"
file_in = "SW_dir_cor_mch_" + icon_res + ".nc"

# Settings
check_plots = False
radiation = 1050.0  # direct beam radiation with 90 deg elevation angle [W m-2]

# Load data
t_beg = perf_counter()
ds = xr.open_dataset(path_in_out + file_in)
f_cor = ds["f_cor"].values # (num_cell_parent, num_hori, num_elev; float32)
ds.close()
t_end = perf_counter()
print(f"Open NetCDF file: {t_end - t_beg:.1f} s")

# Set upper limit for f_cor
f_cor = f_cor.clip(max=10.0)

# Azimuth and elevation angles
azim = np.arange(0.0, 360.0, 15, dtype=np.float64) # [deg]
if f_cor.shape[1] != azim.size:
    raise ValueError("Inconsistency between 'f_cor' and 'azim_ang' size")
elev = np.linspace(0.0, 90.0, 91, dtype=np.float64) # [deg]
if f_cor.shape[2] != elev.size:
    raise ValueError("Inconsistency between 'f_cor' and 'elev_ang' size")
sin_elev = np.sin(np.deg2rad(elev))

# Compute compressed f_cor array
t_beg = perf_counter()
num_elem = 6 # number of elements in compressed f_cor array
f_cor_comp = np.empty((f_cor.shape[0], 24, num_elem), dtype=np.float32)
f_cor_comp.fill(np.nan) # fill with NaN for safety
# for ind_loc in range(200_000):
for ind_loc in range(f_cor.shape[0]):
    for ind_azim in range(24):
        f_cor_loc = f_cor[ind_loc, ind_azim, :]
        ind_0 = np.where(f_cor_loc == 0.0)[0][-1]
        elev_start = elev[ind_0]
        elev_end = elev[-1]

        # elev_spac = np.linspace(elev_start, elev_end, num_elem + 1)
        # even spacing
        elev_spac = spacing_exp(elev_start, elev_end, num_elem + 1, exp=1.0)
        # spacing exponentially decreasing towards right

        f_cor_nodes = np.interp(x=elev_spac, xp=elev, fp=f_cor_loc)
        f_cor_comp[ind_loc, ind_azim, 0] = elev_start
        f_cor_comp[ind_loc, ind_azim, 1:] = f_cor_nodes[1:-1]
    if ind_loc % 20_000 == 0:
        print(f"Progress: {ind_loc / f_cor.shape[0] * 100:.1f} %")
t_end = perf_counter()
print(f"Compress f_cor: {t_end - t_beg:.1f} s")

# Reshape f_cor array
shp_new = (f_cor_comp.shape[0], f_cor_comp.shape[1] * f_cor_comp.shape[2])
f_cor_comp_reshp = f_cor_comp.reshape(shp_new).transpose()

# -----------------------------------------------------------------------------
# Check compressed and reshaped f_cor array
# -----------------------------------------------------------------------------

# Select location and azimuth angle (Fortran: indices + 1)
# ind_loc, ind_azim = 680_000, 0 # always below 1.0
# ind_loc, ind_azim = 580_000, 0  # up to 6.0
# ind_loc, ind_azim = 777125, 11 # total shadow until ca. ~25 deg
ind_loc, ind_azim = np.random.randint(0, f_cor_comp.shape[0], size=1)[0], np.random.randint(0, 24)

# Retrieve f_cor-values
ind_azim_start = ind_azim * num_elem
ind_azim_end = (ind_azim + 1) * num_elem
f_cor_loc = f_cor_comp_reshp[ind_azim_start:ind_azim_end, ind_loc]
if not np.all(f_cor_loc == f_cor_comp[ind_loc, ind_azim, :]):
    raise ValueError("Incorrect data selected")

# Reconstruct f_cor for specific elevation angle
# elev_sel = 85.0 # [deg]
elev_sel = np.random.uniform(0.0, 25.0, 1)[0] # 50.0, 90.0
elev_start = f_cor_comp_reshp[ind_azim_start, ind_loc]
elev_end = 90.0
exp = 1.0 # constant exponent
pos_norm = (elev_sel - elev_start) / (elev_end - elev_start) # normalised position
if pos_norm <= 0.0:
    f_cor_ip = 0.0
    ind_left, ind_right = None, None
    print("Out of bounds (left)")
elif pos_norm >= 1.0:
    f_cor_ip = 1.0
    ind_left, ind_right = None, None
    print("Out of bounds (right)")
else:
    ind_left = int(pos_norm ** (1.0 / (exp + 1.0)) * num_elem) # pos_norm: [0.0 <, < 1.0]
    ind_right = ind_left + 1
    elev_left = elev_start + (elev_end - elev_start) * (ind_left / num_elem) ** (exp + 1)
    elev_right = elev_start + (elev_end - elev_start) * (ind_right / num_elem) ** (exp + 1)
    if ind_left == 0:
        f_cor_left = 0.0
        f_cor_right = f_cor_comp_reshp[ind_azim_start + ind_right, ind_loc]
        print("Used nodes: first two")
    elif ind_right == num_elem:
        f_cor_left = f_cor_comp_reshp[ind_azim_start + ind_left, ind_loc]
        f_cor_right = 1.0
        print("Used nodes: last two")
    else:
        f_cor_left = f_cor_comp_reshp[ind_azim_start + ind_left, ind_loc]
        f_cor_right = f_cor_comp_reshp[ind_azim_start + ind_right, ind_loc]
    f_cor_ip = f_cor_left + (f_cor_right - f_cor_left) * (elev_sel - elev_left) / (elev_right - elev_left)

# Get interpolation nodes
elev_spac = spacing_exp(elev_start, elev_end, num_elem + 1, exp=1.0)
f_cor_nodes = np.concatenate(([0.0], f_cor_loc[1:], [1.0]))

# Check deviation
f_cor_exact = np.interp(x=elev_sel, xp=elev, fp=f_cor[ind_loc, ind_azim, :])
print(f"Deviation {f_cor_ip - f_cor_exact:.5f} (f_cor_ip - f_cor_exact)")

# Plot
plt.figure()
plt.plot(elev, f_cor[ind_loc, ind_azim, :], color="black", lw=3, alpha=0.5)
if ind_left is not None:
    plt.scatter(elev_spac[[ind_left, ind_right]],
                f_cor_nodes[[ind_left, ind_right]], color="black",
                marker="o", s=150)
plt.scatter(elev_spac, f_cor_nodes, color="red", s=50)
plt.scatter(elev_sel, f_cor_ip, color="green", s=100)
plt.title(f"Location index: {ind_loc}, azimuth index: {ind_azim}")
plt.show()
# plt.savefig("/scratch/mch/csteger/HORAYZON_extpar_subgrid/test.png", dpi=250)
# plt.close()

# -----------------------------------------------------------------------------

# Save compressed f_cor array in EXTPAR file
np.save(path_in_out + file_in[:-3] + "_compressed.npy", f_cor_comp_reshp)



# -----------------------------------------------------------------------------
# Outdated stuff below...
# -----------------------------------------------------------------------------

# # Find triangle indices for specific location
# path_ige = "/store_new/mch/msopr/csteger/Data/Miscellaneous/" \
#     + "ICON_grids_EXTPAR/"
# # icon_grid = "MeteoSwiss/icon_grid_0002_R19B07_mch.nc" # 2km
# icon_grid = "MeteoSwiss/icon_grid_0001_R19B08_mch.nc" # 1km
# # icon_grid = "MeteoSwiss/icon_grid_00005_R19B09_DOM02.nc" # 500m
# ds = xr.open_dataset(path_ige + icon_grid)
# vlon_parent = np.rad2deg(ds["vlon"].values)
# vlat_parent = np.rad2deg(ds["vlat"].values)
# if ds["clon"].size != f_cor.shape[0]:
#     raise ValueError("Inconsistent data loaded")
# vertex_of_cell_parent = ds["vertex_of_cell"].values - 1
# triangles = tri.Triangulation(vlon_parent, vlat_parent,
#                               vertex_of_cell_parent.transpose())
# ds.close()
# tri_finder = triangles.get_trifinder()
# ind_tri = int(tri_finder(7.999089, 46.603755))  # type: ignore # (lon, lat)
# print(ind_tri)

# Loop over triangles
num_elem = 6 # number of elements in compressed f_cor array
num_cell = 100_000
ind_locs = np.random.randint(0, f_cor.shape[0], size=num_cell)
# dev_abs_all = np.empty((num_cell, 24, 3, 2), dtype=np.float32)
# exp_opt_all = np.empty((num_cell, 24), dtype=np.float32)
f_cor_compressed = np.empty((num_cell, 24, num_elem), dtype=np.float32)
for ind_loc_0, ind_loc in enumerate(ind_locs):
    for ind_azim in range(24):

        # # Select specific location and azimuth angle
        # # ind_loc, ind_azim = 680_000, 0 # always below 1.0
        # # ind_loc, ind_azim = 580_000, 0  # up to 6.0
        # # ind_loc, ind_azim = 777125, 11 # total shadow until ca. ~25 deg
        # ind_loc, ind_azim = np.random.randint(0, 1_000_000, size=1)[0], 0
        # ind_loc_0 = 0 # arbitrary index

        f_cor_loc = f_cor[ind_loc, ind_azim, :]

        # Index of last elevation angle that is exactly 0.0
        # ind_0 = 0 # np.all(f_cor[:, :, 0] == 0.0)
        # if f_cor_loc.min() == 0.0:
        ind_0 = np.where(f_cor_loc == 0.0)[0][-1]

        # Start and end of elevation angles
        elev_start = elev[ind_0]
        # print(f"elev_start: {elev_start:.2f} deg")
        elev_end = elev[-1]

        # Even spacing
        # elev_even = np.linspace(elev_start, elev_end, num_elem + 1)

        # Spacing exponentially decreasing towards right (const. exponent)
        elev_expc = spacing_exp(elev_start, elev_end, num_elem + 1, exp=1.0)

        # # Spacing exponentially decreasing towards right (opt. exponent)
        # exponents = np.linspace(0.0, 3.0, 100, dtype=np.float64)
        # # spacing: ~0.03 -> large enough to be stored in uint16 (90/65535)
        # elev_expo, exp_opt = spacing_exp_opt(elev_start, elev_end, num_elem,
        #                                      exponents, elev, f_cor_loc,
        #                                      sin_elev)
        # # print(f"exp_opt: {exp_opt:.2f}")
        # exp_opt_all[ind_loc_0, ind_azim] = exp_opt

        # # Compute deviation (and plot f_cor-reconstruction lines)
        # elev_kinds = {
        #     "even": (elev_even, "red"), 
        #     "exp_const": (elev_expc, "blue"),
        #     # "exp_opt": (elev_expo, "green")
        #     }
        # # ---------------------- plotting -------------------------------------
        # # plt.figure()
        # # plt.plot(elev, f_cor_loc, color="black", lw=3, alpha=0.5)
        # # # plt.scatter(elev, f_cor_loc, color="black", s=40, alpha=0.5)
        # # ---------------------- plotting -------------------------------------
        # for ind, i in enumerate(elev_kinds.keys()):
        #     f_cor_nodes = np.interp(x=elev_kinds[i][0], xp=elev, fp=f_cor_loc)
        #     plt.scatter(elev_kinds[i][0], f_cor_nodes, color=elev_kinds[i][1],
        #                 s=50)
        #     if (f_cor_nodes[0] != 0.0) or (f_cor_nodes[-1] != 1.0):
        #         raise ValueError("Unexpected f_cor_nodes start and/or end")
        #     f_cor_rec = np.interp(x=elev, xp=elev_kinds[i][0], fp=f_cor_nodes)
        #     # ---------------------- plotting ---------------------------------
        #     # plt.scatter(elev, f_cor_rec, color=elev_kinds[i][1], s=25,
        #     #             label=i)
        #     # ---------------------- plotting ---------------------------------
        #     dev_abs = np.abs((f_cor_rec - f_cor_loc) * sin_elev * radiation)
        #     dev_abs_mean = dev_abs.mean()
        #     dev_abs_max = dev_abs.max()
        #     dev_abs_all[ind_loc_0, ind_azim, ind, 0] = dev_abs_mean
        #     dev_abs_all[ind_loc_0, ind_azim, ind, 1] = dev_abs_max
        #     # print(f"Abs. dev. (mean, max): {dev_abs_mean:.3f}, " 
        #     #       + f"{dev_abs_max:.3f} W m-2")
        # # ---------------------- plotting -------------------------------------
        # # plt.legend(frameon=False, fontsize=10)
        # # plt.axis((-2.0, 92.0, -0.05, f_cor_loc.max() * 1.02))
        # # plt.xlabel("Elevation angle [deg]")
        # # plt.ylabel("SW_dir correction factor")
        # # plt.show()
        # # ---------------------- plotting -------------------------------------

        # Save relevant information in 'compressed f_cor array'
        f_cor_nodes = np.interp(x=elev_expc, xp=elev, fp=f_cor_loc)
        f_cor_compressed[ind_loc_0, ind_azim, 0] = elev_start
        f_cor_compressed[ind_loc_0, ind_azim, 1:] = f_cor_nodes[1:-1]

        # # Test plot for one location
        # plt.figure()
        # plt.pcolormesh(azim, elev, f_cor[ind[0], :, :].transpose(),
        #                vmin=0.0, vmax=2.0, cmap="RdBu_r")
        # plt.colorbar()
        # plt.show()

# # Optimal exponent
# plt.figure()
# plt.hist(exp_opt_all.flatten(), bins=20, color="blue", edgecolor="black")
# plt.title(f"Min. = {exp_opt_all.min():.3f}, max. = {exp_opt_all.max():.3f}")
# plt.show()

# plt.figure()
# cols = ["red", "blue", "green"]
# for i in range(3):
#      plt.hist(dev_abs_all[:, :, i, 0].flatten(), bins=20, density=False,
#               color=cols[i], edgecolor="black", alpha=0.3)
# plt.yscale("log")
# plt.show()

# -----------------------------------------------------------------------------
# Test code for Fortran implementation
# -----------------------------------------------------------------------------

elev_start = 33.64 # location & azimuth specific (horizon[0])
elev_end = 90.0 # constant
num = 6 # constant
exp = 1.754 # location & azimuth specific (horizon[1])

elev_exp = spacing_exp(elev_start, elev_end, num, exp=exp)
print("elev_exp:", elev_exp)

f_cor_start = 0.0
f_cor_middle = np.array([0.1, 0.25, 0.45, 0.75]) # location & azimuth specific (horizon[2:6])
f_cor_end = 1.0

elev_t = np.random.uniform(0.0, 90.0, 1)[0]
print("elev_t:", elev_t)

if elev_t <= elev_start:
    f_cor_t = 0.0
else:
    ind_left = int(((elev_t - elev_start) / (elev_end - elev_start)) ** (1.0 / (exp + 1.0)) * (num - 1))
    ind_left = np.maximum(ind_left, 0) # 'safety check' to ensure that ind_left is >= 0
    ind_left = np.minimum(ind_left, num - 2)  # in case of 'elev_t = 90.0'
    elev_left = elev_start + (elev_end - elev_start) * (ind_left / (num - 1)) ** (exp + 1)
    ind_right = ind_left + 1
    elev_right = elev_start + (elev_end - elev_start) * (ind_right / (num - 1)) ** (exp + 1)
    if ind_left == 0:
        f_cor_left = f_cor_start
        f_cor_right = f_cor_middle[0]
    elif ind_right == (num - 1):
        f_cor_left = f_cor_middle[num - 3]
        f_cor_right = f_cor_end
    else:
        f_cor_left = f_cor_middle[ind_left - 1]
        f_cor_right = f_cor_middle[ind_left]
    print(ind_left, elev_left, f_cor_left)
    print(ind_right, elev_right, f_cor_right)
    f_cor_t = f_cor_left + (f_cor_right - f_cor_left) * (elev_t - elev_left) / (elev_right - elev_left)
print("f_cor_t:      ", f_cor_t)

f_cor_all = np.concatenate(([f_cor_start], f_cor_middle, [f_cor_end]))
f_cor_t_check = np.interp(x=elev_t, xp=elev_exp, fp=f_cor_all)
print("f_cor_t_check:", f_cor_t_check)