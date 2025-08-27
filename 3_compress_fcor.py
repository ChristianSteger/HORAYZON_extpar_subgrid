# Description: Compress f_cor data for EXTPAR file.
#
# Author: Christian R. Steger, August 2025

from time import perf_counter
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style
from numba import njit, float64, int64

style.use("classic")

# Paths
path_in_out = "/scratch/mch/csteger/temp/ICON_refined_mesh/"
path_plot = "/scratch/mch/csteger/HORAYZON_extpar_subgrid/plots/"

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
         x_spac[i] = x_start + (x_end - x_start) \
            * (float(i) / float(num - 1)) ** (exp + 1.0)
    x_spac[num - 1] = x_end
    return x_spac

def spacing_exp_interp(x_start, x_end, num, exp, x_ip, y):
    """
    Linear interpolation from exponentially spaced data.
    """
    pos_norm = (x_ip - x_start) / (x_end - x_start)
    if pos_norm <= 0.0:
        print("x-value out of bounds (left)")
        y_ip = 0.0
    elif pos_norm >= 0.9999:
        print("x-value out of bounds (right)")
        y_ip =  1.0
    else:
        ind_left = int((num - 1) * pos_norm ** (1.0 / (exp + 1.0)))
        x_left = x_start + (x_end - x_start) \
            * (float(ind_left) / float(num - 1)) ** (exp + 1.0)
        x_right = x_start + (x_end - x_start) \
            * (float(ind_left + 1) / float(num - 1)) ** (exp + 1.0)
        print("Left index: " + str(ind_left))
        print(f"x_left: {x_left:.4f}, x_ip: {x_ip:.4f}, "
              + f"x_right: {x_right:.4f}")
        weight_left = (x_right - x_ip) / (x_right - x_left)
        y_ip = y[ind_left] * weight_left \
            + y[ind_left + 1] * (1.0 - weight_left)
    return y_ip

# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------

# Create exponentially spaced array
x_start = float(np.random.uniform(0.0, 40.0, 1)[0])
print(f"x_start: {x_start:.4f}")
x_end = 90.0
num = 7
exp = 1.2
x_spac = spacing_exp(x_start, x_end, num, exp)

# Check interpolation
x_ip = np.random.uniform(x_start, x_end, 1)[0]
# x_ip = x_start + 0.0000000000001
# x_ip = x_end # - 0.00000000000001
y = np.random.uniform(0.0, 1.0, num)
y[0], y[-1] = 0.0, 1.0
y_ip = spacing_exp_interp(x_start, x_end, num, exp, x_ip, y)
if abs(y_ip - np.interp(x_ip, x_spac, y, left=0.0, right=1.0)) > 1e-10:
    raise ValueError("Interpolation erroneous")
print(f"y_ip: {y_ip:.4f}")

plt.figure()
plt.scatter(np.linspace(x_start, x_end, num), np.ones_like(x_spac) * 0.98,
            s=50, color="blue", edgecolors="black")
plt.scatter(x_spac, np.ones_like(x_spac) * 1.02,
            s=50, color="red", edgecolors="black")
plt.axis((x_start - 5.0, x_end + 5.0, 0.95, 1.05))
plt.show()
# plt.savefig(path_plot + "check_spacing_exp.jpg", dpi=300)
# plt.close()

###############################################################################
# Compress f_cor information and save to EXTPAR NetCDF file
###############################################################################

# Settings
# icon_res = "2km"
icon_res = "1km"
# icon_res = "500m"
exp_const = 1.2
num_interp_nodes = 7

# Input/output files
file_in = "SW_dir_cor_mch_" + icon_res + ".nc"
file_npy = path_in_out + file_in[:-3] + "_compressed.npy"

# Compress f_cor if not yet done
if not os.path.isfile(file_npy):

    # Load data
    t_beg = perf_counter()
    ds = xr.open_dataset(path_in_out + "SW_dir_cor_all_computed/" + file_in)
    f_cor = ds["f_cor"].values # (num_cell_parent, num_hori, num_elev)
    ds.close()
    t_end = perf_counter()
    print(f"Open NetCDF file: {t_end - t_beg:.1f} s")

    # Check/set range of f_cor-values
    f_cor = f_cor.clip(max=10.0) # set upper limit for f_cor
    if (f_cor.min() < 0.0) and (not np.all(f_cor[:, :, 0] == 0.0)):
        raise ValueError("Unexpected values in 'f_cor'")
    print(f"Min/max f_cor-values for elev = 90.0 deg: "
          + f"{f_cor[:, :, -1].min():.3f}, {f_cor[:, :, -1].max():.3f}")

    # Azimuth and elevation angles
    elev = np.linspace(0.0, 90.0, 91, dtype=np.float64) # [deg]
    if f_cor.shape[2] != elev.size:
        raise ValueError("Inconsistency between 'f_cor' and 'elev_ang' size")

    # Compute compressed f_cor
    t_beg = perf_counter()
    f_cor_comp = np.empty((f_cor.shape[0], 24, num_interp_nodes + 1),
                        dtype=np.float32)
    # first elevation angle and array of f_cor-values for interpolation nodes
    for ind_loc in range(f_cor.shape[0]):
        for ind_azim in range(24):
            f_cor_loc = f_cor[ind_loc, ind_azim, :]
            ind_0 = np.where(f_cor_loc == 0.0)[0][-1]
            elev_start = elev[ind_0]
            elev_end = 90.0 # equal to elev[-1]
            elev_spac = spacing_exp(elev_start, elev_end, num_interp_nodes,
                                    exp=exp_const)
            f_cor_ip = np.interp(x=elev_spac, xp=elev, fp=f_cor_loc)
            f_cor_comp[ind_loc, ind_azim, 0] = elev_start
            f_cor_comp[ind_loc, ind_azim, 1:] = f_cor_ip
    np.save(file_npy, f_cor_comp)
    t_end = perf_counter()
    print(f"Compression of f_cor: {t_end - t_beg:.1f} s")

else:
    print("Data already processed - load from file")
    f_cor_comp = np.load(file_npy)
    if f_cor_comp.shape[2] != (num_interp_nodes + 1):
        raise ValueError("Inconsistency between processed data and " \
                         + "'num_interp_nodes'")

# Reshape f_cor array for EXTPAR
shp_new = (f_cor_comp.shape[0], f_cor_comp.shape[1] * f_cor_comp.shape[2])
f_cor_comp_extpar = f_cor_comp.reshape(shp_new).transpose()

# -----------------------------------------------------------------------------
# Check location
# -----------------------------------------------------------------------------

num_gc = f_cor_comp.shape[0]

# Select location and azimuth direction
# ind_loc, ind_azim = 750_000, 13
# ind_loc, ind_azim = 790_610, 5
# ind_loc, ind_azim = 777125, 11 # total shadow until ca. ~25 deg
# ind_loc, ind_azim = 680_000, 0 # always below 1.0
# ind_loc, ind_azim = 580_000, 0  # up to 6.0
ind_loc, ind_azim = np.random.randint(0, num_gc), np.random.randint(0, 24)

# Interpolate f_cor for sun position
# elev_sun = 7.0
elev_sun = np.random.uniform(0.2, 35.0, 1)[0]
f_cor_exact = np.interp(x=elev_sun, xp=elev, fp=f_cor[ind_loc, ind_azim, :])
print(f"f_cor (exact) = {f_cor_exact:.2f}")
x_start = f_cor_comp[ind_loc, ind_azim, 0]
num_elem = num_interp_nodes + 1
ind_azim_start = num_elem * ind_azim + 1
ind_azim_end = num_elem * (ind_azim + 1)
f_cor_loc = f_cor_comp_extpar[ind_azim_start:ind_azim_end, ind_loc]
if not np.all(f_cor_loc == f_cor_comp[ind_loc, ind_azim, 1:]):
    raise ValueError("Incorrect f_cor-values accessed")
f_cor_approx = spacing_exp_interp(x_start, 90.0, num_interp_nodes, exp_const,
                                  elev_sun, f_cor_loc)
print(f"f_cor (approx) = {f_cor_approx:.2f}")

# Plot
plt.figure()
plt.plot(elev, f_cor[ind_loc, ind_azim, :], color="black", lw=1.5)
elev_start = f_cor_comp[ind_loc, ind_azim, 0]
elev_spac = spacing_exp(elev_start, 90.0, num_interp_nodes, exp=exp_const)
f_cor_nodes = f_cor_comp[ind_loc, ind_azim, 1:]
plt.plot(elev_spac, f_cor_nodes, color="red")
plt.scatter(elev_spac, f_cor_nodes, color="red", s=20)
plt.scatter(elev_sun, f_cor_exact, color="green", s=100)
plt.scatter(elev_sun, f_cor_approx, color="black", s=50)
plt.show()

# -----------------------------------------------------------------------------
# Save to EXTPAR NetCDF file
# -----------------------------------------------------------------------------

# EXTPAR file
path_extpar = "/scratch/mch/csteger/ICON-CH1-EPS_copy/"
file = "external_parameter_icon_grid_0001_R19B08_mch_tuned.nc"

# Write f_cor information into 'HORIZON' field
t_beg = perf_counter()
ds = xr.open_dataset(path_extpar + file)
ds = ds.drop("HORIZON")
ds["HORIZON"] = (("nhori", "cell"), f_cor_comp_extpar)
ds["HORIZON"].attrs["standard_name"] = "-"
ds["HORIZON"].attrs["long_name"] = "horizon angle - topography" # rename?
ds["HORIZON"].attrs["units"] = "deg" # rename?
ds["HORIZON"].attrs["CDI_grid_type"] = "unstructured"
ds["HORIZON"].attrs["data_set"] = "ASTER"
encoding = {"time": {"_FillValue": None},
            "HORIZON": {"_FillValue": -1.e+20, "missing_value": -1.e+20}}
ds.to_netcdf(path_extpar + file[:-3] + f"_f_cor_comp.nc", format="NETCDF4",
             encoding=encoding)
t_end = perf_counter()
print(f"Write 'f_cor' to EXTPAR NetCDF file: {t_end - t_beg:.1f} s")

# -----------------------------------------------------------------------------
# Old stuff below (compute optimised f_cor, etc.)
# -----------------------------------------------------------------------------

# Settings
# check_plots = False
# radiation = 1050.0  # direct beam rad. with 90 deg elevation angle [W m-2]

# azim = np.arange(0.0, 360.0, 15, dtype=np.float64) # [deg]
# if f_cor.shape[1] != azim.size:
#     raise ValueError("Inconsistency between 'f_cor' and 'azim_ang' size")
# sin_elev = np.sin(np.deg2rad(elev))

# @njit
# def spacing_exp_opt(elev_start, elev_end, num, exponents, elev, f_cor_loc,
#                     sin_elev):
#     """
#     Computes spacing between x_start and x_end with exponentially increasing
#     spacing towards the right and select the optimal exponent. The output
#     array starts/ends exactly with x_start/x_end.
#     """
#     dev_abs = np.empty(exponents.size, dtype=np.float64)
#     for j in range(exponents.size):
#         elev_expo = spacing_exp(elev_start, elev_end, num, exp=exponents[j])
#         f_cor_nodes = np.interp(x=elev_expo, xp=elev, fp=f_cor_loc)
#         # extrapolation never occurs
#         f_cor_rec = np.interp(x=elev, xp=elev_expo, fp=f_cor_nodes)
#         # extrapolation can occur but boundary values of [0.0, 1.0] correct
#         dev_abs[j] = np.abs((f_cor_rec - f_cor_loc) * sin_elev).mean()
#     exp_opt = exponents[np.argmin(dev_abs)]
#     elev_expo = spacing_exp(elev_start, elev_end, num, exp=exp_opt)
#     return elev_expo, exp_opt


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

# # Loop over triangles
# num_elem = 6 # number of elements in compressed f_cor array
# num_cell = 100_000
# ind_locs = np.random.randint(0, f_cor.shape[0], size=num_cell)
# # dev_abs_all = np.empty((num_cell, 24, 3, 2), dtype=np.float32)
# # exp_opt_all = np.empty((num_cell, 24), dtype=np.float32)
# f_cor_compressed = np.empty((num_cell, 24, num_elem), dtype=np.float32)
# for ind_loc_0, ind_loc in enumerate(ind_locs):
#     for ind_azim in range(24):

#         # # Select specific location and azimuth angle
#         # # ind_loc, ind_azim = 680_000, 0 # always below 1.0
#         # # ind_loc, ind_azim = 580_000, 0  # up to 6.0
#         # # ind_loc, ind_azim = 777125, 11 # total shadow until ca. ~25 deg
#         # ind_loc, ind_azim = np.random.randint(0, 1_000_000, size=1)[0], 0
#         # ind_loc_0 = 0 # arbitrary index

#         f_cor_loc = f_cor[ind_loc, ind_azim, :]

#         # Index of last elevation angle that is exactly 0.0
#         # ind_0 = 0 # np.all(f_cor[:, :, 0] == 0.0)
#         # if f_cor_loc.min() == 0.0:
#         ind_0 = np.where(f_cor_loc == 0.0)[0][-1]

#         # Start and end of elevation angles
#         elev_start = elev[ind_0]
#         # print(f"elev_start: {elev_start:.2f} deg")
#         elev_end = elev[-1]

#         # Even spacing
#         # elev_even = np.linspace(elev_start, elev_end, num_elem + 1)

#         # Spacing exponentially decreasing towards right (const. exponent)
#         elev_expc = spacing_exp(elev_start, elev_end, num_elem + 1, exp=1.0)

#         # # Spacing exponentially decreasing towards right (opt. exponent)
#         # exponents = np.linspace(0.0, 3.0, 100, dtype=np.float64)
#         # # spacing: ~0.03 -> large enough to be stored in uint16 (90/65535)
#         # elev_expo, exp_opt = spacing_exp_opt(elev_start, elev_end, num_elem,
#         #                                      exponents, elev, f_cor_loc,
#         #                                      sin_elev)
#         # # print(f"exp_opt: {exp_opt:.2f}")
#         # exp_opt_all[ind_loc_0, ind_azim] = exp_opt

#         # # Compute deviation (and plot f_cor-reconstruction lines)
#         # elev_kinds = {
#         #     "even": (elev_even, "red"),
#         #     "exp_const": (elev_expc, "blue"),
#         #     # "exp_opt": (elev_expo, "green")
#         #     }
#         # # ---------------------- plotting -------------------------------------
#         # # plt.figure()
#         # # plt.plot(elev, f_cor_loc, color="black", lw=3, alpha=0.5)
#         # # # plt.scatter(elev, f_cor_loc, color="black", s=40, alpha=0.5)
#         # # ---------------------- plotting -------------------------------------
#         # for ind, i in enumerate(elev_kinds.keys()):
#         #     f_cor_nodes = np.interp(x=elev_kinds[i][0], xp=elev, fp=f_cor_loc)
#         #     plt.scatter(elev_kinds[i][0], f_cor_nodes, color=elev_kinds[i][1],
#         #                 s=50)
#         #     if (f_cor_nodes[0] != 0.0) or (f_cor_nodes[-1] != 1.0):
#         #         raise ValueError("Unexpected f_cor_nodes start and/or end")
#         #     f_cor_rec = np.interp(x=elev, xp=elev_kinds[i][0], fp=f_cor_nodes)
#         #     # ---------------------- plotting ---------------------------------
#         #     # plt.scatter(elev, f_cor_rec, color=elev_kinds[i][1], s=25,
#         #     #             label=i)
#         #     # ---------------------- plotting ---------------------------------
#         #     dev_abs = np.abs((f_cor_rec - f_cor_loc) * sin_elev * radiation)
#         #     dev_abs_mean = dev_abs.mean()
#         #     dev_abs_max = dev_abs.max()
#         #     dev_abs_all[ind_loc_0, ind_azim, ind, 0] = dev_abs_mean
#         #     dev_abs_all[ind_loc_0, ind_azim, ind, 1] = dev_abs_max
#         #     # print(f"Abs. dev. (mean, max): {dev_abs_mean:.3f}, "
#         #     #       + f"{dev_abs_max:.3f} W m-2")
#         # # ---------------------- plotting -------------------------------------
#         # # plt.legend(frameon=False, fontsize=10)
#         # # plt.axis((-2.0, 92.0, -0.05, f_cor_loc.max() * 1.02))
#         # # plt.xlabel("Elevation angle [deg]")
#         # # plt.ylabel("SW_dir correction factor")
#         # # plt.show()
#         # # ---------------------- plotting -------------------------------------

#         # Save relevant information in 'compressed f_cor array'
#         f_cor_nodes = np.interp(x=elev_expc, xp=elev, fp=f_cor_loc)
#         f_cor_compressed[ind_loc_0, ind_azim, 0] = elev_start
#         f_cor_compressed[ind_loc_0, ind_azim, 1:] = f_cor_nodes[1:-1]

#         # # Test plot for one location
#         # plt.figure()
#         # plt.pcolormesh(azim, elev, f_cor[ind[0], :, :].transpose(),
#         #                vmin=0.0, vmax=2.0, cmap="RdBu_r")
#         # plt.colorbar()
#         # plt.show()

# # # Optimal exponent
# # plt.figure()
# # plt.hist(exp_opt_all.flatten(), bins=20, color="blue", edgecolor="black")
# # plt.title(f"Min. = {exp_opt_all.min():.3f}, max. = {exp_opt_all.max():.3f}")
# # plt.show()

# # plt.figure()
# # cols = ["red", "blue", "green"]
# # for i in range(3):
# #      plt.hist(dev_abs_all[:, :, i, 0].flatten(), bins=20, density=False,
# #               color=cols[i], edgecolor="black", alpha=0.3)
# # plt.yscale("log")
# # plt.show()

# -----------------------------------------------------------------------------
# Test code for Fortran implementation
# -----------------------------------------------------------------------------

# elev_start = 33.64 # location & azimuth specific (horizon[0])
# elev_end = 90.0 # constant
# num = 6 # constant
# exp = 1.754 # location & azimuth specific (horizon[1])

# elev_exp = spacing_exp(elev_start, elev_end, num, exp=exp)
# print("elev_exp:", elev_exp)

# f_cor_start = 0.0
# f_cor_middle = np.array([0.1, 0.25, 0.45, 0.75]) # location & azimuth specific (horizon[2:6])
# f_cor_end = 1.0

# elev_t = np.random.uniform(0.0, 90.0, 1)[0]
# print("elev_t:", elev_t)

# if elev_t <= elev_start:
#     f_cor_t = 0.0
# else:
#     ind_left = int(((elev_t - elev_start) / (elev_end - elev_start)) ** (1.0 / (exp + 1.0)) * (num - 1))
#     ind_left = np.maximum(ind_left, 0) # 'safety check' to ensure that ind_left is >= 0
#     ind_left = np.minimum(ind_left, num - 2)  # in case of 'elev_t = 90.0'
#     elev_left = elev_start + (elev_end - elev_start) * (ind_left / (num - 1)) ** (exp + 1)
#     ind_right = ind_left + 1
#     elev_right = elev_start + (elev_end - elev_start) * (ind_right / (num - 1)) ** (exp + 1)
#     if ind_left == 0:
#         f_cor_left = f_cor_start
#         f_cor_right = f_cor_middle[0]
#     elif ind_right == (num - 1):
#         f_cor_left = f_cor_middle[num - 3]
#         f_cor_right = f_cor_end
#     else:
#         f_cor_left = f_cor_middle[ind_left - 1]
#         f_cor_right = f_cor_middle[ind_left]
#     print(ind_left, elev_left, f_cor_left)
#     print(ind_right, elev_right, f_cor_right)
#     f_cor_t = f_cor_left + (f_cor_right - f_cor_left) * (elev_t - elev_left) / (elev_right - elev_left)
# print("f_cor_t:      ", f_cor_t)

# f_cor_all = np.concatenate(([f_cor_start], f_cor_middle, [f_cor_end]))
# f_cor_t_check = np.interp(x=elev_t, xp=elev_exp, fp=f_cor_all)
# print("f_cor_t_check:", f_cor_t_check)
