# Description: Analyse and compare computed f_cor and terrain horizon
#
# Author: Christian R. Steger, August 2025

import glob
import datetime as dt
import json

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri, colors
from netCDF4 import Dataset
from scipy.linalg import solve
from scipy import interpolate
from skyfield.api import load, wgs84

style.use("classic")

# Paths
path_in_out = "/scratch/mch/csteger/temp/ICON_refined_mesh/"
path_ige = "/store_new/mch/msopr/csteger/Data/Miscellaneous/" \
    + "ICON_grids_EXTPAR/"
path_plot = "/scratch/mch/csteger/HORAYZON_extpar_subgrid/plots/"

###############################################################################
# Function
###############################################################################

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

###############################################################################
# Compare diurnal cycle of SW_dir
###############################################################################

# Settings
# icon_res = "2km"
icon_res = "1km"
# icon_res = "500m"

# Get available parent grid cell indices
file_mesh = f"ICON_refined_mesh_mch_{icon_res}.nc"
ds = xr.open_dataset(path_in_out + file_mesh)
num_cell_child_per_parent =  int(ds["num_cell_child_per_parent"].values)
ds.close()
file_fcor = f"SW_dir_cor_mch_{icon_res}.nc"
ds = xr.open_dataset(path_in_out + file_fcor)
ind_child = ds["ind_hori_out"].values # index_cell_child
ds.close()
ind_parent = (ind_child[slice(0, None, num_cell_child_per_parent)]
              / num_cell_child_per_parent).astype(int)

# Load locations
file_json = path_in_out + f"locations_sel_{icon_res}.json"
with open(file_json, "r") as f:
    locations = json.load(f)

# Select specific parent grid cell
# 1: Vals
# 2: Piotta (-> radiation only in subgrid-scale cor.) --------------- favourite
# 4: Goeschenen
# 5: Grono
# 10: Limmeren ------------------------------------------------------ favourite
# 12 Gondo (-> radiation only in subgrid-scale cor.)
# 14 Calancatal_1 --------------------------------------------------- favourite
# 23 Lauterbrunnen_1
# 24 Kandertal_S_fac
ind_loc = 14
ind_parent_sel = ind_parent[ind_loc]

# -----------------------------------------------------------------------------
# Uncorrected and grid-scale corrected SW_dir
# -----------------------------------------------------------------------------

# Load uncorrected SW_dir
# module load cdo/2.0.5-gcc
# ls lffm*0.nc | wc -l
# cdo cat -select,name=ASWDIR_S lffm*0.nc ASWDIR_S.nc
path = "/scratch/mch/csteger/wd/24122500_74/lm_coarse/000/"
ds = xr.open_dataset(path + "ASWDIR_S.nc")
time_axis = ds["time"].values # time (UTC)
seconds = (time_axis - time_axis[0]) / (10 ** 9) # seconds since start
sw_dir = ds["ASWDIR_S"][:, ind_parent_sel].values # cumulative values!
sw_dir_uncor = np.diff(sw_dir * seconds) / np.diff(seconds) # [W m-2]
ds.close()

# Load grid-scaled corrected SW_dir
# path = "/scratch/mch/csteger/wd/24122500_73/lm_coarse/000/"  # not 1/cos(slope)
path = "/scratch/mch/csteger/wd/24122500_77/lm_coarse/000/" # 1/cos(slope)
ds = xr.open_dataset(path + "ASWDIR_S.nc")
sw_dir = ds["ASWDIR_S"][:, ind_parent_sel].values # cumulative values!
sw_dir_gs_cor = np.diff(sw_dir * seconds) / np.diff(seconds) # [W m-2]
ds.close()
time_axis = time_axis[:-1] + np.diff(time_axis) / 2.0

# -----------------------------------------------------------------------------
# Subgrid corrected SW_dir (from f_cor ray-tracing data)
# -----------------------------------------------------------------------------

# Compute solar azimuth and elevation angle for time_axis
file_grid = "MeteoSwiss/icon_grid_0001_R19B08_mch.nc"
ds = xr.open_dataset(path_ige + file_grid)
clon = np.rad2deg(ds["clon"].values[ind_parent_sel]) # [deg]
clat = np.rad2deg(ds["clat"].values[ind_parent_sel]) # [deg]
ds.close()
planets = load("de421.bsp")
sun = planets["sun"]
earth = planets["earth"]
loc_obs = earth + wgs84.latlon(clat, clon)
time_axis_dt = [dt.datetime.strptime(str(i)[:19], "%Y-%m-%dT%H:%M:%S")
                .replace(tzinfo=dt.timezone.utc) for i in time_axis]
sun_azim = np.empty(time_axis.size)
sun_elev = np.empty(time_axis.size)
ts = load.timescale()
for ind_i, ta in enumerate(time_axis_dt):
    t = ts.from_datetime(ta)
    astrometric = loc_obs.at(t).observe(sun)
    alt, az, d = astrometric.apparent().altaz()
    sun_azim[ind_i] = az.degrees
    sun_elev[ind_i] = alt.degrees

# Compute interpolated f_cor values from array 'f_cor'
file_in = f"SW_dir_cor_mch_{icon_res}.nc"
ds = xr.open_dataset(path_in_out + file_in)
f_cor_loc = ds["f_cor"][ind_parent_sel, :, :].values # (24, 91)
ds.close()
azim = np.linspace(0.0, 360.0, 25) # cyclic, [deg]
elev = np.linspace(0.0, 90.0, 91) # [deg]
f_cor_loc_cyc = np.vstack((f_cor_loc, f_cor_loc[0:1, :])) # (25, 91)
f_ip = interpolate.RegularGridInterpolator((azim, elev), f_cor_loc_cyc,
                                           bounds_error=False, fill_value=0.0)
f_cor_ip = f_ip(np.vstack((sun_azim, sun_elev)).transpose())

# Compute interpolated f_cor values from array 'f_cor_comp'
exp_const = 1.2
num_interp_nodes = 7
x_end = 90.0
file_npy = path_in_out + file_in[:-3] + "_compressed.npy"
f_cor_comp = np.load(file_npy)[ind_parent_sel, :, :] # (24, 8)
f_cor_ip_comp = np.zeros_like(f_cor_ip)
for i in range(f_cor_ip_comp.size):
    if sun_elev[i] > 0.0:
            # Left f_cor ------------------------------------------------------
            ind_azim_left = int(sun_azim[i] / 15.0)
            x_start = f_cor_comp[ind_azim_left, 0]
            f_cor_left = temp = spacing_exp_interp(
                x_start, x_end, num_interp_nodes, exp_const,
                sun_elev[i], f_cor_comp[ind_azim_left, 1:])
            # Right f_cor -----------------------------------------------------
            ind_azim_right = ind_azim_left + 1 # currently not wrapping around!
            x_start = f_cor_comp[ind_azim_right, 0]
            f_cor_right = temp = spacing_exp_interp(
                x_start, x_end, num_interp_nodes, exp_const,
                sun_elev[i], f_cor_comp[ind_azim_right, 1:])
            # -----------------------------------------------------------------
            azim_left = ind_azim_left * 15.0
            weight_right = (sun_azim[i] - azim_left) / 15.0
            f_cor_ip_comp[i] = (1.0 - weight_right) * f_cor_left \
                + weight_right * f_cor_right

# -----------------------------------------------------------------------------
# Subgrid correction based on separate spatial aggregation of terrain slope
# and terrain horizon
# -----------------------------------------------------------------------------

# Child grid information
file_mesh = f"ICON_refined_mesh_mch_{icon_res}.nc"
ds = xr.open_dataset(path_in_out + file_mesh)
num_cell_child_per_parent = int(ds["num_cell_child_per_parent"])
ds.close()

# Load subgrid terrain slope and horizon
ds = xr.open_dataset(path_in_out + file_in)
slice_sg = slice(ind_loc * num_cell_child_per_parent,
                 (ind_loc + 1) * num_cell_child_per_parent)
horizon = ds["horizon"][slice_sg, :].values # (24, 91)
slope = ds["slope"][slice_sg, :].values # (24, 91)
ds.close()

# Compute horizon percentiles and average terrain slope
q = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
horizon_perc = np.percentile(horizon, q=q, axis=0)
terrain_norm = slope.sum(axis=0)
terrain_norm = terrain_norm / np.linalg.norm(terrain_norm)
slope = np.arccos(terrain_norm[2].clip(max=1.0))
aspect = np.pi / 2.0 - np.arctan2(terrain_norm[1], terrain_norm[0])
print(f"Mean slope: {np.rad2deg(slope):.2f} deg, "
      + f"mean aspect: {np.rad2deg(aspect):.2f} deg")

# # Test plot
# plt.figure()
# for i in range(num_cell_child_per_parent):
#     plt.plot(horizon[i, :], color="gray", lw=0.5)
# for i in range(5):
#     plt.plot(horizon_perc[i, :], color="red", lw=0.5)
# plt.show()

# Compute f_cor
horizontal_norm = np.array([0.0, 0.0, 1.0])
horizon_perc_cyc = np.hstack((horizon_perc, horizon_perc[:, 0:1])) # (5, 25)
f_ip = interpolate.interp1d(azim, horizon_perc_cyc, axis=1)
frac_illuminated = q / 100.0
f_cor_ip_sep = np.zeros_like(f_cor_ip)
for i in range(f_cor_ip_comp.size):
    if sun_elev[i] > 1.0:
        sun = np.array([np.cos(np.deg2rad(sun_elev[i]))
                        * np.sin(np.deg2rad(sun_azim[i])),
                        np.cos(np.deg2rad(sun_elev[i]))
                        * np.cos(np.deg2rad(sun_azim[i])),
                        np.sin(np.deg2rad(sun_elev[i]))])
        dot_ts = np.dot(terrain_norm, sun)
        if dot_ts > 0.0:
            horizon_perc_azim = f_ip(sun_azim[i])
            mask_shadow = np.interp(sun_elev[i], horizon_perc_azim,
                                    frac_illuminated)
            f_cor_ip_sep[i] = (1.0 / np.dot(horizontal_norm, sun)) \
                * (1.0 / np.dot(horizontal_norm, terrain_norm)) \
                    * dot_ts * mask_shadow
f_cor_ip_sep = f_cor_ip_sep.clip(max=10)

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

lw = 2.0
plt.figure(figsize=(8.5, 6.0))
plt.plot(time_axis, sw_dir_uncor, label="Uncorrected", color="black", lw=lw)
plt.plot(time_axis, sw_dir_gs_cor, label="Cor. (grid-scale)",
         color="red", lw=lw)
plt.plot(time_axis, sw_dir_uncor * f_cor_ip,
         label="Cor. (subgrid-scale; full)", color="blue", lw=lw)
plt.plot(time_axis, sw_dir_uncor * f_cor_ip_comp,
         label="Cor. (subgrid-scale; comp.)", color="deepskyblue",
         lw=lw, ls="--")

plt.plot(time_axis, sw_dir_uncor * f_cor_ip_sep,
         label="Cor. (subgrid-scale; sep.)", color="violet",
         lw=lw, ls="--")

plt.legend(frameon=False, fontsize=10)
plt.xlabel("Time (UTC)")
plt.ylabel("Direct beam shortwave radiation [W m-2]")
plt.title(f"Grid cell: {locations[ind_loc][0]}", loc="left", fontsize=11)
plt.title(time_axis_dt[0].strftime("%Y-%m-%d"), loc="right", fontsize=11)
plt.xlim(time_axis[30], time_axis[-31])
# plt.ylim([-5.0, 450.0]) # Limmeren
# plt.show()
plt.savefig(path_plot + f"diurnal_cycle_{locations[ind_loc][0]}.jpg",
            dpi=300, bbox_inches="tight")
plt.close()

###############################################################################
# Compare subgrid f-cor with (sub-)gird terrain horizon of grid cell
###############################################################################

# 1km
file_mesh = "ICON_refined_mesh_mch_1km.nc"
file_hori_fcor = "SW_dir_cor_" + "mch_" + icon_res + ".nc"
file_extpar = "MeteoSwiss/extpar_grid_shift_topo/" \
    + "extpar_icon_grid_0001_R19B08_mch.nc"
file_grid = "MeteoSwiss/icon_grid_0001_R19B08_mch.nc"

# Get initial information
ds = xr.open_dataset(path_in_out + file_hori_fcor)
ind_hori_out = ds["ind_hori_out"].values # index_cell_child
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
horizon_grid_scale = ds["HORIZON"].values[:, ind_cell_parent]
ds.close()

# Select location
ind_loc = 1 # (0, 1, 2, 3) (Vicosoprano, Vals, Piotta, Cevio)

# Compute subgrid-scale horizon statistics
slice_loc = slice(ind_loc * num_cell_child_per_parent,
                  (ind_loc + 1) * num_cell_child_per_parent)
q = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
horizon_perc = np.percentile(horizon_child[slice_loc, :], q=q, axis=0)

# Colormap
levels = np.arange(0.0, 2.1, 0.1)
cmap = plt.get_cmap("RdBu_r")
norm = colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")

# Plot for location
azim = np.arange(0.0, 360.0, 360 // horizon_child.shape[1])
elev = np.linspace(0.0, 90.0, 91)
plt.figure(figsize=(10, 5))
plt.pcolormesh(azim, elev, f_cor[ind_loc, :, :].transpose(), shading="auto",
               cmap=cmap, norm=norm)
cbar = plt.colorbar(pad=0.03)
cbar.set_label("Subgrid SW_dir correction factor [-]", labelpad=8)
for i in range(5):
    plt.plot(azim, horizon_perc[i, :], color="grey", lw=1.0)
plt.plot(azim, horizon_grid_scale[:, ind_loc], color="black", linewidth=2.5)
plt.xlabel("Azimuth angle (clockwise from North) [deg]")
plt.ylabel("Elevation angle [deg]")
plt.axis((-8.0, 352.0, 0.0, 90.0))
plt.title(f"Grid cell: {locations[ind_loc][0]}", loc="left", fontsize=11)
# plt.show()
plt.savefig(path_plot + f"f_cor_vs_sub_grid_horizon_"
            + f"{locations[ind_loc][0]}.jpg", dpi=300, bbox_inches="tight")
plt.close()

###############################################################################
# Compare subgrid-horizon with one from MCH weather station
###############################################################################

# Load mesh data
ds = xr.open_dataset(path_in_out + file_mesh)
vlon = np.rad2deg(ds["vlon"].values)
vlat = np.rad2deg(ds["vlat"].values)
faces = ds["faces"][ind_hori_out, :].values
ds.close()
triangles = tri.Triangulation(vlon, vlat, faces)
tri_finder = triangles.get_trifinder()

# Select location
ind_loc = 3 # (0, 1, 2, 3) (Vicosoprano, Vals, Piotta, Cevio)
ind_tri = int(tri_finder(*locations[ind_loc][1])) # type: ignore

# Plot for location
plt.figure(figsize=(10, 5))
for i in range(num_cell_child_per_parent):
    plt.plot(azim,
             horizon_child[ind_loc * num_cell_child_per_parent + i, :],
             color="grey", alpha=0.5)
plt.plot(azim, horizon_child[ind_tri, :], color="red", alpha=1.0, lw=1.0)
plt.plot(azim, horizon_grid_scale[:, ind_loc], color="black", linewidth=2.5)
plt.xlabel("Azimuth angle (clockwise from North) [deg]")
plt.ylabel("Elevation angle [deg]")
plt.title(f"Grid cell: {locations[ind_loc][0]}", loc="left", fontsize=11)
# plt.show()
plt.savefig(path_plot + f"subgrid_horizon_station_{locations[ind_loc][0]}.jpg",
            dpi=300, bbox_inches="tight")
plt.close()
