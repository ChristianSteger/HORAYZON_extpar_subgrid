# Description: Analyse and compare computed f_cor and terrain horizon
#
# Author: Christian R. Steger, September 2025

import datetime as dt
import json

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri, colors
import matplotlib as mpl
from scipy import interpolate
from skyfield.api import load, wgs84

from functions.icon_implement import interpolate_fcor # type: ignore

style.use("classic")

# Change latex fonts
mpl.rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
mpl.rcParams["mathtext.default"] = "rm"
mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"

# Paths
path_in_out = "/scratch/mch/csteger/temp/ICON_refined_mesh/"
path_ige = "/store_new/mch/msopr/csteger/Data/Miscellaneous/" \
    + "ICON_grids_EXTPAR/"
path_plot = "/scratch/mch/csteger/HORAYZON_extpar_subgrid/plots/"

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

# Compute interpolated f_cor values from dense 'f_cor'-array
file_in = f"SW_dir_cor_mch_{icon_res}.nc"
ds = xr.open_dataset(path_in_out + file_in)
f_cor_loc = ds["f_cor"][ind_parent_sel, :, :].values # (24, 91)
ds.close()
azim = np.linspace(0.0, 360.0, 25) # cyclic, [deg]
elev = np.linspace(0.0, 90.0, 91) # [deg]
f_cor_loc_cyc = np.vstack((f_cor_loc, f_cor_loc[0:1, :])) # (25, 91)
f_ip = interpolate.RegularGridInterpolator((azim, elev), f_cor_loc_cyc,
                                           bounds_error=False, fill_value=0.0)
f_cor_ip_dense = f_ip(np.vstack((sun_azim, sun_elev)).transpose())

# Interpolated f_cor values from array 'f_cor_sparse' (EXTPAR array format)
file = "/scratch/mch/csteger/ICON-CH1-EPS_copy/" \
    + "external_parameter_icon_grid_0001_R19B08_mch_tuned_f_cor_sparse.nc"
ds = xr.open_dataset(file)
num_gc_icon = ds["cell"].size
f_cor_sparse_extpar = ds["HORIZON"][:, ind_parent_sel].values
ds.close()
f_cor_ip_sparse = np.zeros_like(f_cor_ip_dense)
for i in range(f_cor_ip_sparse.size):
    zphi_sun = np.deg2rad(sun_azim[i])
    ztheta_sun = np.deg2rad(sun_elev[i])
    f_cor_ip_sparse[i] = interpolate_fcor(f_cor_sparse_extpar,
                                          ztheta_sun, zphi_sun)

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
f_cor_ip_sep = np.zeros_like(f_cor_ip_dense)
for i in range(f_cor_ip_sep.size):
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
plt.plot(time_axis, sw_dir_gs_cor, label="Cor. (grid)",
         color="red", lw=lw)
plt.plot(time_axis, sw_dir_uncor * f_cor_ip_dense,
         label="Cor. (subgrid; dense)", color="blue", lw=lw)
plt.plot(time_axis, sw_dir_uncor * f_cor_ip_sparse,
         label="Cor. (subgrid; sparse)", color="deepskyblue",
         lw=lw, ls="--")

plt.plot(time_axis, sw_dir_uncor * f_cor_ip_sep,
         label="Cor. (subgrid; separate)", color="violet",
         lw=lw, ls="--")

plt.legend(frameon=False, fontsize=10)
plt.xlabel("Time (UTC)")
plt.ylabel(r"Direct beam shortwave radiation [W m$^{-2}$]")
plt.title(f"Grid cell: {locations[ind_loc][0]}", loc="left", fontsize=11)
plt.title(time_axis_dt[0].strftime("%Y-%m-%d"), loc="right", fontsize=11)
plt.xlim(time_axis[35], time_axis[-36])
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
plt.figure(figsize=(11.0, 5.5))
plt.pcolormesh(azim, elev, f_cor[ind_loc, :, :].transpose(), shading="auto",
               cmap=cmap, norm=norm)
cbar = plt.colorbar(pad=0.03)
cbar.set_label(r"Subgrid SW$_{dir}$ correction factor [-]", labelpad=8)
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

# Compute sun position for specific day
planets = load("de421.bsp")
sun = planets["sun"]
earth = planets["earth"]
loc_lon, loc_lat = locations[ind_loc][1]
loc_obs = earth + wgs84.latlon(loc_lat, loc_lon)
time_axis_dt = [dt.datetime(2025, 9, 1, 4, tzinfo=dt.timezone.utc)
                + dt.timedelta(minutes=5 * i) for i in range(170)]
sun_azim = np.empty(len(time_axis_dt))
sun_elev = np.empty(len(time_axis_dt))
ts = load.timescale()
for ind_i, ta in enumerate(time_axis_dt):
    t = ts.from_datetime(ta)
    astrometric = loc_obs.at(t).observe(sun)
    alt, az, d = astrometric.apparent().altaz()
    sun_azim[ind_i] = az.degrees
    sun_elev[ind_i] = alt.degrees

# Plot for location
plt.figure(figsize=(11.0, 5.5))
for i in range(num_cell_child_per_parent):
    l_sg, = plt.plot(azim,
             horizon_child[ind_loc * num_cell_child_per_parent + i, :],
             color="grey", alpha=0.5)
l_sg_station, = plt.plot(azim, horizon_child[ind_tri, :], color="red", alpha=1.0, lw=1.5)
l_g, = plt.plot(azim, horizon_grid_scale[:, ind_loc], color="black", linewidth=2.0)
plt.plot(sun_azim, sun_elev, color="darkorange", ls="--", lw=2.0)
plt.xlabel("Azimuth angle (clockwise from North) [deg]")
plt.ylabel("Elevation angle [deg]")
plt.title(f"Grid cell: {locations[ind_loc][0]}", loc="left", fontsize=11)
plt.title(f"Sun path: {time_axis_dt[0].strftime("%Y-%m-%d")}", loc="right",
          fontsize=11, color="darkorange")
plt.legend([l_sg, l_sg_station, l_g],
           ["Subgrid horizons", "Subgrid horizon (MeteoSwiss station)",
            "Grid-scale horizon"],
            frameon=False, fontsize=9, loc="upper left")
plt.axis((0.0 - 2.0, 345.0 + 2.0, 0.0, 70.0))
# plt.show()
plt.savefig(path_plot + f"subgrid_horizon_station_{locations[ind_loc][0]}.jpg",
            dpi=300, bbox_inches="tight")
plt.close()
