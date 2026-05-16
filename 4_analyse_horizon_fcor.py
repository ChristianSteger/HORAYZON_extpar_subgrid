# Description: Analyse and compare computed f_cor and terrain horizon
#
# Author: Christian R. Steger, May 2026

import datetime as dt
import json

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri, colors
import matplotlib as mpl
import matplotlib.dates as mdates
from scipy import interpolate
from skyfield.api import load, wgs84

from icon_implement.fortran import interpolate_fcor

style.use("classic")

# Change latex fonts
mpl.rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
mpl.rcParams["mathtext.default"] = "rm"
mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"

# Paths
path_in_out = "/scratch/mch/csteger/temp/ICON_refined_mesh/"
path_icon_grid = "/store_new/mch/msopr/csteger/Data/Miscellaneous/ICON_grids/"
path_plot = "/scratch/mch/csteger/HORAYZON_extpar_subgrid/plots/"

###############################################################################
# Functions
###############################################################################

def resample_average_own(data: xr.DataArray) -> np.ndarray:
    """Resample data array to given time resolution using averaging.

    Parameters
    ----------
    data : xr.DataArray
        Accumulated data array with 'step' or 'lead_time' dimension.

    Returns
    -------
    data_res: np.ndarray
        De-accumulated data array.
    """

    if "step" in data.coords:
        ta_s = data["step"].values / np.timedelta64(1, "s") # [s]
    elif "lead_time" in data.coords:
        ta_s = data["lead_time"].values / np.timedelta64(1, "s") # [s]
    elif "time" in data.coords:
        print("Warning: First time is assumed to be reference time")
        ta_s = (data["time"].values - data["time"].values[0]) \
            / np.timedelta64(1, "s")
    else:
        raise ValueError("Unknown time coordinates in input array")
    data_res = np.diff(data.values * ta_s[:, np.newaxis], axis=0) \
        / np.diff(ta_s)[:, np.newaxis] # [W m-2]
    data_res = np.vstack((np.full((1, data_res.shape[1]), np.nan), data_res))

    return data_res

###############################################################################
# Select parent grid cell and associated child grid cells
###############################################################################

# Settings
icon_dom = "mch_1km"

# Get available parent grid cell indices
file_mesh = f"ICON_refined_mesh_{icon_dom}.nc"
ds = xr.open_dataset(path_in_out + file_mesh)
num_cell_child_per_parent =  int(ds["num_cell_child_per_parent"].values)
ds.close()
file_fcor = f"SW_dir_cor_{icon_dom}_loc_own.nc"
ds = xr.open_dataset(path_in_out + file_fcor)
idx_child = ds["ind_hori_out"].values # index_cell_child
ds.close()
idx_parent = (idx_child[slice(0, None, num_cell_child_per_parent)]
              / num_cell_child_per_parent).astype(int)

# Load locations
file_json = path_in_out + "loc_own.json"
with open(file_json, "r") as f:
    locations = json.load(f)

# Select specific parent grid cell
# 1: Vals
# 2: Piotta (-> radiation only in subgrid-scale cor.) --------------- favourite
# 4: Goeschenen
# 10: Limmeren ------------------------------------------------------ favourite
# 12 Gondo (-> radiation only in subgrid-scale cor.)
# 14 Calancatal_1 --------------------------------------------------- favourite
# 22 Lauterbrunnen_1
# 23 Kandertal_S_fac
idx_cell = 29

# Select indices
idx_parent_sel = idx_parent[idx_cell]
slice_child = slice(num_cell_child_per_parent * idx_cell,
             num_cell_child_per_parent * (idx_cell + 1))

###############################################################################
# Checks
###############################################################################

# -----------------------------------------------------------------------------
# Recompute averaged sub-grid-scale terrain normal
# -----------------------------------------------------------------------------

# Get subgrid correction information
ds = xr.open_dataset(path_in_out + file_fcor)
terrain_normal = ds["terrain_normal"][idx_parent_sel, :].values
ds.close()

# Get child terrain normal vectors
ds = xr.open_dataset(path_in_out + file_fcor)
tri_norms_sg = ds["slope"][slice_child, :].values
ds.close()

# Terrain average normal vector
tri_norm_av = tri_norms_sg.mean(axis=0)
print(np.abs(tri_norm_av - terrain_normal).max()) # check averaged normal

###############################################################################
# Check diurnal cycle of direct beam shortwave radiation
###############################################################################

# Extract and merge direct beam shortwave radiation fluxes for quicker access
# cdo selname,ASWDIR_S,ASWDIR_S_OS,ASWDIR_S_TAN_OS -cat lfff0???0000.nc ASWDIR_S_hourly.nc
# cdo selname,ASWDIR_S,ASWDIR_S_OS,ASWDIR_S_TAN_OS -cat lffm0????000.nc ASWDIR_S_10min.nc

# Load ICON simulation data
path_icon = "/scratch/mch/csteger/alpine_twin/tst/exp/55/" \
    + "wd/24122500_55/lm_coarse/000/"
# file = path_icon + "ASWDIR_S_hourly.nc" # hourly data
file = path_icon + "ASWDIR_S_10min.nc" # 10-min. data
data_icon = {}
var_sel = ("ASWDIR_S", "ASWDIR_S_OS", "ASWDIR_S_TAN_OS")
ds = xr.open_dataset(file)
slice_cell = slice(idx_parent_sel, idx_parent_sel + 1)
# keep second axis for function 'resample_average_own'
for var in var_sel:
    data_icon[var] = resample_average_own(ds[var][:, slice_cell])
time_axis = ds["time"].values
ds.close()
time_axis_dt = [dt.datetime.strptime(str(i)[:19], "%Y-%m-%dT%H:%M:%S")
                .replace(tzinfo=dt.timezone.utc) for i in time_axis]

# Load ICON grid coordinates
file_grid = "MeteoSwiss/icon_grid_0001_R19B08_mch.nc"
ds = xr.open_dataset(path_icon_grid + file_grid)
clon = np.rad2deg(ds["clon"].values[idx_parent_sel]) # [deg]
clat = np.rad2deg(ds["clat"].values[idx_parent_sel]) # [deg]
ds.close()

# Compute sun position for relevant times
planets = load("de421.bsp")
sun = planets["sun"]
earth = planets["earth"]
loc_obs = earth + wgs84.latlon(clat, clon)
sun_azim = np.empty(time_axis.size)
sun_elev = np.empty(time_axis.size)
ts = load.timescale()
for idx_i, ta in enumerate(time_axis_dt):
    t = ts.from_datetime(ta)
    astrometric = loc_obs.at(t).observe(sun)
    alt, az, d = astrometric.apparent().altaz()
    sun_azim[idx_i] = az.degrees
    sun_elev[idx_i] = alt.degrees

# Compute direct beam shortwave correction on sub-grid scale (full fcor data)
file_in = f"SW_dir_cor_{icon_dom}.nc"
ds = xr.open_dataset(path_in_out + file_in)
f_cor_loc = ds["f_cor"][idx_parent_sel, :, :].values # (24, 91)
ds.close()
azim = np.linspace(0.0, 360.0, 25) # cyclic, [deg]
elev = np.linspace(0.0, 90.0, 91) # [deg]
f_cor_loc_cyc = np.vstack((f_cor_loc, f_cor_loc[0:1, :])) # (25, 91)
f_ip = interpolate.RegularGridInterpolator((azim, elev), f_cor_loc_cyc,
                                           bounds_error=False, fill_value=0.0)
f_cor_ip_dense = f_ip(np.vstack((sun_azim, sun_elev)).transpose())

# Compute direct beam shortwave correction on sub-grid scale (compressed data)
file = "/scratch/mch/csteger/ICON-CH1-EPS_copy_inn/" \
    + "external_parameter_icon_grid_0001_R19B08_mch_tuned_horizon_subgrid.nc"
ds = xr.open_dataset(file)
horizon = ds["HORIZON"][:, idx_parent_sel].values # (72)
swdir_cor = ds["SWDIR_COR"][:, idx_parent_sel].values # (96)
terrain_normal = ds["TERRAIN_NORMAL"][:, idx_parent_sel].values # (3)
ds.close()
f_cor_sparse_extpar = ds["HORIZON"][:, idx_parent_sel].values
ds.close()
f_cor_ip_sparse = np.zeros_like(f_cor_ip_dense)
for i in range(f_cor_ip_sparse.size):
    zphi_sun = np.deg2rad(sun_azim[i])
    ztheta_sun = np.deg2rad(sun_elev[i])
    f_cor_ip_sparse[i] = interpolate_fcor(
        horizon, swdir_cor, terrain_normal, ztheta_sun, zphi_sun)

# Get horizon information
ds = xr.open_dataset(file)
horizon = ds["HORIZON"][:, idx_parent_sel].values.reshape(24, 3) # (24, 3)
ds.close()
ds = xr.open_dataset(path_in_out + file_fcor)
horizon_sg = ds["horizon"][slice_child, :].values # (1369, 24)
ds.close()

# # Check subgrid-horizon-distribution with histogram
# idx_azim = 16
# frac_illum_3 = np.array([0.0, 0.5, 1.0])
# plt.figure()
# plt.hist(horizon_sg[:, idx_azim], bins=100, cumulative=True, density=True)
# plt.plot(horizon[idx_azim, :], frac_illum_3, color="red")
# plt.scatter(horizon[idx_azim, :], frac_illum_3, color="red", s=50)
# plt.show()

# Compute shadowing on sub-grid scale
num_azim = 24
azim_spac = float(360.0 / num_azim)
frac_illum = np.empty(sun_azim.size) # subgrid illumination fraction
for i in range(sun_azim.size):

    # Azimuth indices and interpolation weights
    idx_0 = np.minimum(num_azim - 1, int(sun_azim[i] / azim_spac))
    idx_1 = np.mod(idx_0 + 1, num_azim)
    weight_0 = (azim_spac*(idx_0 + 1) - sun_azim[i]) / azim_spac
    weight_1 = 1.0 - weight_0

    # Binary shadow according to median horizon -------------------------------
    horizon_ip = horizon[idx_0, 1] * weight_0 + horizon[idx_1, 1] * weight_1
    frac_illum[i] = float(horizon_ip < sun_elev[i])
    # Fractional shadow -------------------------------------------------------
    # frac_illum_0 = np.interp(sun_elev[i], horizon[idx_0, :], frac_illum_3)
    # frac_illum_1 = np.interp(sun_elev[i], horizon[idx_1, :], frac_illum_3)
    # frac_illum[i] = frac_illum_1 * weight_1 + frac_illum_0 * weight_0
    # -> fractional shadow can also be computed from the full subgrid horizon
    #    information 'horizon_sg' to check how accurate this information is...
    # -------------------------------------------------------------------------
aswdir_s_os_sg = data_icon["ASWDIR_S"].squeeze() * frac_illum

# Compute correction separately for shadow and slope
q = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
horizon_perc = np.percentile(horizon_sg, q=q, axis=0) # (5, 24)
frac_illuminated = q / 100.0
f_cor_ip_sep = np.zeros_like(f_cor_ip_dense)
h_vec = np.array([0.0, 0.0, 1.0])
for i in range(sun_azim.size):
    if sun_elev[i] > 1.0:

        # Azimuth indices and interpolation weights
        idx_0 = np.minimum(num_azim - 1, int(sun_azim[i] / azim_spac))
        idx_1 = np.mod(idx_0 + 1, num_azim)
        weight_0 = (azim_spac*(idx_0 + 1) - sun_azim[i]) / azim_spac
        weight_1 = 1.0 - weight_0

        # Get shadow fractions for two azimuth directions and interpolate
        frac_illum_0 = np.interp(sun_elev[i], horizon_perc[:, idx_0],
                                 frac_illuminated)
        frac_illum_1 = np.interp(sun_elev[i], horizon_perc[:, idx_1],
                                 frac_illuminated)
        frac_illum = frac_illum_0 * weight_0 + frac_illum_1 * weight_1

        sun_elev_rad = np.deg2rad(sun_elev[i])
        sun_azim_rad = np.deg2rad(sun_azim[i])
        sun_vec = np.array([np.cos(sun_elev_rad) * np.sin(sun_azim_rad),
                            np.cos(sun_elev_rad) * np.cos(sun_azim_rad),
                            np.sin(sun_elev_rad)])
        dot_ts = np.dot(tri_norm_av, sun_vec)
        if dot_ts > 0.0:
            f_cor_ip_sep[i] = (1.0 / np.dot(sun_vec, h_vec).clip(min=1e-5)) \
                    * dot_ts * frac_illum
f_cor_ip_sep = f_cor_ip_sep.clip(min=0.0, max=10.0)
aswdir_s_tan_os_sep = data_icon["ASWDIR_S"].squeeze() * f_cor_ip_sep

# Plot diurnal cycle of direct shortwave radiation for different corrections
lw = 2.0
plt.figure(figsize=(8.5, 6.0))
ax = plt.axes()
data_max = np.array([], dtype=np.float32)
# ---------- Uncorrected flux -------------------------------------------------
plt.plot(time_axis, data_icon["ASWDIR_S"].squeeze(), label="ASWDIR_S",
         lw=1.5, color="black")
data_max = np.append(data_max, data_icon["ASWDIR_S"].squeeze())
# ---------- Shadow correction ------------------------------------------------
plt.plot(time_axis, data_icon["ASWDIR_S_OS"].squeeze(),
         label="ASWDIR_S_OS (grid-scale)",
         lw=lw, color="green")
plt.plot(time_axis, aswdir_s_os_sg,
         label="ASWDIR_S_OS (subgrid)",
         lw=lw, ls="--", color="limegreen")
# ---------- Shadow & slope correction ----------------------------------------
plt.plot(time_axis, data_icon["ASWDIR_S_TAN_OS"].squeeze(),
         label="ASWDIR_S_TAN_OS (grid-scale)",
         lw=lw, color="blue")
data_max = np.append(data_max, data_icon["ASWDIR_S_TAN_OS"].squeeze())
plt.plot(time_axis, data_icon["ASWDIR_S"].squeeze() * f_cor_ip_dense,
         label="ASWDIR_S_TAN_OS (subgrid; dense)",
         lw=lw, ls=":", color="cornflowerblue")
data_max = np.append(data_max,
                     data_icon["ASWDIR_S"].squeeze() * f_cor_ip_dense)
plt.plot(time_axis, data_icon["ASWDIR_S"].squeeze() * f_cor_ip_sparse,
         label="ASWDIR_S_TAN_OS (subgrid; sparse)",
         lw=lw, ls="--", color="mediumorchid")
# ---------- Shadow & slope correction (separate) -----------------------------
plt.plot(time_axis, aswdir_s_tan_os_sep,
         label="ASWDIR_S_TAN_OS (subgrid; separate)",
         lw=lw, ls="--", color="red")
# -----------------------------------------------------------------------------
plt.legend(frameon=False, fontsize=9)
plt.xlabel("Time (UTC)")
plt.ylabel(r"Direct beam shortwave radiation [W m$^{-2}$]")
plt.title(f"Grid cell: {locations[idx_cell][0]}", loc="left", fontsize=11)
plt.title(time_axis_dt[0].strftime("%Y-%m-%d"), loc="right", fontsize=11)
plt.xlim(time_axis[35], time_axis[-12])
plt.ylim((-5.0, np.nanmax(data_max) * 1.05))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# plt.show()
plt.savefig(path_plot + f"diurnal_cycle_{locations[idx_cell][0]}.jpg",
            dpi=300, bbox_inches="tight")
plt.close()

###############################################################################
# Compare subgrid f-cor with (sub-)gird terrain horizon of grid cell
###############################################################################

# Load information from SW_dir_cor computation
ds = xr.open_dataset(path_in_out + file_fcor)
horizon_child = ds["horizon"].values # (num_hori_out, num_hori)
f_cor = ds["f_cor"][idx_parent, :, :].values # (num_hori_out, num_hori)
ds.close()

# Load terrain horizon (grid-scale cell)
file_extpar = "/scratch/mch/csteger/ICON-CH1-EPS_copy_inn/" \
    + "external_parameter_icon_grid_0001_R19B08_mch_tuned.nc"
ds = xr.open_dataset(file_extpar)
horizon_grid_scale = ds["HORIZON"].values[:, idx_parent]
ds.close()

# Select location
idx_loc = 1 # (0, 1, 2, 3) (Vicosoprano, Vals, Piotta, Cevio)

# Compute subgrid-scale horizon statistics
slice_loc = slice(idx_loc * num_cell_child_per_parent,
                  (idx_loc + 1) * num_cell_child_per_parent)
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
plt.pcolormesh(azim, elev, f_cor[idx_loc, :, :].transpose(), shading="auto",
               cmap=cmap, norm=norm)
cbar = plt.colorbar(pad=0.03)
cbar.set_label(r"Subgrid SW$_{dir}$ correction factor [-]", labelpad=8)
for i in range(5):
    plt.plot(azim, horizon_perc[i, :], color="grey", lw=1.0)
plt.plot(azim, horizon_grid_scale[:, idx_loc], color="black", linewidth=2.5)
plt.xlabel("Azimuth angle (clockwise from North) [deg]")
plt.ylabel("Elevation angle [deg]")
plt.axis((-8.0, 352.0, 0.0, 90.0))
plt.title(f"Grid cell: {locations[idx_loc][0]}", loc="left", fontsize=11)
# plt.show()
plt.savefig(path_plot + f"f_cor_vs_sub_grid_horizon_"
            + f"{locations[idx_loc][0]}.jpg", dpi=300, bbox_inches="tight")
plt.close()

###############################################################################
# Compare subgrid-horizon with one from MCH weather station
###############################################################################

# Load mesh data
ds = xr.open_dataset(path_in_out + file_mesh)
vlon = np.rad2deg(ds["vlon"].values)
vlat = np.rad2deg(ds["vlat"].values)
faces = ds["faces"][idx_child, :].values
ds.close()
triangles = tri.Triangulation(vlon, vlat, faces)
tri_finder = triangles.get_trifinder()

# Select location
idx_loc = 3 # (0, 1, 2, 3) (Vicosoprano, Vals, Piotta, Cevio)
idx_tri = int(tri_finder(*locations[idx_loc][1])) # type: ignore

# Compute sun position for specific day
planets = load("de421.bsp")
sun = planets["sun"]
earth = planets["earth"]
loc_lon, loc_lat = locations[idx_loc][1]
loc_obs = earth + wgs84.latlon(loc_lat, loc_lon)
time_axis_dt = [dt.datetime(2025, 9, 1, 4, tzinfo=dt.timezone.utc)
                + dt.timedelta(minutes=5 * i) for i in range(170)]
sun_azim = np.empty(len(time_axis_dt))
sun_elev = np.empty(len(time_axis_dt))
ts = load.timescale()
for idx_i, ta in enumerate(time_axis_dt):
    t = ts.from_datetime(ta)
    astrometric = loc_obs.at(t).observe(sun)
    alt, az, d = astrometric.apparent().altaz()
    sun_azim[idx_i] = az.degrees
    sun_elev[idx_i] = alt.degrees

# Plot for location
plt.figure(figsize=(11.0, 5.5))
for i in range(num_cell_child_per_parent):
    l_sg, = plt.plot(azim,
             horizon_child[idx_loc * num_cell_child_per_parent + i, :],
             color="grey", alpha=0.5)
l_sg_station, = plt.plot(azim, horizon_child[idx_tri, :], color="red",
                         alpha=1.0, lw=1.5)
l_g, = plt.plot(azim, horizon_grid_scale[:, idx_loc], color="black",
                linewidth=2.0)
plt.plot(sun_azim, sun_elev, color="darkorange", ls="--", lw=2.0)
plt.xlabel("Azimuth angle (clockwise from North) [deg]")
plt.ylabel("Elevation angle [deg]")
plt.title(f"Grid cell: {locations[idx_loc][0]}", loc="left", fontsize=11)
plt.title(f"Sun path: {time_axis_dt[0].strftime("%Y-%m-%d")}", loc="right",
          fontsize=11, color="darkorange")
plt.legend([l_sg, l_sg_station, l_g],
           ["Subgrid horizons", "Subgrid horizon (MeteoSwiss station)",
            "Grid-scale horizon"],
            frameon=False, fontsize=9, loc="upper left")
plt.axis((0.0 - 2.0, 345.0 + 2.0, 0.0, 70.0))
# plt.show()
plt.savefig(path_plot + f"subgrid_horizon_station_{locations[idx_loc][0]}.jpg",
            dpi=300, bbox_inches="tight")
plt.close()
