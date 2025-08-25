# Description: Analyse and compare computed f_cor and terrain horizon
#
# Author: Christian R. Steger, June 2025

import glob
import datetime as dt

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri, colors
from netCDF4 import Dataset
from scipy.linalg import solve
from scipy import interpolate
from skyfield.api import load, wgs84


from functions import centroid_values

style.use("classic")

# Paths
path_in_out = "/scratch/mch/csteger/temp/ICON_refined_mesh/"
path_ige = "/store_new/mch/msopr/csteger/Data/Miscellaneous/" \
    + "ICON_grids_EXTPAR/"

###############################################################################
# Compare diurnal cycle of SW_dir
###############################################################################

# Get available parent grid cell indices
file_mesh = "ICON_refined_mesh_mch_1km.nc"
ds = xr.open_dataset(path_in_out + file_mesh)
num_cell_child_per_parent =  int(ds["num_cell_child_per_parent"].values)
ds.close()
file_fcor_hori = "SW_dir_cor_mch_1km.nc"
ds = xr.open_dataset(path_in_out + file_fcor_hori)
ind_child = ds["ind_hori_out"].values # index_cell_child
ds.close()
ind_parent = (ind_child[slice(0, None, num_cell_child_per_parent)]
              / num_cell_child_per_parent).astype(int)

# Select specific parent grid cell
ind_sel = 23
ind_parent_sel = ind_parent[ind_sel]
# -> interesting stations:
# 1: Vals
# 2: Piotta (-> no radiation only in grid-scale cor.) --------------- favourite
# 4: Goeschenen
# 5: Grono
# 10: Limmeren ------------------------------------------------------ favourite
# 12 Gondo (-> no radiation only in grid-scale cor.)
# 14 Calancatal_1 --------------------------------------------------- favourite
# 23 Lauterbrunnen_1 (-> strange looking grid-scale cor.) ----------- favourite
# 24 Kandertal_S_fac

# -----------------------------------------------------------------------------
# Uncorrected and grid-scale corrected SW_dir
# -----------------------------------------------------------------------------

# Load uncorrected SW_dir
# module load cdo/2.0.5-gcc
# ls lffm*0.nc | wc -l
# cdo cat -select,name=ASWDIR_S lffm*0.nc ASWDIR_S.nc
path = "/scratch/mch/csteger/wd/24122500_63/lm_coarse/000/"
ds = xr.open_dataset(path + "ASWDIR_S.nc")
time_axis = ds["time"].values # time (UTC)
seconds = (time_axis - time_axis[0]) / (10 ** 9) # seconds since start
sw_dir = ds["ASWDIR_S"].values[:, ind_parent_sel] # cumulative values!
sw_dir_uncor = (sw_dir[1:] * seconds[1:] - sw_dir[:-1] * seconds[:-1]) / np.diff(seconds) # [W m-2]
ds.close()

# Load grid-scaled corrected SW_dir
path = "/scratch/mch/csteger/wd/24122500_64/lm_coarse/000/"
ds = xr.open_dataset(path + "ASWDIR_S.nc")
sw_dir = ds["ASWDIR_S"].values[:, ind_parent_sel] # cumulative values!
sw_dir_gs_cor = (sw_dir[1:] * seconds[1:] - sw_dir[:-1] * seconds[:-1]) / np.diff(seconds) # [W m-2]
ds.close()
time_axis = time_axis[:-1] + np.diff(time_axis) / 2.0

# Drop certain time steps at start/end
sel_ta = slice(6 * 5, -(6 * 5))
time_axis = time_axis[sel_ta]
sw_dir_uncor = sw_dir_uncor[sel_ta]
sw_dir_gs_cor = sw_dir_gs_cor[sel_ta]

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
file_fcor_hori = "SW_dir_cor_all_computed/SW_dir_cor_mch_1km.nc"
ds = xr.open_dataset(path_in_out + file_fcor_hori)
f_cor = ds["f_cor"][ind_parent_sel, :, :].values # (24, 91)
ds.close()
azim = np.arange(0.0, 360.0, 15.0) # [deg]
elev = np.linspace(0.0, 90.0, 91) # [deg]
f_ip = interpolate.RectBivariateSpline(elev, azim, f_cor.transpose(),
                                       kx=1, ky=1)
f_cor_ip_large = np.empty(time_axis.size)
for i in range(len(sun_azim)):
    f_cor_ip_large[i] = f_ip(sun_elev[i], sun_azim[i])[0][0]

# Compute interpolated f_cor values from array 'f_cor_comp'
file_fcor_comp = "SW_dir_cor_all_computed/SW_dir_cor_mch_1km_compressed.npy"
f_cor_comp = np.load(path_in_out + file_fcor_comp)[:, ind_parent_sel] # (144)
f_cor_comp = f_cor_comp.reshape((24, 6))
f_cor_ip_comp = np.empty(time_axis.size)
num_elem = 6
for i in range(len(sun_azim)):
    ind_azim_left = int(sun_azim[i] / 15.0)
    ind_azim_right = ind_azim_left + 1
    # print(azim[ind_azim_left], sun_azim[i], azim[ind_azim_right])
    ind_azim = [ind_azim_left, ind_azim_right]
    elev_sel = sun_elev[i]
    f_cor_ip_lr = np.empty(2)
    for j in range(2):
        elev_start = f_cor_comp[ind_azim[j], 0]
        elev_end = 90.0
        exp = 1.0 # constant exponent
        pos_norm = (elev_sel - elev_start) / (elev_end - elev_start) # normalised position
        if pos_norm <= 0.0:
            f_cor_ip = 0.0
            ind_left, ind_right = None, None
        elif pos_norm >= 1.0:
            f_cor_ip = 1.0
            ind_left, ind_right = None, None
        else:
            ind_left = int(pos_norm ** (1.0 / (exp + 1.0)) * num_elem) # pos_norm: [0.0 <, < 1.0]
            ind_right = ind_left + 1
            elev_left = elev_start + (elev_end - elev_start) * (ind_left / num_elem) ** (exp + 1)
            elev_right = elev_start + (elev_end - elev_start) * (ind_right / num_elem) ** (exp + 1)
            if ind_left == 0:
                f_cor_left = 0.0
                f_cor_right = f_cor_comp[ind_azim[j], ind_right]
            elif ind_right == num_elem:
                f_cor_left = f_cor_comp[ind_azim[j], ind_left]
                f_cor_right = 1.0
            else:
                f_cor_left = f_cor_comp[ind_azim[j], ind_left]
                f_cor_right = f_cor_comp[ind_azim[j], ind_right]
            f_cor_ip = f_cor_left + (f_cor_right - f_cor_left) * (elev_sel - elev_left) / (elev_right - elev_left)
        f_cor_ip_lr[j] = f_cor_ip
    weight_right = (sun_azim[i] - azim[ind_azim_left]) / 15.0
    weight_left = 1.0 - weight_right
    # print(weight_left, weight_right)
    f_cor_ip_comp[i] = (weight_left * f_cor_ip_lr[0] + weight_right * f_cor_ip_lr[1])

# -----------------------------------------------------------------------------
# Subgrid corrected SW_dir (from normal vector and horizon ray-tracing data)
# -----------------------------------------------------------------------------

# # Load data
# file_fcor_hori = "SW_dir_cor_mch_1km.nc"
# ds = xr.open_dataset(path_in_out + file_fcor_hori)
# slic = slice(ind_sel * num_cell_child_per_parent,
#              (ind_sel + 1) * num_cell_child_per_parent)
# horizon = ds["horizon"].values[slic, :]
# slope = ds["slope"].values[slic, :]
# ds.close()

# # Test plot
# plt.figure()
# for i in range(1369):
#     plt.plot(horizon[i, :], color="grey", alpha=0.5)
# #plt.show()
# plt.savefig("/scratch/mch/csteger/HORAYZON_extpar_subgrid/ztemp.png",
#             dpi=250)
# plt.close()

# # Compute f_cor (without shadow mask)
# terrain_slope = slope.mean(axis=0)
# terrain_slope = terrain_slope / np.linalg.norm(terrain_slope) # (t-vector)
# surf_norm = np.array([0.0, 0.0, 1.0]) # (h-vector)
# sun_vector = np.empty((time_axis.size, 3))
# sun_vector[:, 0] = np.cos(np.deg2rad(sun_elev)) * np.sin(np.deg2rad(sun_azim))
# sun_vector[:, 1] = np.cos(np.deg2rad(sun_elev)) * np.cos(np.deg2rad(sun_azim))
# sun_vector[:, 2] = np.sin(np.deg2rad(sun_elev))

# dot_prod_hs = np.dot(sun_vector, surf_norm)
# dot_prod_ts = np.dot(sun_vector, terrain_slope)

# f_cor = np.zeros(time_axis.size)
# mask = (sun_elev > 0.0) & (dot_prod_ts > 0.0)
# f_cor[mask] = (1.0 / dot_prod_hs[mask]) * (1.0 / np.dot(surf_norm, terrain_slope)) * dot_prod_ts[mask]
# f_cor = f_cor.clip(max=10.0)

# -----------------------------------------------------------------------------

lw = 2.0
plt.figure()
plt.plot(time_axis, sw_dir_uncor, label="Uncorrected", color="black", lw=lw)
plt.plot(time_axis, sw_dir_gs_cor, label="Cor. (gs)", color="blue", lw=lw)
plt.plot(time_axis, sw_dir_uncor * f_cor_ip_large, label="Cor. (sgs; full)", color="salmon", lw=lw)
plt.plot(time_axis, sw_dir_uncor * f_cor_ip_comp, label="Cor. (sgs; compressed)", color="red", lw=lw, ls="--")
plt.legend(frameon=False, fontsize=10)
plt.xlabel("Time (UTC)")
plt.ylabel("Direct beam shortwave radiation [W m-2]")
# plt.title("Piotta (Ticino)", loc="left", fontsize=11)
# plt.title("Limmeren (Glarus)", loc="left", fontsize=11)
# plt.title("Val Calanca (Grisons)", loc="left", fontsize=11)
plt.title("Lauterbrunnen (Bern)", loc="left", fontsize=11)
plt.title(time_axis_dt[0].strftime("%Y-%m-%d"), loc="right", fontsize=11)
# plt.ylim([-5.0, 280.0]) # Piotta
# plt.ylim([-5.0, 450.0]) # Limmeren
# plt.ylim([-5.0, 300.0]) # Val Calanca
plt.ylim([-5.0, 300.0]) # Lauterbrunnen
# plt.show()
plt.savefig("/scratch/mch/csteger/HORAYZON_extpar_subgrid/Diurnal_cycle.png",
            dpi=250, bbox_inches="tight")
plt.close()

###############################################################################
# Compare terrain horizon
###############################################################################

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------

# # 1km
# icon_res = "1km"
# file_mesh = "ICON_refined_mesh_mch_1km.nc"
# file_hori_fcor = "SW_dir_cor_" + "mch_" + icon_res + ".nc"
# file_extpar = "MeteoSwiss/extpar_grid_shift_topo/" \
#     + "extpar_icon_grid_0001_R19B08_mch.nc"
# file_grid = "MeteoSwiss/icon_grid_0001_R19B08_mch.nc"

# 500m
icon_res = "500m"
file_mesh = "ICON_refined_mesh_mch_500m.nc"
file_hori_fcor = "SW_dir_cor_" + "mch_" + icon_res + ".nc"
file_extpar = "MeteoSwiss/extpar_grid_shift_topo/" \
    + "extpar_icon_grid_00005_R19B09_DOM02.nc"
file_grid = "MeteoSwiss/icon_grid_00005_R19B09_DOM02.nc"

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
slope_child = ds["slope"].values # (num_hori_out, 3)
f_cor = ds["f_cor"][ind_cell_parent, :, :].values # (num_hori_out, num_hori)
ds.close()

# -----------------------------------------------------------------------------
# Compute slope angle and aspect for specific ICON grid cell
# -----------------------------------------------------------------------------
ind_sel = 8
ind_cell = ind_cell_parent[ind_sel]
ds = xr.open_dataset(path_ige + file_grid)
neighbor_cell_index = ds["neighbor_cell_index"].values[:, ind_cell] - 1
ind_cell_rel = np.append(ind_cell, neighbor_cell_index) # first: centre
clon = ds["clon"].values[ind_cell_rel] # [rad]
clat = ds["clat"].values[ind_cell_rel] # [rad]
ds.close()
ds = xr.open_dataset(path_ige + file_extpar)
topography_c = ds["topography_c"].values[ind_cell_rel] # [m]
ds.close()
rad_earth = 6371229.0 # ICON/COSMO earth radius [m]
x_ecef = (rad_earth + topography_c) * np.cos(clat) * np.cos(clon)
y_ecef = (rad_earth + topography_c) * np.cos(clat) * np.sin(clon)
z_ecef = (rad_earth + topography_c) * np.sin(clat)
x_ecef_orig = x_ecef[0]
y_ecef_orig = y_ecef[0]
z_ecef_orig = z_ecef[0]
sin_lon = np.sin(clon[0])
cos_lon = np.cos(clon[0])
sin_lat = np.sin(clat[0])
cos_lat = np.cos(clat[0])
x_enu = - sin_lon * (x_ecef - x_ecef_orig) \
    + cos_lon * (y_ecef - y_ecef_orig)
y_enu = - sin_lat * cos_lon * (x_ecef - x_ecef_orig) \
    - sin_lat * sin_lon * (y_ecef - y_ecef_orig) \
        + cos_lat * (z_ecef - z_ecef_orig)
z_enu = + cos_lat * cos_lon * (x_ecef - x_ecef_orig) \
    + cos_lat * sin_lon * (y_ecef - y_ecef_orig) \
        + sin_lat * (z_ecef - z_ecef_orig)
points = np.array([x_enu, y_enu, z_enu]).transpose()  # [m]
A = points
b = np.ones(4)
ATA = A.T @ A
ATb = A.T @ b
plane_coeffs = solve(ATA, ATb)
surface_normal = plane_coeffs / np.linalg.norm(plane_coeffs)
if surface_normal[2] < 0:
    surface_normal = -surface_normal  # Ensure normal points upwards
slope = np.arccos(surface_normal[2].clip(max=1.0))
aspect = np.pi / 2.0 - np.arctan2(surface_normal[1], surface_normal[0])
if aspect < 0.0:
    aspect += 2 * np.pi
print(f"Slope angle: {np.rad2deg(slope):.2f} deg")
print(f"Aspect angle: {np.rad2deg(aspect):.2f} deg")
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

# Temporary: check subgrid terrain surface normal vectors
i = ind_sel  # 0 - 3: MeteoSwiss stations,
# 4: south-facing slope, 5: north-facing slope
slic = slice(i * num_cell_child_per_parent,
             (i + 1) * num_cell_child_per_parent)
terrain_normals = slope_child[slic, :]
slope_angle = np.arccos(terrain_normals[:, 2].clip(max=1.0))
slope_aspect = np.pi / 2.0 - np.arctan2(terrain_normals[:, 1],
                                         terrain_normals[:, 0])

terrain_normal_av = terrain_normals.sum(axis=0)
terrain_normal_av = terrain_normal_av / np.linalg.norm(terrain_normal_av)
slope_angle_av = np.arccos(terrain_normal_av[2].clip(max=1.0))
slope_aspect_av = np.pi / 2.0 - np.arctan2(terrain_normal_av[1],
                                           terrain_normal_av[0])

fig = plt.figure()
ax = fig.add_subplot(projection="polar")
ax.set_theta_zero_location("N") # type: ignore
ax.set_theta_direction(-1) # type: ignore
ax.scatter(slope_aspect, np.rad2deg(slope_angle), c="grey",
           alpha=0.5)
ax.scatter(slope_aspect_av, np.rad2deg(slope_angle_av), c="red",
           s=50)
ax.scatter(aspect, np.rad2deg(slope), c="green",
           s=50)
# plt.show()
fig.savefig("/scratch/mch/csteger/HORAYZON_extpar_subgrid/"
            + "terrain_normals_" + icon_res + ".png", dpi=250)
plt.close()

# -----------------------------------------------------------------------------

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
