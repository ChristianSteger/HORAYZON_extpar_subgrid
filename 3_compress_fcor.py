# Description: Compress f_cor data for EXTPAR file.
#
# Author: Christian R. Steger, May 2026

from time import perf_counter

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from functions.fcor_processing import spacing_exp, spacing_exp_interp
from functions.fcor_processing import compute_fcor_sparse
from functions.fcor_processing import dev_bins_default, dev_bins_with_tn

style.use("classic")

# Change latex fonts
mpl.rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
mpl.rcParams["mathtext.default"] = "rm"
mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"

# Paths
path_in_out = "/scratch/mch/csteger/temp/ICON_refined_mesh/"
path_plot = "/scratch/mch/csteger/HORAYZON_extpar_subgrid/plots/"
path_icon_grid = "/store_new/mch/msopr/csteger/Data/Miscellaneous/ICON_grids/"

###############################################################################
# Settings and load data
###############################################################################

# Settings
# ------------ Test 2km ------------
# icon_dom = "test_2km"
# path_extpar = path_icon_grid + "test/"
# file_extpar = "external_parameter_icon_d2_PR444.nc"
# icon_grid = "test/icon_grid_DOM01.nc"
# ------------ MCH 2km ------------
# icon_dom = "mch_2km"
# path_extpar = "/scratch/mch/csteger/ICON-CH2-EPS_copy_inn/"
# file_extpar = "external_parameter_icon_grid_0002_R19B07_mch_tuned.nc"
# icon_grid = "MeteoSwiss/icon_grid_0002_R19B07_mch.nc"
# ------------ MCH 1km ------------
# icon_dom = "mch_1km"
# path_extpar = "/scratch/mch/csteger/ICON-CH1-EPS_copy_inn/"
# file_extpar = "external_parameter_icon_grid_0001_R19B08_mch_tuned.nc"
# icon_grid = "MeteoSwiss/icon_grid_0001_R19B08_mch.nc"
# ------------ MCH 500m -----------
icon_dom = "mch_500m"
path_extpar = "/store_new/mch/msopr/glori/glori-ch500-nested/grid/"
file_extpar = "extpar_icon_grid_00005_R19B09_DOM02.nc"
icon_grid = "MeteoSwiss/icon_grid_00005_R19B09_DOM02.nc"
# ---------------------------------
file_cor = f"SW_dir_cor_{icon_dom}.nc"

# Miscellaneous
check_data = True
compute_error_stat = True

###############################################################################
# Check data (optional)
###############################################################################

if check_data:

    # -------------------------------------------------------------------------
    # Check averaged terrain normals
    # -------------------------------------------------------------------------

    # Load EXTPAR data
    ds = xr.open_dataset(path_extpar + file_extpar)
    topo = ds["topography_c"].values
    ds.close()

    # Load ICON grid data
    ds = xr.open_dataset(path_icon_grid + icon_grid)
    vlon = np.rad2deg(ds["vlon"].values)
    vlat = np.rad2deg(ds["vlat"].values)
    vertex_of_cell = ds["vertex_of_cell"].values - 1
    ds.close()

    # Compute slope angle and aspect
    ds = xr.open_dataset(path_in_out + file_cor)
    terrain_normal = ds["terrain_normal"].values
    ds.close()
    terrain_normal /= np.linalg.norm(terrain_normal, axis=1, keepdims=True)
    slope = np.rad2deg(np.arccos(terrain_normal[:, 2]))
    aspect = np.rad2deg(np.pi / 2.0 - np.arctan2(terrain_normal[:, 1],
                                                 terrain_normal[:, 0]))
    aspect[aspect < 0.0] += 360.0

    # Plot
    triangles = tri.Triangulation(vlon, vlat, vertex_of_cell.transpose())
    map_extent = (vlon.min() - 0.05, vlon.max() + 0.05,
                  vlat.min() - 0.05, vlat.max() + 0.05)
    plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 3, left=0.1, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.2, hspace=0.1)
    # ------------------------
    ax = plt.subplot(gs[0])
    plt.tripcolor(triangles, topo, cmap="terrain", vmin=0.0, vmax=3500.0)
    plt.axis(map_extent)
    plt.colorbar(orientation="horizontal")
    # ------------------------
    ax = plt.subplot(gs[1])
    plt.tripcolor(triangles, slope, cmap="YlOrRd", vmin=0.0, vmax=25.0)
    plt.axis(map_extent)
    plt.colorbar(orientation="horizontal")
    # ------------------------
    ax = plt.subplot(gs[2])
    plt.tripcolor(triangles, aspect, cmap="twilight", vmin=0.0, vmax=360.0)
    plt.axis(map_extent)
    plt.colorbar(orientation="horizontal")
    # ------------------------
    plt.show()

    # -------------------------------------------------------------------------
    # Check shadow angles and reconstructed f_cor-values
    # -------------------------------------------------------------------------

    # Load data
    t_beg = perf_counter()
    ds = xr.open_dataset(path_in_out + file_cor)
    f_cor_dense = ds["f_cor"].values # (num_cell_parent, num_hori, num_elev)
    shadow_angle_idx = ds["shadow_angle_idx"].values
    terrain_normal = ds["terrain_normal"].values
    ds.close()
    t_end = perf_counter()
    print(f"Open NetCDF file: {t_end - t_beg:.1f} s")

    # Check range of shadow_angle indices
    print(f"Range of shadow_angle_idx indices:"
          f" {shadow_angle_idx.min()} - {shadow_angle_idx.max()}")
    print((shadow_angle_idx > 75).sum())
    # -> values above ca. 75 deg are suspicious -> probably due to artefacts
    #    in ASTER DEM...

    # Arrays with elevation and azimuth angle
    azim = np.arange(0.0, 360.0, 15, dtype=np.float32)
    elev_dense = np.linspace(0.0, 90.0, 91, dtype=np.float32)

    # Estimates for direct-beam radiation to compute errors
    rad_zenith = 900.0 # [W m-2]
    radiation = np.sin(np.deg2rad(elev_dense)) * rad_zenith

    # Select location and azimuth angle
    # --------------------------------------------
    # idx_cell, idx_azim = 9208, 0
    # --------------------------------------------
    idx_cell = np.random.randint(0, 18416, 1)[0] # random
    idx_azim = np.random.randint(0, 24, 1)[0] # random
    # --------------------------------------------

    # Check f_cor-value at 90 deg
    f_cor_sel =  f_cor_dense[idx_cell, idx_azim, :]
    print("Slope angle: {:.2f} deg".format(slope[idx_cell]))
    print(f"f-cor-value @ elevation angle of 90 deg: {f_cor_sel[-1]:.3f}")
    # if the (1.0 / cos(slope)) term is not considered for f-cor computation,
    # than the f-cor-value at 90 deg is only 1.0 for a horizontal surface!

    # Recompute f_cor from average terrain normal
    h_vec = np.array([0.0, 0.0, 1.0])
    s_vec = np.empty((91, 3), dtype=np.float32)
    s_vec[:, 0] = np.cos(np.deg2rad(elev_dense)) \
        * np.sin(np.deg2rad(azim[idx_azim]))
    s_vec[:, 1] = np.cos(np.deg2rad(elev_dense)) \
        * np.cos(np.deg2rad(azim[idx_azim]))
    s_vec[:, 2] = np.sin(np.deg2rad(elev_dense))
    t_vec = terrain_normal[idx_cell, :]
    f_cor_tn = np.zeros(len(elev_dense))
    f_cor_tn[1:] = (1.0 / np.dot(s_vec[1:, :], h_vec)) \
        * np.dot(s_vec[1:, :], t_vec) # [1:] -> avoid division by zero

    # Check deviations in reconstructed f_cor (total shadow and illumination)
    idx_start = int(shadow_angle_idx[idx_cell, idx_azim, 0])
    idx_end = int(shadow_angle_idx[idx_cell, idx_azim, 2])
    diff_abs = np.array([], dtype=np.float32)
    diff_abs = np.append(diff_abs, np.abs(f_cor_sel[:(idx_start + 1)] - 0.0))
    diff_abs = np.append(diff_abs,
                         np.abs(f_cor_tn[idx_end:] - f_cor_sel[idx_end:]))
    print(f"Max. abs. difference in f_cor (total shadow/illumination zone):"
          f" {diff_abs.max():.8f}")

    # Define number of nodes
    num_nodes = 7
    # total 8 array elements required: (7 - 2 = 5) f_cor-values, 3 elev. angles

    # Recompute f_cor (old method)
    elev_nodes_old = spacing_exp(elev_dense[idx_start], 90.0, num_nodes,
                                 eta=2.0)
    f_cor_nodes_old = np.interp(elev_nodes_old, elev_dense, f_cor_sel)
    f_cor_rec_old = np.interp(elev_dense, elev_nodes_old, f_cor_nodes_old)
    dev = (np.abs(f_cor_rec_old - f_cor_sel) * radiation)[:66] # only to 65 deg
    print("Ab. error [W m-2]:   mean    max")
    print(f"Old method:          {dev.mean():.3f}   {dev.max():.3f}")

    # Recompute f_cor (new method; with additionally using terrain normal)
    elev_nodes_new = spacing_exp(elev_dense[idx_start], elev_dense[idx_end],
                                 num_nodes, eta=2.0)
    f_cor_nodes_new = np.interp(elev_nodes_new, elev_dense, f_cor_sel)
    f_cor_rec = np.zeros(91)
    f_cor_rec[(idx_start + 1):idx_end] \
        = np.interp(elev_dense[(idx_start + 1):idx_end], elev_nodes_new,
                    f_cor_nodes_new)
    f_cor_rec[idx_end:] = f_cor_tn[idx_end:]
    dev = (np.abs(f_cor_rec - f_cor_sel) * radiation)[:66] # only to 65 deg
    print(f"New method:          {dev.mean():.3f}   {dev.max():.3f}")

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(elev_dense, f_cor_dense[idx_cell, idx_azim, :], color="blue",
             lw=1.5)
    plt.plot(elev_dense, f_cor_rec, color="red", lw=1.5, ls="--")
    # -------------------------------------------------------------------------
    plt.plot(elev_dense, f_cor_tn, color="red", lw=1.5, ls=":")
    # # -----------------------------------------------------------------------
    plt.plot(elev_dense, f_cor_rec_old, color="gray", lw=1.5, ls="--")
    plt.scatter(elev_nodes_old, f_cor_nodes_old, s=50, color="gray",
                marker="d")
    # -------------------------------------------------------------------------
    plt.plot(elev_dense, f_cor_rec, color="green", lw=1.5, ls="--")
    plt.scatter(elev_nodes_new, f_cor_nodes_new, s=50, color="green")
    # -------------------------------------------------------------------------
    f_cor_max = f_cor_dense[idx_cell, idx_azim, :].max() * 1.05
    plt.fill_between(x=[-2.0, elev_dense[idx_start]], y1=-0.1, y2=f_cor_max,
                     color="darkgrey", alpha=0.5)
    plt.fill_between(x=[elev_dense[idx_end], 92.0], y1=-0.1, y2=f_cor_max,
                     color="gold", alpha=0.4)
    plt.vlines(x=shadow_angle_idx[idx_cell, idx_azim, 1], ymin=-0.1,
               ymax=f_cor_max, color="black", linewidth=1.5, linestyle="-")
    plt.xticks(range(0, 91, 10))
    plt.xlabel("Elevation angle [deg]")
    plt.ylabel("Correction factor (f_cor) [-]")
    plt.axis((-2.0, 92.0, -0.1, f_cor_max))
    plt.show()
    # plt.savefig(path_plot + f"elev_f_cor_example_3.jpg",
    #             dpi=300, bbox_inches="tight")
    # plt.close()

###############################################################################
# Compute error statistics (optional) and save relevant data for
# EXTPAR file / ICON simulation
###############################################################################

# Load data
t_beg = perf_counter()
ds = xr.open_dataset(path_in_out + file_cor)
f_cor_dense = ds["f_cor"].values # (num_cell_parent, num_hori, num_elev)
shadow_angle_idx = ds["shadow_angle_idx"].values
terrain_normal = ds["terrain_normal"].values
ds.close()
t_end = perf_counter()
print(f"Open NetCDF file: {t_end - t_beg:.1f} s")

# Set upper limit of f_cor-values and check general range
print(f"Maximal f-cor-value: {f_cor_dense.max():.2f}")
f_cor_dense = f_cor_dense.clip(max=10.0) # set upper limit for f_cor
if ((f_cor_dense.min() < 0.0)
    or (not np.all(f_cor_dense[:, :, 0] == 0.0))):
    raise ValueError("Unexpected values in 'f_cor'")

# Azimuth and elevation angles
elev_dense = np.linspace(0.0, 90.0, 91, dtype=np.float32) # [deg]
if f_cor_dense.shape[2] != elev_dense.size:
    raise ValueError("Inconsistency between 'f_cor' and 'elev_ang' size")

# Settings
num_nodes = 7
# total 8 array elements required: (7 - 2 = 5) f_cor-values, 3 elev. angles

# -----------------------------------------------------------------------------

if compute_error_stat:

    # Settings
    eta_range = np.arange(1.0, 4.5, 0.5, dtype=np.float32)
    bin_size = 1_000_000
    scaling = 100
    rad_zenith = 900.0 # rather lower end because direct beam radiation at
    # low solar elevation angles overestimated with 'rad * sin(elev)' approach

    # Compute error statistics
    cdf_values = [[], []]
    for i in range(2):
        t_beg = perf_counter()
        if i == 0:
            idx_elev_start = shadow_angle_idx[:, :, 0]
            idx_elev_end = np.full(idx_elev_start.shape, 90, dtype=np.int32)
            dev_bins = dev_bins_default
        else:
            idx_elev_start = shadow_angle_idx[:, :, 0]
            idx_elev_end = shadow_angle_idx[:, :, 2]
            dev_bins = dev_bins_with_tn
        for eta in eta_range:
            f_cor_sparse = compute_fcor_sparse(
                f_cor_dense, elev_dense, idx_elev_start, idx_elev_end,
                num_nodes, eta)
            bin_counts = dev_bins(
                f_cor_dense, elev_dense, f_cor_sparse, idx_elev_start,
                idx_elev_end, terrain_normal, num_nodes, eta, rad_zenith,
                bin_size, scaling)
            shp = np.array(f_cor_dense.shape)
            # shp[2] = elev_dense[0:71].size
            # -> only consider elevation angles up to 70 degrees
            if bin_counts.sum() != np.prod(shp):
                raise ValueError("'bin_size' was chosen too small")
            cum_dist_func = np.cumsum(bin_counts) / bin_counts.sum()
            cdf_values[i].append(cum_dist_func)
        t_end = perf_counter()
        print(f"Compute error for different 'etas': {t_end - t_beg:.1f} s")

    # Plot error statistics
    titles = (
        "f_cor reconstructed only from f_cor",
        "f_cor reconstructed from f_cor and terrain_normal"
        )
    bin_edges = np.linspace(0.0, bin_size / scaling, bin_size + 1)
    cmap = plt.get_cmap("turbo")
    colors = [cmap(i) for i in np.linspace(0, 1, eta_range.size)]
    q = np.array([50.0, 90.0, 95.0, 99.0, 99.9, 99.99, 99.999])
    plt.figure(figsize=(14.0, 7.0))
    gs = gridspec.GridSpec(1, 2, left=0.1, bottom=0.1, right=0.9, top=0.9,
                       wspace=0.1, hspace=0.1)
    for j in range(2):
        ax = plt.subplot(gs[j])
        with np.printoptions(precision=3, suppress=True):
            print("qs:", q)
        for idx_i, i in enumerate(cdf_values[j]):
            plt.plot(bin_edges[1:], i * 100.0, lw=1.5, color=colors[idx_i],
                    label=rf"$\eta$ = {eta_range[idx_i]:.1f}")
            with np.printoptions(precision=2, suppress=True):
                print(eta_range[idx_i], np.interp(q / 100.0, i, bin_edges[1:]))
        plt.hlines(y=[90.0, 95, 99], xmin=0.0, xmax=100.0, colors="black",
                   lw=0.5, ls="--")
        # plt.axis((0.0, 40.0, 40.0, 100.0))
        plt.axis((0.0, 30.0, 75.0, 100.0))
        # plt.axis((0.0, 15.0, 85.0, 100.0))
        plt.xlabel(r"Absolute deviation [W m$^{-2}$]")
        if j == 0:
            plt.ylabel("Cumulative distribution function [%]")
            plt.legend(frameon=False, fontsize=10, loc="lower right", ncol=2)
        plt.title(titles[j], fontsize=11, loc="left")
    # plt.show()
    plt.savefig(path_plot + f"f_cor_error_stat_{icon_dom}.jpg",
                dpi=300, bbox_inches="tight")
    plt.close()

# -----------------------------------------------------------------------------

# Compute 'f_cor_sparse' and save relevant data for ICON
eta_sel = 2.0 # 2.0, 2.5 -> set hard-coded value in ICON code accordingly
idx_elev_start = shadow_angle_idx[:, :, 0]
idx_elev_end = shadow_angle_idx[:, :, 2]
f_cor_sparse = compute_fcor_sparse(
    f_cor_dense, elev_dense, idx_elev_start, idx_elev_end, num_nodes, eta_sel)

# Check that f_cor can be correctly recomputed for the lower and upper
# shadow angle
if check_data:

    num_check = 50_000 # number of random locations / azimuth angles to check
    idx_cell = np.random.randint(0, f_cor_sparse.shape[0], num_check)
    idx_azim = np.random.randint(0, 24, num_check)
    h_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    azim = np.arange(0.0, 360.0, 15, dtype=np.float32)
    for idx in range(num_check):
        idx_ang = idx_elev_start[idx_cell[idx], idx_azim[idx]]
        f_cor_sel = f_cor_dense[idx_cell[idx], idx_azim[idx], :][idx_ang]
        if f_cor_sel != 0.0:
            raise ValueError("Erroneous f_cor @ lower shadow angle")
        idx_ang = idx_elev_end[idx_cell[idx], idx_azim[idx]]
        t_vec = terrain_normal[idx_cell[idx], :]
        f_cor_sel = f_cor_dense[idx_cell[idx], idx_azim[idx], :][idx_ang]
        elev_ang = np.deg2rad(elev_dense[idx_ang]) # [rad]
        azim_ang = np.deg2rad(azim[idx_azim[idx]]) # [rad]
        s_vec = np.array(
            [np.cos(elev_ang) * np.sin(azim_ang),
             np.cos(elev_ang) * np.cos(azim_ang),
             np.sin(elev_ang)
             ], dtype=np.float32)
        denom = np.dot(s_vec, h_vec)
        if denom != 0.0:
            f_cor_rec = (1.0 / denom) * np.dot(s_vec, t_vec)
        else:
            f_cor_rec = 0.0
        f_cor_rec = np.minimum(f_cor_rec, 10.0) # same upper limit is above
        dev_abs = np.abs(f_cor_sel - f_cor_rec)
        if dev_abs > 1e-3:
            print("Warning: significant deviation in f_cor @ "
                  + "upper shadow angle")
            print(f_cor_sel, f_cor_rec, dev_abs, idx_ang)

# Notes from above check:
# - Maybe save terrain_normal as double precision for better accuracy?

# Save 'f_cor' data as numpy array
np.savez(path_in_out + f"f_cor_sparse_{icon_dom}.npz",
         shadow_angle=elev_dense[shadow_angle_idx],
         f_cor_sparse=f_cor_sparse[:, :, 1:-1],
         terrain_normal=terrain_normal)

############################################################################### code below not yet updated !!!
# Check f_cor for specific locations
###############################################################################

# -------------- 1km data -----------------
# ind_loc, ind_azim = 750_000, 13
# ind_loc, ind_azim = 790_610, 5
ind_loc, ind_azim = 777125, 11 # total shadow until ca. ~25 deg
# ind_loc, ind_azim = 680_000, 0 # always below 1.0
# ind_loc, ind_azim = 580_000, 0  # up to 6.0
# -------------- 2km data -----------------
# ind_loc, ind_azim = 100_000, 0  # 2km
# -----------------------------------------

# Plot
plt.figure()
plt.plot(elev_dense, f_cor_dense[ind_loc, ind_azim, :], color="black", lw=1.5)
# -----------------------------------------------------------------------------
elev_start = f_cor_sparse_global[ind_loc, ind_azim, 0]
elev_end = 90.0
elev_sparse = spacing_exp(elev_start, elev_end, num_elem - 1, eta_global)
plt.plot(elev_sparse, f_cor_sparse_global[ind_loc, ind_azim, 1:], color="green",
         lw=1.5)
plt.scatter(elev_sparse, f_cor_sparse_global[ind_loc, ind_azim, 1:],
            color="green", s=80)
# -----------------------------------------------------------------------------
eta_loc = f_cor_sparse_local[ind_loc, ind_azim, 0]
elev_start = f_cor_sparse_local[ind_loc, ind_azim, 1]
elev_end = 90.0
elev_sparse = spacing_exp(elev_start, elev_end, num_elem - 2, eta_loc)
plt.plot(elev_sparse, f_cor_sparse_local[ind_loc, ind_azim, 2:],
         color="red", lw=1.5)
plt.scatter(elev_sparse, f_cor_sparse_local[ind_loc, ind_azim, 2:],
            color="red", s=80)
# -----------------------------------------------------------------------------
plt.show()

###############################################################################
# Check that interpolation of f_cor from sparse data works correctly and save
# compressed f_cor information
###############################################################################

# Select sparse f_cor data
f_cor_sparse = f_cor_sparse_global
eta = eta_global
print(f"Shape of f_cor_sparse: {f_cor_sparse.shape}")
print(f"Size of f_cor_sparse: {(f_cor_sparse.nbytes / 10 ** 6):.1f} MB")
print(f"Global eta: {eta:.2f}")
num_gc = f_cor_sparse.shape[0]

# Reshape f_cor array for EXTPAR
shp_extpar = (f_cor_sparse.shape[0], f_cor_sparse.shape[1]
              * f_cor_sparse.shape[2])
f_cor_sparse_extpar = f_cor_sparse.reshape(shp_extpar).transpose()

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
f_cor_exact = np.interp(x=elev_sun, xp=elev_dense,
                        fp=f_cor_dense[ind_loc, ind_azim, :])
print(f"f_cor (exact) = {f_cor_exact:.2f}")
elev_start = f_cor_sparse[ind_loc, ind_azim, 0]
elev_end = 90.0
# ------------------------------
f_cor_loc = f_cor_sparse[ind_loc, ind_azim, 1:]
ind_azim_start = num_elem * ind_azim + 1
ind_azim_end = num_elem * (ind_azim + 1)
f_cor_loc_extpar = f_cor_sparse_extpar[ind_azim_start:ind_azim_end, ind_loc]
if np.any(f_cor_loc != f_cor_loc_extpar):
    raise ValueError("Incorrect f_cor-values accessed")
# ------------------------------
f_cor_approx = spacing_exp_interp(elev_start, elev_end, num_elem - 1,
                                  eta, elev_sun, f_cor_loc)
print(f"f_cor (approx) = {f_cor_approx:.2f}")

# Save to EXTPAR NetCDF file (write 'f_cor' to 'HORIZON' field)
t_beg = perf_counter()
ds = xr.open_dataset(path_extpar + file_extpar)
ds = ds.drop_vars("HORIZON")
ds["HORIZON"] = (("nhori", "cell"), f_cor_sparse_extpar)
ds["HORIZON"].attrs["standard_name"] = "-"
ds["HORIZON"].attrs["long_name"] = "horizon angle - topography" # rename?
ds["HORIZON"].attrs["units"] = "deg" # rename?
ds["HORIZON"].attrs["CDI_grid_type"] = "unstructured"
ds["HORIZON"].attrs["data_set"] = "ASTER"
encoding = {"time": {"_FillValue": None},
            "HORIZON": {"_FillValue": -1.e+20, "missing_value": -1.e+20}}
ds.to_netcdf(path_extpar + file_extpar[:-3] + f"_f_cor_sparse.nc", format="NETCDF4",
             encoding=encoding)
t_end = perf_counter()
print(f"Write 'f_cor' to EXTPAR NetCDF file: {t_end - t_beg:.1f} s")
