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

from functions.fcor_processing import spacing_exp, compute_fcor_sparse
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
icon_dom = "mch_1km"
path_extpar = "/scratch/mch/csteger/ICON-CH1-EPS_copy_inn/"
file_extpar = "external_parameter_icon_grid_0001_R19B08_mch_tuned.nc"
icon_grid = "MeteoSwiss/icon_grid_0001_R19B08_mch.nc"
# ------------ MCH 500m -----------
# icon_dom = "mch_500m"
# path_extpar = "/store_new/mch/msopr/glori/glori-ch500-nested/grid/"
# file_extpar = "extpar_icon_grid_00005_R19B09_DOM02.nc"
# icon_grid = "MeteoSwiss/icon_grid_00005_R19B09_DOM02.nc"
# ---------------------------------
file_cor = f"SW_dir_cor_{icon_dom}.nc"

# Settings
num_nodes = 7
# total 7 array elements required: (6 - 2 = 4) f_cor-values, 3 elev. angles
rad_zenith = 900.0 # direct radiation at zenith for error estimation [W m-2]
# -> real vales likely lower - particularly at low solar elevation angles
#    because the diffuse part increase...
check_data = True
compute_error_stat = True

###############################################################################
# Load and check data
###############################################################################

# Load data
t_beg = perf_counter()
ds = xr.open_dataset(path_in_out + file_cor)
f_cor_dense = ds["f_cor"].values # modified in script (set upper limit)
shadow_angle_idx = ds["shadow_angle_idx"].values # modified in script
# (some ICON cells - 2 for ICON MCH 1km - show sub-grid-shadow up to 90 deg???)
terrain_normal = ds["terrain_normal"].values # not modified in script
ds.close()
t_end = perf_counter()
print(f"Open NetCDF file: {t_end - t_beg:.1f} s")

# Check f-cor-values and set upper limit
if ((f_cor_dense.min() < 0.0) or (not np.all(f_cor_dense[:, :, 0] == 0.0))):
    raise ValueError("Unexpected values in 'f_cor'")
print(f"Maximal f-cor-value: {f_cor_dense.max():.2f}")
f_cor_max = 10.0 # maximal allowed f-cor-value
f_cor_dense = f_cor_dense.clip(max=f_cor_max)
print(f"Set upper limit of f-cor-values to {f_cor_max:.2f}")

# Check shadow_angle indices
if not np.all(np.diff(shadow_angle_idx, axis=2) >= 0):
    raise ValueError("Some shadow angle indices are decreasing")
mask = (shadow_angle_idx == 999)
if np.any(mask):
    print("Warning: Some shadow_angle_idx values are missing (999)")
    print(f"Number of values being 999: {mask.sum()}")
    i, j, k = np.where(mask)
    print(i)
    print(j)
    print(k)
    shadow_angle_idx[mask] = 90
    print("Set missing values to 90 (upper limit)")
num_values = (shadow_angle_idx > 75).sum()
print(f"Number of values above 75 deg: {num_values}")
# -> values above ca. 75 deg are suspicious -> probably due to artefacts
#    in the ASTER DEM...
print(f"Range of shadow_angle_idx indices:"
        f" {shadow_angle_idx.min()} - {shadow_angle_idx.max()}")
if np.any(shadow_angle_idx[:, :, 0] == shadow_angle_idx[:, :, 2]):
    raise ValueError("Some shadow angle indies (0) and (2) are identical")

# Arrays with elevation/azimuth angles and surface horizontal vector
azim = np.arange(0.0, 360.0, 15, dtype=np.float32)
elev_dense = np.linspace(0.0, 90.0, 91, dtype=np.float32) # [deg]
if f_cor_dense.shape[2] != elev_dense.size:
    raise ValueError("Inconsistency between 'f_cor' and 'elev_ang' size")
h_vec = np.array([0.0, 0.0, 1.0])

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
    terrain_normal_norm = terrain_normal.copy()
    terrain_normal_norm /= np.linalg.norm(terrain_normal_norm, axis=1,
                                          keepdims=True)
    slope = np.rad2deg(np.arccos(terrain_normal_norm[:, 2]))
    aspect = np.rad2deg(np.pi / 2.0 - np.arctan2(terrain_normal_norm[:, 1],
                                                 terrain_normal_norm[:, 0]))
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

    # Select location and azimuth angle
    # --------------------------------------------
    # idx_cell, idx_azim = 9208, 0
    # idx_cell, idx_azim = 1_052_692, 23 # 1km; 0-23; upper shadow angle 999
    # idx_cell, idx_azim = 1_052_697, 23 # 1km; 0-23; upper shadow angle 999
    # --------------------------------------------
    idx_cell = np.random.randint(0, f_cor_dense.shape[0], 1)[0] # random
    idx_azim = np.random.randint(0, 24, 1)[0] # random
    # --------------------------------------------

    # Check f_cor-value at 90 deg
    f_cor_sel =  f_cor_dense[idx_cell, idx_azim, :]
    print("Slope angle: {:.2f} deg".format(slope[idx_cell]))
    print(f"f-cor-value @ elevation angle of 90 deg: {f_cor_sel[-1]:.3f}")
    # if the (1.0 / cos(slope)) term is not considered for f-cor computation,
    # than the f-cor-value at 90 deg is only 1.0 for a horizontal surface!

    # Recompute f_cor from average terrain normal
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

    # Recompute f_cor (old method)
    elev_nodes_old = spacing_exp(elev_dense[idx_start], 90.0, num_nodes,
                                 eta=2.0)
    f_cor_nodes_old = np.interp(elev_nodes_old, elev_dense, f_cor_sel)
    f_cor_rec_old = np.interp(elev_dense, elev_nodes_old, f_cor_nodes_old)
    radiation = np.sin(np.deg2rad(elev_dense)) * rad_zenith
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
# Compute error statistics (optional)
###############################################################################

if compute_error_stat:

    # Settings
    eta_range = np.arange(1.0, 4.5, 0.5, dtype=np.float32)
    bin_size = 1_000_000
    scaling = 100

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
        plt.axis((0.0, 25.0, 80.0, 100.0))
        plt.xlabel(r"Absolute deviation [W m$^{-2}$]")
        if j == 0:
            plt.ylabel("Cumulative distribution function [%]")
        if j == 1:
            plt.legend(frameon=False, fontsize=10, loc="lower right", ncol=2)
        plt.title(titles[j], fontsize=11, loc="left")
    # plt.show()
    file_name = f"f_cor_error_{icon_dom}_nodes_{num_nodes}.jpg"
    plt.savefig(path_plot + file_name, dpi=300, bbox_inches="tight")
    plt.close()

###############################################################################
# Compute sparse 'f_cor'-values, check recomputation of correction values and
# save relevant data for EXTPAR/ICON
###############################################################################

# Compute 'f_cor_sparse' and save relevant data for ICON
eta_sel = 2.0 # set hard-coded value in ICON code accordingly
idx_elev_start = shadow_angle_idx[:, :, 0]
idx_elev_end = shadow_angle_idx[:, :, 2]
f_cor_sparse = compute_fcor_sparse(
    f_cor_dense, elev_dense, idx_elev_start, idx_elev_end, num_nodes, eta_sel)

# -----------------------------------------------------------------------------
# Check that f_cor can be correctly recomputed for the lower and upper
# shadow angle
# -----------------------------------------------------------------------------

if check_data:

    num_check = 50_000 # number of random locations / azimuth angles to check
    idx_cell = np.random.randint(0, f_cor_sparse.shape[0], num_check)
    idx_azim = np.random.randint(0, 24, num_check)
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

# -----------------------------------------------------------------------------

# Save 'f_cor' data as numpy array
shadow_angle = elev_dense[shadow_angle_idx]
f_cor_sparse_no_bound = f_cor_sparse[:, :, 1:-1]
np.savez(path_in_out + f"f_cor_sparse_{icon_dom}.npz",
         shadow_angle=shadow_angle,
         f_cor_sparse=f_cor_sparse_no_bound,
         terrain_normal=terrain_normal)

# Reshape arrays for Fortran/EXTPAR
num_cells = f_cor_sparse_no_bound.shape[0]
num_azim = 24
num_f_cor = f_cor_sparse_no_bound.shape[2]
shp = (num_cells, num_azim * num_f_cor)
f_cor_sparse_extpar = f_cor_sparse_no_bound.reshape(shp).transpose()
# e.g. (96, 1147980)
shp = (num_cells, num_azim * 3)
shadow_angle_extpar = shadow_angle.reshape(shp).transpose()
# e.g. (72, 1147980)
terrain_normal_extpar = terrain_normal.transpose() # e.g. (3, 1147980)

# -----------------------------------------------------------------------------
# Check that correction values can be correctly recomputed
# -----------------------------------------------------------------------------

if check_data:

    # Select location and azimuth direction (for MCH 1km)
    # idx_cell, idx_azim = 750_000, 13
    # idx_cell, idx_azim = 790_610, 5
    # idx_cell, idx_azim = 777125, 11 # total shadow until ca. ~25 deg
    # idx_cell, idx_azim = 680_000, 0 # always below 1.0
    # idx_cell, idx_azim = 580_000, 0  # up to 6.0
    # idx_cell, idx_azim = 1052692, 0 # upper shadow angle originally missing
    # idx_cell, idx_azim = 1052697, 12 # upper shadow angle originally missing
    idx_cell = np.random.randint(0, num_cells)
    idx_azim = np.random.randint(0, 24)

    elev_sun = np.linspace(0.0, 90.0, 90 * 5 + 1, dtype=np.float32) # [deg]
    # all possible sun positions at or above the horizon
    # -> same spacing as 'elev_dense': elev_sun[0:None:5]

    # Loop through sun positions
    f_cor_rec = np.zeros(elev_sun.size, dtype=np.float32)
    horizon_min = shadow_angle_extpar[idx_azim * 3 + 0, idx_cell] # [deg]
    horizon_max = shadow_angle_extpar[idx_azim * 3 + 2, idx_cell] # [deg]
    t_vec = terrain_normal_extpar[:, idx_cell]
    for idx in range(elev_sun.size):

        # Position relative to lower and upper shadow angles (nodes)
        pos_norm = (elev_sun[idx] - horizon_min) / (horizon_max - horizon_min)

        # ---------------------------------------------------------------------
        # Total shadow -> f_cor = 0.0
        # ---------------------------------------------------------------------
        if pos_norm <= 0.0:
            f_cor_rec[idx] = 0.0
        # ---------------------------------------------------------------------
        # No shadow -> compute f_cor from terrain normal
        # ---------------------------------------------------------------------
        elif pos_norm >= 1.0:

            # Sun position vector
            elev_ang = np.deg2rad(elev_sun[idx]) # [rad]
            azim_ang = np.deg2rad(azim[idx_azim]) # [rad]
            s_vec = np.array(
                [np.cos(elev_ang) * np.sin(azim_ang),
                 np.cos(elev_ang) * np.cos(azim_ang),
                 np.sin(elev_ang)
                 ], dtype=np.float32)

            dot_prod_s_h = np.dot(s_vec, h_vec).clip(min=1e-5)
            # avoid division by zero
            f_cor_rec[idx] = (1.0 / dot_prod_s_h) * np.dot(s_vec, t_vec)

        # ---------------------------------------------------------------------
        # Partial shadow -> interpolate f_cor from saved data
        # ---------------------------------------------------------------------
        else:

            idx_left = int((num_nodes - 1) * pos_norm ** (1.0 / eta_sel))
            if idx_left >= (num_nodes - 1): # not sure if actually needed...
                print("Warning: 'idx_left' too large -> limit")
                idx_left = num_nodes - 2
            idx_right = idx_left + 1
            angle_left = (horizon_min + (horizon_max - horizon_min)
                        * (float(idx_left) / float(num_nodes - 1)) ** eta_sel)
            angle_right = (horizon_min + (horizon_max - horizon_min)
                        * (float(idx_right) / float(num_nodes - 1)) ** eta_sel)
            if idx_left == 0:
                f_cor_left = 0.0
                idx_lin = idx_azim * num_f_cor + (idx_right - 1)
                f_cor_right = f_cor_sparse_extpar[idx_lin, idx_cell]
            elif idx_right == (num_nodes - 1):

                # Sun position vector (sun @ horizon_max)
                elev_ang = np.deg2rad(angle_right) # [rad]
                azim_ang = np.deg2rad(azim[idx_azim]) # [rad]
                s_vec = np.array(
                    [np.cos(elev_ang) * np.sin(azim_ang),
                     np.cos(elev_ang) * np.cos(azim_ang),
                     np.sin(elev_ang)
                    ], dtype=np.float32)

                idx_lin = idx_azim * num_f_cor + (idx_left - 1)
                f_cor_left = f_cor_sparse_extpar[idx_lin, idx_cell]
                dot_prod_s_h = np.dot(s_vec, h_vec).clip(min=1e-5)
                f_cor_right = (1.0 / dot_prod_s_h) * np.dot(s_vec, t_vec)
            else:
                idx_lin = idx_azim * num_f_cor + (idx_left - 1)
                f_cor_left = f_cor_sparse_extpar[idx_lin, idx_cell]
                idx_lin = idx_azim * num_f_cor + (idx_right - 1)
                f_cor_right = f_cor_sparse_extpar[idx_lin, idx_cell]

            weight_left = ((angle_right - elev_sun[idx])
                           / (angle_right - angle_left))
            f_cor_rec[idx] = (f_cor_left * weight_left
                              + f_cor_right * (1.0 - weight_left))

        # ---------------------------------------------------------------------

    f_cor_rec = f_cor_rec.clip(max=10.0)

    # Compute mean error in radiation
    radiation = np.sin(np.deg2rad(elev_dense)) * rad_zenith
    rad_dev = np.abs(f_cor_rec[0:None:5]
                     - f_cor_dense[idx_cell, idx_azim, :]) * radiation
    print("Absolute error:  mean         max")
    print(" " * 17 +f"{rad_dev.mean():.3f} W m-2  {rad_dev.max():.3f} W m-2")

    # Plot
    plt.figure(figsize=(12.0, 6.0))
    plt.plot(elev_dense, f_cor_dense[idx_cell, idx_azim, :],
             color="gray", lw=1.5)
    idx_elev_start = shadow_angle_idx[idx_cell, idx_azim, 0]
    idx_elev_end = shadow_angle_idx[idx_cell, idx_azim, 2]
    elev_sparse = spacing_exp(elev_dense[idx_elev_start],
                              elev_dense[idx_elev_end],
                              num_nodes, eta_sel)
    plt.plot(elev_sparse, f_cor_sparse[idx_cell, idx_azim, :],
             color="green", lw=1.5)
    plt.scatter(elev_sparse, f_cor_sparse[idx_cell, idx_azim, :],
                color="green", s=80)
    plt.plot(elev_sun, f_cor_rec, color="red", lw=2.5, ls="--")
    f_cor_max = f_cor_dense[idx_cell, idx_azim, :].max() * 1.05
    plt.axis((-2.0, 92.0, -0.1, f_cor_max))
    plt.xlabel("Elevation angle [deg]")
    plt.ylabel("Correction factor (f_cor) [-]")
    plt.show()

# -----------------------------------------------------------------------------

# Save to EXTPAR NetCDF file
t_beg = perf_counter()
ds = xr.open_dataset(path_extpar + file_extpar)
# ----------------------------
ds = ds.drop_vars("HORIZON")
ds["HORIZON"] = (("nhori", "cell"), shadow_angle_extpar)
# data_all = np.vstack((shadow_angle_extpar, f_cor_sparse_extpar,
#                       terrain_normal_extpar))
# ds["HORIZON"] = (("nhori", "cell"), data_all)
ds["HORIZON"].attrs["standard_name"] = "-"
ds["HORIZON"].attrs["long_name"] = "horizon angle - topography"
ds["HORIZON"].attrs["units"] = "deg"
ds["HORIZON"].attrs["CDI_grid_type"] = "unstructured"
ds["HORIZON"].attrs["data_set"] = "ASTER"
# ----------------------------
ds["SWDIR_COR"] = (("nelem", "cell"), f_cor_sparse_extpar)
ds["SWDIR_COR"].attrs["standard_name"] = "-"
ds["SWDIR_COR"].attrs["long_name"] \
    = "correction factor for direct shortwave radiation"
ds["SWDIR_COR"].attrs["units"] = "-"
ds["SWDIR_COR"].attrs["CDI_grid_type"] = "unstructured"
ds["SWDIR_COR"].attrs["data_set"] = "ASTER"
# ----------------------------
ds["TERRAIN_NORMAL"] = (("comp", "cell"), terrain_normal_extpar)
ds["TERRAIN_NORMAL"].attrs["standard_name"] = "-"
ds["TERRAIN_NORMAL"].attrs["long_name"] \
    = "sub-grid averaged terrain normals (not normalised)"
ds["TERRAIN_NORMAL"].attrs["units"] = "-"
ds["TERRAIN_NORMAL"].attrs["CDI_grid_type"] = "unstructured"
ds["TERRAIN_NORMAL"].attrs["data_set"] = "ASTER"
# ----------------------------
encoding = {"time": {"_FillValue": None},
            "HORIZON": {"_FillValue": None},
            "SWDIR_COR": {"_FillValue": None},
            "TERRAIN_NORMAL": {"_FillValue": None}
            }
# "HORIZON": {"_FillValue": -1.e+20, "missing_value": -1.e+20}
ds.to_netcdf(path_extpar + file_extpar[:-3] + f"_horizon_subgrid.nc",
             format="NETCDF4", encoding=encoding)
t_end = perf_counter()
print(f"Write 'f_cor' to EXTPAR NetCDF file: {t_end - t_beg:.1f} s")
