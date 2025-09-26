# Description: Compress f_cor data for EXTPAR file.
#
# Author: Christian R. Steger, September 2025

from time import perf_counter

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

from functions.fcor_processing import spacing_exp, spacing_exp_interp
from functions.fcor_processing import fcor_sparse_eta_const
from functions.fcor_processing import fcor_sparse_eta_opt
from functions.fcor_processing import dev_bins_eta_const
from functions.fcor_processing import dev_bins_eta_opt

style.use("classic")

# Change latex fonts
mpl.rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
mpl.rcParams["mathtext.default"] = "rm"
mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"

# Paths
path_in_out = "/scratch/mch/csteger/temp/ICON_refined_mesh/"
path_plot = "/scratch/mch/csteger/HORAYZON_extpar_subgrid/plots/"

###############################################################################
# Settings and load/check data
###############################################################################

# Settings
# icon_res = "2km"
icon_res = "1km"
# icon_res = "500m"

# Load data
file_in = f"SW_dir_cor_mch_{icon_res}.nc"
t_beg = perf_counter()
ds = xr.open_dataset(path_in_out + file_in)
f_cor_dense = ds["f_cor"].values # (num_cell_parent, num_hori, num_elev)
ds.close()
t_end = perf_counter()
print(f"Open NetCDF file: {t_end - t_beg:.1f} s")

# Set upper limit of f_cor-values and check general range
print(f"Maximal f-cor-value: {f_cor_dense.max():.2f}")
f_cor_dense = f_cor_dense.clip(max=10.0) # set upper limit for f_cor
if ((f_cor_dense.min() < 0.0) or (not np.all(f_cor_dense[:, :, 0] == 0.0))
    or (np.abs(f_cor_dense[:, :, -1] - 1.0).max() > 1e-8)):
    raise ValueError("Unexpected values in 'f_cor'")

# Azimuth and elevation angles
elev_dense = np.linspace(0.0, 90.0, 91, dtype=np.float32) # [deg]
if f_cor_dense.shape[2] != elev_dense.size:
    raise ValueError("Inconsistency between 'f_cor' and 'elev_ang' size")

# Settings
eta_range_const = np.array([1.0, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 3.0, 4.0],
                           dtype=np.float32)
eta_range_opt = np.arange(1.5, 3.6, 0.1, dtype=np.float32)
num_elem = 8 # number of array elements (incl. elev. angle and optional eta)
bin_size = 1_000_000
scaling = 100
rad_zenith = 900.0 # rather lower end because direct beam radiation at
# low solar elevation angles overestimated with 'rad * sin(elev)' approach

###############################################################################
# Compute optimal exponents 'eta' (globally and locally)
###############################################################################

# Find globally optimal exponent 'eta'
t_beg = perf_counter()
cdf_exp_global = []
for eta in eta_range_const:
    f_cor_sparse = fcor_sparse_eta_const(f_cor_dense, elev_dense, num_elem,
                                         eta)
    bin_counts = dev_bins_eta_const(f_cor_dense, elev_dense, f_cor_sparse,
                                       num_elem, eta, rad_zenith, bin_size,
                                       scaling)
    shp = np.array(f_cor_dense.shape)
    shp[2] = elev_dense[0:71].size
    # -> only consider elevation angles up to 70 degrees
    if bin_counts.sum() != np.prod(shp):
        raise ValueError("'bin_size' was chosen too small")
    bin_edges = np.linspace(0.0, bin_size / scaling, bin_size + 1)
    cum_dist_func = np.cumsum(bin_counts) / bin_counts.sum()
    cdf_exp_global.append(cum_dist_func)
t_end = perf_counter()
print(f"Find global optimum exponent: {t_end - t_beg:.1f} s")

# Find locally optimal exponent 'eta'
f_cor_sparse_local = fcor_sparse_eta_opt(f_cor_dense, elev_dense, num_elem,
                                         eta_range_opt, rad_zenith)
bin_counts = dev_bins_eta_opt(f_cor_dense, elev_dense, f_cor_sparse_local,
                              num_elem, rad_zenith, bin_size, scaling)
cdf_local = np.cumsum(bin_counts) / bin_counts.sum()

# Colors for lines
cmap = plt.get_cmap("turbo")
colors = [cmap(i) for i in np.linspace(0, 1, len(cdf_exp_global))]

q = np.array([50.0, 90.0, 95.0, 99.0, 99.9, 99.99, 99.999])
plt.figure()
with np.printoptions(precision=3, suppress=True):
    print("qs:", q)
# Global ----------------------------------------------------------------------
for ind_i, i in enumerate(cdf_exp_global):
    plt.plot(bin_edges[1:], i * 100.0, lw=1.5, color=colors[ind_i],
             label=rf"$\eta$ = {eta_range_const[ind_i]:.1f}")
    with np.printoptions(precision=2, suppress=True):
        print(eta_range_const[ind_i], np.interp(q / 100.0, i, bin_edges[1:]))
# Local -----------------------------------------------------------------------
plt.plot(bin_edges[1:], cdf_local * 100.0, lw=2.5, color="black", ls="--",
         label=r"local $\eta$")
with np.printoptions(precision=2, suppress=True):
    print("loc", np.interp(q / 100.0, cdf_local, bin_edges[1:]))
# -----------------------------------------------------------------------------
plt.hlines(y=[90.0, 95, 99], xmin=0.0, xmax=100.0, colors="black", lw=0.5,
           ls="--")
# plt.axis((0.0, 40.0, 40.0, 100.0))
plt.axis((0.0, 30.0, 75.0, 100.0))
# plt.axis((0.0, 15.0, 85.0, 100.0))
plt.xlabel(r"Absolute deviation [W m$^{-2}$]")
plt.ylabel("Cumulative distribution function [%]")
plt.legend(frameon=False, fontsize=10, loc="lower right", ncol=2)
# plt.show()
plt.savefig(path_plot + "f_cor_optimal_exp.jpg", dpi=300, bbox_inches="tight")
plt.close()

# Compute 'f_cor_sparse' with globally optimal exponent 'eta'
eta_global = 2.1
f_cor_sparse_global = fcor_sparse_eta_const(f_cor_dense, elev_dense, num_elem,
                                            eta_global)

###############################################################################
# Check f_cor for specific locations
###############################################################################

# ind_loc, ind_azim = 750_000, 13
# ind_loc, ind_azim = 790_610, 5
ind_loc, ind_azim = 777125, 11 # total shadow until ca. ~25 deg
# ind_loc, ind_azim = 680_000, 0 # always below 1.0
# ind_loc, ind_azim = 580_000, 0  # up to 6.0

plt.figure()
plt.plot(elev_dense, f_cor_dense[ind_loc, ind_azim, :], color="black", lw=1.5)
# -----------------------------------------------------------------------------
elev_start = f_cor_sparse_global[ind_loc, ind_azim, 0]
elev_end = 90.0
elev_sparse = spacing_exp(elev_start, elev_end, num_elem - 1, eta_global)
plt.plot(elev_sparse, f_cor_sparse_global[ind_loc, ind_azim, 1:], color="red",
         lw=1.5)
plt.scatter(elev_sparse, f_cor_sparse_global[ind_loc, ind_azim, 1:],
            color="red", s=80)
# -----------------------------------------------------------------------------
eta_loc = f_cor_sparse_local[ind_loc, ind_azim, 0]
elev_start = f_cor_sparse_local[ind_loc, ind_azim, 1]
elev_end = 90.0
elev_sparse = spacing_exp(elev_start, elev_end, num_elem - 2, eta_loc)
plt.plot(elev_sparse, f_cor_sparse_local[ind_loc, ind_azim, 2:], color="blue",
         lw=1.5)
plt.scatter(elev_sparse, f_cor_sparse_local[ind_loc, ind_azim, 2:],
            color="blue", s=80)
# -----------------------------------------------------------------------------
plt.show()
# plt.savefig("test.jpg", dpi=200, bbox_inches="tight")
# plt.close()

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

# Save 'f_cor' data as numpy array
np.save(path_in_out + f"f_cor_sparse_{icon_res}.npy", f_cor_sparse)

# Save to EXTPAR NetCDF file (write 'f_cor' to 'HORIZON' field)
path_extpar = "/scratch/mch/csteger/ICON-CH1-EPS_copy/"
# file = "external_parameter_icon_grid_0001_R19B08_mch_tuned.nc"
file = "extpar_icon_grid_0001_R19B08_mch_copernicus_ray.nc"
t_beg = perf_counter()
ds = xr.open_dataset(path_extpar + file)
ds = ds.drop_vars("HORIZON")
ds["HORIZON"] = (("nhori", "cell"), f_cor_sparse_extpar)
ds["HORIZON"].attrs["standard_name"] = "-"
ds["HORIZON"].attrs["long_name"] = "horizon angle - topography" # rename?
ds["HORIZON"].attrs["units"] = "deg" # rename?
ds["HORIZON"].attrs["CDI_grid_type"] = "unstructured"
ds["HORIZON"].attrs["data_set"] = "ASTER"
encoding = {"time": {"_FillValue": None},
            "HORIZON": {"_FillValue": -1.e+20, "missing_value": -1.e+20}}
ds.to_netcdf(path_extpar + file[:-3] + f"_f_cor_sparse.nc", format="NETCDF4",
             encoding=encoding)
t_end = perf_counter()
print(f"Write 'f_cor' to EXTPAR NetCDF file: {t_end - t_beg:.1f} s")
