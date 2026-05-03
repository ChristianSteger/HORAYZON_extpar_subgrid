# Description: Auxiliary functions for f_cor processing (compression)
#
# Author: Christian R. Steger, September 2025

import functools
from multiprocessing.pool import Pool
import time

import numpy as np
import xarray as xr
from scipy import optimize
import matplotlib.pyplot as plt
from numba import njit, float32, int64
from numba import prange, set_num_threads, get_num_threads

from functions.fcor_processing import spacing_exp
from functions.fcor_processing import dev_bins_eta_global

# Paths
path_in_out = "/scratch/mch/csteger/temp/ICON_refined_mesh/"

###############################################################################
# Test further optimisation by shifting vertical positions of f_cor-values
###############################################################################

# -----------------------------------------------------------------------------
# Functions to optimise vertical positions of f_cor-values
# -----------------------------------------------------------------------------

def opt_least_squares(x_sparse, y_sparse, x_dense, y_dense, weights):

    y_sparse_beg = y_sparse[0]
    y_sparse_end = y_sparse[-1]

    def residuals(y_sparse_inner):
        y_sparse = np.concatenate(
            [[y_sparse_beg], y_sparse_inner, [y_sparse_end]])
        y_interp = np.interp(x_dense, x_sparse, y_sparse)
        return (y_interp - y_dense) * weights

    res = optimize.least_squares(
        residuals, x0=y_sparse[1:-1], bounds=(0.0, 10.0), method='trf')

    y_opt = np.concatenate([[y_sparse_beg], res.x, [y_sparse_end]])

    return y_opt

# -----------------------------------------------------------------------------

def interp_matrix(x_dense, x_sparse):
    m = len(x_dense)
    n = len(x_sparse)
    A = np.zeros((m, n))

    j = 0
    for i, x in enumerate(x_dense):
        while j + 1 < n and x > x_sparse[j + 1]:
            j += 1
        t = (x - x_sparse[j]) / (x_sparse[j + 1] - x_sparse[j])
        A[i, j] = 1 - t
        A[i, j + 1] = t
    return A


def opt_least_squares_lin(x_sparse, y_sparse, x_dense, y_dense, weights):

    y_sparse_beg = y_sparse[0]
    y_sparse_end = y_sparse[-1]

    A = interp_matrix(x_dense, x_sparse)

    W = weights[:, None]
    # W = np.sqrt(weights)[:, None]
    
    A_w = W * A
    y_w = W[:, 0] * y_dense

    A_inner = A_w[:, 1:-1]
    y_adj = y_w - A_w[:, 0] * y_sparse_beg - A_w[:, -1] * y_sparse_end

    v_inner, *_ = np.linalg.lstsq(A_inner, y_adj, rcond=None)

    y_opt = np.concatenate([[y_sparse_beg], v_inner, [y_sparse_end]])

    return y_opt

def opt_least_squares_lin_bounds(x_sparse, y_sparse, x_dense, y_dense, weights,
                                 y_min=0.0, y_max=10.0, max_iter=10):

    y0 = y_sparse[0]
    yN = y_sparse[-1]

    A = interp_matrix(x_dense, x_sparse)
    W = weights[:, None]
    A_w = W * A
    y_w = W[:, 0] * y_dense

    n_inner = len(y_sparse) - 2
    free_idx = np.arange(n_inner)  # indices of nodes we solve for
    v_inner = y_sparse[1:-1].copy()  # initial guess

    for _ in range(max_iter):
        if len(free_idx) == 0:
            break

        A_inner = A_w[:, 1:-1][:, free_idx]
        y_adj = y_w - A_w[:, 0] * y0 - A_w[:, -1] * yN
        # subtract contributions of already fixed nodes
        fixed_mask = np.ones(n_inner, bool)
        fixed_mask[free_idx] = False
        if fixed_mask.any():
            y_adj -= A_w[:, 1:-1][:, fixed_mask] @ v_inner[fixed_mask]

        # solve least squares for free nodes
        v_sol, *_ = np.linalg.lstsq(A_inner, y_adj, rcond=None)
        v_inner[free_idx] = v_sol

        # check bounds
        out_lower = v_inner < y_min
        out_upper = v_inner > y_max
        out_of_bounds = out_lower | out_upper

        if not out_of_bounds.any():
            break  # all within bounds

        # clip out-of-bounds nodes and remove from free set
        v_inner[out_lower] = y_min
        v_inner[out_upper] = y_max
        free_idx = np.where(~out_of_bounds)[0]

    # reconstruct full y vector
    y_opt = np.concatenate([[y0], v_inner, [yN]])
    return y_opt

# -----------------------------------------------------------------------------
# Other functions
# -----------------------------------------------------------------------------

def all_close_to_0_or_1(x, tol=1e-2):
    return np.all(np.isclose(x, 0, atol=tol) | np.isclose(x, 1, atol=tol))

def error_stat(elev_dense, f_cor_dense, elev_sparse, f_cor_sparse, 
               weights, rad_zenith):
    f_cor_ip = np.interp(x=elev_dense, xp=elev_sparse, fp=f_cor_sparse)
    dev = ((f_cor_ip - f_cor_dense) * weights * rad_zenith)
    err_mean = dev.mean()
    err_abs_mean = np.abs(dev).mean()
    err_abs_max = np.abs(dev).max()
    rmse = np.sqrt((dev ** 2).mean())
    return err_mean, err_abs_mean, err_abs_max, rmse

# -----------------------------------------------------------------------------

# Select ICON model resolution
icon_res = "500m"

# Load 'f_cor_dense'
file_in = f"SW_dir_cor_mch_{icon_res}.nc"
ds = xr.open_dataset(path_in_out + file_in)
f_cor_dense = ds["f_cor"].values # (num_cell_parent, num_hori, num_elev)
ds.close()
elev_dense = np.linspace(0.0, 90.0, 91, dtype=np.float32) # [deg]

# Load '# Load 'f_cor_dense'
f_cor_sparse = np.load(path_in_out + f"f_cor_sparse_global_{icon_res}.npy")
num_elem = f_cor_sparse.shape[2]
eta = 2.0

# -----------------------------------------------------------------------------
# Optimise a random sub-sample and check influence on error metric
# -----------------------------------------------------------------------------

# Settings
num = 2_000 # (-> * 24)

# Random points
np.random.seed(23)
ind_loc_arr = np.random.randint(0, f_cor_dense.shape[0], num)

# Points in complex terrain
# path = "/scratch/mch/csteger/ICON-CH1-EPS_copy_inn/"
# file = "external_parameter_icon_grid_0001_R19B08_mch_tuned.nc"
# ds = xr.open_dataset(path + file)
# svf =  ds["SKYVIEW"].values
# ds.close()
# ind_loc_arr = np.where(svf < 0.93)[0]
# # ind_loc_arr = np.where(svf > 0.9999)[0][:1000]  # flat terrain test
# num = ind_loc_arr.size
# print(num)
# path = "/scratch/mch/csteger/ICON-CH2-EPS_copy_inn/"
# file = "external_parameter_icon_grid_0002_R19B07_mch_tuned.nc"
# ds = xr.open_dataset(path + file)
# svf =  ds["SKYVIEW"].values
# ds.close()
# ind_loc_arr = np.where(svf < 0.98)[0]
# # ind_loc_arr = np.where(svf > 0.9999)[0][:1000]  # flat terrain test
# num = ind_loc_arr.size
# print(num)
path = "/store_new/mch/msopr/glori/glori-ch500-nested/grid//"
file = "extpar_icon_grid_00005_R19B09_DOM02.nc"
ds = xr.open_dataset(path + file)
svf =  ds["SKYVIEW"].values
ds.close()
ind_loc_arr = np.where(svf < 0.86)[0]
# ind_loc_arr = np.where(svf > 0.9999)[0][:1000]  # flat terrain test
num = ind_loc_arr.size
print(num)


# Weights and zenith radiation
weights = np.sin(np.deg2rad(elev_dense))
# weights[0] = weights[1]  # avoid zero weight at the first point
weights = weights.clip(min=0.1)
# weights = np.ones_like(weights) # set all to 1.0 (testing)
rad_zenith = 900.0 # rather lower end because direct beam radiation at
# low solar elevation angles overestimated with 'rad * sin(elev)' approach

# Loop through locations and azimuths
f_cor_sparse_sel = f_cor_sparse[ind_loc_arr, :, :]
f_cor_sparse_sel_opt = f_cor_sparse_sel.copy()
mask_impr = np.zeros(f_cor_sparse_sel.shape[:-1], dtype=bool)
out_of_bounds = 0
for idx, ind_loc in enumerate(ind_loc_arr): 
    for ind_azim in range(24):

        # Compute optimal vertical position of inner f_cor-values
        elev_start = f_cor_sparse[ind_loc, ind_azim, 0]
        elev_end = 90.0
        elev_sparse = spacing_exp(elev_start, elev_end, 
                                  num_elem - 1, eta)
        f_cor_sparse_opt = opt_least_squares_lin_bounds(
            elev_sparse, f_cor_sparse[ind_loc, ind_azim, 1:], elev_dense, 
            f_cor_dense[ind_loc, ind_azim, :], weights)
        
        # Check error metrics
        err_mean, err_abs_mean, err_abs_max, rmse = error_stat(
            elev_dense, f_cor_dense[ind_loc, ind_azim, :], 
            elev_sparse, f_cor_sparse[ind_loc, ind_azim, 1:], 
            weights, rad_zenith)
        err_mean_opt, err_abs_mean_opt, err_abs_max_opt, rmse_opt = error_stat(
            elev_dense, f_cor_dense[ind_loc, ind_azim, :], 
            elev_sparse, f_cor_sparse_opt, 
            weights, rad_zenith)
        if ((rmse_opt <= rmse) and (err_abs_mean_opt <= err_abs_mean) and (err_abs_max_opt <= err_abs_max * 1.5)
            and (f_cor_sparse_opt.min() >= 0.0 
                 and f_cor_sparse_opt.max() <= 10.0)):
            f_cor_sparse_sel_opt[idx, ind_azim, 1:] = f_cor_sparse_opt
            mask_impr[idx, ind_azim] = True
        # f_cor_sparse_sel_opt[idx, ind_azim, 1:] = f_cor_sparse_opt
        # mask_impr[idx, ind_azim] = True
        flag = (f_cor_sparse_opt.min() >= 0.0 and f_cor_sparse_opt.max() <= 10.0)
        if not flag:
            out_of_bounds += 1

        # # Print results
        # print(f"ME: {me:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        # print(f"ME_opt: {me_opt:.3f}, MAE_opt: {mae_opt:.3f}, RMSE_opt: {rmse_opt:.3f}")

        # # Plot
        # plt.figure()
        # plt.plot(elev_dense, f_cor_dense[ind_loc, ind_azim, :], 
        #          color="black", lw=1.5)
        # # ---------------------------------------------------------------------
        # elev_start = f_cor_sparse[ind_loc, ind_azim, 0]
        # elev_end = 90.0
        # elev_sparse = spacing_exp(elev_start, elev_end, num_elem - 1, eta)
        # plt.plot(elev_sparse, f_cor_sparse[ind_loc, ind_azim, 1:], 
        #          color="blue", lw=1.5)
        # plt.scatter(elev_sparse, f_cor_sparse[ind_loc, ind_azim, 1:],
        #             color="blue", s=80)
        # # ---------------------------------------------------------------------
        # plt.plot(elev_sparse, f_cor_sparse_opt, color="green", lw=1.5)
        # plt.scatter(elev_sparse, f_cor_sparse_opt, color="green", s=80)
        # # ---------------------------------------------------------------------
        # plt.show()

# Settings
bin_size = 1_000_000
scaling = 100



# Plot
bin_edges = np.linspace(0.0, bin_size / scaling, bin_size + 1)
cmap = plt.get_cmap("turbo")
q = np.array([50.0, 90.0, 95.0, 99.0, 99.9, 99.99, 99.999])
plt.figure()
with np.printoptions(precision=3, suppress=True):
    print("qs:", q)
# ------------------------------- local eta  ----------------------------------
f_cor_dense_sel = f_cor_dense[ind_loc_arr, :, :]
bin_counts = dev_bins_eta_global(f_cor_dense_sel, elev_dense, f_cor_sparse_sel,
                                 num_elem, eta, rad_zenith, bin_size,
                                 scaling)
cum_dist_func = np.cumsum(bin_counts) / bin_counts.sum()
plt.plot(bin_edges[1:], cum_dist_func * 100.0, lw=2.5, color="black", ls="-",
         label="default")
with np.printoptions(precision=2, suppress=True):
    print("loc", np.interp(q / 100.0, cum_dist_func, bin_edges[1:]))
# ------------------------------- local eta  ----------------------------------
f_cor_dense_sel = f_cor_dense[ind_loc_arr, :, :]
bin_counts = dev_bins_eta_global(f_cor_dense_sel, elev_dense, f_cor_sparse_sel_opt,
                                 num_elem, eta, rad_zenith, bin_size,
                                 scaling)
cum_dist_func = np.cumsum(bin_counts) / bin_counts.sum()
plt.plot(bin_edges[1:], cum_dist_func * 100.0, lw=2.5, color="black", ls="--",
         label="optimised")
with np.printoptions(precision=2, suppress=True):
    print("loc", np.interp(q / 100.0, cum_dist_func, bin_edges[1:]))
# -----------------------------------------------------------------------------
plt.hlines(y=[90.0, 95, 99], xmin=0.0, xmax=100.0, colors="black", lw=0.5,
           ls="--")
# plt.axis((0.0, 40.0, 40.0, 100.0))
plt.axis((0.0, 30.0, 75.0, 100.0))
# plt.axis((0.0, 15.0, 85.0, 100.0))
plt.xlabel(r"Absolute deviation [W m$^{-2}$]")
plt.ylabel("Cumulative distribution function [%]")
plt.legend(frameon=False, fontsize=10, loc="lower right", ncol=2)
plt.show()
# plt.savefig(path_plot + f"f_cor_exp_{icon_res}.jpg", 
#             dpi=300, bbox_inches="tight")
# plt.close()













###############################################################################
# Parallelised function to optimise 'f_cor_sparse_global'
###############################################################################




@njit((float32[:])(float32, float32, int64, float32))
def spacing_exp(x_start, x_end, num_nodes, eta):
    """
    Computes spacing between x_start and x_end with increasing spacing towards
    the right. The output array starts/ends exactly with x_start/x_end.

    Parameters
    ----------
    x_start : float
        Start of the spacing.
    x_end : float
        End of the spacing.
    num_nodes : int
        Number of points in the spacing.
    eta : float
        Exponent for the spacing. Must be >= 1.0.

    Returns
    -------
    x_spac : ndarray
        Array of size num with the spacing.
    """
    x_spac = np.empty(num_nodes, dtype=np.float32)
    x_spac[0] = x_start
    for i in range(1, num_nodes - 1):
         x_spac[i] = x_start + (x_end - x_start) \
            * (float(i) / float(num_nodes - 1)) ** eta
    x_spac[num_nodes - 1] = x_end
    return x_spac

@njit((float32)(float32, float32, int64, float32, float32, float32[:]))
def spacing_exp_interp(x_start, x_end, num_nodes, eta, x_ip, y):
    """
    Linear interpolation from spacing increasing towards the right.

    Parameters
    ----------
    x_start : float
        Start of the spacing.
    x_end : float
        End of the spacing.
    num_nodes : int
        Number of points in the spacing.
    eta : float
        Exponent for the spacing. Must be >= 1.0.
    x_ip : float
        x-value for interpolation.
    y : ndarray
        y-values at the spacing points. 

    Returns
    -------
    y_ip : float
        Interpolated y-value at x_ip.
    """
    pos_norm = (x_ip - x_start) / (x_end - x_start)
    if pos_norm <= 0.0:
        # -> intercept negative 'pos_norm' values -> issue for 'pos_norm ** m',
        #    guarantees that 'ind_left' is >= 0
        # print("x-value out of bounds (left)")
        return 0.0
    ind_left = int((num_nodes - 1) * pos_norm ** (1.0 / eta))
    if ind_left >= (num_nodes - 1):
        # -> handle values when 'ind_left' would be rightmost index or larger
        # print("x-value out of bounds (right)")
        return 1.0
    x_left = x_start + (x_end - x_start) \
        * (float(ind_left) / float(num_nodes - 1)) ** eta
    x_right = x_start + (x_end - x_start) \
        * (float(ind_left + 1) / float(num_nodes - 1)) ** eta
    # print("Left index: " + str(ind_left))
    # print(f"x_left: {x_left:.4f}, x_ip: {x_ip:.4f}, "
    #       + f"x_right: {x_right:.4f}")
    weight_left = (x_right - x_ip) / (x_right - x_left)
    y_ip = y[ind_left] * weight_left \
        + y[ind_left + 1] * (1.0 - weight_left)
    return y_ip

# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Create exponentially spaced array
    x_start = float(np.random.uniform(0.0, 40.0, 1)[0])
    print(f"x_start: {x_start:.4f}")
    x_end = 90.0
    num_nodes = 7
    eta = 2.2
    x_spac = spacing_exp(x_start, x_end, num_nodes, eta)

    # Check interpolation
    x_ip = np.random.uniform(x_start, x_end, 1)[0]
    # x_ip = x_start - 0.0000000000001
    # x_ip = x_end - 0.00000000000001
    y = np.random.uniform(0.0, 1.0, num_nodes).astype(np.float32)
    y[0], y[-1] = 0.0, 1.0
    y_ip = spacing_exp_interp(x_start, x_end, num_nodes, eta, x_ip, y)
    if abs(y_ip - np.interp(x_ip, x_spac, y, left=0.0, right=1.0)) > 1e-6:
        raise ValueError("Interpolation erroneous")
    print(f"y_ip: {y_ip:.4f}")

###############################################################################
# Compress f_cor information for entire domain and compute error statistics
###############################################################################

def measure_time(func):
    """
    Decorator to measure the execution time of a method.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        time_start = time.perf_counter()
        result = func(*args, **kwargs)
        time_end = time.perf_counter()
        print(f"{func.__name__}: {time_end - time_start:.1f} s")
        return result
    return wrapper

set_num_threads(8)
print("Using", get_num_threads(), "threads")

@measure_time
@njit((float32[:, :, :])(float32[:, :, :], float32[:], int64, float32),
      parallel=True)
def fcor_sparse_eta_global(f_cor_dense, elev_dense, num_elem, eta):
    """
    Compress f_cor information using a global exponent 'eta'.
    """
    f_cor_sparse = np.empty((f_cor_dense.shape[0], 24, num_elem),
                            dtype=np.float32)
    # axis 2: elevation angle and array of f_cor-values
    for ind_loc in prange(f_cor_dense.shape[0]):
        for ind_azim in range(24):
            ind_start \
                = np.where(f_cor_dense[ind_loc, ind_azim, :] == 0.0)[0][-1]
            elev_start = elev_dense[ind_start]
            elev_end = 90.0 # equal to elev_dense[-1]
            elev_sparse = spacing_exp(elev_start, elev_end, num_elem - 1, eta)
            f_cor_ip = np.interp(x=elev_sparse, xp=elev_dense,
                                 fp=f_cor_dense[ind_loc, ind_azim, :])
            f_cor_sparse[ind_loc, ind_azim, 0] = elev_start
            f_cor_sparse[ind_loc, ind_azim, 1:] = f_cor_ip
    return f_cor_sparse

@measure_time
@njit((float32[:, :, :])(float32[:, :, :], float32[:], int64, float32[:], 
                         float32), parallel=True)
def fcor_sparse_eta_local(f_cor_dense, elev_dense, num_elem, eta_range,
                          rad_zenith):
    """
    Compress f_cor information using an local optimal exponent 'eta'.
    """
    f_cor_sparse = np.empty((f_cor_dense.shape[0], 24, num_elem),
                            dtype=np.float32)
    # axis 2: optimal exponent, elevation angle and array of f_cor-values
    sol_ang_sin = np.sin(np.deg2rad(elev_dense))
    for ind_loc in prange(f_cor_dense.shape[0]):
        for ind_azim in range(24):
            ind_start \
                = np.where(f_cor_dense[ind_loc, ind_azim, :] == 0.0)[0][-1]
            elev_start = elev_dense[ind_start]
            elev_end = 90.0 # equal to elev_dense[-1]
            error_metric = np.zeros(eta_range.size, dtype=np.float32)
            for ind_eta in range(eta_range.size):
                elev_sparse = spacing_exp(elev_start, elev_end, num_elem - 2,
                                          eta_range[ind_eta])
                f_cor_sparse_temp = np.interp(
                    x=elev_sparse, xp=elev_dense,
                    fp=f_cor_dense[ind_loc, ind_azim, :])
                f_cor_dense_rec = np.interp(elev_dense, elev_sparse,
                                            f_cor_sparse_temp)
                rad = f_cor_dense_rec * sol_ang_sin * rad_zenith
                rad_ref = f_cor_dense[ind_loc, ind_azim, :] * sol_ang_sin \
                    * rad_zenith
                # ----- Sum of absolute difference ----------------------------
                # error_metric[ind_eta] = np.abs(rad - rad_ref).sum()
                # ----- Sum of squared difference -----------------------------
                error_metric[ind_eta] = ((rad - rad_ref) ** 2).sum()
                # -------------------------------------------------------------
            eta_opt = eta_range[np.argmin(error_metric)]
            elev_sparse = spacing_exp(elev_start, elev_end, num_elem - 2,
                                      eta_opt)
            f_cor_ip = np.interp(x=elev_sparse, xp=elev_dense,
                                 fp=f_cor_dense[ind_loc, ind_azim, :])
            f_cor_sparse[ind_loc, ind_azim, 0] = eta_opt
            f_cor_sparse[ind_loc, ind_azim, 1] = elev_start
            f_cor_sparse[ind_loc, ind_azim, 2:] = f_cor_ip

    return f_cor_sparse

###############################################################################
# Compute error statistics for compressed f_cor information
###############################################################################

@measure_time
@njit((int64[:])(float32[:, :, :], float32[:], float32[:, :, :], int64,
                 float32, float32, int64, float32), parallel=True)
def dev_bins_eta_global(f_cor_dense, elev_dense, f_cor_sparse,
                        num_elem, eta, rad_zenith, bin_size, scaling):
    """
    Compute binned deviations for 'f_cor_sparse' with respect to reference 
    data ('f_cor_dense'). Parallel version.
    """
    num_threads = get_num_threads()
    bin_counts = np.zeros((num_threads, bin_size), dtype=np.int64)
    sol_ang_sin = np.sin(np.deg2rad(elev_dense))
    shape_0 = f_cor_dense.shape[0]
    chunk_size = (shape_0 + num_threads - 1) // num_threads
    for tid in prange(num_threads):
        start = tid * chunk_size
        end = min(shape_0, start + chunk_size)
        for ind_loc in range(start, end):
            for ind_azim in range(24):
                elev_start = f_cor_sparse[ind_loc, ind_azim, 0]
                elev_end = 90.0
                elev_sparse = spacing_exp(elev_start, elev_end,
                                          num_elem - 1, eta)
                f_cor_ip = np.interp(elev_dense, elev_sparse,
                                    f_cor_sparse[ind_loc, ind_azim, 1:])
                # -> out-of-bounds interpolation behaviour:
                #    x < xp[0]  -> fp[0]
                #    x > xp[-1] -> fp[-1]
                f_cor_diff = np.abs(f_cor_ip
                                    - f_cor_dense[ind_loc, ind_azim, :])
                deviations = f_cor_diff * sol_ang_sin * rad_zenith
                indices = np.floor(deviations * scaling).astype(np.int64)[0:71]
                # -> only consider elevation angles up to 70 degrees
                for ind in indices:
                    if (ind >= 0) and (ind < bin_size):
                        bin_counts[tid, ind] += 1
    return np.sum(bin_counts, axis=0)

@measure_time
@njit((int64[:])(float32[:, :, :], float32[:], float32[:, :, :], int64,
                 float32, int64, float32), parallel=True)
def dev_bins_eta_local(f_cor_dense, elev_dense, f_cor_sparse,
                       num_elem, rad_zenith, bin_size, scaling):
    """
    Compute binned deviations for 'f_cor_sparse' with respect to reference 
    data ('f_cor_dense'). Parallel version.
    """
    num_threads = get_num_threads()
    bin_counts = np.zeros((num_threads, bin_size), dtype=np.int64)
    sol_ang_sin = np.sin(np.deg2rad(elev_dense))
    shape_0 = f_cor_dense.shape[0]
    chunk_size = (shape_0 + num_threads - 1) // num_threads
    for tid in prange(num_threads):
        start = tid * chunk_size
        end = min(shape_0, start + chunk_size)
        for ind_loc in range(start, end):
            for ind_azim in range(24):
                eta = f_cor_sparse[ind_loc, ind_azim, 0]
                elev_start = f_cor_sparse[ind_loc, ind_azim, 1]
                elev_end = 90.0
                elev_sparse = spacing_exp(elev_start, elev_end,
                                          num_elem - 2, eta)
                f_cor_ip = np.interp(elev_dense, elev_sparse,
                                    f_cor_sparse[ind_loc, ind_azim, 2:])
                # -> out-of-bounds interpolation behaviour:
                #    x < xp[0]  -> fp[0]
                #    x > xp[-1] -> fp[-1]
                f_cor_diff = np.abs(f_cor_ip
                                    - f_cor_dense[ind_loc, ind_azim, :])
                deviations = f_cor_diff * sol_ang_sin * rad_zenith
                indices = np.floor(deviations * scaling).astype(np.int64)[0:71]
                # -> only consider elevation angles up to 70 degrees
                for ind in indices:
                    if (ind >= 0) and (ind < bin_size):
                        bin_counts[tid, ind] += 1
    return np.sum(bin_counts, axis=0)

###############################################################################
# Enhance 'f_cor_sparse' further by optimising vertical position of
# f-cor-values
###############################################################################

