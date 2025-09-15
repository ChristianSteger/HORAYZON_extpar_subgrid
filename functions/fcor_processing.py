# Description: Auxiliary functions for f_cor processing (compression)
#
# Author: Christian R. Steger, September 2025

import functools
import time

import numpy as np
from numba import njit, float32, int64
from numba import prange, set_num_threads, get_num_threads

###############################################################################
# Generate uneven spacing and interpolate linearly from the spacing
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
def fcor_sparse_eta_const(f_cor_dense, elev_dense, num_elem, eta):
    """
    Compress f_cor information using constant exponent 'eta' for all locations.
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
def fcor_sparse_eta_opt(f_cor_dense, elev_dense, num_elem, eta_range,
                        rad_zenith):
    """
    Compress f_cor information using an optimal exponent 'eta' for every
    location.
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
def dev_bins_eta_const(f_cor_dense, elev_dense, f_cor_sparse,
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
                f_cor_diff = np.abs(f_cor_dense[ind_loc, ind_azim, :]
                                    - f_cor_ip)
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
def dev_bins_eta_opt(f_cor_dense, elev_dense, f_cor_sparse,
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
                f_cor_diff = np.abs(f_cor_dense[ind_loc, ind_azim, :]
                                    - f_cor_ip)
                deviations = f_cor_diff * sol_ang_sin * rad_zenith
                indices = np.floor(deviations * scaling).astype(np.int64)[0:71]
                # -> only consider elevation angles up to 70 degrees
                for ind in indices:
                    if (ind >= 0) and (ind < bin_size):
                        bin_counts[tid, ind] += 1
    return np.sum(bin_counts, axis=0)
