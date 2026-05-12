# Description: Auxiliary functions for f_cor processing (compression)
#
# Author: Christian R. Steger, September 2025

import functools
import time

import numpy as np
from numba import njit, float32, int64, int32
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
        #    guarantees that 'idx_left' is >= 0
        # print("x-value out of bounds (left)")
        return 0.0
    idx_left = int((num_nodes - 1) * pos_norm ** (1.0 / eta))
    if idx_left >= (num_nodes - 1):
        # -> handle values when 'idx_left' would be rightmost index or larger
        # print("x-value out of bounds (right)")
        return 1.0
    x_left = x_start + (x_end - x_start) \
        * (float(idx_left) / float(num_nodes - 1)) ** eta
    x_right = x_start + (x_end - x_start) \
        * (float(idx_left + 1) / float(num_nodes - 1)) ** eta
    # print("Left index: " + str(idx_left))
    # print(f"x_left: {x_left:.4f}, x_ip: {x_ip:.4f}, "
    #       + f"x_right: {x_right:.4f}")
    weight_left = (x_right - x_ip) / (x_right - x_left)
    y_ip = y[idx_left] * weight_left \
        + y[idx_left + 1] * (1.0 - weight_left)
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
@njit((float32[:, :, :])(float32[:, :, :], float32[:], int32[:, :],
                         int32[:, :], int64, float32), parallel=True)
def compute_fcor_sparse(f_cor_dense, elev_dense, idx_elev_start, idx_elev_end,
                        num_nodes, eta):
    """
    Compress f_cor information using the exponent 'eta'.
    """
    f_cor_sparse = np.empty((f_cor_dense.shape[0], 24, num_nodes),
                            dtype=np.float32)
    for idx_cell in prange(f_cor_dense.shape[0]):
        for idx_azim in range(24):
            elev_start = elev_dense[idx_elev_start[idx_cell, idx_azim]]
            elev_end = elev_dense[idx_elev_end[idx_cell, idx_azim]]
            elev_sparse = spacing_exp(elev_start, elev_end, num_nodes, eta)
            f_cor_sparse[idx_cell, idx_azim, :] \
                = np.interp(x=elev_sparse, xp=elev_dense,
                            fp=f_cor_dense[idx_cell, idx_azim, :])
    return f_cor_sparse

@measure_time
@njit((int64[:])(float32[:, :, :], float32[:], float32[:, :, :], int32[:, :],
                 int32[:, :], float32[:, :], int64, float32, float32, int64,
                 float32), parallel=True)
def dev_bins_default(f_cor_dense, elev_dense, f_cor_sparse, idx_elev_start,
                     idx_elev_end, terrain_normal, num_nodes, eta, rad_zenith,
                     bin_size, scaling):
    """
    Compute binned deviations for 'f_cor_sparse' with respect to reference 
    data 'f_cor_dense'. Only use 'f_cor_sparse' information for recomputing
    'f_cor' ('terrain_normal' is not used).
    """
    num_threads = get_num_threads()
    bin_counts = np.zeros((num_threads, bin_size), dtype=np.int64)
    radiation = np.sin(np.deg2rad(elev_dense)) * rad_zenith
    shape_0 = f_cor_dense.shape[0]
    chunk_size = (shape_0 + num_threads - 1) // num_threads
    for tid in prange(num_threads):
        start = tid * chunk_size
        end = min(shape_0, start + chunk_size)
        for idx_cell in range(start, end):
            for idx_azim in range(24):
                elev_start = elev_dense[idx_elev_start[idx_cell, idx_azim]]
                elev_end = elev_dense[idx_elev_end[idx_cell, idx_azim]]
                elev_sparse = spacing_exp(elev_start, elev_end, num_nodes, eta)
                f_cor_rec = np.interp(elev_dense, elev_sparse,
                                     f_cor_sparse[idx_cell, idx_azim, :])
                # -> out-of-bounds interpolation behaviour:
                #    x < xp[0]  -> fp[0]
                #    x > xp[-1] -> fp[-1]
                f_cor_dev = np.abs(f_cor_rec
                                   - f_cor_dense[idx_cell, idx_azim, :])
                radiation_dev = f_cor_dev * radiation
                indices = np.floor(radiation_dev * scaling).astype(np.int64)
                # indices = np.floor(radiation_dev * scaling).astype(np.int64)[0:71]
                # -> only consider elevation angles up to 70 degrees
                for idx in indices:
                    if (idx >= 0) and (idx < bin_size):
                        bin_counts[tid, idx] += 1
    return np.sum(bin_counts, axis=0)

@measure_time
@njit((int64[:])(float32[:, :, :], float32[:], float32[:, :, :], int32[:, :],
                 int32[:, :], float32[:, :], int64, float32, float32, int64,
                 float32)) # parallel: , parallel=True
def dev_bins_with_tn(f_cor_dense, elev_dense, f_cor_sparse, idx_elev_start,
                     idx_elev_end, terrain_normal, num_nodes, eta, rad_zenith,
                     bin_size, scaling):
    """
    Compute binned deviations for 'f_cor_sparse' with respect to reference 
    data 'f_cor_dense'. Use 'f_cor_sparse' and 'terrain_normal' information
    for recomputing 'f_cor'.
    """
    # Note: parallelised version yields incorrect results for unknown reasons

    # Horizontal normal and sun vector(s)
    h_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    s_vec = np.empty((24, 91, 3), dtype=np.float32)
    s_vec.fill(np.nan)
    azim = np.arange(0.0, 360.0, 15, dtype=np.float32)
    for idx_azim in range(24):
        for idx_elev in range(91):
            s_vec[idx_azim, idx_elev, 0] \
                = np.cos(np.deg2rad(elev_dense[idx_elev])) \
                    * np.sin(np.deg2rad(azim[idx_azim]))
            s_vec[idx_azim, idx_elev, 1] \
                = np.cos(np.deg2rad(elev_dense[idx_elev])) \
                    * np.cos(np.deg2rad(azim[idx_azim]))
            s_vec[idx_azim, idx_elev, 2] \
                = np.sin(np.deg2rad(elev_dense[idx_elev]))

    # Compute binned deviations
    num_threads = get_num_threads()
    bin_counts = np.zeros((num_threads, bin_size), dtype=np.int64)
    radiation = np.sin(np.deg2rad(elev_dense)) * rad_zenith
    shape_0 = f_cor_dense.shape[0]
    chunk_size = (shape_0 + num_threads - 1) // num_threads
    for tid in range(num_threads): # parallel: prange
        start = tid * chunk_size
        end = min(shape_0, start + chunk_size)
        for idx_cell in range(start, end):
            t_vec = terrain_normal[idx_cell, :]
            for idx_azim in range(24):
                f_cor_rec = np.zeros(91, dtype=np.float32)
                # ---------- Recompute from 'f_cor_sparse' --------------------
                idx_start = idx_elev_start[idx_cell, idx_azim]
                idx_end = idx_elev_end[idx_cell, idx_azim]
                elev_sparse = spacing_exp(elev_dense[idx_start],
                                          elev_dense[idx_end], num_nodes, eta)
                f_cor_ip = np.interp(elev_dense[(idx_start + 1):idx_end],
                                     elev_sparse,
                                     f_cor_sparse[idx_cell, idx_azim, :])
                f_cor_rec[(idx_start + 1):idx_end] = f_cor_ip
                # ---------- Recompute from 'terrain normal' ------------------
                for idx_elev in range(np.maximum(idx_end, 1), 91):
                    # ignore case 'idx_end = 0' -> dot product in denominator
                    # becomes 0.0 -> keep 'f_cor_rec = 0.0' for this case
                    s_vec_sel = s_vec[idx_azim, idx_elev, :]
                    f_cor_rec[idx_elev] = (1.0 / np.dot(s_vec_sel, h_vec)) \
                        * np.dot(s_vec_sel, t_vec)
                # -------------------------------------------------------------
                f_cor_dev = np.abs(f_cor_rec
                                   - f_cor_dense[idx_cell, idx_azim, :])
                radiation_dev = f_cor_dev * radiation
                indices = np.floor(radiation_dev * scaling).astype(np.int64)
                # indices = np.floor(radiation_dev * scaling).astype(np.int64)[0:71]
                # -> only consider elevation angles up to 70 degrees
                for idx in indices:
                    if (idx >= 0) and (idx < bin_size):
                        bin_counts[tid, idx] += 1
    return np.sum(bin_counts, axis=0)
