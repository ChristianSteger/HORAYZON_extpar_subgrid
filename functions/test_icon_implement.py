# Description: Test ICON implementation of subgrid-f_cor correction
#
# Author: Christian R. Steger, September 2025

import numpy as np
import xarray as xr

from functions.fcor_processing import spacing_exp, spacing_exp_interp
from functions.icon_implement import interpolate # type: ignore

###############################################################################
# Test
###############################################################################

# # Artificial input data
# num_azim = 24
# num_elem = 8
# f_cor_sparse = np.random.uniform(0.0, 5.0, num_azim * num_elem) \
#     .astype(np.float32) # [-]
# f_cor_sparse[0:None:num_elem] = 10.0 # starting elevation angle [degrees]
# # f_cor_sparse[0:None:num_elem] = np.random.uniform(0.0, 35.0, num_azim)
# f_cor_sparse[1:None:num_elem] = 0.0 # f_cor @ elev_start
# f_cor_sparse[7:None:num_elem] = 1.0 # f_cor @ elev_end

# Real input data from EXTPAR
num_azim = 24
num_elem = 8
file = "/scratch/mch/csteger/ICON-CH1-EPS_copy/" \
    + "external_parameter_icon_grid_0001_R19B08_mch_tuned_f_cor_sparse.nc"
ds = xr.open_dataset(file)
num_gc_icon = ds["cell"].size
# ind_loc = 750_000
# ind_loc = 790_610
# ind_loc = 777125 # total shadow until ca. ~25 deg
# ind_loc = 680_000 # always below 1.0
ind_loc = 580_000  # up to 6.0
ind_loc = np.random.randint(0, num_gc_icon, 1)[0]
f_cor_sparse = ds["HORIZON"][:, ind_loc].values
ds.close()

# Settings
zphi_sun = np.deg2rad(np.random.uniform(0.0, 360.0, 1)[0])
# zphi_sun = np.deg2rad(360.0)
# sun azimuth angle [rad]
ztheta_sun = np.deg2rad(np.random.uniform(0.0, 90.0, 1)[0])
# ztheta_sun = np.deg2rad(f_cor_sparse[0]) + 0.0000000001
# ztheta_sun = np.deg2rad(+90.3001)
# sun elevation angle [rad]
print(f"Sun position: azimuth {np.rad2deg(zphi_sun):.2f} deg," +
      f" elevation {np.rad2deg(ztheta_sun):.2f} deg")

# -----------------------------------------------------------------------------
# Python (Numba) implementation
# -----------------------------------------------------------------------------
print("Python (Numba) implementation".center(60, '-'))

# Azimuth indices
ind_azim_left = np.minimum(int(zphi_sun / np.deg2rad(360.0 / num_azim)),
                           num_azim - 1)
ind_azim_right = (ind_azim_left + 1) % num_azim

# Fixed settings
eta = 2.1
elev_end = 90.0

# Left and right f_cor-values
f_cor_values = np.empty(2, dtype=np.float32)
for ind_i, ind_azim in enumerate([ind_azim_left, ind_azim_right]):
    ind_azim_start = ind_azim * num_elem
    elev_start = f_cor_sparse[ind_azim_start]
    f_cor_sparse_azim \
        = f_cor_sparse[ind_azim_start + 1:ind_azim_start + num_elem]
    f_cor_values[ind_i] = spacing_exp_interp(
        elev_start, elev_end, num_elem - 1, eta, np.rad2deg(ztheta_sun),
        f_cor_sparse_azim)
    x_spac = spacing_exp(elev_start, elev_end, num_elem - 1, eta)
    f_cor_check = np.interp(np.rad2deg(ztheta_sun), x_spac, f_cor_sparse_azim,
                            left=0.0, right=1.0)
    if abs(f_cor_check - f_cor_values[ind_i]) > 1e-6:
        raise ValueError("Interpolation erroneous")
with np.printoptions(precision=4, suppress=True):
    print(f"f_cor values (left, right): {f_cor_values}")

# Compute f_cor for sun position
azim_spac = 360.0 / num_azim
weight_right = (np.rad2deg(zphi_sun) - ind_azim_left * azim_spac) / azim_spac
f_cor_sun = f_cor_values[0] * (1.0 - weight_right) \
    + f_cor_values[1] * weight_right
print(f"fcor_sun: {f_cor_sun:.4f}")

# -----------------------------------------------------------------------------
# ICON (Fortran) implementation
# -----------------------------------------------------------------------------
print("ICON (Fortran) implementation".center(60, '-'))

f_cor_sun = interpolate(f_cor_sparse, ztheta_sun, zphi_sun)
print(f"fcor_sun: {f_cor_sun:.4f}")
