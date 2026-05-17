# Description: Test ICON implementation of subgrid-f_cor correction
#
# Author: Christian R. Steger, May 2026

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors

from icon_implement.python import interpolate_fcor as ip_fcor_python
from icon_implement.fortran import interpolate_fcor as ip_fcor_fortran

###############################################################################
# Test
###############################################################################

# Constant settings
num_azim = 24
eta = 2.0
num_nodes = 7 # all interpolation nodes including bounds

# # Artificial input data
# horizon = np.empty(num_azim * 3, dtype=np.float32)
# horizon[0:None:3] = 15.0
# horizon[1:None:3] = 30.0
# horizon[2:None:3] = 45.0
# horizon[0:3] = [25.0, 47.5, 68.3] # first azimuth sector
# terrain_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
# swdir_cor = np.empty(num_azim * (num_nodes - 2), dtype=np.float32)
# swdir_cor[0:None:4] = 0.1
# swdir_cor[1:None:4] = 0.2
# swdir_cor[2:None:4] = 0.6
# swdir_cor[3:None:4] = 1.3
# swdir_cor[0:4] = [0.2, 0.5, 0.7, 1.8] # first azimuth sector

# Real input data from EXTPAR
file = "/scratch/mch/csteger/ICON-CH1-EPS_copy_inn/" \
    + "external_parameter_icon_grid_0001_R19B08_mch_tuned_horizon_subgrid.nc"
ds = xr.open_dataset(file)
num_gc_icon = ds["cell"].size
# idx_cell = 750_000
# idx_cell = 790_610
idx_cell = 777125 # total shadow until ca. ~25 deg
# idx_cell = 680_000 # always below 1.0
# idx_cell = 580_000  # up to 6.0
# idx_cell = np.random.randint(0, num_gc_icon, 1)[0]
horizon = ds["HORIZON"][:, idx_cell].values
swdir_cor = ds["SWDIR_COR"][:, idx_cell].values
terrain_normal = ds["TERRAIN_NORMAL"][:, idx_cell].values
ds.close()

# -----------------------------------------------------------------------------
# Single sun position
# -----------------------------------------------------------------------------

# Sun position
# zphi_sun = np.deg2rad(225.5) # sun azimuth angle [rad]
zphi_sun = np.deg2rad(np.random.uniform(0.0, 360.0, 1)[0])
# zphi_sun = np.deg2rad(360.0)
# ztheta_sun = np.deg2rad(90.1) # sun elevation angle [rad]
ztheta_sun = np.deg2rad(np.random.uniform(0.0, 90.0, 1)[0])
print(f"Sun position: azimuth {np.rad2deg(zphi_sun):.2f} deg," +
      f" elevation {np.rad2deg(ztheta_sun):.2f} deg")

# Compute
print("Python (Numba) implementation".center(60, '-'))
f_cor_sun = ip_fcor_python(horizon, swdir_cor, terrain_normal,
                             ztheta_sun, zphi_sun)
print(f"fcor_sun: {f_cor_sun:.4f}")
print("ICON (Fortran) implementation".center(60, '-'))
f_cor_sun, zha_sun = ip_fcor_fortran(horizon, swdir_cor, terrain_normal,
                                     ztheta_sun, zphi_sun)
print(f"fcor_sun: {f_cor_sun:.4f}")
print(f"zha_sun: {zha_sun:.1f} deg")

# -----------------------------------------------------------------------------
# Sample entire azimuth/elevation space
# -----------------------------------------------------------------------------

# Sun positions
zphi_sun = np.arange(0.0, 360.0, 1.0)
ztheta_sun = np.arange(-3.0, 91.0, 1.0)
f_cor_sun = np.empty((2, ztheta_sun.size, zphi_sun.size), dtype=np.float32)
zha_sun = np.empty((ztheta_sun.size, zphi_sun.size), dtype=np.float32)
for i in range(ztheta_sun.size):
    for j in range(zphi_sun.size):
        f_cor_sun[0, i, j] = ip_fcor_python(
            horizon, swdir_cor, terrain_normal,
            np.deg2rad(ztheta_sun[i]), np.deg2rad(zphi_sun[j]))
        f_cor_sun[1, i, j], zha_sun[i, j] = ip_fcor_fortran(
            horizon, swdir_cor, terrain_normal,
            np.deg2rad(ztheta_sun[i]), np.deg2rad(zphi_sun[j]))

# Check maximal absolute deviation between two implementations
dev_abs_max = np.abs(np.diff(f_cor_sun, axis=0).max()).max()
print(f"Maximal absolute deviation: {dev_abs_max:.8f}")

# Check median horizon values
if np.any(np.diff(zha_sun, axis=0) != 0.0):
    raise ValueError("Computation of median horizon values erroneous")

# Plot
levels = np.arange(0.0, 2.1, 0.1)
cmap = plt.get_cmap("RdGy_r")
norm = colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
plt.figure()
plt.pcolormesh(zphi_sun, ztheta_sun, f_cor_sun[1, :, :], cmap=cmap, norm=norm)
plt.colorbar()
plt.plot(zphi_sun, zha_sun[0, :], lw=1.5, color="blue")
plt.show()
