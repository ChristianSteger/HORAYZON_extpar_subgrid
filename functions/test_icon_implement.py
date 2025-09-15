# Description: Test ICON implementation of subgrid-f_cor correction
#
# Author: Christian R. Steger, September 2025

import numpy as np

from functions.fcor_processing import spacing_exp, spacing_exp_interp
from functions.icon_implement import interpolate # type: ignore

###############################################################################
# Test
###############################################################################

# Create exponentially spaced array
x_start = float(np.random.uniform(0.0, 40.0, 1)[0])
# x_start = 34.6
print(f"x_start: {x_start:.4f}")
x_end = 90.0
num_nodes = 7
eta = 2.1
x_spac = spacing_exp(x_start, x_end, num_nodes, eta)

# Check interpolation
x_ip = np.random.uniform(x_start, x_end, 1)[0]
# x_ip = 85.7
# x_ip = x_start - 0.00001
x_ip = x_end - 0.000000000000001
y = np.random.uniform(0.0, 1.0, num_nodes).astype(np.float32)
y[0], y[-1] = 0.0, 1.0
y_ip = spacing_exp_interp(x_start, x_end, num_nodes, eta, x_ip, y)
if abs(y_ip - np.interp(x_ip, x_spac, y, left=0.0, right=1.0)) > 1e-6:
    raise ValueError("Interpolation erroneous")
print(f"y_ip: {y_ip:.4f}")

horizon = np.empty(24 * 8, dtype=np.float32)
horizon[0] = x_start
horizon[1:8] = y
horizon[8] = x_start
horizon[9:16] = y

zphi_sun = np.deg2rad(7.5) # sun azimuth angle [rad]
ztheta_sun = np.deg2rad(x_ip) # sun elevation angle [rad]

fcor_sun = interpolate(horizon, ztheta_sun, zphi_sun)
print(f"fcor_sun: {fcor_sun:.4f}")