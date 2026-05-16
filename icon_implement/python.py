# Description: Interpolate sub-grid shortwave correction factor for specific
#              sun position (Python/Numba version)
#
# Author: Christian R. Steger, May 2026

import numpy as np

from functions.fcor_processing import spacing_exp, spacing_exp_interp

def interpolate_fcor(horizon, swdir_cor, terrain_normal,
                     ztheta_sun, zphi_sun):
  
    # Constant values
    num_azim = 24
    eta = 2.0
    num_nodes = 7 # all interpolation nodes including bounds

    # Azimuth indices
    idx_azim_0 = np.minimum(int(zphi_sun / np.deg2rad(360.0 / num_azim)),
                               num_azim - 1)
    idx_azim_1 = (idx_azim_0 + 1) % num_azim
    azim_idx = np.array([idx_azim_0, idx_azim_1])

    # Compute direct shortwave correction factors for both azimuth directions
    fcor = np.empty(2, dtype=np.float32)
    azim = np.arange(0.0, 360.0, 360.0 / num_azim)
    h_vec = np.array([0.0, 0.0, 1.0])
    num_nodes_in = num_nodes - 2
    for i in range(2):

        horizon_min = horizon[azim_idx[i] * 3 + 0] # min. horizon angle [deg]
        horizon_max = horizon[azim_idx[i] * 3 + 2] # max. horizon angle [deg]

        # Sun position is below maximal horizon
        if np.rad2deg(ztheta_sun) <= horizon_max:

            nodes_elev = spacing_exp(horizon_min, horizon_max, num_nodes, eta)
            nodes_fcor = np.empty(num_nodes, dtype=np.float32)
            nodes_fcor[0] = 0.0
            nodes_fcor[1:(num_nodes - 1)] \
                = swdir_cor[azim_idx[i] * num_nodes_in:azim_idx[i] \
                            * num_nodes_in + num_nodes_in]
            sun_vec = np.array([
                np.cos(np.deg2rad(horizon_max)) \
                    * np.sin(np.deg2rad(azim[azim_idx[i]])),
                np.cos(np.deg2rad(horizon_max)) \
                    * np.cos(np.deg2rad(azim[azim_idx[i]])),
                np.sin(np.deg2rad(horizon_max))
            ])
            dot_prod_s_h = np.dot(sun_vec, h_vec).clip(min=1e-5)
            nodes_fcor[num_nodes - 1] = (1.0 / dot_prod_s_h) \
                * np.dot(sun_vec, terrain_normal)
            fcor[i] = np.interp(np.rad2deg(ztheta_sun), nodes_elev, nodes_fcor)

        # Sun position is above maximal horizon
        else:

            sun_vec = np.array([
                np.cos(ztheta_sun) * np.sin(np.deg2rad(azim[azim_idx[i]])),
                np.cos(ztheta_sun) * np.cos(np.deg2rad(azim[azim_idx[i]])),
                np.sin(ztheta_sun)
            ])
            dot_prod_s_h = np.dot(sun_vec, h_vec).clip(min=1e-5)
            fcor[i] = (1.0 / dot_prod_s_h) * np.dot(sun_vec, terrain_normal)

    # Interpolate fcor-value at sun's azimuth angle
    azim_spac = 360.0 / num_azim
    weight_left = (azim_spac * (idx_azim_0 + 1) - np.rad2deg(zphi_sun)) \
        / azim_spac
    weight_right = (np.rad2deg(zphi_sun) - azim_spac * idx_azim_0) \
        / azim_spac
    fcor_sun = weight_left * fcor[0] + weight_right * fcor[1]

    return fcor_sun
