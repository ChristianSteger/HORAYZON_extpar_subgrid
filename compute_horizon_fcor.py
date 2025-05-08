# Description: Compute terrain horizon and slope angle/aspect for subgrid
#              cells, spatially aggregated correction factors (f_cor) and
#              compress f_cor information in discrete values.
#
# Author: Christian R. Steger, May 2025

import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri, colors

from functions import centroid_values

style.use("classic")

# Paths
path_in = "/scratch/mch/csteger/temp/ICON_refined_mesh/"
# path_in = "/Users/csteger/Desktop/"

# Path to Cython/C++ functions
sys.path.append("/scratch/mch/csteger/HORAYZON_extpar_subgrid/")
# from horizon_svf import horizon_svf_comp_py

###############################################################################
# Load refined ICON mesh
###############################################################################

# Select mesh file
file_mesh = "ICON_refined_mesh_test_2km.nc"
# file_mesh = "ICON_refined_mesh_mch_500m.nc"
# file_mesh = "ICON_refined_mesh_mch_1km.nc"
# file_mesh = "ICON_refined_mesh_mch_2km.nc"

# Load data
ds = xr.open_dataset(path_in + file_mesh)
vlon = ds["vlon"].values # (num_vertex; float64)
vlat = ds["vlat"].values # (num_vertex; float64)
elevation = ds["elevation"].values # (num_vertex; float32)
faces = ds["faces"].values # (num_cell, 3; int32) (transposed 'vertex_of_cell')
parent_tri_id = ds["parent_tri_id"].values # (num_cell; int32)
ds.close()

# -----------------------------------------------------------------------------
# Test plot
# -----------------------------------------------------------------------------

# num_tri = 150_000
num_tri = (parent_tri_id == 0).sum() * 25
triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                 faces[:num_tri, :])

# Elevation
elevation_centroids = centroid_values(elevation, faces)
plt.figure(figsize=(10, 10))
plt.tripcolor(triangles, elevation_centroids[:num_tri], cmap="terrain",
              vmin=elevation_centroids[:num_tri].min(),
              vmax=elevation_centroids[:num_tri].max(),
              edgecolors="black", linewidth=0.1)
plt.show()

# Parent ID
plt.figure(figsize=(10, 10))
plt.tripcolor(triangles, parent_tri_id[:num_tri], cmap="Spectral",
              vmin=parent_tri_id[:num_tri].min(),
              vmax=parent_tri_id[:num_tri].max(),
              edgecolors="black", linewidth=0.1)
plt.show()

###############################################################################
# Compute spatially aggregated correction factors (f_cor)
###############################################################################

# Settings 
num_hori = 2 # number of azimuth angles
dist_search = 40_000.0 #  horizon search distance [m]
ray_org_elev = 0.2 # 0.1, 0.2 [m]

# Compute f_cor
cells_of_vertex = np.ascontiguousarray(faces)
del faces
f_cor = horizon_svf_comp_py(
        vlon, vlat, elevation.astype(np.float64),
        cells_of_vertex,
        num_hori, dist_search,
        ray_org_elev)

###############################################################################
# Compress f_cor information in discrete values
###############################################################################
