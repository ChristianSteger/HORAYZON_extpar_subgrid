# Load modules
import os
import sys
import time
import numpy as np
import xarray as xr

# Path to folders
path_extpar = "/scratch/snx3000/csteger/EXTPAR_HORAYZON/ICON_grids_EXTPAR/"
path_out = "/scratch/snx3000/csteger/EXTPAR_HORAYZON/output/"

# Path to Cython/C++ functions
sys.path.append("/scratch/snx3000/csteger/EXTPAR_HORAYZON/Semester_Project/")
from horizon_svf import horizon_svf_comp_py

# -----------------------------------------------------------------------------
# Real data (resolutions from ~2km to ~30 m)
# -----------------------------------------------------------------------------

# Load grid information
file_grid = "Resolutions/icon_grid_res0032m.nc"
# file_grid = "Resolutions/icon_grid_res0130m.nc"
# file_grid = "Resolutions/icon_grid_res0519m.nc"
# file_grid = "Resolutions/icon_grid_res2076m.nc"
ds = xr.open_dataset(path_extpar + file_grid)
vlon = ds["vlon"].values  # (num_vertex; float64)
vlat = ds["vlat"].values  # (num_vertex; float64)
clon = ds["clon"].values  # (num_cell; float64)
clat = ds["clat"].values  # (num_cell; float64)
vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
edge_of_cell = ds["edge_of_cell"].values - 1  # (3, num_cell)
adjacent_cell_of_edge = ds["adjacent_cell_of_edge"].values - 1  # (2, num_edge)
cells_of_vertex = ds["cells_of_vertex"].values - 1  # (6, num_vertex)
ds.close()

# Load elevation of cell vertices
file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res0032m.nc"
# file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res0130m.nc"
# file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res0519m.nc"
# file_topo = "Resolutions/topography_buffer_extpar_v5.8_icon_grid_res2076m.nc"
ds = xr.open_dataset(path_extpar + file_topo)
topography_v = ds["topography_v"].values.squeeze()  # (num_vertex; float32)
hsurf = ds["HSURF"].values.squeeze()  # (num_cell)
ds.close()

# Further settings
num_hori = 24
refine_factor = 10
svf_type = 3

# Find outermost cells and indices of cell vertices
mask_cell_outer = np.zeros(clon.size, dtype=bool)
for i in range(clon.size):
    if np.any(adjacent_cell_of_edge[:, edge_of_cell[:, i]] == -2):
        mask_cell_outer[i] = True
ind_vertices_outer = np.unique(vertex_of_cell[:, np.where(mask_cell_outer)[0]])

# Adjust erroneous elevation values of outermost cell vertices
for i in ind_vertices_outer:
    mask = cells_of_vertex[:, i] != -2
    topography_v[i] = hsurf[cells_of_vertex[:, i][mask]].mean()

# -----------------------------------------------------------------------------
# Compute horizon and sky view factor
# -----------------------------------------------------------------------------

t_beg = time.perf_counter()
horizon, skyview = horizon_svf_comp_py(vlon, vlat, topography_v,
                                       clon, clat, vertex_of_cell,
                                       num_hori, refine_factor, svf_type)
print("Total elapsed time: %.5f" % (time.perf_counter() - t_beg) + " s")

# Check range of computes values
print("Terrain horizon range [deg]: %.5f" % np.min(horizon)
      + ", %.5f" % np.max(horizon))
print("Sky view factor range [-]: %.8f" % np.min(skyview)
      + ", %.8f" % np.max(skyview))

# -----------------------------------------------------------------------------
# Save output to NetCDF file
# -----------------------------------------------------------------------------

ds = xr.Dataset({
    "HORIZON": xr.DataArray(
        data=horizon,
        dims=["nhori", "cell"],
        attrs={
            "units": "deg"
        }
    ),
    "SKYVIEW": xr.DataArray(
        data=skyview,
        dims=["cell"],
        attrs={
            "units": "-"
        }
    )
},
    attrs={"settings": "refine_factor: " + str(refine_factor)
                       + ", svf_type: " + str(svf_type)}
)
file_out = file_grid.split("/")[-1].split(".")[0] + "_horizon_svf.nc"
ds.to_netcdf(path_out + file_out)
