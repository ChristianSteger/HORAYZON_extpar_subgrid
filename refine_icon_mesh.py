# Description: Refined ICON mesh for terrain horizon computation
#
# Author: Christian R. Steger, May 2025

import math
from time import perf_counter

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import style, tri
from scipy.spatial import KDTree
import pyinterp
from netCDF4 import Dataset

from functions import refine_mesh_nc, alpha_minmax, centroid_values

style.use("classic")

# Paths
path_ige = "/store_new/mch/msopr/csteger/Data/Miscellaneous/" \
    + "ICON_grids_EXTPAR/"
path_dem = "/store_new/mch/msopr/csteger/Data/DEMs/Copernicus_DEM/"
path_plot = "/scratch/mch/csteger/HORAYZON_extpar/plots/"
path_out = "/scratch/mch/csteger/temp/"
# path_ige = "/Users/csteger/Dropbox/MeteoSwiss/Data/Miscellaneous/" \
#     + "ICON_grids_EXTPAR/"
# path_dem = "/Users/csteger/Dropbox/MeteoSwiss/Data/DEMs/Copernicus_DEM/"
# path_plots = "/Users/csteger/Desktop/"
# path_out = "/Users/csteger/Desktop/"

###############################################################################
# Load ICON grid data and pre-process
###############################################################################

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Select ICON resolution
icon_res = "2km"  # "2km", "1km", "500m"

# ICON grids
# icon_grids = {"2km": "MeteoSwiss/icon_grid_0002_R19B07_mch.nc",
#               "1km": "MeteoSwiss/icon_grid_0001_R19B08_mch.nc",
#               "500m": "MeteoSwiss/icon_grid_00005_R19B09_DOM02.nc"}
icon_grids = {"2km": "test/icon_grid_DOM01.nc"}

# Output file name (part)
# file_out = "ICON_refined_mesh_" + "test_" + icon_res + ".nc"
file_out = "ICON_refined_mesh_" + "mch_" + icon_res + ".nc"

# Optional (computational intensive) mesh checking steps
check_mesh = False

# Mesh refinement level
n_sel = 73 # Test 2km -> 'faces' < 16 GB
# n_sel = 33 # MCH 1km -> 'faces' < 16 GB

# -----------------------------------------------------------------------------

# Resolution of DEM
res_dem = (40075.0 * 1000.0 / 360.0) * (1.0 / 3600.0)  # ~max. resolution [m]
print(f"DEM resolution: {(res_dem):.1f} m")
cell_area_dem = res_dem ** 2  # [m2]

# Load ICON grid with specific resolution
ds = xr.open_dataset(path_ige + icon_grids[icon_res])
vlon = ds["vlon"].values
vlat = ds["vlat"].values
vertex_of_cell = ds["vertex_of_cell"].values - 1  # (3, num_cell; int32)
# ordered counter-clockwise
edge_of_cell = ds["edge_of_cell"].values - 1  # (3, num_cell; int32)
# ordered counter-clockwise
edge_vertices = ds["edge_vertices"].values - 1  # (2, num_edge; int32)
grid_level = int(ds.attrs["grid_level"])
grid_root = int(ds.attrs["grid_root"])
ds.close()
res_icon = 5500.0 / (grid_root * 2 ** grid_level) * 1000  # [m]
print(f"ICON resolution: {res_icon:.1f} m")
cell_area_icon = res_icon ** 2  # [m2]

# Cartesian coordinates (unit sphere)
vx = np.cos(vlat) * np.cos(vlon)
vy = np.cos(vlat) * np.sin(vlon)
vz = np.sin(vlat)
vector_v = np.ascontiguousarray(np.vstack([vx, vy, vz]).T)

###############################################################################
# Refine mesh by splitting into n ** 2 child triangles
###############################################################################

# Theoretical optimal refinement level
print(" Theoretical refinement level ".center(60, "-"))
# n ** 2 = cell_area_icon / cell_area_dem -> solve for n
n_theo = math.sqrt(cell_area_icon / cell_area_dem)
n_theo = round(n_theo) # closest to DEM resolution
# n_theo = math.ceil(n_theo) # first higher resolution than DEM
print(f"Division steps (n): {n_theo}")
res_icon_ref = math.sqrt(cell_area_icon / (n_theo ** 2))
print(f"ICON resolution: {res_icon_ref:.1f} m")
num_tri_ref = vertex_of_cell.shape[1] * (n_theo ** 2)
print(f"Number of resulting triangles: {num_tri_ref:,}".replace(",", "'"))
print(f"Size of 'faces_child' array: {(12 * num_tri_ref / 1e9):.2f} GB")

# Select refinement level
print(" Selected refinement level ".center(60, "-"))
print(f"Division steps (n): {n_sel}")
res_icon_ref = math.sqrt(cell_area_icon / (n_sel ** 2))
print(f"ICON resolution: {res_icon_ref:.1f} m")
num_tri_ref = vertex_of_cell.shape[1] * (n_sel ** 2)
print(f"Number of resulting triangles: {num_tri_ref:,}".replace(",", "'"))
print(f"Size of 'faces_child' array: {(12 * num_tri_ref / 1e9):.2f} GB")
if (12 * num_tri_ref / 1e9) > 16.0:
    raise ValueError("Array 'faces_child' is too large (> 16 GB)")
print("-" * 60)

# Refine ICON triangle mesh
vertices = vector_v.copy()
t_beg = perf_counter()
vertices_child, faces_child = refine_mesh_nc(vertices, vertex_of_cell,
                                             edge_of_cell, edge_vertices,
                                             n_sel)
t_end = perf_counter()
print(f"ICON mesh refinement: {t_end - t_beg:.1f} s")

# Check that vertices are correctly connected into triangles
if ((faces_child.min() != 0)
    or (vertices_child.shape[0] != faces_child.max() + 1)):
    raise ValueError("Array 'faces_child' is erroneous")
# -------------------- dot product check --------------------
alpha_min, alpha_max, centroids = alpha_minmax(vertices_child, faces_child)
rad_earth = 6_378_000.0 # [m]
dist_per_deg = 2.0 * np.pi * rad_earth / 360.0  # [m / deg]
print(f"Minimal chord distance {(alpha_min * dist_per_deg):.1f} m")
print(f"Maximal chord distance {(alpha_max * dist_per_deg):.1f} m")
# ----------------- nearest neighbour check -----------------
if check_mesh:
    tree = KDTree(vertices_child)
    dist, ind = tree.query(centroids, k=3, workers=4)
    if np.any(np.sort(faces_child, axis=1) != np.sort(ind, axis=1)):
        raise ValueError("Array 'faces_child' is erroneous")
    del tree, dist, ind
# -----------------------------------------------------------
del centroids

# ID of parent triangle
parent_tri_id = np.repeat(np.arange(vertex_of_cell.shape[1]), n_sel ** 2) \
    .astype(np.int32)

# Compute spherical coordinates (longitude/latitude) of child vertices
t_beg = perf_counter()
vlon_child = np.arctan2(vertices_child[:, 1], vertices_child[:, 0])
vlat_child = np.arcsin(vertices_child[:, 2])
del vertices_child
t_end = perf_counter()
print(f"Coordinate transformation: {t_end - t_beg:.1f} s")

# Check part of the mesh
if check_mesh:
    plt.figure(figsize=(15, 15))
    triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                vertex_of_cell.transpose())
    plt.triplot(triangles, color="black", lw=0.8, ls="-")
    num_tri_parent = 10
    num_tri_child = num_tri_parent * (n_sel ** 2)
    triangles_child = tri.Triangulation(np.rad2deg(vlon_child),
                                        np.rad2deg(vlat_child),
                                        faces_child[:num_tri_child, :])
    vmin = parent_tri_id[:num_tri_child].min()
    vmax = parent_tri_id[:num_tri_child].max()
    plt.tripcolor(triangles_child, facecolors=parent_tri_id[:num_tri_child],
                cmap="Spectral", vmin=vmin, vmax=vmax, alpha=0.5)
    plt.triplot(triangles_child, color="black", lw=0.8, ls=":")
    plt.show()
    del triangles, triangles_child

# Load raw DEM data (required domain)
t_beg = perf_counter()
add = 0.02 # 'safety margin' [deg]
files_dem = ("Copernicus_DEM_N50-N40_W020-E000.nc",
             "Copernicus_DEM_N50-N40_E000-E020.nc",
             "Copernicus_DEM_N60-N50_W020-E000.nc",
             "Copernicus_DEM_N60-N50_E000-E020.nc")
ds = xr.open_mfdataset([path_dem + i for i in files_dem],
                       mask_and_scale=False)
ds = ds.sel(lon=slice(np.rad2deg(vlon.min()) - add,
                      np.rad2deg(vlon.max()) + add),
            lat=slice(np.rad2deg(vlat.max()) + add,
                      np.rad2deg(vlat.min()) - add))
lon_dem = np.deg2rad(ds["lon"].values) # float64, [rad]
lat_dem = np.deg2rad(ds["lat"].values) # float64, [rad]
elevation_dem = ds["elevation"].values # int16, [m]
ds.close()
t_end = perf_counter()
print(f"Load raw DEM: {t_end - t_beg:.1f} s")

# Interpolate elevation data on refined mesh vertices
t_beg = perf_counter()
x_axis = pyinterp.Axis(lon_dem) # type: ignore
y_axis = pyinterp.Axis(lat_dem) # type: ignore
grid = pyinterp.Grid2D(x_axis, y_axis, elevation_dem.transpose())
velev_child = pyinterp.bivariate(
    grid, vlon_child, vlat_child, interpolator="bilinear", bounds_error=True,
    num_threads=8).astype(np.float32) # [m]
del x_axis, y_axis, grid
t_end = perf_counter()
print(f"Interpolate elevation data: {t_end - t_beg:.1f} s")

# Check interpolated elevation
if check_mesh:

    # Values at mesh cell centroids
    clon_child = centroid_values(vlon_child, faces_child)
    clat_child = centroid_values(vlat_child, faces_child)
    celev_child = centroid_values(velev_child, faces_child)
    sub_dom = (6.65, 7.15, 45.75, 45.95)
    mask = ((np.rad2deg(clon_child) > sub_dom[0])
            & (np.rad2deg(clon_child) < sub_dom[1])
            & (np.rad2deg(clat_child) > sub_dom[2])
            & (np.rad2deg(clat_child) < sub_dom[3]))
    del clon_child, clat_child
    print(f"Triangle number in sub-domain: {mask.sum()}")

    # Test plot
    plt.figure(figsize=(15, 10))
    triangles_child = tri.Triangulation(np.rad2deg(vlon_child),
                                        np.rad2deg(vlat_child),
                                        faces_child[mask, :])
    plt.tripcolor(triangles_child, facecolors=celev_child[mask],
                cmap="terrain", vmin=celev_child.min(), vmax=celev_child.max(),
                edgecolor="black", linewidth=0.1)
    plt.colorbar()
    triangles = tri.Triangulation(np.rad2deg(vlon), np.rad2deg(vlat),
                                vertex_of_cell.transpose())
    plt.triplot(triangles, color="black", lw=0.8, ls="-")
    plt.scatter(6.864325, 45.832544, s=100, marker="^", color="black")
    # Mont Blanc
    plt.axis(sub_dom)
    plt.show()
    del mask, triangles, triangles_child, celev_child

# Write refined mesh to netCDF file
t_beg = perf_counter()
ncfile = Dataset(filename=path_out + file_out, mode="w", format="NETCDF4")
ncfile.createDimension(dimname="num_vertex", size=vlon_child.size)
ncfile.createDimension(dimname="num_cell", size=faces_child.shape[0])
ncfile.createDimension(dimname="tri_vertex", size=3)
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="vlon", datatype="f8",
                                dimensions=("num_vertex"))
nc_data.units = "radian"
nc_data.long_name = "longitude of vertices"
nc_data[:] = vlon_child
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="vlat", datatype="f8",
                                dimensions=("num_vertex"))
nc_data.units = "radian"
nc_data.long_name = "latitude of vertices"
nc_data[:] = vlat_child
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="elevation", datatype="f4",
                                dimensions=("num_vertex"))
nc_data.units = "m"
nc_data.long_name = "elevation of vertices"
nc_data[:] = velev_child
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="faces", datatype="i4",
                                dimensions=("num_cell", "tri_vertex"))
nc_data.long_name = "transposed vertex_of_cell"
nc_data[:] = faces_child
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="parent_tri_id", datatype="i4",
                                dimensions=("num_cell"))
nc_data.long_name = "ID of parent triangle"
nc_data[:] = parent_tri_id
# -----------------------------------------------------------------------------
ncfile.close()
t_end = perf_counter()
print(f"Write output NetCDF file: {t_end - t_beg:.1f} s")
