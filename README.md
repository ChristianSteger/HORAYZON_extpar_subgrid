# HORAYZON_extpar_subgrid

Implement HORAYZON algorithm in pre-processing tool EXTPAR (on subgrid-scale).

# Installation

 Create Conda environment:
```bash
conda create -n horayzon_extpar_subgrid -c conda-forge embree tbb-devel cython setuptools numpy xarray netcdf4 matplotlib cartopy pyproj scipy numba pyinterp trimesh ipython skyfield
```
activate this environment, clone **HORAYZON_extpar_subgrid** and compile with:
```bash
git clone git@github.com:ChristianSteger/HORAYZON_extpar_subgrid.git
cd HORAYZON_extpar_subgrid
python setup.py build_ext --inplace

