cimport numpy as np
import numpy as np

cdef extern from "mo_lradtopo_horayzon.h":
    void horizon_svf_comp(double* vlon, double* vlat,
                          double* elevation,
                          np.npy_int32* faces,
                          np.npy_int32* ind_hori_out,
                          float* f_cor,
                          double* horizon_out,
                          int num_vertex, int num_cell, int num_hori_out,
                          int num_cell_parent, int num_cell_child_per_parent,
                          int num_hori, double dist_search,
                          double ray_org_elev, int num_elev,
                          double sw_dir_cor_max, int cons_area_factor)

# Interface for Python function
def horizon_svf_comp_py(np.ndarray[np.float64_t, ndim = 1] vlon,
                        np.ndarray[np.float64_t, ndim = 1] vlat,
                        np.ndarray[np.float64_t, ndim = 1] elevation,
                        np.ndarray[np.int32_t, ndim = 2] faces,
                        np.ndarray[np.int32_t, ndim = 1] ind_hori_out,
                        int num_cell_parent,
                        int num_cell_child_per_parent,
                        int num_hori,
                        double dist_search,
                        double ray_org_elev,
                        int num_elev,
                        double sw_dir_cor_max,
                        int cons_area_factor):
    """
    Compute the terrain horizon and sky view factor.

    Parameters
    ----------
    vlon : ndarray of double
        Array with longitude of cell vertices (num_vertex) [rad]
    vlat : ndarray of double
        Array with latitude of cell vertices (num_vertex) [rad]
    elevation : ndarray of double
        Array with elevation of cell vertices (num_vertex) [m]
    faces : ndarray of int
        Array with indices of cell vertices. Indices start with 0 and are
        contiguous in memory as (tri0_v0, tri0_v1, tri0_v2, tri1_v0, ...)
        (num_cell, 3)
    ind_hori_out : ndarray of int
        Array with cell indices for which the computed horizon is outputted
    num_cell_parent : int
        Number of parent cells
    num_cell_child_per_parent : int
        Number of child cells per parent cell
    num_hori : int
        Number of terrain horizon sampling directions
    dist_search : double
        Radial search distance for horizon computation [m]
    ray_org_elev : double
        Vertical elevation of ray origin above surface [m]
    num_elev : int
        Number of elevation angles for SW_dir correction factor computation
    sw_dir_cor_max : double
        Maximum value for individual SW_dir correction factor [-]
    cons_area_factor : int
        Flag for using surface increase area factor for computing SW_dir
        correction factor (0: off, 1: on)

    Returns
    -------
    f_cor : ndarray of float
        Array with SW_dir correction factors
        (num_cell_parent, num_hori, num_elev) [-]
    horizon_out: ndarray of double
        Array with terrain horizon for selected cells (num_hori_out, num_hori)
        [deg]
    """

    # Check consistency and validity of input arguments
    if (vlon.size != vlat.size) or (vlat.size != elevation.size):
        raise ValueError("Inconsistent lengths of input arrays 'vlon', "
                         "'vlat' and 'elevation'")
    if faces.shape[1] != 3:
        raise ValueError("Second dimension of 'faces' must have length 3")
    if (faces.min() < 0) or (faces.max() > vlon.size - 1):
        raise ValueError("Indices of 'faces' out of range")
    if faces.shape[0] != num_cell_parent * num_cell_child_per_parent:
        raise ValueError("Inconsistency between shape of 'faces' and" \
                         + "'num_cell_parent' and 'num_cell_child_per_parent'")
    if (ind_hori_out.min() < 0) or (ind_hori_out.max() > faces.shape[0] - 1):
        raise ValueError("Indices of 'ind_hori_out' out of range")
    if np.unique(ind_hori_out).size != ind_hori_out.size:
        raise ValueError("Indices of 'ind_hori_out' must be unique")
    if (num_hori < 4) or (num_hori > 360):
        raise ValueError("'num_hori' must be in the range [4, 360]")
    if (dist_search < 1_000.0) or (dist_search > 500_000.0):
        raise ValueError("'dist_search' must be in the range " \
        + "[1'000, 500'000] km")
    if ray_org_elev < 0.1:
        raise TypeError("Minimal allowed value for 'ray_org_elev' is 0.1 m")
    if (num_elev < 11) or (num_elev > 361):
        raise ValueError("'num_elev' must be in the range [11, 361]")
    if (sw_dir_cor_max < 2) or (sw_dir_cor_max > 50):
        raise ValueError("'sw_dir_cor_max' must be in the range [2, 50]")
    if cons_area_factor not in (0, 1):
        raise ValueError("'cons_area_factor' must be either 0 or 1")

    # Allocate array for output
    cdef np.ndarray[np.float32_t, ndim = 3, mode = "c"] \
        f_cor = np.zeros((num_cell_parent, num_hori, num_elev),
        dtype=np.float32)
    cdef np.ndarray[np.float64_t, ndim = 2, mode = "c"] \
        horizon_out = np.empty((ind_hori_out.size, num_hori), dtype=np.float64)

    # Ensure that passed (multi-dimensional) arrays are contiguous in memory
    faces = np.ascontiguousarray(faces)

    # Call C++ function and pass arguments
    horizon_svf_comp(&vlon[0], &vlat[0],
                     &elevation[0],
                     &faces[0, 0],
                     &ind_hori_out[0],
                     &f_cor[0, 0, 0],
                     &horizon_out[0, 0],
                     vlon.size, faces.shape[0], ind_hori_out.size,
                     num_cell_parent, num_cell_child_per_parent,
                     num_hori, dist_search,
                     ray_org_elev, num_elev,
                     sw_dir_cor_max, cons_area_factor)

    return f_cor, horizon_out
