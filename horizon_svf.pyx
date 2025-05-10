cimport numpy as np
import numpy as np

cdef extern from "mo_lradtopo_horayzon.h":
    void horizon_svf_comp(double* vlon, double* vlat,
                          double* elevation,
                          np.npy_int32* faces,
                          np.npy_int32* parent_indptr,
                          float* f_cor,
                          int num_vertex, int num_cell, int num_cell_parent,
                          int num_hori, double dist_search,
                          double ray_org_elev)

# Interface for Python function
def horizon_svf_comp_py(np.ndarray[np.float64_t, ndim = 1] vlon,
                        np.ndarray[np.float64_t, ndim = 1] vlat,
                        np.ndarray[np.float64_t, ndim = 1] elevation,
                        np.ndarray[np.int32_t, ndim = 2] faces,
                        np.ndarray[np.int32_t, ndim = 1] parent_indptr,
                        int num_hori,
                        double dist_search,
                        double ray_org_elev):
    """Compute the terrain horizon and sky view factor.

    Parameters
    ----------
    vlon : ndarray of double
        Array with longitude of cell vertices
        (num_vertex) [rad]
    vlat : ndarray of double
        Array with latitude of cell vertices
        (num_vertex) [rad]
    elevation : ndarray of double
        Array with elevation of cell vertices
        (num_vertex) [m]
    faces : ndarray of int
        Array with indices of cell vertices. Indices start with 0 and are
        contiguous in memory as (tri0_v0, tri0_v1, tri0_v2, tri1_v0, ...)
        (num_cell, 3)
    parent_indptr : ndarray of int
        Index pointer linking child to parent triangles (num_cell_parent + 1)
    num_hori : int
        Number of terrain horizon sampling directions
    dist_search : double
        Radial search distance for horizon computation [m]
    ray_org_elev : double
        Vertical elevation of ray origin above surface [m]

    Returns
    -------
    f_cor : ndarray of float
        Array (two-dimensional) with terrain horizon [deg]"""

    # Check consistency and validity of input arguments
    if (vlon.size != vlat.size) or (vlat.size != elevation.size):
        raise ValueError("Inconsistent lengths of input arrays 'vlon', "
                         "'vlat' and 'elevation'")
    if faces.shape[1] != 3:
        raise ValueError("Second dimension of 'faces' must "
            + "have length 3")
    if (faces.min() < 0) or (faces.max() > vlon.size - 1):
        raise ValueError("Indices of 'faces' out of range")
    if (num_hori < 4) or (num_hori > 90):
        raise ValueError("'num_hori' must be in the range [4, 90]")
    if (dist_search < 1_000.0) or (dist_search > 500_000.0):
        raise ValueError("'dist_search' must be in the range " \
        + "[1'000, 500'000] km")
    if ray_org_elev < 0.1:
        raise TypeError("Minimal allowed value for 'ray_org_elev' is 0.1 m")

    # Allocate array for output
    cdef np.ndarray[np.float32_t, ndim = 3, mode = "c"] \
        f_cor = np.empty((10, num_hori, 91), dtype=np.float32) ################ temporary
    # (parent_indptr.size - 1, num_hori, 91)

    # Ensure that passed arrays are contiguous in memory
    faces = np.ascontiguousarray(faces)

    # Call C++ function and pass arguments
    horizon_svf_comp(&vlon[0], &vlat[0],
                     &elevation[0],
                     &faces[0, 0],
                     &parent_indptr[0],
                     &f_cor[0, 0, 0],
                     vlon.size, faces.shape[0],
                     parent_indptr.size - 1,
                     num_hori, dist_search,
                     ray_org_elev)

    return f_cor
