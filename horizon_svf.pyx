cimport numpy as np
import numpy as np

cdef extern from "mo_lradtopo_horayzon.h":
    void horizon_svf_comp(double* clon, double* clat, double* hsurf,
                          double* vlon, double* vlat,
                          np.npy_int32* cells_of_vertex,
                          double* horizon, double* skyview,
                          int num_cell, int num_vertex, int num_hori,
                          int grid_type, double dist_search,
                          double ray_org_elev, int refine_factor,
                          int svf_type)

# Interface for Python function
def horizon_svf_comp_py(np.ndarray[np.float64_t, ndim = 1] clon,
                        np.ndarray[np.float64_t, ndim = 1] clat,
                        np.ndarray[np.float64_t, ndim = 1] hsurf,
                        np.ndarray[np.float64_t, ndim = 1] vlon,
                        np.ndarray[np.float64_t, ndim = 1] vlat,
                        np.ndarray[np.int32_t, ndim = 2] cells_of_vertex,
                        int num_hori,
                        int grid_type,
                        double dist_search,
                        double ray_org_elev,
                        int refine_factor,
                        int svf_type):
    """Compute the terrain horizon and sky view factor.

    Parameters
    ----------
    clon : ndarray of double
        Array with longitude of ICON grid cell circumcenters
        (number of ICON cells) [rad]
    clat : ndarray of double
        Array with latitude of ICON grid cell circumcenters
        (number of ICON cells) [rad]
    hsurf : ndarray of double
        Array with elevation of ICON grid cell circumcenters
        (number of ICON cells) [m]
    vlon : ndarray of double
        Array with longitude of ICON grid cell vertices
        (number of ICON vertices) [rad]
    vlat : ndarray of double
        Array with latitude of ICON grid cell vertices
        (number of ICON vertices) [rad]
    cells_of_vertex : ndarray of int
        Array with indices of ICON cells adjacent to ICON vertices. Indices
        start with 0 (6, number of ICON vertices)
    num_hori : int
        Number of terrain horizon sampling directions
    grid_type : int
        Triangle mesh construction method
        - 0: Build triangle mesh solely from ICON grid cell circumcenters
             (non-unique triangulation of hexa- and pentagons; relatively
             long triangle edges can cause artefacts in horizon computation)
        - 1: Build triangle mesh from ICON grid cell circumcenters and vertices
             (elevation at vertices is computed as mean from adjacent cell
             circumcenters; triangulation is unique and artefacts are reduced)
    dist_search : double
        Radial search distance for horizon computation [m]
    ray_org_elev : double
        Vertical elevation of ray origin above surface [m]
    refine_factor : int
        Refinement factor that subdivides 'num_hori' for more robust results
    svf_type : int
        Method for computing the Sky View Factor (SVF)
            0: Visible sky fraction; pure geometric skyview-factor
            1: SVF for horizontal surface; geometric scaled with sin(horizon)
            2: ?; geometric scaled with sin(horizon)**2

    Returns
    -------
    horizon : ndarray of float
        Array (two-dimensional) with terrain horizon [deg]
    skyview : ndarray of float
        Array (one-dimensional) with sky view factor [-]"""

    # Check consistency and validity of input arguments
    if (clon.size != clat.size) or (clat.size != hsurf.size):
        raise ValueError("Inconsistent lengths of input arrays 'clon', "
                         "'clat' and 'hsurf'")
    if (vlon.size != vlat.size):
        raise ValueError("Inconsistent lengths of input arrays 'vlon' and "
                         "'vlat'")
    if cells_of_vertex.shape[0] != 6:
        raise ValueError("First dimension of 'cells_of_vertex' must "
            + "have length 6")
    if not np.all((cells_of_vertex >= 0) & (cells_of_vertex <= clon.size - 1)
        | (cells_of_vertex == -2)):
        raise ValueError("Indices of 'cells_of_vertex' out of range")
    if (num_hori < 4) or (num_hori > 1440):
        raise ValueError("'num_hori' must be in the range [4, 1440]")
    if (grid_type < 0) or (grid_type > 1):
        raise ValueError("'grid_type' must be in the range [0, 1]")
    if (dist_search < 1_000.0) or (dist_search > 500_000.0):
        raise ValueError("'dist_search' must be in the range " \
        + "[1'000, 500'000] km")
    if ray_org_elev < 0.1:
        raise TypeError("Minimal allowed value for 'ray_org_elev' is 0.1 m")
    if (refine_factor < 1) or (refine_factor > 50):
        raise ValueError("'refine_factor' must be in the range [1, 50]")
    if (svf_type < 0) or (svf_type > 2):
        raise ValueError("'svf_type' must be in the range [0, 2]")

    # Allocate array for output
    cdef np.ndarray[np.float64_t, ndim = 2, mode = "c"] \
        horizon = np.empty((num_hori, clon.size), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] \
        skyview = np.empty(clon.size, dtype=np.float64)

    # Ensure that passed arrays are contiguous in memory
    cells_of_vertex = np.ascontiguousarray(cells_of_vertex)

    # Call C++ function and pass arguments
    horizon_svf_comp(&clon[0], &clat[0], &hsurf[0],
                     &vlon[0], &vlat[0],
                     &cells_of_vertex[0, 0],
                     &horizon[0, 0], &skyview[0],
                     clon.size, vlon.size, num_hori,
                     grid_type, dist_search,
                     ray_org_elev, refine_factor,
                     svf_type)

    return horizon, skyview
