#ifndef TESTLIB_H
#define TESTLIB_H

void horizon_svf_comp(double* clon, double* clat, double* hsurf,
    double* vlon, double* vlat,
    int* cells_of_vertex,
    double* horizon, double* skyview,
    int num_cell, int num_vertex, int num_hori,
    int grid_type, double dist_search,
    double ray_org_elev, int refine_factor,
    int svf_type);

#endif
