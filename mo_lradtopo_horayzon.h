#ifndef TESTLIB_H
#define TESTLIB_H

void horizon_svf_comp(double* vlon, double* vlat,
    double* elevation,
    int* faces,
    float* f_cor,
    int num_vertex, int num_cell,
    int num_cell_parent, int num_cell_child_per_parent,
    int num_hori, double dist_search,
    double ray_org_elev);

#endif
