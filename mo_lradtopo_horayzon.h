#ifndef TESTLIB_H
#define TESTLIB_H

void horizon_svf_comp(double* vlon, double* vlat,
    double* elevation,
    int* faces,
    int* ind_hori_out,
    float* f_cor,
    double* horizon_out,
    int num_vertex, int num_cell, int num_hori_out,
    int num_cell_parent, int num_cell_child_per_parent,
    int num_hori, double dist_search,
    double ray_org_elev, int num_elev,
    double sw_dir_cor_max, int cons_area_factor);

#endif
