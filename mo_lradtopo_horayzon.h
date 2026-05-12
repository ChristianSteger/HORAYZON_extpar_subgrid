#ifndef TESTLIB_H
#define TESTLIB_H

void horizon_svf_comp(double* vlon, double* vlat,
    double* elevation,
    unsigned int* faces,
    unsigned int* idx_hori_out,
    float* f_cor,
    int* shadow_angle_idx,
    float* terrain_normal,
    double* horizon_out,
    double* slope_out,
    int num_vertex, int num_cell,
    int num_hori_out,
    int num_cell_parent, int num_cell_child_per_parent,
    int num_hori, double dist_search,
    double ray_org_elev, int num_elev,
    double sw_dir_cor_max, int cons_area_factor);

#endif
