// C++ program to compute topographic horizon and sw_dir_cor

#define _USE_MATH_DEFINES
#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <embree4/rtcore.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// Namespace
#if defined(RTC_NAMESPACE_USE)
    RTC_NAMESPACE_USE
#endif

//-----------------------------------------------------------------------------
// Definition of geometries
//-----------------------------------------------------------------------------

// Point in 3D space
struct geom_point{
    double x, y, z;
    // also used for geographic coordinates: lon (x), lat (y), elevation (z)
};

// Vector in 3D space
struct geom_vector{
    double x, y, z;
};

// Vertex (for Embree)
struct Vertex{
    float x, y, z;
};

// Triangle specified by vertex indices (for Embree)
struct Triangle{
    int v0, v1, v2;
};
// Indices should be 32-bit unsigned integers according to the Embree
// documentation. However, until 2'147'483'647, the binary representation
// between signed/unsigned integers is identical.

//-----------------------------------------------------------------------------
// Functions (not dependent on Embree)
//-----------------------------------------------------------------------------

/**
 * @brief Converts degree to radian.
 * @param ang Input angle [deg].
 * @return Output angle [rad].
 */
inline double deg2rad(double ang) {
	return ((ang / 180.0) * M_PI);
}

/**
 * @brief Converts radian to degree.
 * @param ang Input angle [rad].
 * @return Output angle [deg].
 */
inline double rad2deg(double ang) {
	return ((ang / M_PI) * 180.0);
}

/**
 * @brief Computes the dot product between two vectors.
 * @param a Vector a.
 * @param b Vector b.
 * @return Resulting dot product.
 */
inline double dot_product(geom_vector a, geom_vector b) {
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

/**
 * @brief Computes cross dot product between two vectors.
 * @param a Vector a.
 * @param b Vector b.
 * @return Resulting cross product.
 */
inline geom_vector cross_product(geom_vector a, geom_vector b) {
    geom_vector c = {a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x};
    return c;
}

/**
 * @brief Computes the unit vector (normalised vector) of a vector in-place.
 * @param a Vector a.
 */
void unit_vector(geom_vector& a) {
    double vector_mag = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    a.x /= vector_mag;
    a.y /= vector_mag;
    a.z /= vector_mag;
}

/**
 * @brief Rotates vector v around unit vector k with a given angle.
 *
 * This function rotates vector v around a unit vector k with a given angle
 * according to the Rodrigues' rotation formula. For performance reasons,
 * trigonometric function have to be pre-computed.
 *
 * @param v Vector that should be rotated.
 * @param k Unit vector specifying the rotation axis.
 * @param ang_rot_sin Sine of the rotation angle.
 * @param ang_rot_cos Cosine of the rotation angle.
 * @return Rotated vector.
 */
inline geom_vector vector_rotation(geom_vector v, geom_vector k,
    double ang_rot_sin, double ang_rot_cos) {
    geom_vector v_rot;
    double term = dot_product(k, v) * (1.0 - ang_rot_cos);
    v_rot.x = v.x * ang_rot_cos + (k.y * v.z - k.z * v.y) * ang_rot_sin
        + k.x * term;
    v_rot.y = v.y * ang_rot_cos + (k.z * v.x - k.x * v.z) * ang_rot_sin
        + k.y * term;
    v_rot.z = v.z * ang_rot_cos + (k.x * v.y - k.y * v.x) * ang_rot_sin
        + k.z * term;
    return v_rot;
}

/**
 * @brief Returns index of element in array. If not found, returns -1.
 * @param element Search element.
 * @param array Input array.
 * @param array_len Length of array.
 * @return Index of element in array.
 */
int element_in_array(int element, int* array, int array_len){
    for (int i = 0; i < array_len; i++){
        if (array[i] == element){
            return i;
        }
    }
    return -1;
}

/**
 * @brief Computes the linear index from subscripts (3D-array)
 * @param dim_1 second dimension length of the array
 * @param dim_2 third dimension length of the array
 * @param ind_0 first array indices
 * @param ind_1 second array indices
 * @param ind_2 third array indices
 * @return Linear index.
 */
inline size_t lin_ind_3d(size_t dim_1, size_t dim_2, size_t ind_0,
    size_t ind_1, size_t ind_2) {
	return (ind_0 * (dim_1 * dim_2) + ind_1 * dim_2 + ind_2);
}

/**
 * @brief Transforms geographic to ECEF coordinates in-place.
 *
 * This function transforms geographic longitude/latitude to earth-centered,
 * earth-fixed (ECEF) coordinates. A spherical Earth is assumed.
 *
 * @param points Points (lon, lat elevation) in geographic coordinates
 *               [rad, rad, m].
 * @param rad_earth Radius of Earth [m].
 */
void lonlat2ecef(std::vector<geom_point>& points, double rad_earth){
    for (size_t i = 0; i < points.size(); i++){
        double sin_lon = sin(points[i].x);
        double cos_lon = cos(points[i].x);
        double sin_lat = sin(points[i].y);
        double cos_lat = cos(points[i].y);
        double elevation = points[i].z;
        points[i].x = (rad_earth + elevation) * cos_lat * cos_lon;
        points[i].y = (rad_earth + elevation) * cos_lat * sin_lon;
        points[i].z = (rad_earth + elevation) * sin_lat;
    }
}

/**
 * @brief Transforms points from ECEF to ENU coordinates in-place.
 * @param points Points (x, y, z) in ECEF coordinates [m].
 * @param lon_orig Longitude of ENU coordinate system origin [rad].
 * @param lat_orig Latitude of ENU coordinate system origin [rad].
 * @param rad_earth Radius of Earth [m].
 */
void ecef2enu_point(std::vector<geom_point>& points, double lon_orig,
    double lat_orig, double rad_earth){
    double sin_lon = sin(lon_orig);
    double cos_lon = cos(lon_orig);
    double sin_lat = sin(lat_orig);
    double cos_lat = cos(lat_orig);
    double x_ecef_orig = rad_earth * cos(lat_orig) * cos(lon_orig);
    double y_ecef_orig = rad_earth * cos(lat_orig) * sin(lon_orig);
    double z_ecef_orig = rad_earth * sin(lat_orig);
    double x_enu, y_enu, z_enu;
    for (size_t i = 0; i < points.size(); i++){
        x_enu = - sin_lon * (points[i].x - x_ecef_orig)
            + cos_lon * (points[i].y - y_ecef_orig);
        y_enu = - sin_lat * cos_lon * (points[i].x - x_ecef_orig)
            - sin_lat * sin_lon * (points[i].y - y_ecef_orig)
            + cos_lat * (points[i].z - z_ecef_orig);
        z_enu = + cos_lat * cos_lon * (points[i].x - x_ecef_orig)
            + cos_lat * sin_lon * (points[i].y - y_ecef_orig)
            + sin_lat * (points[i].z - z_ecef_orig);
        points[i].x = x_enu;
        points[i].y = y_enu;
        points[i].z = z_enu;
    }
}

//-----------------------------------------------------------------------------
// Functions (Embree related)
//-----------------------------------------------------------------------------

/**
 * @brief Error function for device initialiser.
 * @param userPtr
 * @param error
 * @param str
 */
void errorFunction(void* userPtr, enum RTCError error, const char* str) {
    printf("error %d: %s\n", error, str);
}

/**
 * @brief Initialises device and registers error handler
 * @return Device instance.
 */
RTCDevice initializeDevice() {
    RTCDevice device = rtcNewDevice(NULL);
    if (!device) {
        printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));
    }
    rtcSetDeviceErrorFunction(device, errorFunction, NULL);
    return device;
}

/**
 * @brief Initialises the Embree scene.
 * @param device Initialised device.
 * @param vertex_of_triangle Indices of the triangle vertices.
 * @param num_triangle Number of triangles.
 * @param vertices Vertices of the triangles [m].
 * @return Embree scene.
 */
RTCScene initializeScene(RTCDevice device, int* vertex_of_triangle,
    int num_triangle, std::vector<geom_point>& vertices){

    RTCScene scene = rtcNewScene(device);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Vertices
    Vertex* vertices_embree = (Vertex*) rtcSetNewGeometryBuffer(geom,
        RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex),
        vertices.size());
    for (size_t i = 0; i < vertices.size(); i++) {
        vertices_embree[i].x = (float)vertices[i].x;
        vertices_embree[i].y = (float)vertices[i].y;
        vertices_embree[i].z = (float)vertices[i].z;
    }

    // Cell (triangle) indices to vertices
    Triangle* triangles_embree = (Triangle*) rtcSetNewGeometryBuffer(geom,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle),
        num_triangle);
    for (int i = 0; i < num_triangle; i++) {
        triangles_embree[i].v0 = vertex_of_triangle[(i * 3) + 0];
        triangles_embree[i].v1 = vertex_of_triangle[(i * 3) + 1];
        triangles_embree[i].v2 = vertex_of_triangle[(i * 3) + 2];
    }
    // -> improvement: pass buffer directly instead of copying

    auto start = std::chrono::high_resolution_clock::now();

    // Commit geometry and scene
    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end - start;
    std::cout << std::setprecision(2) << std::fixed;
    std::cout << "Building bounding volume hierarchy (BVH): " << time.count()
        << " s" << std::endl;

    return scene;

}

/**
 * @brief Ray casting with occlusion testing (hit / no hit).
 * @param scene Embree scene.
 * @param ox x-coordinate of the ray origin [m].
 * @param oy y-coordinate of the ray origin [m].
 * @param oz z-coordinate of the ray origin [m].
 * @param dx x-component of the ray direction [m].
 * @param dy y-component of the ray direction [m].
 * @param dz z-component of the ray direction [m].
 * @param dist_search Search distance for potential collision [m].
 * @return Collision status (true: hit, false: no hit).
 */
bool castRay_occluded1(RTCScene scene, float ox, float oy, float oz, float dx,
    float dy, float dz, float dist_search){
    struct RTCRay ray;
    ray.org_x = ox;
    ray.org_y = oy;
    ray.org_z = oz;
    ray.dir_x = dx;
    ray.dir_y = dy;
    ray.dir_z = dz;
    ray.tnear = 0.0;
    ray.tfar = dist_search;
    ray.mask = 1;
    rtcOccluded1(scene, &ray); // intersect ray with scene
    return (ray.tfar < 0.0);
}

/**
 * @brief Computes the terrain horizon for a specific point.
 *
 * This function computes the terrain horizon for a specific point on the
 * triangle mesh. It iteratively samples a certain azimuth direction with rays
 * until the horizon is found. For all but the first azimuth direction, the
 * elevation angle for the search is initialised with a value equal to the
 * horizon from the previous azimuth direction +/- the horizon accuracy value.
 *
 * @param ray_org_x x-coordinate of the ray origin [m].
 * @param ray_org_y y-coordinate of the ray origin [m].
 * @param ray_org_z z-coordinate of the ray origin [m].
 * @param hori_acc Horizon accuracy [rad].
 * @param dist_search Search distance for potential collision [m].
 * @param elev_ang_thresh Threshold angle for sampling in negative elevation
 *                        angle direction [rad].
 * @param scene Embree scene.
 * @param num_rays Number of rays casted.
 * @param horizon_cell Horizon array [rad].
 * @param azim_num Length of the horizon array.
 * @param sphere_normal Sphere normal at the point location [m].
 * @param north_direction North direction at the point location [m].
 * @param azim_sin Sine of the azimuth angle spacing.
 * @param azim_cos Cosine of the azimuth angle spacing.
 * @param elev_sin_2ha Sine of the double elevation angle spacing.
 * @param elev_cos_2ha Cosine of the double elevation angle spacing.
 */
void terrain_horizon(float ray_org_x, float ray_org_y, float ray_org_z,
    double hori_acc, float dist_search, double elev_ang_thresh,
    RTCScene scene, size_t &num_rays,
    double* horizon_cell, int azim_num,
    geom_vector sphere_normal, geom_vector north_direction,
    double azim_sin, double azim_cos,
    double elev_sin_2ha, double elev_cos_2ha){

    // Initial ray direction
    geom_vector ray_dir;
    ray_dir.x = north_direction.x;
    ray_dir.y = north_direction.y;
    ray_dir.z = north_direction.z;

    // Sample along azimuth
    double elev_ang = 0.0;
    for (int i = 0; i < azim_num; i++){

        // Rotation axis
        geom_vector rot_axis = cross_product(ray_dir, sphere_normal);
        unit_vector(rot_axis);
        // not necessarily a unit vector because vectors are mostly not
        // perpendicular

        // Find terrain horizon by iterative ray sampling
        bool hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
            ray_org_z, (float)ray_dir.x, (float)ray_dir.y, (float)ray_dir.z,
            dist_search);
        num_rays += 1;
        if (hit) { // terrain hit -> increase elevation angle
            while (hit){
                elev_ang += (2.0 * hori_acc);
                ray_dir = vector_rotation(ray_dir, rot_axis, elev_sin_2ha,
                    elev_cos_2ha);
                hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
                ray_org_z, (float)ray_dir.x, (float)ray_dir.y,
                (float)ray_dir.z, dist_search);
                num_rays += 1;
            }
            horizon_cell[i] = elev_ang - hori_acc;
        } else { // terrain not hit -> decrease elevation angle
            while ((!hit) && (elev_ang > elev_ang_thresh)){
                elev_ang -= (2.0 * hori_acc);
                ray_dir = vector_rotation(ray_dir, rot_axis, -elev_sin_2ha,
                    elev_cos_2ha); // sin(-x) == -sin(x), cos(x) == cos(-x)
                hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
                ray_org_z, (float)ray_dir.x, (float)ray_dir.y,
                (float)ray_dir.z, dist_search);
                num_rays += 1;
            }
            horizon_cell[i] = elev_ang + hori_acc;
        }

        // Azimuthal rotation of ray direction (clockwise; first to east)
        ray_dir = vector_rotation(ray_dir, sphere_normal, -azim_sin,
            azim_cos);  // sin(-x) == -sin(x), cos(x) == cos(-x)

    }

}

//-----------------------------------------------------------------------------
// Main function
//-----------------------------------------------------------------------------

void horizon_svf_comp(double* vlon, double* vlat,
    double* elevation,
    int* faces,
    int* ind_hori_out,
    float* f_cor,
    double* horizon_out,
    int num_vertex, int num_cell, int num_hori_out,
    int num_cell_parent, int num_cell_child_per_parent,
    int azim_num, double dist_search_dp,
    double ray_org_elev){

    // Fixed settings
    double hori_acc = deg2rad(0.25); // horizon accuracy [rad]
    double elev_ang_thresh = deg2rad(-85.0);
    // threshold for sampling in negative elevation angle direction [rad]
    // - relevant for 'void sampling directions' at edge of mesh
    // - necessary requirement: (elev_ang_thresh - (2.0 * hori_acc)) > -90.0  # probably obsolete when ray casting at triangle centroids...
    int elev_num = 91; // number of elevation angles for sw_dir_cor computation
    double sw_dir_cor_max = 25.0; // maximum value for sw_dir_cor [-]
    bool cons_area_factor = true; // use area factor for sw_dir_cor

    // Constants
    double rad_earth = 6371229.0;  // ICON/COSMO earth radius [m]

    // Type casting
    float dist_search = (float)dist_search_dp;

    std::cout << "------------------------------------------------------------"
        << "-------------------" << std::endl;
    std::cout << "Horizon and SVF computation with Intel Embree (v1.1)"
        << std::endl;
    std::cout << "------------------------------------------------------------"
        << "-------------------" << std::endl;

    // In-place transformation from geographic to ECEF coordinates
    std::vector<geom_point> vertices(num_vertex);
    for (int i = 0; i < num_vertex; i++){
        vertices[i].x = vlon[i];
        vertices[i].y = vlat[i];
        vertices[i].z = elevation[i];
    }
    lonlat2ecef(vertices, rad_earth);

    // Earth center and North Pole in ECEF coordinates
    std::vector<geom_point> earth_centre(1);
    earth_centre[0].x = 0.0;
    earth_centre[0].y = 0.0;
    earth_centre[0].z = 0.0;
    std::vector<geom_point> north_pole(1);
    north_pole[0].x = 0.0;
    north_pole[0].y = 0.0;
    north_pole[0].z = rad_earth;

    // Origin of ENU coordinate system
    double x_orig = 0.0;
    double y_orig = 0.0;
    double z_orig = 0.0;
    for (int i = 0; i < num_vertex; i++){
        x_orig += vertices[i].x;
        y_orig += vertices[i].y;
        z_orig += vertices[i].z;
    }
    double radius = sqrt(x_orig * x_orig + y_orig * y_orig + z_orig * z_orig);
    double lon_orig = atan2(y_orig, x_orig);
    double lat_orig = asin(z_orig / radius);
    // works correctly for domains containing the North/South Pole and/or
    // crossing the +/- 180 deg meridian

    // In-place transformation from ECEF to ENU coordinates
    std::cout << std::setprecision(4) << std::fixed;
    std::cout << "Origin of ENU coordinate system: " << rad2deg(lat_orig)
        << " deg lat, "  << rad2deg(lon_orig) << " deg lon" << std::endl;
    ecef2enu_point(vertices, lon_orig, lat_orig, rad_earth);
    ecef2enu_point(earth_centre, lon_orig, lat_orig, rad_earth);
    ecef2enu_point(north_pole, lon_orig, lat_orig, rad_earth);

    // Build bounding volume hierarchy (BVH)
    RTCDevice device = initializeDevice();
    RTCScene scene = initializeScene(device, faces, num_cell, vertices);

    // Evaluated trigonometric functions for rotation along azimuth/elevation
    // angle
    double azim_sin = sin(deg2rad(360.0) / (double)azim_num);
    double azim_cos = cos(deg2rad(360.0) / (double)azim_num);
    double elev_sin_2ha = sin(2.0 * hori_acc);
    double elev_cos_2ha = cos(2.0 * hori_acc);
    // Note: sin(-x) == -sin(x), cos(x) == cos(-x)
    double elev_spac = deg2rad(90.0) / (double)(elev_num - 1);
    double elev_sin_sun = sin(elev_spac);
    double elev_cos_sun = cos(elev_spac);

    auto start_ray = std::chrono::high_resolution_clock::now();
    size_t num_rays = 0;

    num_rays += tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, 5000), 0.0,
    [&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel (100 should be 'num_cell_parent') temporary!!!

    // Loop through parent cells
    // for (size_t i = 0; i < (size_t)num_cell_parent; i++){ // serial  ########## temporary !!!
    //for (size_t i = 0; i < 100; i++){ // serial
    for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel

        // Loop through child cells
         for (size_t j = 0; j < (size_t)num_cell_child_per_parent; j++){  // temporary
        // for (size_t j = 0; j < 1; j++){

            int ind_cell = i * num_cell_child_per_parent + j;

            // Compute cell (triangle) centroid
            geom_point vertex_0 = {vertices[faces[(ind_cell * 3) + 0]].x,
                                   vertices[faces[(ind_cell * 3) + 0]].y,
                                   vertices[faces[(ind_cell * 3) + 0]].z};
            geom_point vertex_1 = {vertices[faces[(ind_cell * 3) + 1]].x,
                                   vertices[faces[(ind_cell * 3) + 1]].y,
                                   vertices[faces[(ind_cell * 3) + 1]].z};
            geom_point vertex_2 = {vertices[faces[(ind_cell * 3) + 2]].x,
                                   vertices[faces[(ind_cell * 3) + 2]].y,
                                   vertices[faces[(ind_cell * 3) + 2]].z};
            geom_point cell_centroid = {
                (vertex_0.x + vertex_1.x + vertex_2.x) / 3.0,
                (vertex_0.y + vertex_1.y + vertex_2.y) / 3.0,
                (vertex_0.z + vertex_1.z + vertex_2.z) / 3.0
            };

            // Compute sphere normal
            geom_vector sphere_normal = {
                (cell_centroid.x - earth_centre[0].x),
                (cell_centroid.y - earth_centre[0].y),
                (cell_centroid.z - earth_centre[0].z)
            };
            unit_vector(sphere_normal);

            // Compute north direction (orthogonal to sphere normal)
            geom_vector north_direction;
            geom_vector v_n;
            double dot_prod;
            v_n.x = north_pole[0].x - cell_centroid.x;
            v_n.y = north_pole[0].y - cell_centroid.y;
            v_n.z = north_pole[0].z - cell_centroid.z;
            dot_prod = dot_product(v_n, sphere_normal);
            north_direction.x = v_n.x - dot_prod * sphere_normal.x;
            north_direction.y = v_n.y - dot_prod * sphere_normal.y;
            north_direction.z = v_n.z - dot_prod * sphere_normal.z;
            unit_vector(north_direction);

            // Elevate origin for ray tracing by 'safety margin'
            float ray_org_x = (float)(cell_centroid.x
                + sphere_normal.x * ray_org_elev);
            float ray_org_y = (float)(cell_centroid.y
                + sphere_normal.y * ray_org_elev);
            float ray_org_z = (float)(cell_centroid.z
                + sphere_normal.z * ray_org_elev);
            // The origin of the ray is slightly elevated to avoid potential
            // ray-terrain collisions near the origin due to numerical
            // imprecisions.

            double* horizon_cell = new double[azim_num];  // [rad]

            // Compute terrain horizon
            terrain_horizon(ray_org_x, ray_org_y, ray_org_z,
                hori_acc, dist_search, elev_ang_thresh,
                scene, num_rays,
                horizon_cell, azim_num,
                sphere_normal, north_direction,
                azim_sin, azim_cos,
                elev_sin_2ha, elev_cos_2ha);

            // Store horizon for certain cells in output array
            int index = element_in_array(ind_cell, ind_hori_out, num_hori_out); // inefficient for larger 'ind_hori_out'!
            if (index != -1){
                for (int k = 0; k < azim_num; k++){
                    horizon_out[index * azim_num + k] = rad2deg(horizon_cell[k]);
                }
            }

            // Compute triangle normal
            geom_vector a = {vertex_2.x - vertex_1.x,
                             vertex_2.y - vertex_1.y,
                             vertex_2.z - vertex_1.z};
            geom_vector b = {vertex_0.x - vertex_1.x,
                             vertex_0.y - vertex_1.y,
                             vertex_0.z - vertex_1.z};
            geom_vector triangle_normal = cross_product(a, b);
            unit_vector(triangle_normal);
            std::cout << std::setprecision(4) << std::fixed;
            if (triangle_normal.z < 0.0){
                std::cout << "Warning: z-component of triangle normal is smaller than 0.0!" << std::endl; // temporary
            }

            // Define area factor
            double area_factor;
            if (cons_area_factor){
                area_factor = 1.0 / dot_product(triangle_normal,
                                                sphere_normal);
            } else {
                area_factor = 1.0;
            }

            // ----------------------------------------------------------------
            // Compute SW correction factor (sw_dir_cor)
            // ----------------------------------------------------------------

            // Initial ray direction
            geom_vector sun_dir_xyp; // sun direction in the x-y plane
            sun_dir_xyp.x = north_direction.x;
            sun_dir_xyp.y = north_direction.y;
            sun_dir_xyp.z = north_direction.z;

            // Loop through azimuth angles
            for (int k = 0; k < azim_num; k++){

                // Rotation axis
                geom_vector rot_axis = cross_product(sun_dir_xyp,
                                                     sphere_normal);
                // cross product of 2 perpendicular unit vectors -> unit vector

                // Iteratively increase elevation angle and compute sw_dir_cor
                geom_vector sun_dir = sun_dir_xyp;
                double elev_ang = 0.0;  // -> sw_dir_cor = 0.0 for this angle
                for (int m = 0; m < (elev_num - 1); m++){
                    elev_ang += elev_spac;
                    sun_dir = vector_rotation(sun_dir, rot_axis, elev_sin_sun,
                        elev_cos_sun);
                    // check for self-shadowing (triangle)
                    double dot_prod_ts = dot_product(sun_dir, triangle_normal);
                    if (dot_prod_ts <= 0.0){
                        continue; // sw_dir_cor += 0.0
                    }
                    double dot_prod_hs = dot_product(sun_dir, sphere_normal);
                    double mask_shadow = double(elev_ang > horizon_cell[k]);
                    double sw_dir_cor = (1.0 / dot_prod_hs) * area_factor *
                        mask_shadow * dot_prod_ts;
                    if (sw_dir_cor < 0.0){
                        std::cout << "Warning: 'sw_dir_cor' is smaller than 0.0!" << std::endl; // temporary
                    }
                    size_t ind_hori = lin_ind_3d(azim_num, elev_num,
                                                 i, k, m + 1);
                    f_cor[ind_hori] += std::min(sw_dir_cor, sw_dir_cor_max);
                }

            // Azimuthal rotation of ray direction (clockwise; first to east)
            sun_dir_xyp = vector_rotation(sun_dir_xyp, sphere_normal,
                -azim_sin, azim_cos);  // sin(-x) == -sin(x), cos(x) == cos(-x)

            }

            // ----------------------------------------------------------------

            delete[] horizon_cell;

        }

        // Compute average SW correction factor for parent cell
        for (int k = 0; k < azim_num; k++){
            for (int m = 0; m < elev_num; m++){
                size_t ind_hori = lin_ind_3d(azim_num, elev_num,
                                             i, k, m);
                f_cor[ind_hori] /= (float)num_cell_child_per_parent;

            }
        }

    }

    return num_rays;  // parallel
    }, std::plus<size_t>());  // parallel

    auto end_ray = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ray = end_ray - start_ray;
    std::cout << std::setprecision(2) << std::fixed;
    std::cout << "Ray tracing: " << time_ray.count() << " s" << std::endl;

    // Print number of rays needed for location and azimuth direction
    std::cout << "Number of rays shot: " << num_rays << std::endl;
    // double ratio = (double)num_rays / (double)(num_cell * azim_num); // ############# temporary
    double ratio = (double)num_rays / (double)(5000 * num_cell_child_per_parent * azim_num);
    std::cout << std::setprecision(2) << std::fixed;
    std::cout << "Average number of rays per cell and azimuth sector: "
        << ratio << std::endl;

    std::cout << "------------------------------------------------------------"
        << "-------------------" << std::endl;

}
