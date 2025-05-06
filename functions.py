# Description: Auxiliary functions
#
# Author: Christian R. Steger, May 2025

import numpy as np
from numba import njit, prange
from numba.typed import Dict
from numba.types import int64 # type: ignore

###############################################################################
# Refines a triangle mesh on a sphere surface by subdividing each parent 
# triangle into 4 child triangles.
###############################################################################

# Notes:
# - dividing the chord length in two equivalent parts also divides the arc
#   length in two equivalent parts (-> splitting line creates a symmetry axis)

# -----------------------------------------------------------------------------
# Trimesh function
# -----------------------------------------------------------------------------

def refine_mesh_4c_trimesh(mesh, level=1):
    """
    Iteratively subdivides a triangle mesh on a unit sphere into four child 
    triangles.
    """
    for _ in range(level):
        mesh = mesh.subdivide()
    mesh.vertices = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1) \
        [:, None]
    return mesh

# -----------------------------------------------------------------------------
# Own Python implementation
# -----------------------------------------------------------------------------

def refine_mesh_4c_py(vertices, faces):
    """
    Refines a triangle mesh on a sphere surface by subdividing each parent 
    triangle into 4 child triangles.

    Parameters:
    vertices : array of float (num_vertex, 3)
        Cartesian coordinates (x, y, z) of vertices [-]
    faces : array of int (num_faces, 3)
        Indices of triangle faces
    
    Returns:
    vertices_ref : array of float (num_vertex_ref, 3)
        Cartesian coordinates (x, y, z) of refined vertices [-]
    faces_ref : array of int (num_faces_ref, 3)
        Indices of refined triangle faces
    """
    num_vert = vertices.shape[0]
    num_faces = faces.shape[0]
    vertices_ref = np.empty((num_vert + num_faces * 3, 3),
                            dtype=vertices.dtype)
    # allocate maximal possible size
    vertices_ref[:num_vert, :] = vertices
    faces_ref = np.empty((num_faces * 4, 3), dtype=faces.dtype)    
    # allocate exact size
    midpoint_cache = {}  # Cache for midpoints (to only compute them once)
    ind_mp = [0, 0, 0]
    ind = 0
    for ind_faces in range(num_faces):

        # Get indices of edge midpoints of parent triangle (and add midpoints
        # to new vertices if not already done)
        for k in range(3):
            i = faces[ind_faces, k]
            j = faces[ind_faces, (k + 1) % 3]
            key = tuple(sorted((i, j)))
            if key in midpoint_cache:
                ind_mp[k] = midpoint_cache[key]
            else:
                midpoint = (vertices_ref[i, :] + vertices_ref[j, :]) / 2.0
                midpoint /= np.sqrt((midpoint ** 2).sum())
                vertices_ref[num_vert, :] = midpoint
                midpoint_cache[key] = num_vert
                ind_mp[k] = num_vert
                num_vert += 1

        # Add four child triangles (counter-clockwise orientated)
        ind_0, ind_1, ind_2 = faces[ind_faces, :]
        faces_ref[ind, :] = [ind_0, ind_mp[0], ind_mp[2]]
        ind += 1
        faces_ref[ind, :] = [ind_1, ind_mp[1], ind_mp[0]]
        ind += 1
        faces_ref[ind, :] = [ind_2, ind_mp[2], ind_mp[1]]
        ind += 1
        faces_ref[ind, :] = [ind_mp[0], ind_mp[1], ind_mp[2]]
        ind += 1

    return vertices_ref[:num_vert, :], faces_ref

# -----------------------------------------------------------------------------
# Own Numba implementation
# -----------------------------------------------------------------------------

@njit
def refine_mesh_4c_nb(vertices, faces):
    """
    Refines a triangle mesh on a sphere surface by subdividing each parent 
    triangle into 4 child triangles.

    Parameters:
    vertices : array of float (num_vertex, 3)
        Cartesian coordinates (x, y, z) of vertices [-]
    faces : array of int (num_faces, 3)
        Indices of triangle faces
    
    Returns:
    vertices_ref : array of float (num_vertex_ref, 3)
        Cartesian coordinates (x, y, z) of refined vertices [-]
    faces_ref : array of int (num_faces_ref, 3)
        Indices of refined triangle faces
    """
    num_vert = vertices.shape[0]
    num_faces = faces.shape[0]
    vertices_ref = np.empty((num_vert + num_faces * 3, 3),
                            dtype=vertices.dtype)
    # allocate maximal possible size
    vertices_ref[:num_vert, :] = vertices
    faces_ref = np.empty((num_faces * 4, 3), dtype=faces.dtype)    
    # allocate exact size
    midpoint_cache = Dict.empty(
        key_type=int64,
        value_type=int64
    ) # Cache for midpoints (to only compute them once)
    ind_mp = np.empty(3, dtype=np.int64)
    ind = 0
    for ind_faces in range(num_faces):

        # Get indices of edge midpoints of parent triangle (and add midpoints
        # to new vertices if not already done)
        for k in range(3):
            i = int64(faces[ind_faces, k])
            j = int64(faces[ind_faces, (k + 1) % 3])
            key = min(i, j) * 1_000_000_000 + max(i, j)
            if key in midpoint_cache:
                ind_mp[k] = midpoint_cache[key]
            else:
                midpoint = (vertices_ref[i, :] + vertices_ref[j, :]) / 2.0
                midpoint /= np.sqrt((midpoint ** 2).sum())
                vertices_ref[num_vert, :] = midpoint
                midpoint_cache[key] = num_vert
                ind_mp[k] = num_vert
                num_vert += 1

        # Add four child triangles (counter-clockwise orientated)
        ind_0, ind_1, ind_2 = faces[ind_faces, :]
        faces_ref[ind, :] = [ind_0, ind_mp[0], ind_mp[2]]
        ind += 1
        faces_ref[ind, :] = [ind_1, ind_mp[1], ind_mp[0]]
        ind += 1
        faces_ref[ind, :] = [ind_2, ind_mp[2], ind_mp[1]]
        ind += 1
        faces_ref[ind, :] = [ind_mp[0], ind_mp[1], ind_mp[2]]
        ind += 1

    return vertices_ref[:num_vert, :], faces_ref

###############################################################################
# Refines a triangle mesh on a sphere surface by subdividing each parent
# triangle into n ** 2 child triangles.
###############################################################################

# -----------------------------------------------------------------------------
# Python/Numba implementation
# -----------------------------------------------------------------------------

@njit
def refine_mesh_nc(vertices, vertex_of_cell, edge_of_cell, edge_vertices, n):
    """
    Refines a triangle mesh on a sphere surface by subdividing each parent
    triangle n ** 2 child triangles.

    Parameters:
    vertices : array of float (num_vert, 3)
        Cartesian coordinates (x, y, z) of vertices [-]
    vertex_of_cell : array of int (3, num_cells)
        Indices of triangle faces
    edge_of_cell : array of int (3, num_cells)
        Indices of triangle edges
    edge_vertices : array of int (2, num_edges)
        Indices of vertices of edges
    n : int
        Triangle division number

    Returns:
    vertices_child : array of float (num_vertex_ref, 3)
        Cartesian coordinates (x, y, z) of refined vertices [-]
    faces_child : array of int (num_faces_ref, 3)
        Indices of refined triangle faces
    """

    # Compute number of new vertices
    num_vertex_in = vertices.shape[0]
    num_vertex_edge = edge_vertices.shape[1] * (n - 1)
    num_vertex_interior_pgc = 0  # number of interior vertices per grid cell
    for i in range(n - 1):
        num_vertex_interior_pgc += i
    num_vertex_interior = vertex_of_cell.shape[1] * num_vertex_interior_pgc
    num_vertex_ref = num_vertex_in + num_vertex_edge + num_vertex_interior

    # Mapping of triangle vertex indices
    num_vert_per_tri = 3 + 3 * (n - 1) + num_vertex_interior_pgc
    # print(f"Number of vertices per triangle: {num_vert_per_tri}")
    mapping = np.empty(num_vert_per_tri, dtype=np.int32)
    ind = 0
    mapping[ind] = 0
    ind += 1
    mapping[ind:(ind + (n - 1))] \
        = np.arange(3 + 2 * (n - 1), 3 + 2 * (n - 1) + (n - 1), dtype=np.int32)
    ind += (n - 1)
    mapping[ind] = 2
    ind += 1
    start = n * 3
    for i in range(0, n - 1):
        mapping[ind] = 3 + i
        ind += 1
        mapping[ind:(ind + n - 2 - i)] \
            = np.arange(start, start + n - 2 - i, dtype=np.int32)
        start = start + n - 2 - i
        ind += (n - 2 - i)
        mapping[ind] = 3 + (n - 1) + i
        ind += 1
    mapping[ind] = 1
    ind += 1
    if not np.all(np.diff(np.unique(mapping)) == 1) \
        or (num_vert_per_tri != ind):
        raise ValueError("Error while computing the remapping array")

    # Compute faces
    index_2d = np.empty((n + 1, n + 1), dtype=np.int32)
    index_2d.fill(-999)
    ind_vertex = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            index_2d[i, j] = ind_vertex
            ind_vertex += 1
    faces = np.empty((n ** 2, 3), dtype=np.int32)
    ind_face = 0
    for i in range(n):
        for j in range(n - i):
            ind_v0 = index_2d[i, j]
            ind_v1 = index_2d[i + 1, j]
            ind_v2 = index_2d[i, j + 1]
            faces[ind_face, :] = (ind_v0, ind_v1, ind_v2)
            ind_face += 1
            if i + j + 1 < n:
                inv_v3 = index_2d[i + 1, j + 1]
                faces[ind_face, :] = (ind_v1, inv_v3, ind_v2)
                ind_face += 1

    # Allocate arrays for refined mesh
    vertices_child = np.empty((num_vertex_ref, 3), dtype=np.float64)
    vertices_child.fill(np.nan) # temporary
    faces_child = np.empty((n ** 2 * vertex_of_cell.shape[1], 3),
                           dtype=np.int32)

    # Add vertices from base mesh
    vertices_child[:num_vertex_in, :] = vertices

    # Add vertices located on the edge of bash mesh (shared)
    t = np.linspace(0.0, 1.0, num=(n + 1))[1:-1]
    ind_vertex = num_vertex_in
    for i in range(edge_vertices.shape[1]):  # loop through all edges
        vertex_0 = vertices[edge_vertices[0, i], :]
        vertex_1 = vertices[edge_vertices[1, i], :]
        for j in range(n - 1):
            vertices_child[ind_vertex, :] \
                = vertex_0 + t[j] * (vertex_1 - vertex_0)
            ind_vertex += 1

    # Add vertices located in the interior of base mesh triangles
    for ind_cell in range(vertex_of_cell.shape[1]): # loop through all cells
        vertex_0 = vertices[vertex_of_cell[0, ind_cell]]
        vertex_1 = vertices[vertex_of_cell[1, ind_cell]]
        vertex_2 = vertices[vertex_of_cell[2, ind_cell]]
        for i in range(1, n):
            for j in range(1, n - i):
                k = n - i - j
                vertices_child[ind_vertex, :] \
                    = (k * vertex_0 + i * vertex_1 + j * vertex_2) / n
                ind_vertex += 1
    for i in range(vertices_child.shape[0]):
        vertices_child[i, :] /= np.linalg.norm(vertices_child[i, :])
    # unit vectors

    # Connect vertices into child triangle
    indices = np.empty(num_vert_per_tri, dtype=np.int32)
    ind_face = 0
    for ind_cell in range(vertex_of_cell.shape[1]):
    # for ind_cell in range(500):
        indices[:3] = vertex_of_cell[:, ind_cell]
        # counter-clockwise ordered
        ind_vertex = 0
        for ind in edge_of_cell[:, ind_cell]:
            indices_edge = np.arange(num_vertex_in + ind * (n - 1),
                                    num_vertex_in + (ind + 1) * (n - 1))
            # ordering (clockwise vs. counter-clockwise) not consistent
            if ind_vertex == 0: # order: 0 -> 1
                if vertex_of_cell[ind_vertex, ind_cell] \
                    != edge_vertices[0, edge_of_cell[ind_vertex, ind_cell]]:
                    indices_edge = indices_edge[::-1]
            else: # order: 2 -> 1, 0 -> 2
                if vertex_of_cell[ind_vertex, ind_cell] \
                    == edge_vertices[0, edge_of_cell[ind_vertex, ind_cell]]:
                    indices_edge = indices_edge[::-1]
            slice_v = slice(3 + ind_vertex * (n - 1),
                            3 + (ind_vertex + 1) * (n - 1))
            indices[slice_v] = indices_edge
            ind_vertex += 1
        indices[slice_v.stop:] \
            = np.arange(num_vertex_in + num_vertex_edge
                        + ind_cell * num_vertex_interior_pgc,
                        num_vertex_in + num_vertex_edge
                        + (ind_cell + 1) * num_vertex_interior_pgc)
        for i in range(n ** 2):
            faces_child[ind_face, :] = indices[mapping[faces[i, :]]]
            ind_face += 1

    return vertices_child, faces_child

# -----------------------------------------------------------------------------
# Auxiliary function to compute min/max angle between triangles centroids and
# vertices
# -----------------------------------------------------------------------------

@njit(parallel=True)
def alpha_minmax(vertices_child, faces_child):
    num_tri = faces_child.shape[0]
    centroids = np.empty((num_tri, 3), dtype=np.float64)
    dot_prod_min = np.empty(num_tri, dtype=np.float64)
    dot_prod_max = np.empty(num_tri, dtype=np.float64)
    for i in prange(num_tri):
        ind_0, ind_1, ind_2 = faces_child[i, :]
        v0 = vertices_child[ind_0, :]
        v1 = vertices_child[ind_1, :]
        v2 = vertices_child[ind_2, :]
        centroid = (v0 + v1 + v2) / 3.0
        centroid /= np.linalg.norm(centroid) # unit vector
        centroids[i, :] = centroid
        dot_prod_0 = np.dot(v0, centroid)
        dot_prod_1 = np.dot(v1, centroid)
        dot_prod_2 = np.dot(v2, centroid)
        dot_prod_min[i] = np.minimum(np.minimum(dot_prod_0, dot_prod_1),
                                     dot_prod_2)
        dot_prod_max[i] = np.maximum(np.maximum(dot_prod_0, dot_prod_1),
                                     dot_prod_2)
    alpha_min = np.rad2deg(np.arccos(dot_prod_max.max()))
    alpha_max = np.rad2deg(np.arccos(dot_prod_min.min()))
    return alpha_min, alpha_max, centroids

###############################################################################
