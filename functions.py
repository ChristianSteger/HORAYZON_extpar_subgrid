# Description: Auxiliary functions
#
# Author: Christian R. Steger, May 2025

import numpy as np
from numba import njit
from numba.typed import Dict
from numba.types import int64 # type: ignore
import trimesh

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
    vertices : array of float (num_vert, 3)
        Cartesian coordinates (x, y, z) of vertices [-]
    faces : array of int (num_faces, 3)
        Indices of triangle faces
    
    Returns:
    vertices_ref : array of float (num_vert_ref, 3)
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
    vertices : array of float (num_vert, 3)
        Cartesian coordinates (x, y, z) of vertices [-]
    faces : array of int (num_faces, 3)
        Indices of triangle faces
    
    Returns:
    vertices_ref : array of float (num_vert_ref, 3)
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
