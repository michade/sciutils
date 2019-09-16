# cython: infer_types=True
import cython
import numpy as np

from cython import boundscheck, wraparound


@boundscheck(False)
@wraparound(False)
def points_in_disjoint_regions(
        const long [:] point_ids,
        const int [:] point_coords,
        const long [:] region_ids,
        const int [:] region_starts,
        const int [:] region_ends
):
    cdef int pt_N = len(point_ids)
    cdef int reg_N = len(region_ids)

    id_pairs = np.empty((pt_N, 2), 'long')  # max size is number of points
    cdef long [:, ::1] id_pairs_mv = id_pairs

    cdef int i = 0  # array index for anchors
    cdef int j = 0  # array index for regions
    cdef int k = 0  # array index for pre-allocated output array
    while i < pt_N and j < reg_N:
        pt_coord = point_coords[i]
        if pt_coord < region_starts[j]:
            i += 1
        elif pt_coord > region_ends[j]:
            j += 1
        else:
            id_pairs_mv[k, 0] = point_ids[i]
            id_pairs_mv[k, 1] = region_ids[j]
            k += 1
            i += 1

    del id_pairs_mv  # let go of the memview, so that the array can be resized
    id_pairs.resize((k, 2))  # shrink to match actual size
    return id_pairs