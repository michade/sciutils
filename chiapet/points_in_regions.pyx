# cython: infer_types=True
import cython
import numpy as np
import pandas as pd


@cython.boundscheck(False)
@cython.wraparound(False)
def anchor_midpoints_in_regions(anchors, regions):
    # get columns
    # TODO: not sure how to control the dtype of categoricals
    cdef const char [:] an_chrom_col = anchors.chrom.cat.codes.values
    cdef const int [:] an_x_col = anchors.mid.values
    cdef const long [:] an_id_col = anchors.anchor_id.values
    cdef const long [:] reg_id_col = regions.region_id.values
    cdef const char [:] reg_chrom_col = regions.chrom.cat.codes.values
    cdef const int [:] reg_s_col = regions.start.values
    cdef const int [:] reg_e_col = regions.end.values

    cdef int pt_N = len(anchors)
    cdef int reg_N = len(regions)

    id_pairs = np.empty((pt_N, 2), np.int)  # max size is number of points
    cdef long [:, ::1] id_pairs_mv = id_pairs

    cdef int i = 0  # array index for anchors
    cdef int j = 0  # array index for regions
    cdef int k = 0  # array index for pre-allocatedd output array
    while i < pt_N and j < reg_N:
        pt_chrom = an_chrom_col[i]
        pt_x = an_x_col[i]
        reg_chrom = reg_chrom_col[j]
        reg_s = reg_s_col[j]
        reg_e = reg_e_col[j]
        if pt_chrom < reg_chrom:
            i += 1
        elif pt_chrom > reg_chrom:
            j += 1
        else:
            if pt_x < reg_s:
                i += 1
            elif pt_x > reg_e:
                j += 1
            else:
                id_pairs_mv[k, 0] = an_id_col[i]
                id_pairs_mv[k, 1] = reg_id_col[j]
                k += 1
                i += 1

    del id_pairs_mv  # let go of the memview, so that the array can be resized
    id_pairs.resize((k, 2))  # shrink to match actual size
    df = pd.DataFrame(id_pairs, copy=False)
    df.columns = ['anchor_id', 'region_id']
    return df
