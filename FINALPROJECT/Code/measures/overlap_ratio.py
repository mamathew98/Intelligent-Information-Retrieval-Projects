import numpy as np


def OverlappingRatio(matrix):
    nz_mtx = []
    output = []
    for row in matrix:
        x = np.array(row)
        indices = np.where(x > 0)[0]
        nz_mtx.append(indices)
    for arr in nz_mtx[1:]:
        common = list(np.intersect1d(nz_mtx[0], arr))
        output.append(len(common) / (len(arr) + len(nz_mtx[0]) - len(common)))
    return output
