import numpy as np


def Informativeness(matrix):
    nz_mtx = []
    output = []
    for row in matrix:
        x = np.array(row)
        indices = np.where(x > 0)[0]
        nz_mtx.append(indices)
    for arr in nz_mtx[1:]:
        un_common = list(np.setdiff1d(arr, nz_mtx[0]))
        output.append(len(un_common))
    return output
