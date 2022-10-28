import numpy as np


def QueryIDFSum(matrix, idf):
    nz_mtx = []
    output = []
    for row in matrix:
        x = np.array(row)
        indices = np.where(x > 0)[0]
        nz_mtx.append(indices)
    for arr in nz_mtx[1:]:
        output.append(sum(idf[i] for i in nz_mtx[0]))
    return output


def AnswerIDFSum(matrix, idf):
    nz_mtx = []
    output = []
    for row in matrix:
        x = np.array(row)
        indices = np.where(x > 0)[0]
        nz_mtx.append(indices)
    for arr in nz_mtx[1:]:
        output.append(sum(idf[i] for i in arr))
    return output


def QueryTFSum(matrix):
    nz_mtx = []
    output = 10 * [np.sum(matrix[0])]
    return output


def AnswerTFSum(matrix):
    nz_mtx = []
    output = [np.sum(x) for x in matrix]
    return output


def QueryTFIDFSum(matrix):
    nz_mtx = []
    output = 10 * [np.sum(matrix[0])]
    return output


def AnswerTFIDFSum(matrix):
    nz_mtx = []
    output = [np.sum(x) for x in matrix]
    return output


def CommonTFIDFSum(matrix):
    nz_mtx = []
    output = []
    for row in matrix:
        x = np.array(row)
        indices = np.where(x > 0)[0]
        nz_mtx.append(indices)
    for i in range(len(nz_mtx[1:])):
        common = list(np.intersect1d(nz_mtx[0], nz_mtx[i + 1]))
        output.append(sum(matrix[i + 1][j] for j in common) + sum(matrix[0][j] for j in common))
    return output


def CommonTFSum(matrix):
    nz_mtx = []
    output = []
    for row in matrix:
        x = np.array(row)
        indices = np.where(x > 0)[0]
        nz_mtx.append(indices)
    for i in range(len(nz_mtx[1:])):
        common = list(np.intersect1d(nz_mtx[0], nz_mtx[i + 1]))
        output.append(sum(matrix[i + 1][j] for j in common) + sum(matrix[0][j] for j in common))
    return output


def CommonIDFSum(matrix):
    nz_mtx = []
    output = []
    for row in matrix:
        x = np.array(row)
        indices = np.where(x > 0)[0]
        nz_mtx.append(indices)
    for i in range(len(nz_mtx[1:])):
        common = list(np.intersect1d(nz_mtx[0], nz_mtx[i + 1]))
        output.append(sum(matrix[i + 1][j] for j in common) + sum(matrix[0][j] for j in common))
    return output


