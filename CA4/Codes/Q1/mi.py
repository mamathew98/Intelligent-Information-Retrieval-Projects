import numpy as np
from tqdm import tqdm
from itertools import combinations
from tqdm.contrib import tzip
from data_manager import save_object
from data_manager import load_object
import time
import csv

# loading saved data
vocabulary = load_object('vocab.pickle')
occurrence_matrix = load_object('occurrence_matrix.pickle')
co_occurrence_matrix = load_object('co_occurrence_matrix.pickle')

N = occurrence_matrix.shape[0]  # number of documents
occurrence_NO = {}  # number of documents containing a word


def find_index(arr, val):
    index = np.where(arr == val)
    return index[0][0]


# calculate number of documents containing a word
for w in vocabulary:
    i = find_index(vocabulary, w)
    occurrence_NO[w] = np.sum(occurrence_matrix[:, i])


# calculate probabilities
def p(w1, present1, w2=None, present2=None, p_co=None):

    p_w1 = (occurrence_NO[w1] + 0.5) / (N + 1)

    # if it's singular probability
    if w2 is None:
        if present1:
            return p_w1
        else:
            return 1 - p_w1

    p_w2 = (occurrence_NO[w2] + 0.5) / (N + 1)

    if present1 and present2:  # p(w1 = 1, w2 = 1)
        return p_co
    elif present1 and not present2:  # p(w1 = 1, w2 = 0)
        return p_w1 - p_co
    elif not present1 and present2:  # p(w1 = 0, w2 = 1)
        return p_w2 - p_co
    elif not present1 and not present2:  # p(w1 = 0, w2 = 0)
        return 1 - (p_co + (p_w1 - p_co) + (p_w2 - p_co))


def mi(w1, w2, p_co):
    summation = 0
    for u in [False, True]:
        for v in [False, True]:
            numerator = p(w1, u, w2, v, p_co)
            denominator = p(w1, u) * p(w2, v)
            summation += numerator * np.log2(numerator / denominator)

    return summation


def main():
    pairs = list(combinations(vocabulary, 2))
    mi_matrix = np.zeros(shape=(len(vocabulary), len(vocabulary)))  # matrix to save MI score of words
    sorted_matrix = co_occurrence_matrix.sorted_indices()
    cx = sorted_matrix.tocoo()
    file = open("mi_data.csv", "w", newline='')  # file to write MI score of words
    header = ['word1', 'word2', 'MI']
    writer = csv.writer(file)
    writer.writerow(header)
    for i, j, v in tzip(cx.row, cx.col, cx.data):  # TQDM lib used for exec time estimation
        p_co = (v + 0.25) / (N + 1)
        mi_value = mi(vocabulary[i], vocabulary[j], p_co)
        mi_matrix[i, j] = mi_value
        row = [vocabulary[i], vocabulary[j], str(mi_value)]
        writer.writerow(row)

    save_object(mi_matrix, 'mi_matrix.pickle')
    file.close()


start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))  # calculate exec time
