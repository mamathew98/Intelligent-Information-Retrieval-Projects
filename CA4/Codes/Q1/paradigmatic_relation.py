import numpy as np
import sys
from data_manager import load_object
from data_manager import save_object
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from nltk.stem import SnowballStemmer

# restore pre saved data
mi_matrix = load_object("mi_matrix.pickle")
vocabulary = load_object('vocab.pickle')


def find_index(arr, val):
    index = np.where(arr == val)
    if len(index[0]) == 0:
        sys.exit("Word you entered wasn't found in any document")
    return index[0][0]


def paradigmatic(word):
    idx = find_index(vocabulary, word)
    sparse_mi = csr_matrix(mi_matrix)
    sim_matrix = load_object('sim_matrix.pickle')
    if sim_matrix is None:
        sim_matrix = cosine_similarity(sparse_mi)  # calculate pair similarity
        save_object(sim_matrix, 'sim_matrix.pickle')
    sim = sim_matrix[idx, :]
    ind = np.argpartition(sim, -10)[-10:]  # get top 10 similarities indexes
    count = 1
    for j in ind:
        print(count, ':', vocabulary[j], "-->", round(sim[j], 5))
        count += 1


def main():
    stemmer = SnowballStemmer('english')

    word = str(input("Enter your word: "))
    paradigmatic(stemmer.stem(word))


main()
