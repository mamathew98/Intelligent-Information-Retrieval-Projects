import numpy as np
import sys
from data_manager import load_object
from nltk.stem import SnowballStemmer

# restore pre saved data
mi_matrix = load_object("mi_matrix.pickle")
vocabulary = load_object('vocab.pickle')


def find_index(arr, val):
    index = np.where(arr == val)
    if len(index[0]) == 0:
        sys.exit("Word you entered wasn't found in any document")
    return index[0][0]


def syntagmatic(word):
    idx = find_index(vocabulary, word)
    sim = mi_matrix[idx, :]
    ind = np.argpartition(sim, -10)[-10:]  # get top 10 MI scores indexes
    count = 1
    for j in ind:
        print(count, ':', vocabulary[j], "-->", round(sim[j], 5))
        count += 1


def main():
    stemmer = SnowballStemmer('english')

    word = str(input("Enter your word: "))
    syntagmatic(stemmer.stem(word))


main()
