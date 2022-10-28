import csv
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from data_manager import save_object


nltk.download('punkt')
# 2 of the topic had no docs related so they were removed
topics = ["acq", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade"]
ground_truth = []


def read_csv(address):
    file = open(address)
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    rows = []
    for row in csv_reader:
        rows.append(row)
    file.close()
    return header, rows


def get_corpus(address):
    header, documents = read_csv(address)
    corpus = []
    for doc in documents:
        corpus.append(doc[1])
        ground_truth.append(topics.index(doc[2]))
    save_object(ground_truth, "ground_truth.pickle")
    return corpus


def tokenizer(text):
    stemmer = SnowballStemmer('english')
    nltk_tokenizer = RegexpTokenizer(r'\w+')
    tokens = nltk_tokenizer.tokenize(re.sub(r'\d+', '', text))
    return (stemmer.stem(w) for w in tokens)


def custom_preprocessor(text):
    # lowering the text case
    text = text.lower()
    # remove special chars
    text = re.sub(r'\b\w{1,3}\b', '', text)
    return text


def find_index(arr, val):
    index = np.where(arr == val)
    # print(index[0][0])
    return index[0][0]


def py_matmul(a, b):
    ra, ca = a.shape
    rb, cb = b.shape
    assert ca == rb, f"{ca} != {rb}"

    return np.matmul(a, b)


def pre_process(corpus):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.1, max_features=3000,
                                 binary=True, tokenizer=tokenizer, preprocessor=custom_preprocessor)
    X = vectorizer.fit_transform(corpus)
    vocabulary = vectorizer.get_feature_names_out()
    tfidf_matrix = np.array(X.toarray())
    sparse_tfidf_matrix = csr_matrix(tfidf_matrix)
    print(sparse_tfidf_matrix)

    print("Shape: ", X.shape)
    save_object(vocabulary, 'vocab.pickle')
    save_object(tfidf_matrix, 'sparse_tfidf_matrix.pickle')


def main():
    pre_process(get_corpus('news.csv'))


main()
