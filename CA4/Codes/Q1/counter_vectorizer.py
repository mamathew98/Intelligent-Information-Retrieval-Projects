import csv
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from data_manager import save_object


nltk.download('punkt')  # download punkt if not downloaded


def read_csv(address):  # read csv file into array of documents data
    file = open(address)
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    rows = []
    for row in csv_reader:
        rows.append(row)
    file.close()
    return header, rows


def get_corpus(address):  # create a corpus file from a file
    header, documents = read_csv(address)
    corpus = []
    for doc in documents:
        corpus.append(doc[1])
    return corpus


def tokenizer(text):  # tokenizer used to tokenize word while vectorizing
    stemmer = SnowballStemmer('english')
    nltk_tokenizer = RegexpTokenizer(r'\w+')
    # text = (PorterStemmer.stem(w) for w in text)
    tokens = nltk_tokenizer.tokenize(re.sub(r'\d+', '', text))
    # tokens = word_tokenize(text)
    return (stemmer.stem(w) for w in tokens)


def custom_preprocessor(text):  # pre processing words
    # lowering the text case
    text = text.lower()
    # remove special chars
    text = re.sub(r'\b\w{1,3}\b', '', text)
    return text


def find_index(arr, val):  # find index of a word in vocabulary
    index = np.where(arr == val)
    # print(index[0][0])
    return index[0][0]


def py_matmul(a, b):  # efficient way to multiply matrixes
    ra, ca = a.shape
    rb, cb = b.shape
    assert ca == rb, f"{ca} != {rb}"

    return np.matmul(a, b)


def pre_process(corpus):  # main functionality
    vectorizer = CountVectorizer(stop_words='english', max_df=0.5,
                                 binary=True, tokenizer=tokenizer, preprocessor=custom_preprocessor)
    X = vectorizer.fit_transform(corpus)
    vocabulary = vectorizer.get_feature_names_out()
    # print(vocabulary)
    occurrence_matrix = np.array(X.toarray())
    sparse_W = csr_matrix(occurrence_matrix)  # transform occ_matrix to sparse for optimization
    # calculating co_occ_matrix by multiplying transpose(occ_matrix) and occ_matrix
    co_occurrence_matrix = sparse_W.transpose().dot(sparse_W)
    print("Shape: ", X.shape)
    # save everything we get in a file so we don't need to calculate again
    save_object(vocabulary, 'vocab.pickle')
    save_object(occurrence_matrix, 'occurrence_matrix.pickle')
    save_object(co_occurrence_matrix, 'co_occurrence_matrix.pickle')
    # return vocabulary, occurrence_matrix, co_occurrence_matrix


def main():
    pre_process(get_corpus('news.csv'))


main()


# section below was the 1st idea used for this question using loops, that would took a LONG time and wasn't efficient
# ====================================================================================================================
# def doc_tokenize(csv_documents):
#     nltk_tokenizer = RegexpTokenizer(r'\w+')
#
#     for index, document in enumerate(csv_documents):
#         csv_documents[index] = nltk_tokenizer.tokenize(re.sub(r'\d+', '', document[1]))
#
#     return csv_documents
#
#
# def doc_normalize(documents):
#     stop_words = set(stopwords.words("english"))
#     words = []
#
#     for index, document in enumerate(documents):
#         # documents[index] = map(lambda x: ps.stem(x).lower(), document)
#         documents[index] = [x.lower() for x in document if x not in stop_words and len(x) > 3]
#         words = words + documents[index]
#         documents[index] = Counter(documents[index])
#     return documents, words
#
#
# def doc_get_words_distribution(all_words):
#     unique_words = nltk.FreqDist(all_words)
#     return unique_words
#
#
# def process_csv(address):
#     header, documents = read_csv(address)
#
#     documents = doc_tokenize(documents)
#
#     documents, all_words = doc_normalize(documents)
#
#     distributions = doc_get_words_distribution(all_words)
#
#     return documents, all_words, distributions
