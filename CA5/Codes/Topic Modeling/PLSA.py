from numpy import zeros, int8, log
from pylab import random
import sys
import codecs
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import numpy as np
from file_manager import save_object, load_object
import pickle

nltk.download('stopwords')


# segmentation, stopwords filtering and document-word matrix generating
# return:
# N : number of documents
# M : Number of Words in Vocabulary 
# word2id : a map mapping terms to their corresponding ids
# id2word : a map mapping ids to terms
# X : document-word matrix, N*M, each line is the number of terms that show up in the document
def preprocessing(datasetFilePath):
    # read the documents
    file = codecs.open(datasetFilePath, 'r', 'utf-8')
    documents = [document.strip() for document in file]
    file.close()

    # stopwords
    stops = set(stopwords.words("english"))

    # number of documents
    N = len(documents)

    # generate the word2id and id2word maps and generate the document-word matrix X
    ####complete this part

    docs = []

    for doc in range(0, len(documents)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(documents[doc]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        nltk_tokenizer = RegexpTokenizer(r'\w+')
        tokens = nltk_tokenizer.tokenize(re.sub(r'\d+', '', document))

        document = ' '.join(tokens)

        docs.append(document)

    # vectorizer = CountVectorizer(max_df=0.75, stop_words=stopwords.words('english'))

    try:
        vectorizer = pickle.load(open("vec.pickle", "rb"))
    except (OSError, IOError) as e:
        vectorizer = CountVectorizer(max_features=1500, min_df=2, max_df=0.85, stop_words=stopwords.words('english'))
        pickle.dump(vectorizer, open("vec.pickle", "wb"))

    X = vectorizer.fit_transform(docs).toarray()

    vocabulary = vectorizer.get_feature_names()
    M = len(vocabulary)

    word2id = {}
    for i in range(M):
        word2id[vocabulary[i]] = i

    id2word = {v: k for k, v in word2id.items()}

    return N, M, word2id, id2word, X


def initializeParameters():
    for i in range(0, N):
        normalization = sum(PI[i, :])
        for j in range(0, K):
            PI[i, j] /= normalization

    for i in range(0, K):
        normalization = sum(theta[i, :])
        for j in range(0, M):
            theta[i, j] /= normalization


def EStep():
    for d in range(0, N):
        for w in range(0, M):
            denominator = 0
            for k in range(0, K):
                p[d, w, k] = theta[k, w] * PI[d, k]
                denominator += p[d, w, k]
            if denominator == 0:
                for k in range(0, K):
                    p[d, w, k] = 0
            else:
                for k in range(0, K):
                    p[d, w, k] /= denominator


def MStep():
    # update theta
    for k in range(0, K):
        denominator = 0
        for w in range(0, M):
            theta[k, w] = 0
            for d in range(0, N):
                theta[k, w] += X[d, w] * p[d, w, k]
            denominator += theta[k, w]
        if denominator == 0:
            for w in range(0, M):
                theta[k, w] = 1.0 / M
        else:
            for w in range(0, M):
                theta[k, w] /= denominator

    # update PI
    for d in range(0, N):
        for k in range(0, K):
            PI[d, k] = 0
            denominator = 0
            for w in range(0, M):
                PI[d, k] += X[d, w] * p[d, w, k]
                denominator += X[d, w]
            if denominator == 0:
                PI[d, k] = 1.0 / K
            else:
                PI[d, k] /= denominator


# calculate the log likelihood
def LogLikelihood():
    loglikelihood = 0
    for d in range(0, N):
        for w in range(0, M):
            tmp = 0
            for k in range(0, K):
                tmp += theta[k, w] * PI[d, k]
            if tmp > 0:
                loglikelihood += X[d, w] * log(tmp)
    return loglikelihood


# output the params of model and top words of topics to files
def output():
    # document-topic distribution
    file = codecs.open(docTopicDist, 'w', 'utf-8')
    for i in range(0, N):
        tmp = ''
        for j in range(0, K):
            tmp += str(PI[i, j]) + ' '
        file.write(tmp + '\n')
    file.close()

    # topic-word distribution
    file = codecs.open(topicWordDist, 'w', 'utf-8')
    for i in range(0, K):
        tmp = ''
        for j in range(0, M):
            tmp += str(theta[i, j]) + ' '
        file.write(tmp + '\n')
    file.close()

    # dictionary
    file = codecs.open(dictionary, 'w', 'utf-8')
    for i in range(0, M):
        file.write(id2word[i] + '\n')
    file.close()

    # top words of each topic
    file = codecs.open(topicWords, 'w', 'utf-8')
    for i in range(0, K):
        topicword = []
        ids = theta[i, :].argsort()
        for j in ids:
            topicword.insert(0, id2word[j])
        tmp = ''
        for word in topicword[0:min(topicWordsNum, len(topicword))]:
            tmp += word + ' '
        file.write(tmp + '\n')
    file.close()


# set the default params and read the params from cmd
datasetFilePath = "dataset.txt"
K = 10  # number of topic
maxIteration = 30
threshold = 0.0001
topicWordsNum = 10
docTopicDist = "docTopicDist.txt"
topicWordDist = 'topicWordDistribution.txt'
dictionary = 'dictionary.dic'
topicWords = "topics.txt"

# preprocessing
N, M, word2id, id2word, X = preprocessing(datasetFilePath)

# PI[i, j] : p(zj|di)
PI = random([N, K])

# theta[i, j] : p(wj|zi)
theta = random([K, M])

# p[i, j, k] : p(zk|di,wj)
p = zeros([N, M, K])

initializeParameters()

# EM algorithm
oldLoglikelihood = 1
newLoglikelihood = 1
for i in range(0, maxIteration):
    EStep()
    MStep()
    newLoglikelihood = LogLikelihood()
    delta = (oldLoglikelihood - newLoglikelihood) / oldLoglikelihood  # relative change
    print(i + 1, " iteration  ", str(newLoglikelihood), delta)
    if (oldLoglikelihood != 1 and delta < threshold):
        break
    oldLoglikelihood = newLoglikelihood

output()
