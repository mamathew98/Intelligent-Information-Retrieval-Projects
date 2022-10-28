import numpy as np
import pandas
import pandas as pd
from sklearn import metrics
from data_manager import load_object
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import rand_score
from sklearn.cluster import AgglomerativeClustering

vocabulary = load_object('vocab.pickle')
sparse_tfidf_matrix = load_object('sparse_tfidf_matrix.pickle')
ground_truth = load_object('ground_truth.pickle')
topics = ["acq", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade"]


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def k_means(sparse_matrix):
    model = KMeans(n_clusters=8).fit(sparse_matrix)

    NMI = normalized_mutual_info_score(ground_truth, model.labels_)
    RI = rand_score(ground_truth, model.labels_)
    PURITY = purity_score(ground_truth, model.labels_)
    F1 = f1_score(ground_truth, model.labels_, average='micro')

    df = pandas.DataFrame([[NMI, F1, RI, PURITY]], index=["K_MEANS"], columns=["NMI", "F1", "RI", "PURITY"])
    df.to_csv('stat.csv', index=True, mode='a', header=True)

    CONFUSION = confusion_matrix(ground_truth, model.labels_)
    mat = pandas.DataFrame(CONFUSION, index=topics, columns=topics)
    mat.to_csv('k_mean_conf.csv', index=True, mode='a', header=True)
    print("k-means", "confusion matrix")
    print(mat)
    print()


def agglomerative(sparse_matrix, linkage):  # linkage shows which algorithm to use
    model = AgglomerativeClustering(linkage=linkage ,n_clusters=8).fit(sparse_matrix)

    NMI = normalized_mutual_info_score(ground_truth, model.labels_)
    RI = rand_score(ground_truth, model.labels_)
    PURITY = purity_score(ground_truth, model.labels_)
    F1 = f1_score(ground_truth, model.labels_, average='micro')

    df = pandas.DataFrame([[NMI, F1, RI, PURITY]], index=[linkage], columns=["NMI", "F1", "RI", "PURITY"])
    df.to_csv('stat.csv', index=True, mode='a', header=False)

    CONFUSION = confusion_matrix(ground_truth, model.labels_)
    mat = pandas.DataFrame(CONFUSION, index=topics, columns=topics)
    mat.to_csv('{}_conf.csv'.format(linkage), index=True, mode='a', header=True)
    print(linkage, "confusion matrix")
    print(mat)
    print()


def main():
    k_means(sparse_tfidf_matrix)

    agglomerative(sparse_tfidf_matrix, "single")

    agglomerative(sparse_tfidf_matrix, "average")

    agglomerative(sparse_tfidf_matrix, "complete")

    t = np.array(ground_truth)
    unique, counts = np.unique(t, return_counts=True)
    topic_counts = dict(zip(unique, counts))
    print("Topic Counts: ", topic_counts)


main()

