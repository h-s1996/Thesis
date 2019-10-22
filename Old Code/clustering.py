from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import numpy


class Clustering:

    def __init__(self, lsa, n_phrases):
        self.lsa = lsa
        self.n_phrases = n_phrases

    def cluster(self):
        return dendrogram(linkage(self.lsa, 'single'),
                          orientation='top',
                          labels=numpy.array(range(1, self.n_phrases + 1)))  # clustering measures

    def get_clusters(self, n_clusters):
        c = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='single')
        c.fit(self.lsa)
        labels = c.labels_
        clusters = []
        cluster = []
        for i in range(0, n_clusters):
            for j in range(0, len(labels)):
                if labels[j] == i:
                    cluster.append(j + 1)
            clusters.append(cluster)
            cluster = []
        return clusters
