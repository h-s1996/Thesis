from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import numpy as np

N_CLUSTERS = 3
X = np.array([[-1.27e-31, 3.43e-15, 2.47e-01],
    [-9.09e-03, 1.00e00, -1.08e-14],
    [1.00e00, 9.09e-03, -2.10e-15],
    [1.11e-15, 4.27e-15, 4.16e-01],
    [8.85e-16, 4.19e-15, 3.94e-01],
    [6.63e-16, 4.01e-15, 3.94e-01],
    [8.85e-16, 4.24e-15, 3.94e-01],
    [3.55e-16, 3.39e-15, 3.10e-01],
    [7.34e-16, 4.55e-15, 4.51e-01]])


linked = linkage(X, 'single') #clustering measures
labelList = range(1, 10) #number of elements

figure(1)
dendrogram(linked, orientation='top', labels=labelList)

cluster = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='euclidean', linkage='single')
cluster.fit_predict(X)

#3D
fig = figure(2)
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster.labels_, cmap='rainbow')

#2D
#scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster.labels_, cmap='rainbow')

show()
