from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import numpy


X = [[3, 0, 0, 1],
     [0, 1, 0, 2],
     [0, 1, 5, 0],
     [0, 3, 1, 1]]


def normalizer(X):
     minimum = X[0]
     maximum = X[0]

     for x in X:
          if x < minimum:
               minimum = x

          if x > maximum:
               maximum = x

     X_new = []
     for x in X:
          X_new.append((x - minimum)/(maximum-minimum))

     return X_new


u, t, z = numpy.linalg.svd(X, compute_uv=True)
svd = TruncatedSVD(n_components=2)
lsa = make_pipeline(svd, Normalizer(copy=False))  # normalize data
X_reduced = svd.fit_transform(X)
aux = numpy.matrix.dot(X_reduced, svd.components_)
print(aux)
X_naives = []
for x in aux:
     X_naives.append(normalizer(x))

y_naives = numpy.array([1, 2, 3, 4])
print(X_naives)
ml = MultinomialNB()
ml.fit(X_naives, y_naives)
print(ml)
print(ml.predict([normalizer([1, 0, 1, 1])]))
print(ml.predict([normalizer([1, 0, 1, 1])]))
print(ml.predict([normalizer([1, 0, 0, 0])]))
print(ml.predict([normalizer([5, 0, 0, 0])]))



