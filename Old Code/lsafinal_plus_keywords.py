from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.mplot3d import Axes3D
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from pylab import *
import numpy
import nltk
import string

###############################################################################
#  Initializing
##############################################################################

example_pt = [ "Machine Learning é super divertido", #1
               "Python é super, super cool",
               "Estatística também é cool", #3
               "Data Science é divertido",
               "Python é ótimo para Machine Learning", #5
               "Eu odeio futebol",
               "Futebol é aborrecido de ver", #7
               "Hoje o tempo está de chuva",
               "Este tempo anda completamente louco"] #9

keywords = ['super divertido',
            'super cool',
            'machine learning',
            'python',
            'futebol',
            'data science',
            'estatística',
            'chuva',
            'louco']

#        MUTABLE VARIABLES     #

MIN_FREQ_TERMS = 1
NGRAM_MAX = 2
P_EIGENVALUES = 0.45  # 50%

#          CONSTANTS               #

NGRAM_MIN = 1
N_IMPORTANT_WORDS = 15
n_phrases = len(example_pt)
language = "portuguese"
stopwords = stopwords.words(language)
stopwords.append('é')


def tokenize(t):
    sentence = t.lower()
    sentence = nltk.word_tokenize(sentence)
    aux = []
    for word in sentence:
        if word not in stopwords and word not in string.punctuation:
            aux.append(RSLPStemmer().stem(word.lower()))
    phrase = []
    for word in aux:
        phrase.append(word)
    return phrase


keys = []
for i in keywords:
    keys.extend(tokenize(i))

t_key = []
for i in keys:
    repeat = False
    for v in t_key:
        if i == v:
            repeat = True
            break
    if not repeat:
        t_key.append(i)

###############################################################################
#  TFIDF
###############################################################################
vec_prev = TfidfVectorizer(min_df=MIN_FREQ_TERMS,
                           stop_words=stopwords,
                           tokenizer=tokenize,
                           token_pattern=r'\w{1,}',
                           ngram_range=(NGRAM_MIN, NGRAM_MAX))
Y = vec_prev.fit_transform(example_pt)
vocabulary = vec_prev.get_feature_names()
vocabulary.extend(t_key)
for i in range(0, len(vocabulary) - len(t_key)):
    for j in range(len(vocabulary) - len(t_key), len(vocabulary)):
        if i != j:
            if vocabulary[i] == vocabulary[j]:
                vocabulary.pop(j)
                break

vec = TfidfVectorizer(stop_words=stopwords,
                      vocabulary=vocabulary,
                      tokenizer=tokenize,
                      token_pattern=r'\w{1,}',
                      ngram_range=(NGRAM_MIN, NGRAM_MAX))

X = vec.fit_transform(example_pt)
feat_names = vec.get_feature_names()
print(feat_names)

###############################################################################
# CHOOSING EIGENVALUES BY COMPONENT WEIGHT AND PLOTTING THEM
###############################################################################

sum = 0
eigenvalues = numpy.linalg.svd(X.todense(), compute_uv=False)
normalized_eigenvalues = eigenvalues/numpy.sum(eigenvalues)
figure(0)
for i in range(0, len(eigenvalues)):
    sum += normalized_eigenvalues[i]
    if sum >= P_EIGENVALUES:
        N_COMPONENTS = i + 1
        break
barh(numpy.arange(1, len(eigenvalues) + 1, 1), normalized_eigenvalues)
yticks(numpy.arange(1, len(eigenvalues) + 1, 1), numpy.arange(1, len(eigenvalues) + 1, 1))
xlabel('Singular Values')
title('Importance of Each Singular Value')
grid(True)
print(N_COMPONENTS)

###############################################################################
# SINGULAR VALUE DECOMPOSITION --> DIMENSIONALITY REDUCED
###############################################################################
svd = TruncatedSVD(n_components=N_COMPONENTS, algorithm="arpack") #n_dimensions desired
lsa = make_pipeline(svd, Normalizer(copy=False)) #normalize data
X_reduced = lsa.fit_transform(X.T)#transpose so you can take the matrix documents vs components

###############################################################################
# SIMILARITY COMPUTATION
###############################################################################

similarity = cosine_similarity(X.T, X.T)
#print(numpy.matrix.round(similarity, decimals=2))

###############################################################################
#   PLOTS ACCORDING TO COMPONENTS
###############################################################################

for compNum in range(0, N_COMPONENTS):
    aux = TruncatedSVD(n_components=N_COMPONENTS)
    make_pipeline(aux, Normalizer(copy=False)).fit_transform(X)
    comp = aux.components_[compNum]

    # Sort the weights in the first component, and get the indeces
    indices = numpy.argsort(comp).tolist()

    # Reverse the indeces, so we have the largest weights first.
    indices.reverse()

    # Grab the top N_IMPORTANT_WORDS terms which have the highest weight in this component.
    if len(feat_names) >= N_IMPORTANT_WORDS:
        terms = [feat_names[weightIndex] for weightIndex in indices[0:N_IMPORTANT_WORDS]]
        weights = [comp[weightIndex] for weightIndex in indices[0:N_IMPORTANT_WORDS]]
        positions = arange(N_IMPORTANT_WORDS) + .5  # the bar centers on the y axis
    else:
        terms = [feat_names[weightIndex] for weightIndex in indices[0:(len(feat_names))]]
        weights = [comp[weightIndex] for weightIndex in indices[0:(len(feat_names))]]
        positions = arange(len(feat_names)) + .5  # the bar centers on the y axis
    # Display these terms and their weights as a horizontal bar graph.
    # The horizontal bar graph displays the first item on the bottom; reverse
    # the order of the terms so the biggest one is on top.
    terms.reverse()
    weights.reverse()

    figure(compNum + 1)
    barh(positions, weights, align='center')
    yticks(positions, terms)
    xlabel('Weight')
    title('Strongest terms for component %d' % compNum)
    grid(True)

###############################################################################
#   CLUSTERING
###############################################################################
# multiply the eigenvalues matrix by V (right eigenvectors)
# this way we can obtain the coordinates in the left-eigenvector axis
X_cluster = numpy.dot(diag(svd.singular_values_), svd.components_).T

#clustering measures
figure(N_COMPONENTS + 1)
dendrogram(linkage(X_cluster, 'single'), orientation='top', labels=range(1, n_phrases + 1))
cluster = AgglomerativeClustering(n_clusters=N_COMPONENTS, affinity='euclidean', linkage='single')
cluster.fit_predict(X_cluster)

if N_COMPONENTS == 2:
    figure(N_COMPONENTS + 2)
    scatter(X_cluster[:, 0], X_cluster[:, 1], c=cluster.labels_, cmap='rainbow')
elif N_COMPONENTS == 3:
    ax = Axes3D(figure(N_COMPONENTS + 2))
    ax.scatter(X_cluster[:, 0], X_cluster[:, 1], X_cluster[:, 2], c=cluster.labels_, cmap='rainbow')

show()
