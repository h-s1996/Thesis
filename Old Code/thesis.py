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
###############################################################################

example_pt = ["Bom dia. Dormiu bem esta noite?", #1
              "Mas levantou-se muito cedo",
              "Bom dia Como está hoje",
              "Agora o quê é que está a planear para o resto do dia", #4
              "Parece-me um dia em cheio Faça bom proveito",#5
              "Sim está ótimo para passear e apanhar sol",
              "Desfrute porque amanhã o sol vai-se embora e começa a chuva",
              "Boa noite O seu dia correu bem",
              "Passou-se alguma coisa Quer falar sobre isso",
              "Conversa com quem",
              "Não falou com a sua família", #11
              "Durma bem boa noite"]

keywords = ['sol', 'cheio', 'proveito', 'desfrute', 'chuva', 'resto', 'coisa', 'conversa', 'família', 'cedo']



#        MUTABLE VARIABLES     #

NGRAM_MAX = 3
MIN_FREQ_TERMS = 1
P_EIGENVALUES = 0.5  # 50%

#          CONSTANTS               #

N_IMPORTANT_WORDS = 15
NGRAM_MIN = 1
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


###############################################################################
# TFIDF
###############################################################################

vec = TfidfVectorizer(min_df=MIN_FREQ_TERMS,
                      stop_words=stopwords,
                      tokenizer=tokenize,
                      token_pattern=r'\w{1,}',
                      ngram_range=(NGRAM_MIN, NGRAM_MAX))
X = vec.fit_transform(example_pt)
feat_names = vec.get_feature_names()
n_feat_names = len(feat_names)
print("%d feature Names:" % n_feat_names)
print(feat_names)

###############################################################################
# CHOOSING N_COMPONENTS
###############################################################################

N_COMPONENTS = 1
res = 0
eigenvalues = numpy.linalg.svd(X.todense(), compute_uv=False)
normalized_eigenvalues = eigenvalues/numpy.sum(eigenvalues)
for i in range(0, len(eigenvalues) - 1):
    res += normalized_eigenvalues[i]
    if res >= P_EIGENVALUES:
        N_COMPONENTS = i + 1
        break


###############################################################################
# SINGULAR VALUE DECOMPOSITION --> DIMENSIONALITY REDUCED
###############################################################################
svd = TruncatedSVD(n_components=N_COMPONENTS, algorithm="arpack") #n_dimensions desired
lsa = make_pipeline(svd, Normalizer(copy=False)) #normalize data
U_reduced = lsa.fit_transform(X)
X_reduced = numpy.dot(U_reduced, svd.components_)
print(X_reduced)

###############################################################################
# LSA Keywords
###############################################################################

tokens = []
for i in keywords:
    tokens.extend(tokenize(i))

vocabulary = []
for i in tokens:
    repeat = False
    for v in vocabulary:
        if i == v:
            repeat = True
            break
    if not repeat:
        vocabulary.append(i)

print("KEY\n\n")

vec_key = TfidfVectorizer(stop_words=stopwords,
                          vocabulary=vocabulary,
                          tokenizer=tokenize,
                          token_pattern=r'\w{1,}',
                          ngram_range=(NGRAM_MIN, NGRAM_MAX))

X_keywords = vec_key.fit_transform(example_pt).todense()
print(X_keywords)

X_final = numpy.matrix.round(numpy.concatenate([X_reduced.T, X_keywords.T]).T, 2)

print("FINAL\n\n")

print(X_final)

figure(0)
X_cluster = X_final
d = dendrogram(linkage(X_cluster, 'single'), orientation='top', labels=range(1, n_phrases + 1))  # clustering measures
cluster = AgglomerativeClustering(n_clusters=N_COMPONENTS, affinity='euclidean', linkage='single')
cluster.fit_predict(X_cluster)

if N_COMPONENTS == 2:
    figure(N_COMPONENTS + 2)
    scatter(X_cluster[:, 0], X_cluster[:, 1], c=cluster.labels_, cmap='rainbow')
elif N_COMPONENTS == 3:
    ax = Axes3D(figure(N_COMPONENTS + 2))
    ax.scatter(X_cluster[:, 0], X_cluster[:, 1], X_cluster[:, 2], c=cluster.labels_, cmap='rainbow')

show()