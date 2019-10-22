from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import nltk
import string
import numpy


class LSA:

    def __init__(self, ngram_max, min_freq, p_eig, phrases):
        self.ngram_max = ngram_max
        self.min_freq = min_freq
        self.p_eig = p_eig
        self.ngram_min = 1
        self.stopwords = stopwords.words("portuguese")
        self.stopwords.append('é')
        self.phrases = phrases
        self.u = []
        self.features_utterance = self.get_features_utterance()
        self.tfidf = []

    @staticmethod
    def normalizer(x_abnormal):
        minimum = x_abnormal.min()
        maximum = x_abnormal.max()
        x_new = (x_abnormal - minimum) /(maximum - minimum)
        return x_new

    def tokenize(self, t):
        if t in self.stopwords:
            return []
        sentence = t.lower()
        sentence = nltk.tokenize.word_tokenize(sentence)
        aux = []
        for word in sentence:
            if word not in self.stopwords and word not in string.punctuation:
                aux.append(RSLPStemmer().stem(word.lower()))
        phrase = []
        for word in aux:
            phrase.append(word)
        return phrase

    def manage_keywords(self, keywords):
        tokens, vocabulary = [], []
        for i in keywords:
            t = self.tokenize(i)
            if len(t) > 1:
                key_str = ''
                for j in t:
                    key_str = key_str + ' ' + j
                tokens.append(key_str[1:])
            else:
                tokens.extend(t)
        for i in tokens:
            repeat = False
            for v in vocabulary:
                if i == v:
                    repeat = True
                    break
            if not repeat:
                vocabulary.append(i)
        return vocabulary

    def get_features_utterance(self):
        vec = TfidfVectorizer(min_df=self.min_freq,
                              stop_words=self.stopwords,
                              tokenizer=self.tokenize,
                              ngram_range=(self.ngram_min, self.ngram_max))
        vec.fit_transform(self.phrases)
        return vec.get_feature_names()

    def tf_idf(self, phrase, keywords):
        examples = []
        examples.extend(self.phrases)
        if phrase:
            examples.append(phrase)
        vec = TfidfVectorizer(stop_words=self.stopwords,
                              vocabulary=keywords,
                              tokenizer=self.tokenize,
                              ngram_range=(self.ngram_min, self.ngram_max))
        x = vec.fit_transform(examples)
        return x.todense()

    def choose_dimensionality(self):
        res = 0
        eigenvalues = numpy.linalg.svd(self.tfidf, compute_uv=False)
        normalized_eigenvalues = eigenvalues / numpy.sum(eigenvalues)
        for i in range(0, len(eigenvalues)):
            res += normalized_eigenvalues[i]
            if res >= self.p_eig:
                return i + 1

    def reduce_dimensionality(self):
        svd = TruncatedSVD(n_components=self.choose_dimensionality(), algorithm="arpack", tol=0)
        lsa = make_pipeline(svd, Normalizer(copy=False))  # normalize data
        lsa.fit(self.tfidf)
        u = lsa.transform(self.tfidf)
        x = numpy.matrix.dot(u, svd.components_)
        return x, u

    def eliminate_dimensions(self):
        res = 0
        u, eigen, v = numpy.linalg.svd(self.tfidf, compute_uv=True)
        normalized_eigenvalues = eigen / numpy.sum(eigen)
        eigenvalues = numpy.diag(eigen)
        for i in range(0, len(eigen)):
            res += normalized_eigenvalues[i]
            if res >= self.p_eig:
                k = i+1
                x = numpy.matrix.dot(numpy.matrix.dot(u[-1, 0:k], eigenvalues[0:k, 0:k]), v[0:k, :])
                return x

    def process_phrase(self, phrase, keywords):
        tfidf_utterance = numpy.array(self.tf_idf(phrase, self.features_utterance))
        tfidf_keywords = numpy.array(self.tf_idf(phrase, keywords))
        self.tfidf = numpy.empty([len(tfidf_utterance), len(tfidf_utterance[0]) + len(tfidf_keywords[0])])
        self.tfidf = numpy.concatenate([tfidf_utterance, tfidf_keywords], axis=1)
        x = numpy.round(self.eliminate_dimensions(), 10)
        return x

    def process_examples(self, keywords):
        lsa = []
        for phrase in self.phrases:
            lsa.append(self.process_phrase(phrase, keywords).tolist())
        return lsa



