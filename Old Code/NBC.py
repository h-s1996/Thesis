from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import string
import numpy
import nltk


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

    @staticmethod
    def normalizer(x_abnormal):
        minimum = x_abnormal.min()
        maximum = x_abnormal.max()
        x_new = (x_abnormal - minimum) /(maximum - minimum)
        return x_new

    def tokenize(self, t):
        sentence = t.lower()
        sentence = nltk.word_tokenize(sentence)
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
            tokens.extend(self.tokenize(i))
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
                              token_pattern=r'\w{1,}',
                              ngram_range=(self.ngram_min, self.ngram_max))
        vec.fit_transform(self.phrases)
        return vec.get_feature_names()

    def tf_idf(self, phrase, keywords):
        examples = self.phrases
        examples.append(phrase)
        vec = TfidfVectorizer(stop_words=self.stopwords,
                              vocabulary=keywords,
                              tokenizer=self.tokenize,
                              token_pattern=r'\w{1,}',
                              ngram_range=(self.ngram_min, self.ngram_max))
        x = vec.fit_transform(examples)
        return x

    def choose_dimensionality(self, phrase, keywords):
        res = 0
        eigenvalues = numpy.linalg.svd(self.tf_idf(phrase, keywords).todense(), compute_uv=False)
        normalized_eigenvalues = eigenvalues / numpy.sum(eigenvalues)
        for i in range(0, len(eigenvalues)):
            res += normalized_eigenvalues[i]
            if res >= self.p_eig:
                return i + 1

    def reduce_dimensionality(self, phrase, keywords):
        svd = TruncatedSVD(n_components=self.choose_dimensionality(phrase, keywords), algorithm="arpack")
        lsa = make_pipeline(svd, Normalizer(copy=False))  # normalize data
        u = lsa.fit_transform(self.tf_idf(phrase, keywords))
        x = numpy.matrix.dot(u, svd.components_)
        return x[len(x) - 1], u[len(u) - 1]

    def process_phrase(self, phrase, keywords):
        x_utterance, u_utterance = self.reduce_dimensionality(phrase, self.features_utterance)
        x_keywords, u_keywords = self.reduce_dimensionality(phrase, keywords)
        self.u = numpy.round(numpy.concatenate([u_utterance.T, u_keywords.T]).T, 10)
        return numpy.round(numpy.concatenate([x_utterance.T, x_keywords.T]).T, 10)

    def process_examples(self, keywords):
        lsa = []
        for phrase in self.phrases:
            lsa.append(self.process_phrase(phrase, keywords))


class LSA2:

    def __init__(self, ngram_max, min_freq, p_eig, phrases):
        self.ngram_max = ngram_max
        self.min_freq = min_freq
        self.p_eig = p_eig
        self.ngram_min = 1
        self.stopwords = stopwords.words("portuguese")
        self.stopwords.append('é')
        self.phrases = phrases
        self.x_utterance = []
        self.x_keywords = []
        self.features_utterance = []
        self.features_keyword = []

    @staticmethod
    def normalizer(x_abnormal):
        minimum = x_abnormal.min()
        maximum = x_abnormal.max()
        x_new = (x_abnormal - minimum) /(maximum - minimum)
        return x_new

    def tokenize(self, t):
        sentence = t.lower()
        sentence = nltk.word_tokenize(sentence)
        aux = []
        for word in sentence:
            if word not in self.stopwords and word not in string.punctuation:
                aux.append(RSLPStemmer().stem(word.lower()))
        phrase = []
        for word in aux:
            phrase.append(word)
        return phrase

    def tf_idf(self, variant, keywords):
        if variant == 0:
            vec = TfidfVectorizer(min_df=self.min_freq,
                                  stop_words=self.stopwords,
                                  tokenizer=self.tokenize,
                                  token_pattern=r'\w{1,}',
                                  ngram_range=(self.ngram_min, self.ngram_max))
            x = vec.fit_transform(self.phrases)
            self.features_utterance = vec.get_feature_names()
        else:
            vec = TfidfVectorizer(stop_words=self.stopwords,
                                  vocabulary=keywords,
                                  tokenizer=self.tokenize,
                                  token_pattern=r'\w{1,}',
                                  ngram_range=(self.ngram_min, self.ngram_max))
            x = vec.fit_transform(self.phrases)
            self.features_keyword = vec.get_feature_names()
        return x

    def choose_dimensionality(self, variant, keywords):
        res = 0
        eigenvalues = numpy.linalg.svd(self.tf_idf(variant, keywords).todense(), compute_uv=False)
        normalized_eigenvalues = eigenvalues / numpy.sum(eigenvalues)
        for i in range(0, len(eigenvalues)):
            res += normalized_eigenvalues[i]
            if res >= self.p_eig:
                return i + 1

    def reduce_dimensionality(self, variant, keywords):
        svd = TruncatedSVD(n_components=self.choose_dimensionality(variant, keywords), algorithm="arpack")
        lsa = make_pipeline(svd, Normalizer(copy=False))  # normalize data
        u = lsa.fit_transform(self.tf_idf(variant, keywords))
        aux = numpy.matrix.dot(u, svd.components_)

        if variant == 0:
            self.x_utterance = aux
        else:
            self.x_keywords = aux
        return u

    def manage_keywords(self, keywords):
        tokens = []
        for i in keywords:
            tokens.extend(self.tokenize(i))

        vocabulary = []
        for i in tokens:
            repeat = False
            for v in vocabulary:
                if i == v:
                    repeat = True
                    break
            if not repeat:
                vocabulary.append(i)
        return vocabulary

    def process_examples(self, keywords):
        u_utterance = self.reduce_dimensionality(0, [])
        u_keywords = self.reduce_dimensionality(1, keywords)
        return numpy.concatenate([u_utterance.T, u_keywords.T]).T

    def process_new_phrase(self, utterance_keywords, keywords_keywords):

        aux1 = self.tf_idf(1, utterance_keywords).todense()
        if not aux1.size:
            aux1 = numpy.zeros((1, len(utterance_keywords)))

        aux2 = self.tf_idf(1, keywords_keywords).todense()
        if not aux2.size:
            aux2 = numpy.zeros((1, len(keywords_keywords)))
        svd1 = TruncatedSVD(n_components=5, algorithm="arpack")
        lsa1 = make_pipeline(svd1, Normalizer(copy=False))  # normalize data
        x1 = numpy.matrix.dot(lsa1.fit_transform(aux1), svd1.components_)

        svd2 = TruncatedSVD(n_components=3, algorithm="arpack")
        lsa2 = make_pipeline(svd2, Normalizer(copy=False))  # normalize data
        x2 = numpy.matrix.dot(lsa2.fit_transform(aux2), svd2.components_)

        x = numpy.concatenate([x1.T, x2.T]).T
        return numpy.round(x[len(x) - 1], 10)

    def get_reduced_matrix(self):
        return numpy.round(numpy.concatenate([self.x_utterance.T, self.x_keywords.T]).T, 10)