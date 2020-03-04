# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import concurrent.futures
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
        self.features_utterance = self.get_features_utterance()

    @staticmethod
    def normalizer(x_abnormal):
        minimum = x_abnormal.min()
        maximum = x_abnormal.max()

        if minimum == maximum:
            return x_abnormal
        else:
            x_new = (x_abnormal - minimum) / (maximum - minimum)
            return x_new

    def tokenize(self, t):
        if t in self.stopwords:
            return []
        sentence = t.lower()
        sentence = word_tokenize(sentence)
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

    def tf_idf(self, examples, keywords):
        vec = TfidfVectorizer(stop_words=self.stopwords,
                              vocabulary=keywords,
                              tokenizer=self.tokenize,
                              ngram_range=(self.ngram_min, self.ngram_max))
        x = vec.fit_transform(examples)
        return x.todense()

    def eliminate_dimensions(self, tfidf):
        if self.p_eig == 1:
            return tfidf
        res = 0
        eigen = numpy.linalg.svd(tfidf, compute_uv=False)
        normalized_eigenvalues = eigen / numpy.sum(eigen)
        eigenvalues = numpy.diag(eigen)
        for i in range(0, len(eigen)):
            res += normalized_eigenvalues[i]
            if res >= self.p_eig:
                svd = TruncatedSVD(n_components= (i+1), algorithm="arpack", tol=0)
                svd.fit(tfidf)
                u = svd.transform(tfidf)
                x = numpy.matrix.dot(u, svd.components_)
                return x

    def train_phrases(self, keywords):
        tfidf_utterance = numpy.array(self.tf_idf(self.phrases, self.features_utterance))
        tfidf_keywords = numpy.array(self.tf_idf(self.phrases, keywords))
        x = numpy.round(self.eliminate_dimensions(numpy.concatenate([tfidf_utterance, tfidf_keywords], axis=1)), 10)
        return x

