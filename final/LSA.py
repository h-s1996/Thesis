# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
import string
import numpy


class LSA:
    """
    A class that represents all the steps of the Latent Semantic Analysis and wraps them in one class. Through the
    method process_utterances_through_lsa it is possible to process the utterances present in the database. It enables
    the employment of different values of the parameters ngram_max, min_freq and p_eig.
    ...

    Methods
    -------
    tokenize(utterance)
        Tokenizes the utterance received, eliminates the terms that correspond to a stop word or to a punctuation sign
        and withdraws the stem of the remaining terms. Finally it returns a list of the stemmed terms that characterize
        the utterance received as input.
    tf_idf()
        Build the TF-IDF based on the human utterances from the database.
    dimensionality_reduction(tfidf)
        Reduces the dimensionality of the TF-IDF matrix given as input based on the percentage of the cumulative
        eigenvalues.
    process_utterances_through_lsa(human_utterances)
        Processes the human utterances of the database through the Latent Semantic Analysis and returns a TF-IDF matrix
    process_new_human_phrase(new_human_utterance, human_utterances)
        Processes a new human utterance through the Latent Semantic Analysis and returns the corresponding LSA vector.
    @staticmethod
    normalizer(x_abnormal)
        Normalizes the numpy array received as input between 0 and 1.
    """
    def __init__(self, ngram_max, min_freq, p_eig):
        """
        :param ngram_max: the maximum value of the NGram
        :type ngram_max: int
        :param min_freq: the minimum document frequency threshold
        :type min_freq: int
        :param p_eig: the percentage of cumulative eigenvalues
        :type p_eig: float
        """
        self.ngram_max = ngram_max
        self.min_freq = min_freq
        self.p_eig = p_eig
        self.ngram_min = 1
        self.stopwords = ["e", "de", "da", "do", "dos", "das", "em", "o", "a", "os", "as", "que", "um", "uma", "para",
                          "com", "no", "na", "nos", "nas", "por", "por", "mais", "se", "como", "mais", "à", "às", "ao",
                          "aos", "ou", "quando", "muito", "pela", "pelas", "pelos", "pelo", "isso", "esse", "essa",
                          "esses", "essas", "num", "numa", "nuns", "numas", "este", "esta", "estes", "estas", "isto",
                          "aquilo", "aquele", "aquela", "aqueles", "aquelas", "sem", "entre", "nem", "quem", "qual",
                          "depois", "só", "mesmo", "mas"]
        self.features_utterance = []

    @staticmethod
    def normalizer(x_abnormal):
        """
        Normalizes the numpy array received as input between 0 and 1.
        :param x_abnormal: array to be normalized
        :type x_abnormal: numpy array
        :return: array normalized between 0 and 1
        :rtype: numpy array
        """
        minimum = x_abnormal.min()
        maximum = x_abnormal.max()
        if minimum == maximum:
            return x_abnormal
        else:
            x_new = (x_abnormal - minimum) / (maximum - minimum)
            return x_new

    def tokenize(self, utterance):
        """
        Tokenizes the utterance received, eliminates the terms that correspond to a stop word or to a punctuation sign
        and withdraws the stem of the remaining terms. Finally it returns a list of the stemmed terms that characterize
        the utterance received as input.
        :param utterance: utterance to be tokenized
        :type utterance: str
        :return: set of terms that characterizes the utterance received
        :rtype: list
        """
        sentence = utterance.lower()
        sentence = word_tokenize(sentence)
        aux = []
        for word in sentence:
            if self.stopwords:
                if word not in self.stopwords and word not in string.punctuation:
                    aux.append(RSLPStemmer().stem(word.lower()))
            else:
                if word not in string.punctuation:
                    aux.append(RSLPStemmer().stem(word.lower()))
        terms = []
        for word in aux:
            terms.append(word)
        return terms

    def tf_idf(self, human_utterances):
        """
        Build the TF-IDF based on the human utterances from the database and the terms bigger the minimum document
        frequency.
        :return: the TF-IDF matrix (utterances by terms) that corresponds to the human utterance of the database
        :rtype: list
        """
        vec = TfidfVectorizer(min_df=self.min_freq,
                              stop_words=self.stopwords,
                              tokenizer=self.tokenize,
                              ngram_range=(self.ngram_min, self.ngram_max))
        x = vec.fit_transform(human_utterances)
        self.features_utterance = vec.get_feature_names()
        return x.todense()

    def dimensionality_reduction(self, tfidf):
        """
        Reduces the dimensionality of the TF-IDF matrix given as input based on the percentage of the cumulative
        eigenvalues.
        :param tfidf: the TF-IDF matrix (utterances by terms) that corresponds to the human utterance of the database
        :type tfidf: numpy array
        :return: the TF-IDF matrix (utterances by terms) after the dimensionality reduction step
        :rtype: numpy array
        """
        if self.p_eig == 1:
            return tfidf
        res = 0
        u, eigen, v = numpy.linalg.svd(tfidf, compute_uv=True)
        normalized_eigenvalues = eigen / numpy.sum(eigen)
        eigenvalues = numpy.diag(eigen)
        for i in range(0, len(eigen)):
            res += normalized_eigenvalues[i]
            if res >= self.p_eig:
                k = i+1
                x = numpy.matrix.dot(numpy.matrix.dot(u[:, 0:k], eigenvalues[0:k, 0:k]), v[0:k, :])
                return x

    def process_utterances_through_lsa(self, human_utterances):
        """
        Processes the human utterances of the database through the Latent Semantic Analysis and returns a TF-IDF matrix
        (utterances by terms)
        :param human_utterances: a list of all the human utterances available in the database
        :type human_utterances: list
        :return: the TF-IDF matrix (utterances by terms) after being processed by Latent Semantic Analysis (LSA)
        :rtype: numpy array
        """
        tfidf_utterance = numpy.array(self.tf_idf(human_utterances))
        return numpy.round(self.dimensionality_reduction(tfidf_utterance), 10)

    def process_new_human_utterance(self, new_human_utterance, human_utterances):
        """
        Processes a new human utterance through the Latent Semantic Analysis and returns the corresponding LSA vector.
        :param new_human_utterance: new human utterance to be predicted
        :type new_human_utterance: str
        :param human_utterances: a list of all the human utterances available in the database
        :type human_utterances: list
        :return: the LSA vector of the new human utterance
        :rtype: numpy array
        """
        aux = human_utterances
        aux.append(new_human_utterance)
        return self.process_utterances_through_lsa(aux)[-1]
