from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import speech_recognition as sr
from gtts import gTTS
import string
import numpy
import nltk
import sys
import os


class Examples:

    def __init__(self, phrases, lsa_s):
        self.phrases = phrases
        self.lsa_s = numpy.array(lsa_s)
        self.examples = self.populate()

    def populate(self):
        examples = []
        for i in range(0, len(self.phrases)):
            id_phrase = self.check_phrase(self.phrases[i], i)
            examples.append(Example(self.phrases[i], self.lsa_s[i], id_phrase + 1))
        return examples

    def check_phrase(self, phrase, id_phrase):
        for j in range(0, len(self.phrases)):
            if id_phrase != j:
                if self.phrases[j] == phrase:
                    if id_phrase > j:
                        return j
                    else:
                        return id_phrase
        return id_phrase

    def cluster(self, clusters):
        for e in self.examples:
            for cluster in clusters:
                for elem in cluster:
                    if elem == e.id:
                        e.id = cluster[0]

    def get_ids(self):
        aux = []
        for e in self.examples:
            aux.append(e.id)
        return aux

    def search_for_phrase(self, lsa, classifier, phrase, keywords):
        lsa_result = lsa.process_phrase(phrase, lsa.manage_keywords(keywords))
        phrase_id = classifier.predict(LSA.normalizer(lsa_result))
        for e in self.examples:
            if e.id == phrase_id:
                return e.phrase


class Example:

    def __init__(self, phrase, lsa, number):
        self.id = number
        self.phrase = phrase
        self.lsa = lsa


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
                              norm=None,
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
                              norm=None,
                              tokenizer=self.tokenize,
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
        return x, u

    def process_phrase(self, phrase, keywords):
        x_utterance, u_utterance = self.reduce_dimensionality(phrase, self.features_utterance)
        x_keywords, u_keywords = self.reduce_dimensionality(phrase, keywords)
        return numpy.round(numpy.concatenate([x_utterance[len(x_utterance) - 1], x_keywords[len(x_keywords) - 1]]), 10)

    def process_examples(self, keywords):
        lsa = []
        for phrase in self.phrases:
            lsa.append(self.process_phrase(phrase, keywords).tolist())
        return lsa

    def process_robot_examples(self, keywords):
        x_utterance, u_utterance = self.reduce_dimensionality([], self.features_utterance)
        x_keywords, u_keywords = self.reduce_dimensionality([], keywords)
        return numpy.round(numpy.concatenate([x_utterance.T, x_keywords.T]).T, 10).tolist(), numpy.round(
            numpy.concatenate([u_utterance.T, u_keywords.T]).T, 10).tolist()


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


class NaivesClassifier:

    def __init__(self):
        self.classifier = MultinomialNB(alpha=1.0e-10)

    def train(self, x_train, y_train):
        x_naive = numpy.empty(x_train.shape)
        for i in range(0, len(x_train)):
            x_naive[i] = LSA.normalizer(x_train[i])
        self.classifier.fit(x_naive, y_train)

    def predict(self, value):
        aux = self.classifier.predict(numpy.reshape(value, (1, len(value))))
        return aux


class SpeakWithTheRobot:
    def __init__(self, human_lsa, naives, human_keywords, robot_vectors):
        self.slow = False
        self.device_id = 0
        self.lang = 'pt-pt'
        self.naives = naives
        self.chunk_size = 2048
        self.r = sr.Recognizer()
        self.sample_rate = 48000
        self.human_lsa = human_lsa
        self.robot_vectors = robot_vectors
        self.human_keywords = human_keywords

    def hear(self):
        with sr.Microphone(device_index=self.device_id, sample_rate=self.sample_rate, chunk_size=self.chunk_size) as source:
            self.r.adjust_for_ambient_noise(source)
            print("Say Something")
            audio = self.r.listen(source)
            try:
                text = self.r.recognize_google(audio, language="pt-PT")
                print("you said: " + text)
                return text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service;{0}".format(e))

    def speak(self, phrase):
        tts = gTTS(text=phrase, lang=self.lang, slow=self.slow)
        tts.save("soundfile.mp3")
        os.system("mpg123 soundfile.mp3")
        return

    def speaking_to_the_robot(self):
        while True:
            print("Press a character")
            c = sys.stdin.read(1)
            if c == 's':
                self.speak(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, self.hear(),
                                                                self.human_keywords))
            elif c == 'q':
                break
