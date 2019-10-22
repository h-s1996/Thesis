from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import speech_recognition as sr
from gtts import gTTS
import random
import string
import numpy
import nltk
import sys
import os


class File:

    def __init__(self):
        self.examples = []
        file = open('newtextfile.txt', 'r')
        groups = []
        group = False
        try:
            line = file.readline()
            while line != '':  # The EOF char is an empty string
                if line[0].isdigit():
                    group = Group()
                    groups.append(group)
                else:
                    if group:
                        if line[0] == 'H':
                            group.human.append(line[2:-1])
                        if line[0] == 'R':
                            group.robot.append(line[2:-1])
                line = file.readline()
            #TODO HOW MANY RANDOMIZE SHOULD IT DO
            for g in groups:
                self.randomize(g)
                self.randomize(g)
        finally:
            file.close()

    def randomize(self, group):
        for i in range(0, len(group.human)):
            self.examples.append(group.human[i])
            self.examples.append(random.choice(group.robot))


class Group:
    def __init__(self):
        self.human = []
        self.robot = []


class Examples:

    def __init__(self, phrases):
        self.phrases = phrases
        self.examples = self.populate()

    def populate(self):
        examples = []
        for i in range(0, len(self.phrases)):
            id_phrase = self.check_phrase(self.phrases[i], i)
            examples.append(Example(self.phrases[i], id_phrase + 1))
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

    def __init__(self, phrase, number):
        self.id = number
        self.phrase = phrase


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


class NaivesClassifier:

    def __init__(self):
        self.classifier = MultinomialNB(alpha=0.01)

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
            c = sys.stdin.read(2)
            if c[0] == 's':
                self.speak(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, self.hear(),
                                                                self.human_keywords))
            elif c[0] == 't':
                print(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, "Olá", self.human_keywords))
                print(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, "Como está tudo a andar?",
                                                           self.human_keywords))
                print(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, "Comigo está tudo fantástico.",
                                                           self.human_keywords))
                print(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, "Como está tudo a andar?",
                                                           self.human_keywords))
                print(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, "Qual é o tempo para hoje?",
                                                           self.human_keywords))
                print(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, "Hoje vou almoçar com o meu filho.",
                                                           self.human_keywords))
                print(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, "Vou ao centro comercial à tarde.",
                                                           self.human_keywords))
            elif c[0] == 'q':
                break
