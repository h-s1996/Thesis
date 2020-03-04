#!/usr/bin/env python
from File import File
from LSA import LSA
from Set import Set
from NaiveBayesClassifier import NaiveBayesClassifier
from nltk.corpus import stopwords
import numpy
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

###############################################################################
#  Initializing
###############################################################################

f = File()

print("Data imported.")

MIN_FREQ = 3
MAX_GRAM = 5
P_EIG = 0.95
y = []
yerrormin = []
yerrormax = []

aux = stopwords.words("portuguese")
stop_words = [aux]
aux.append('é')
stop_words.append(aux)
stop_words.append(["e", "de", "da", "do", "dos", "das", "em", "o", "a", "os", "as", "que", "um", "uma", "para", "com", "no", "na", "nos", "nas",
                          "por", "por", "mais", "se", "como", "mais", "à", "às", "ao", "aos", "ou", "quando", "muito", "pela", "pelas", "pelos",
                          "pelo", "isso", "esse", "essa", "esses", "essas", "num", "numa", "nuns", "numas", "este", "esta", "estes", "estas", "isto",
                          "aquilo", "aquele", "aquela", "aqueles", "aquelas", "sem", "entre", "nem", "quem", "qual", "depois", "só", "mesmo"])
stop_words.append([])

for s in stop_words:
    l = LSA(MAX_GRAM, MIN_FREQ, P_EIG, f.x, s)
    test_score = []
    print("LSA created.")

    ###########################
    # LSA
    human_keywords = l.manage_keywords(f.keywords)
    lsa_results = l.train_phrases([])
    print("LSA Results computed.")
    for time in range(50):
        sets = Set(lsa_results, f.y, f.x)
        for i in range(len(sets.x_train)):
            ###########################

            ###########################
            # NAIVE BAYES
            naive = NaiveBayesClassifier(alpha=0.01)
            naive.train(numpy.array(sets.x_train[i]), sets.y_train[i])
            test_score.append(naive.test_score(numpy.array(sets.x_test[i]), numpy.array(sets.y_test[i])))
    avg = numpy.round(numpy.average(numpy.array(test_score)), 2)
    y.append(avg)
    min_ = numpy.round(numpy.array(test_score).min(), 2)
    yerrormin.append(numpy.round(avg - min_, 2))
    max_ = numpy.round(numpy.array(test_score).max(), 2)
    yerrormax.append(numpy.round(max_ - avg, 2))
    print("Avg test performance: ", avg)
    print(min_)
    print(max_)
    print('\n'*3)

print("y = ", y)
print("yerrormin = ", yerrormin)
print("yerrormax = ", yerrormax)

