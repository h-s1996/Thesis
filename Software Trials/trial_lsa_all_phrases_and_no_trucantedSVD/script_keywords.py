#!/usr/bin/env python
from File import File
from LSA import LSA
from Set import Set
from NaiveBayesClassifier import NaiveBayesClassifier
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

human_keywords = [""]
human_keywords.append(f.keywords)

for o in range(10):
    y = []
    yerrormin = []
    yerrormax = []
    for h in human_keywords:
        l = LSA(MAX_GRAM, MIN_FREQ, P_EIG, f.x)
        test_score = []
        print("LSA created.")

        ###########################
        # LSA
        aux = l.manage_keywords(h)
        lsa_results = l.train_phrases(aux)
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


