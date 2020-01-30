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
P_EIG = 0.5
time_score = []
lsa = []
min_freq = [1, 2, 3, 4, 5, 6]
max_gram = [1, 2, 3, 4, 5, 6]
p_eig = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
y = []
yerrormin = []
yerrormax = []

for mi in min_freq:
    lsa.append(LSA(MAX_GRAM, mi, P_EIG, f.x))
for ma in max_gram:
    lsa.append(LSA(ma, MIN_FREQ, P_EIG, f.x))
for p in p_eig:
    lsa.append(LSA(MAX_GRAM, MIN_FREQ, p, f.x))

for l in lsa:
    print("Parameters: Min_freq =", l.min_freq,"NGram_max =", l.ngram_max, "P_eig =", l.p_eig*100)
    test_score = []
    print("LSA created.")

    ###########################
    # LSA
    human_keywords = l.manage_keywords(f.keywords)
    lsa_results = l.train_phrases(human_keywords)
    print("LSA Results computed.")
    for j in range(50):
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

