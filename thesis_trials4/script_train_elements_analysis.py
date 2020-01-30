#!/usr/bin/env python
from File import File
from LSA import LSA
from Set import Set
from NaiveBayesClassifier import NaiveBayesClassifier
from itertools import groupby
import random
import numpy
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
ALPHA = 1e-10
lsa = []
classification = []
elements = []
classificationerrormin = []
classificationerrormax = []
n_elements = 5
n_labels = [len(list(group)) for key, group in groupby(f.y)]

while True:
    test_score = []
    for o in range(10):
        x = []
        y = []
        need_labels = [n >= n_elements for n in n_labels]
        count = 0
        for n in need_labels:
            if n == True:
                count = count + 1 

        label = 0
        k = 0
        for n in n_labels:
            if n >= n_elements:
                x.extend(numpy.random.choice(f.x[k:k+n], n_elements, replace=False))
                y.extend(numpy.random.choice(f.y[k:k+n], n_elements, replace=False))
            k = k + n

        if not x:
            break

        print("total phrases per class = ", n_elements)
        print("total number of classes = ", count)

        l = LSA(MAX_GRAM, MIN_FREQ, P_EIG, x)
        print("Parameters: Min_freq =", l.min_freq,"NGram_max =", l.ngram_max, "P_eig =", l.p_eig*100)
        print("LSA created.")

        ###########################
        # LSA
        human_keywords = l.manage_keywords(f.keywords)
        lsa_results = l.train_phrases(human_keywords)
        print("LSA Results computed.")
        sets = Set(lsa_results, numpy.array(y), numpy.array(x))
        for i in range(len(sets.x_train)):
            ###########################

            ###########################
            # NAIVE BAYES
            naive = NaiveBayesClassifier(alpha=ALPHA)
            naive.train(numpy.array(sets.x_train[i]), sets.y_train[i])
            test_score.append(naive.test_score(numpy.array(sets.x_test[i]), numpy.array(sets.y_test[i])))
    if not test_score:
        break
    elements.append(n_elements)
    avg = numpy.round(numpy.average(numpy.array(test_score)), 2)
    classification.append(avg)
    min_ = numpy.round(numpy.array(test_score).min(), 2)
    classificationerrormin.append(numpy.round(avg - min_, 2))
    max_ = numpy.round(numpy.array(test_score).max(), 2)
    classificationerrormax.append(numpy.round(max_ - avg, 2))
    print("Avg test performance: ", avg)
    print(min_)
    print(max_)
    print('\n'*3)
    n_elements = n_elements + 1

print("x = ", elements)
print("y = ", classification)
print("yerrormin = ", classificationerrormin)
print("yerrormax = ", classificationerrormax)

