#!/usr/bin/env python
from File import File
from LSA import LSA
from Set import Set
from NaiveBayesClassifier import NaiveBayesClassifier
import numpy
import random
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
ALPHA = 1e-10
lsa = []
classification = []
labels = []
classificationerrormin = []
classificationerrormax = []
n_labels = f.y[-1] + 2

for n in range(1, n_labels):
    labels.append(n)
    print("labels = ", n)
    test_score = []
    for o in range(10):
        x = []
        y = []
        aux = [random.choice(f.y)]
        i = 1
        while True:
            if i == n:
                break
            
            random_class = random.choice(f.y)
            i = i + 1
            for a in aux:
                if a == random_class:
                    i = i - 1
                    break
                if aux[-1] == a:
                    aux.append(random_class)
                    break

        for i in range(len(f.y)):
            for a in aux:
                if f.y[i] == a:
                    x.append(f.x[i])
                    y.append(f.y[i])
                    break

        l = LSA(MAX_GRAM, MIN_FREQ, P_EIG, x)
        print("Parameters: Min_freq =", l.min_freq,"NGram_max =", l.ngram_max, "P_eig =", l.p_eig*100)
        print("LSA created.")

        ###########################
        # LSA
        lsa_results = l.train_phrases([])
        print("LSA Results computed.")
        sets = Set(lsa_results, numpy.array(y), numpy.array(x))
        for i in range(len(sets.x_train)):
            ###########################

            ###########################
            # NAIVE BAYES
            naive = NaiveBayesClassifier(alpha=ALPHA)
            naive.train(numpy.array(sets.x_train[i]), sets.y_train[i])
            test_score.append(naive.test_score(numpy.array(sets.x_test[i]), numpy.array(sets.y_test[i])))
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

print("x = ", labels)
print("y = ", classification)
print("yerrormin = ", classificationerrormin)
print("yerrormax = ", classificationerrormax)

