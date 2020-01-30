#!/usr/bin/env python
from File import File
from LSA import LSA
from Set import Set
from NaiveBayesClassifier import NaiveBayesClassifier
import numpy
import datetime

###############################################################################
#  Initializing
###############################################################################

f = File()

print("Data imported.")

MIN_FREQ = 3
MAX_GRAM = 5
P_EIG = 0.95
time_score = []
lsa = []
alpha = [1e-10, 1, 0.5, 0.1, 0.05, 0.01, 0.005] 

y = []
yerrormin = []
yerrormax = []

#for mi in min_freq:
#    lsa.append(LSA(MAX_GRAM, mi, P_EIG, f.x))
#for ma in max_gram:
#    lsa.append(LSA(ma, MIN_FREQ, P_EIG, f.x))
l = LSA(MAX_GRAM, MIN_FREQ, P_EIG, f.x)

test_score = []
print("LSA created.")

###########################
# LSA
human_keywords = l.manage_keywords(f.keywords)
lsa_results = l.train_phrases(human_keywords)
print("LSA Results computed.")
sets = Set(lsa_results, f.y, f.x)
for a in alpha:
    print("Parameters: Min_freq =", l.min_freq,"NGram_max =", l.ngram_max, "P_eig =", l.p_eig*100, "alpha = ", a)
    for i in range(len(sets.x_train)):
        ###########################

        ###########################
        # NAIVE BAYES
        naive = NaiveBayesClassifier(alpha=a)
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
