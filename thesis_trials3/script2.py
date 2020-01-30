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
MAX_GRAM = 3
P_EIG = 0.65
time_score = []
lsa = []
min_freq = [1, 2, 3, 4, 5, 6]
max_gram = [1, 2, 3, 4, 5, 6]
p_eig = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

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
    print("Start", datetime.datetime.now())
    aux1 = datetime.datetime.now()
    lsa_results = l.train_phrases(human_keywords)
    sets = Set(lsa_results, f.y)
    print("LSA Results computed.")
    for i in range(len(sets.x_train)):
        ###########################

        ###########################
        # NAIVE BAYES
        naive = NaiveBayesClassifier(alpha=0.01)
        naive.train(numpy.array(sets.x_train[i]), sets.y_train[i])
        test_score.append(naive.test_score(numpy.array(sets.x_test[i]), numpy.array(sets.y_test[i]), "test"))
        naive.test_score(numpy.array(sets.x_train[i]), numpy.array(sets.y_train[i]), "train")
        print("End", datetime.datetime.now())
        aux2 = datetime.datetime.now()
        time_score.append(aux2-aux1)
        print("Difference", aux2-aux1)
    print("Avg test performance: ", numpy.average(numpy.array(test_score)))
    print('\n'*5)

