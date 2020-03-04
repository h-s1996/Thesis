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
ALPHA = 1e-10

###########################
# LSA
l = LSA(MAX_GRAM, MIN_FREQ, P_EIG, f.x)
print("Parameters: Min_freq =", l.min_freq,"NGram_max =", l.ngram_max, "P_eig =", l.p_eig*100)
human_keywords = l.manage_keywords(f.keywords)
lsa_results = l.train_phrases(human_keywords)

print("LSA Results computed.")
while True:
    test_score  = []
    sets = Set(lsa_results, f.y, f.x)
    for i in range(len(sets.x_train)):

        ###########################
        # NAIVE BAYES
        naive = NaiveBayesClassifier(alpha=ALPHA)
        naive.train(numpy.array(sets.x_train[i]), sets.y_train[i])
        test_score.append(naive.test_score(numpy.array(sets.x_test[i]), numpy.array(sets.y_test[i])))

    avg = numpy.round(numpy.average(numpy.array(test_score)), 2)
    max_index = test_score.index(max(test_score))
    print("Test Score:", avg)
    print(max_index)

    naive = NaiveBayesClassifier(alpha=ALPHA)
    naive.train(numpy.array(sets.x_train[max_index]), sets.y_train[max_index])

    j = 0
    for i in range(len(sets.x_test[max_index])):
        if ((sets.y_test[max_index][i] == f.search_for_phrase(naive, sets.x_test[max_index][i])) and (sets.y_test[max_index][i] == j)):
            print(sets.test_phrases[max_index][i])
            print(f.get_phrase(sets.y_test[max_index][i]))
            j = j+1
    
    if j == 22:
        break
    

