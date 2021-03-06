#!/usr/bin/env python
from File import File
from LSA import LSA
from Set import Set
from NaiveBayesClassifier import NaiveBayesClassifier
import numpy
import time

###############################################################################
#  Initializing
###############################################################################

f = File()

print("Data imported.")

MIN_FREQ = 2
MAX_GRAM = 2
P_EIG = 0.5
test_score = []

#min_freq = [2]
#max_gram = [1, 2, 3, 4, 5, 6]
#p_eig = [0.2]

#for mi in min_freq:
#    lsa = LSA(MAX_GRAM, mi, P_EIG, train_set.phrases)
# for ma in max_gram:
#    lsa.append(LSA(ma, MIN_FREQ, P_EIG, train_set.phrases))
# for p in p_eig:
#lsa = LSA(MAX_GRAM, MIN_FREQ, p, train_set.phrases)

for train_index, test_index in f.splits:
    train_set = Set(f.x[train_index], f.y[train_index])
    test_set = Set(f.x[test_index], f.y[test_index])
    print("Data imported.")
    lsa = LSA(MAX_GRAM, MIN_FREQ, P_EIG, train_set.phrases)
    print("LSA created.")

    ###########################
    # LSA
    human_keywords = lsa.manage_keywords(f.keywords)
    print("Start Train LSA", time.ctime(time.time()))
    ex1 = lsa.process_examples(human_keywords, train_set)
    ex1.shutdown(wait=True)
    print("End Train LSA", time.ctime(time.time()))

    print("LSA Results computed.")
    ###########################

    ###########################
    # NAIVE BAYES

    naive = NaiveBayesClassifier()
    print("Start Test LSA", time.ctime(time.time()))
    ex2 = lsa.process_examples(human_keywords, test_set)
    print("End Test LSA", time.ctime(time.time()))
    naive.train(numpy.array(train_set.get_lsa_results()), train_set.get_class_labels())
    ex2.shutdown(wait=True)

    test_score.append(naive.test_score(numpy.array(test_set.get_lsa_results()), numpy.array(test_set.get_class_labels()), "test"))
    naive.test_score(numpy.array(train_set.get_lsa_results()), numpy.array(train_set.get_class_labels()), "train")

print("Avg test performance: ", numpy.average(numpy.array(test_score)))

