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

MIN_FREQ = 2
MAX_GRAM = 2
P_EIG = 1.0
time_score = []
train_set = []
test_set = []
#min_freq = [2]
max_gram = [1, 3, 4, 5, 6]
#p_eig = [0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
alpha = [0, 1, 0.5, 0.1, 0.05, 0.01, 0.005] 
#for mi in min_freq:
#    lsa = LSA(MAX_GRAM, mi, P_EIG, train_set.phrases)
# for ma in max_gram:
#    lsa.append(LSA(ma, MIN_FREQ, P_EIG, train_set.phrases))
# for p in p_eig:
#lsa = LSA(MAX_GRAM, MIN_FREQ, p, train_set.phrases)
for train_index, test_index in f.splits:
  train_set.append(Set(f.x[train_index], f.y[train_index]))
  test_set.append(Set(f.x[test_index], f.y[test_index]))



for a in alpha:
    test_score = []
    print("alpha", a)
    for i in range(len(train_set)):
        lsa = LSA(MAX_GRAM, MIN_FREQ, P_EIG, train_set[i].phrases)
        print("LSA created.")

        ###########################
        # LSA
        human_keywords = lsa.manage_keywords(f.keywords)
        print("Start", datetime.datetime.now())
        aux1 = datetime.datetime.now()
        ex1 = lsa.process_examples(human_keywords, train_set[i])
        ex1.shutdown(wait=True)
        print("LSA Results computed.")
        ###########################

        ###########################
        # NAIVE BAYES

        naive = NaiveBayesClassifier(a)
        ex2 = lsa.process_examples(human_keywords, test_set[i])
        naive.train(numpy.array(train_set[i].get_lsa_results()), train_set[i].get_class_labels())
        ex2.shutdown(wait=True)
        test_score.append(naive.test_score(numpy.array(test_set[i].get_lsa_results()), numpy.array(test_set[i].get_class_labels()), "test"))
        naive.test_score(numpy.array(train_set[i].get_lsa_results()), numpy.array(train_set[i].get_class_labels()), "train")
        print("End", datetime.datetime.now())
        aux2 = datetime.datetime.now()
        time_score.append(aux2-aux1)
        print("Difference", aux2-aux1)
    print("Avg test performance: ", numpy.average(numpy.array(test_score)))
    print('\n'*5)

