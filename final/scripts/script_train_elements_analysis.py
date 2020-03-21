#!/usr/bin/env python
from final.Database import Database
from final.LSA import LSA
from final.Set import Set
from final.NaiveBayesClassifier import NaiveBayesClassifier
from itertools import groupby
import numpy

###############################################################################
#  Initializing
###############################################################################

db = Database()

print("Data imported.")

MIN_FREQ = 3
MAX_GRAM = 5
P_EIG = 0.95
ALPHA = 1e-10
lsa = []
classification = []
elements = []
classification_error_min = []
classification_error_max = []
n_elements = 5
n_labels = [len(list(group)) for key, group in groupby(db.robot_ids)]


while True:
    test_score = []
    for o in range(10):
        x = []
        y = []
        need_labels = [n >= n_elements for n in n_labels]
        count = 0
        for n in need_labels:
            if n:
                count = count + 1

        label = 0
        k = 0
        for n in n_labels:
            if n >= n_elements:
                x.extend(numpy.random.choice(db.human_utterances[k:k+n], n_elements, replace=False))
                y.extend(numpy.random.choice(db.robot_ids[k:k+n], n_elements, replace=False))
            k = k + n

        if not x:
            break

        print("total phrases per class = ", n_elements)
        print("total number of classes = ", count)

        ###############################################################################

        ###############################################################################
        # Latent Semantic Analysis
        ###############################################################################
        lsa = LSA(MAX_GRAM, MIN_FREQ, P_EIG)
        lsa_results = lsa.process_utterances_through_lsa(db.human_utterances)
        print("LSA Results computed.")
        ###############################################################################

        ###############################################################################
        # Data Division
        sets = Set(lsa_results, db.robot_ids, db.human_utterances, n_splits=5)
        ###############################################################################

        ###############################################################################
        # Naive Bayes Classifier
        ###############################################################################
        for i in range(len(sets.lsa_vectors_train)):
            naive = NaiveBayesClassifier(alpha=ALPHA)
            naive.learning_phase(numpy.array(sets.lsa_vectors_train[i]), sets.robot_ids_train[i])
            ###############################################################################

            ###############################################################################
            # Computing the results of the experiment
            ###############################################################################
            test_score.append(
                naive.test_score(numpy.array(sets.lsa_vectors_test[i]), numpy.array(sets.robot_ids_test[i])))
    if not test_score:
        break
    elements.append(n_elements)
    avg = numpy.round(numpy.average(numpy.array(test_score)), 2)
    classification.append(avg)
    min_ = numpy.round(numpy.array(test_score).min(), 2)
    classification_error_min.append(numpy.round(avg - min_, 2))
    max_ = numpy.round(numpy.array(test_score).max(), 2)
    classification_error_max.append(numpy.round(max_ - avg, 2))
    print("Avg test performance: ", avg)
    print(min_)
    print(max_)
    print('\n'*3)
    n_elements = n_elements + 1

print("x = ", elements)
print("y = ", classification)
print("yerrormin = ", classification_error_min)
print("yerrormax = ", classification_error_max)
###############################################################################

