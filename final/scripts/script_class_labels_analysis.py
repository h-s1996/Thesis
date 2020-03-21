#!/usr/bin/env python
from final.Database import Database
from final.LSA import LSA
from final.Set import Set
from final.NaiveBayesClassifier import NaiveBayesClassifier
import numpy
import random

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
labels = []
classification_error_min = []
classification_error_max = []
print(db.robot_ids)
n_labels = db.robot_ids[-1]

for n in range(1, n_labels):
    labels.append(n)
    print("labels = ", n)
    test_score = []
    for o in range(10):
        x = []
        y = []
        aux = [random.choice(db.robot_ids)]
        i = 1
        while True:
            if i == n:
                break
            random_class = random.choice(db.robot_ids)
            for a in aux:
                if a == random_class:
                    break
                if aux[-1] == a:
                    i = i + 1
                    aux.append(random_class)
                    break

        for i in range(len(db.robot_ids)):
            for a in aux:
                if db.robot_ids[i] == a:
                    x.append(db.human_utterances[i])
                    y.append(db.robot_ids[i])
                    break
###############################################################################

###############################################################################
# Latent Semantic Analysis
###############################################################################

        lsa = LSA(MAX_GRAM, MIN_FREQ, P_EIG)
        lsa_results = lsa.process_utterances_through_lsa(x)
###############################################################################

###############################################################################
# Data Division
###############################################################################
        sets = Set(lsa_results, numpy.array(y), numpy.array(x), n_splits=5)
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
            test_score.append(naive.test_score(numpy.array(sets.lsa_vectors_test[i]),
                                               numpy.array(sets.robot_ids_test[i])))
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

print("x = ", labels)
print("y = ", classification)
print("yerrormin = ", classification_error_min)
print("yerrormax = ", classification_error_max)
###############################################################################
