#!/usr/bin/env python
from final.Database import Database
from final.LSA import LSA
from final.Set import Set
from final.NaiveBayesClassifier import NaiveBayesClassifier
import numpy

###############################################################################
#  Initializing
###############################################################################

db = Database()
MIN_FREQ = 3
MAX_GRAM = 5
P_EIG = 0.95
ALPHA = 1e-10
test_score = []
print("Data imported.")

###############################################################################

###############################################################################
# Latent Semantic Analysis
###############################################################################
lsa = LSA(MAX_GRAM, MIN_FREQ, P_EIG)
lsa_results = lsa.process_utterances_through_lsa(db.human_utterances)
print("LSA Results computed.")
###############################################################################

test_score = []
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
    test_score.append(numpy.round(naive.test_score(numpy.array(sets.lsa_vectors_test[i]),
                                                   numpy.array(sets.robot_ids_test[i])), 2))

avg = numpy.round(numpy.average(numpy.round(test_score)), 2)
print("All tests results: ")
print(test_score)
print("Average performance: ", avg)

###############################################################################
