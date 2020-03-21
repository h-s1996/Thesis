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

print("Data imported.")

MIN_FREQ = 3
MAX_GRAM = 5
P_EIG = 0.5
ALPHA = 1e-10
time_score = []
lsas = []
min_freq = [1, 2, 3, 4, 5, 6]
max_gram = [1, 2, 3, 4, 5, 6]
p_eig = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
y = []
y_error_min = []
y_error_max = []

###############################################################################

###############################################################################
# Latent Semantic Analysis
###############################################################################
for mi in min_freq:
    lsas.append(LSA(MAX_GRAM, mi, P_EIG))
for ma in max_gram:
    lsas.append(LSA(ma, MIN_FREQ, P_EIG))
for p in p_eig:
    lsas.append(LSA(MAX_GRAM, MIN_FREQ, p))

for lsa in lsas:
    test_score = []
    print("Parameters: Min_freq =", lsa.min_freq, "NGram_max =", lsa.ngram_max, "P_eig =", lsa.p_eig*100)
    lsa_results = lsa.process_utterances_through_lsa(db.human_utterances)
    print("LSA Results computed.")
###############################################################################

    for j in range(50):
        ###############################################################################
        # Data Division
        ###############################################################################
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
            test_score.append(naive.test_score(numpy.array(sets.lsa_vectors_test[i]),
                                               numpy.array(sets.robot_ids_test[i])))
    avg = numpy.round(numpy.average(numpy.array(test_score)), 2)
    y.append(avg)
    min_ = numpy.round(numpy.array(test_score).min(), 2)
    y_error_min.append(numpy.round(avg - min_, 2))
    max_ = numpy.round(numpy.array(test_score).max(), 2)
    y_error_max.append(numpy.round(max_ - avg, 2))
    print("Avg test performance: ", avg)
    print(min_)
    print(max_)
    print('\n'*3)

print("y = ", y)
print("yerrormin = ", y_error_min)
print("yerrormax = ", y_error_max)
###############################################################################

