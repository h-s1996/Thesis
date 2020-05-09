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

while True:
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
        test_score.append(naive.test_score(numpy.array(sets.lsa_vectors_test[i]), numpy.array(sets.robot_ids_test[i])))

    avg = numpy.round(numpy.average(numpy.array(test_score)), 2)
    max_index = test_score.index(max(test_score))
    print("Test Score:", avg)
    print(max_index)

    naive = NaiveBayesClassifier(alpha=ALPHA)
    naive.learning_phase(numpy.array(sets.lsa_vectors_train[max_index]), sets.robot_ids_train[max_index])

    j = 0
    for i in range(len(sets.lsa_vectors_test[max_index])):
        if ((sets.robot_ids_test[max_index][i] == naive.predict_new_robot_id(sets.lsa_vectors_test[max_index][i])) and
                (sets.robot_ids_test[max_index][i] == j)):
            print(sets.test_phrases[max_index][i])
            print(db.get_robot_utterance(sets.robot_ids_test[max_index][i]))
            j = j + 1
    
    if j == 22:
        break
###############################################################################
