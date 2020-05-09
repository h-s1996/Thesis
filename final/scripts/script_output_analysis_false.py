#!/usr/bin/env python
from final.Database import Database
from final.LSA import LSA
from final.Set import Set
from final.NaiveBayesClassifier import NaiveBayesClassifier
import numpy
import math

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

###############################################################################
# Data Division
sets = Set(lsa_results, db.robot_ids, db.human_utterances, n_splits=5)
###############################################################################§

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
    probability = naive.classifier.predict_proba(sets.lsa_vectors_test[i])
    for j in range(len(sets.lsa_vectors_test[i])):
        predicted_class = naive.predict_new_robot_id(sets.lsa_vectors_test[i][j])[0]
        r_class = sets.robot_ids_test[i][j]
        if r_class != predicted_class:
            print("Correct total prob", numpy.round(probability[j][r_class], 3))
            print("Pred total prob", numpy.round(probability[j][predicted_class], 3))
            print("Prob of real class:", numpy.round(math.exp(naive.classifier.class_log_prior_[r_class]), 3))
            print("Prob of of predicted class:",
                  numpy.round(math.exp(naive.classifier.class_log_prior_[predicted_class]), 3))
            for o in range(len(sets.lsa_vectors_test[i][j])):
                if sets.lsa_vectors_test[i][j][o] > 0.1:
                    print("Term:" + lsa.features_utterance[o])
                    print("Prob of term given real class:",
                          math.exp(naive.classifier.feature_log_prob_[r_class][o]))
                    print("Prob of term given predicted class:",
                          math.exp(naive.classifier.feature_log_prob_[predicted_class][o]))
                    print(sets.lsa_vectors_test[i][j][o])
                    print('\n')
            print("Human phrase: " + sets.test_phrases[i][j])
            print("Correct phrase: " + db.get_robot_utterance(r_class))
            print("Predicted phrase: " + db.get_robot_utterance(predicted_class))
            print('\n'*3)
avg = numpy.round(numpy.average(numpy.array(test_score)), 2)
print("Test Score:", avg)
###############################################################################
