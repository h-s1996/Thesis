#!/usr/bin/env python
from File import File
from LSA import LSA
from Set import Set
from NaiveBayesClassifier import NaiveBayesClassifier
import numpy

###############################################################################
#  Initializing
###############################################################################

f = File()

print("Data imported.")

MIN_FREQ = 3
MAX_GRAM = 5
P_EIG = 0.95
ALPHA = 1e-10
test_score = []

###########################
# LSA
l = LSA(MAX_GRAM, MIN_FREQ, P_EIG, f.x)
print("Parameters: Min_freq =", l.min_freq,"NGram_max =", l.ngram_max, "P_eig =", l.p_eig*100)
lsa_results = l.train_phrases()

print("LSA Results computed.")
sets = Set(lsa_results, f.y, f.x)
for i in range(len(sets.x_train)):

    ###########################
    # NAIVE BAYES
    naive = NaiveBayesClassifier(alpha=ALPHA)
    naive.train(numpy.array(sets.x_train[i]), sets.y_train[i])
    test_score.append(naive.test_score(numpy.array(sets.x_test[i]), numpy.array(sets.y_test[i])))
    probs = naive.classifier.predict_proba(sets.x_test[i])
    for j in range(len(sets.x_test[i])):
        aux = numpy.argsort(probs[j])[::-1]
        predicted_class = f.search_for_phrase(naive, sets.x_test[i][j])
        r_class = sets.y_test[i][j]
        if r_class != predicted_class:
            if r_class == 6:
                print("Correct prob", probs[j][r_class])
                print("Pred prob", probs[j][predicted_class])
                print(numpy.where(aux == r_class))
                print("Human phrase: " + sets.test_phrases[i][j])
                print("Correct phrase: " + f.get_phrase(r_class))
                print("Predicted phrase: " + f.get_phrase(predicted_class))
                print('\n'*3)

avg = numpy.round(numpy.average(numpy.array(test_score)), 2)
print("Test Score:", avg)
