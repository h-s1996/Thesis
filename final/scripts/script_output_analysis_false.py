#!/usr/bin/env python
from final.File import File
from final.LSA import LSA
from final.Set import Set
from final.NaiveBayesClassifier import NaiveBayesClassifier
import numpy
import math

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
#n_labels = [len(list(group)) for key, group in groupby(f.y)]

print("LSA Results computed.")
sets = Set(lsa_results, f.y, f.x)
for i in range(len(sets.x_train)):
    #error_per_class = numpy.zeros(22)
    #errors = 0
    ###########################
    # NAIVE BAYES
    naive = NaiveBayesClassifier(alpha=ALPHA)
    naive.train(numpy.array(sets.x_train[i]), sets.y_train[i])
    test_score.append(naive.test_score(numpy.array(sets.x_test[i]), numpy.array(sets.y_test[i])))
    for j in range(len(sets.x_test[i])):
        predicted_class = f.search_for_phrase(naive, sets.x_test[i][j])
        r_class = sets.y_test[i][j]
        if r_class != predicted_class:
            if r_class == 4:
                print("Prob of real class:", math.exp(naive.classifier.class_log_prior_[r_class]))
                print("Prob of of predicted class:", math.exp(naive.classifier.class_log_prior_[predicted_class]))
                print(numpy.round(naive.all_classes_result(sets.x_test[i][j]),2))
                for o in range(len(sets.x_test[i][j])):
                    if sets.x_test[i][j][o] > 0.1:
                        print("Term:" + l.features_utterance[o])
                        print("Prob of term given real class:", math.exp(naive.classifier.feature_log_prob_[r_class][o]))
                        print("Prob of term given predicted class:", math.exp(naive.classifier.feature_log_prob_[predicted_class][0][o]))
                        print(sets.x_test[i][j][o])
                        print('\n')
                #print(l.features_utterance[o])
                #errors = errors + 1
                #error_per_class[r_class] = error_per_class[r_class] + 1
                print("Human phrase: " + sets.test_phrases[i][j])
                print("Correct phrase: " + f.get_phrase(r_class))
                print("Predicted phrase: " + f.get_phrase(predicted_class))
                print('\n'*3)
    #print(errors)
    #print(list(error_per_class))
    #print(numpy.round(error_per_class*100/n_labels, 2))

#print(n_labels)
avg = numpy.round(numpy.average(numpy.array(test_score)), 2)
print("Test Score:", avg)
