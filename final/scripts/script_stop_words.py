#!/usr/bin/env python
from final.Database import Database
from final.LSA import LSA
from final.Set import Set
from final.NaiveBayesClassifier import NaiveBayesClassifier
from nltk.corpus import stopwords
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
y = []
y_error_min = []
y_error_max = []
aux = stopwords.words("portuguese")
stop_words = [aux]
aux.append('é')
stop_words.append(aux)
stop_words.append(["e", "de", "da", "do", "dos", "das", "em", "o", "a", "os", "as", "que", "um", "uma", "para", "com",
                   "no", "na", "nos", "nas", "por", "por", "mais", "se", "como", "mais", "à", "às", "ao", "aos", "ou",
                   "quando", "muito", "pela", "pelas", "pelos", "pelo", "isso", "esse", "essa",
                   "esses", "essas", "num", "numa", "nuns", "numas", "este", "esta", "estes", "estas", "isto",
                   "aquilo", "aquele", "aquela", "aqueles", "aquelas", "sem", "entre", "nem", "quem", "qual",
                   "depois", "só", "mesmo", "mas"])
stop_words.append([])
print("Data imported.")

###############################################################################


for s in stop_words:
    ###############################################################################
    # Latent Semantic Analysis
    ###############################################################################
    lsa = LSA(MAX_GRAM, MIN_FREQ, P_EIG, s) # different LSA class needed
    lsa_results = lsa.process_utterances_through_lsa(db.human_utterances)
    print("LSA Results computed.")
    ###############################################################################
    for time in range(50):
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
