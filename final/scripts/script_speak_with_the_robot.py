#!/usr/bin/env python
from final.Database import Database
from final.LSA import LSA
from final.Set import Set
from final.NaiveBayesClassifier import NaiveBayesClassifier
from final.SpeakWithTheRobot import  SpeakWithTheRobot
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

###############################################################################
# Data Division
sets = Set(lsa_results, db.robot_ids, db.human_utterances, n_splits=5)
###############################################################################

###############################################################################
# Naive Bayes Classifier
###############################################################################

naive = NaiveBayesClassifier(alpha=ALPHA)
naive.learning_phase(numpy.array(lsa_results), db.robot_ids)
###############################################################################

speak = SpeakWithTheRobot()
speak.speaking_to_the_robot(lsa, naive, db)

###############################################################################
