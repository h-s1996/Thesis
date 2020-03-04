#!/usr/bin/env python
from File import File
from LSA import LSA
from Set import Set
from NaiveBayesClassifier import NaiveBayesClassifier
import numpy
import datetime
from itertools import groupby
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

###############################################################################
#  Initializing
###############################################################################

f = File()

print("Data imported.")

MIN_FREQ = 3
MAX_GRAM = 5
P_EIG = 0.95
ALPHA = 1e-10
test_score  = []
phrase = ["na", "maior", "parte", "das", "vezes", "o", "meu", "local", "favorito", "é", "o", "centro", "comercial", "muitas",
          "vezes", "não", "compro", "nada", "mas", "adoro", "estar", "a", "par", "das", "novidades"]


###########################
# LSA
l = LSA(MAX_GRAM, MIN_FREQ, P_EIG, f.x)
for p in phrase:
    word = l.manage_keywords([p])
    if word:
        for v in l.features_utterance:
           if v == word[0]:
                print(word[0])
