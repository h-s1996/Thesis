# coding: utf-8
from sklearn.model_selection import StratifiedKFold # import KFold
import numpy

class Set:

    def __init__(self, lsa_results, class_labels):
        self.splits = StratifiedKFold(n_splits = 5, shuffle=True).split(lsa_results, class_labels)
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.split_set(lsa_results, class_labels)

    def split_set(self, lsa_results, class_labels):
        for train_index, test_index in self.splits:
            self.x_train.append(lsa_results[train_index])
            self.x_test.append(lsa_results[test_index])
            self.y_train.append(class_labels[train_index])
            self.y_test.append(class_labels[test_index])  
