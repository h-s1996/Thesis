# coding: utf-8
from sklearn.naive_bayes import MultinomialNB
import numpy


class NaiveBayesClassifier:

    def __init__(self, alpha):
        self.classifier = MultinomialNB(alpha=alpha)

    @staticmethod
    def normalizer(x_abnormal):
        minimum = x_abnormal.min()
        maximum = x_abnormal.max()
        if minimum == maximum:
            return x_abnormal
        else:
            x_new = (x_abnormal - minimum) / (maximum - minimum)
            return x_new

    def train(self, x_train, y_train):
        x_naive = numpy.empty(x_train.shape)
        for i in range(0, len(x_train)):
            x_naive[i] = self.normalizer(x_train[i])
        self.classifier.fit(x_naive, y_train)

    def predict(self, value):
        return self.classifier.predict(numpy.reshape(value, (1, len(value))))

    def all_classes_result(self, value):
        return self.classifier.predict_proba(numpy.reshape(value, (1, len(value))))

    def test_score(self, x_test, y_test):
        aux = self.classifier.score(x_test, y_test)*100
        return aux
    
