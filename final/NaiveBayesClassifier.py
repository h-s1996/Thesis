# coding: utf-8
from sklearn.naive_bayes import MultinomialNB
from final.LSA import LSA
import numpy


class NaiveBayesClassifier:
    """
    A class that illustrates the classifier used in this system to generate the robot utterances. It contains a
    Multinomial Naive Bayes Classifier and enables the smoothing (or not) of the probabilities through the Laplace
    parameter.
    ...

    Methods
    -------
    learning_phase(human_vectors, robot_ids)
        The Naive Bayes Classifier computes the class and conditional probabilities needed to predict new LSA vectors.
    predict_new_robot_id(new_human_vector)
        Predicts the robot ID of a new human vector through the classifier received
    test_score(test_human_vectors, test_robot_ids)
        Computes the performance of the system according to a certain testing set (human vectors and robot ids)
    """
    def __init__(self, alpha):
        """
        :param alpha: the Laplace Smoothing Parameter
        :type alpha: float
        """
        self.classifier = MultinomialNB(alpha=alpha)

    def learning_phase(self, human_vectors, robot_ids):
        """
        The Naive Bayes Classifier computes the class and conditional probabilities needed to predict new LSA vectors.
        :param human_vectors: LSA vectors that describe the textual information of the human utterances of the database
        :type human_vectors: numpy array
        :param robot_ids: robot IDs of each corresponding LSA vector
        :type robot_ids: list
        """
        x_naive = numpy.empty(human_vectors.shape)
        for i in range(len(human_vectors)):
            x_naive[i] = LSA.normalizer(human_vectors[i])
        self.classifier.fit(x_naive, robot_ids)

    def predict_new_robot_id(self, new_human_vector):
        """
        Predicts the robot ID of a new human vector through the classifier received
        :param new_human_vector: the LSA vector that numerically describes the human utterance to be predicted
        :type new_human_vector: array
        :return: the robot utterance ID predicted
        :rtype: int
        """
        lsa_vector = LSA.normalizer(new_human_vector)
        robot_utterance_id = self.classifier.predict(numpy.reshape(lsa_vector, (1, len(lsa_vector))))
        return robot_utterance_id

    def test_score(self, test_human_vectors, test_robot_ids):
        """
        Computes the performance of the system according to a certain testing set (human vectors and robot ids)
        :param test_human_vectors: testing human LSA vectors (x)
        :type test_human_vectors: numpy array
        :param test_robot_ids: testing corresponding robot IDs (y)
        :type test_robot_ids: numpy array
        :return: the performance of the system when confronted with the testing set received as input.
        :rtype: float
        """
        aux = self.classifier.score(test_human_vectors, test_robot_ids)*100
        return aux
