# coding: utf-8
from sklearn.model_selection import StratifiedKFold


class Set:
    """
    A class that depicts the division of the database into the training and testing sets based on the StratifiedKFold.
    This class returns K different splits of the same database. The K is symbolized by the variable n_splits.
    ...
    """
    def __init__(self, lsa_vectors, robot_ids, human_utterances, n_splits):
        """
        :param lsa_vectors: the vectors obtained after LSA that numerically describe the textual information of the
        human utterances available in the database.
        :type lsa_vectors: numpy array
        :param robot_ids: the ids of the robot utterances that correspond to each human lsa vector
        :type robot_ids: list
        :param human_utterances: the human utterances available in the database
        :type human_utterances: list
        :param n_splits: how many parts the same database is divided and how many different splits are executed. In each
        different split a different portion of the database is used as testing set.
        :type n_splits: int
        """
        self.splits = StratifiedKFold(n_splits=n_splits, shuffle=True).split(lsa_vectors, robot_ids)
        self.lsa_vectors_train = []
        self.lsa_vectors_test = []
        self.robot_ids_train = []
        self.robot_ids_test = []
        self.test_phrases = []
        for train_index, test_index in self.splits:
            self.lsa_vectors_train.append(lsa_vectors[train_index])
            self.lsa_vectors_test.append(lsa_vectors[test_index])
            self.robot_ids_train.append(robot_ids[train_index])
            self.robot_ids_test.append(robot_ids[test_index])
            self.test_phrases.append(human_utterances[test_index])
