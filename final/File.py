# coding: utf-8
import random
from final.LSA import LSA
import numpy as np
import os


class File:
    def __init__(self):
        self.x = []
        self.y = []
        self.robots_vectors = []
        this_folder = os.path.dirname(os.path.abspath(__file__))
        print(this_folder)
        my_file = os.path.join(this_folder, 'files/database.txt')
        print(my_file)
        file = open(my_file, 'r')
        groups = []
        group = False
        examples = []
        try:
            line = file.readline()
            while line != '':  # The EOF char is an empty string
                if line[0].isdigit():
                    group = Group()
                    groups.append(group)
                else:
                    if group:
                        if line[0] == 'H':
                            group.human.append(line[2:-1])
                        if line[0] == 'R':
                            group.robot.append(line[2:-1])
                line = file.readline()
            self.transform(groups)  # transform robot phrases into numbers and save them
            for g in groups:
                for i in range(0, len(g.human)):
                    examples.append(g.human[i])
                    examples.append(g.robot[0])
        finally:
            file.close()
            self.split_utterances(examples)

    def transform(self, groups):
        i = 0
        for g in groups:
            self.robots_vectors.append(RobotVector(g.robot[0], i))
            g.robot[0] = i
            i = i + 1

    def split_utterances(self, examples):
        self.x = np.array(examples[::2])
        self.y = np.array(examples[1::2])

    def get_ids(self):
        aux = []
        for r in self.robots_vectors:
            aux.append(r.id)
        return aux

    def get_phrase(self, number):
        for e in self.robots_vectors:
            if e.id == number:
                return e.phrase

    @staticmethod
    def search_for_phrase(classifier, phrase):
        phrase_id = classifier.predict(LSA.normalizer(phrase))
        return phrase_id


class Group:
    def __init__(self):
        self.human = []
        self.robot = []


class RobotVector:
    def __init__(self, phrase, number):
        self.id = number
        self.phrase = phrase
