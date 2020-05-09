# coding: utf-8
import numpy as np
import os


class Database:
    """
    A class used to represent the Database used in the system proposed in this thesis
    ...

    Methods
    -------
    extract_data_from_file()
        Extracts all the data from the file and prepares it to be used in the next steps
    get_all_robot_ids()
        Returns a list of all the IDs of the utterances of the robot present in the file
    get_robot_utterance(robot_id)
        Returns the utterance of the robot whose ID corresponds to the ID received
    """
    def __init__(self):
        self.human_utterances = []
        self.robot_ids = []
        self.groups = []
        self.extract_data_from_file()

    def extract_data_from_file(self):
        """
        Opens the file, withdraws the data and groups them by:
            - different robot utterance (each robot utterance corresponds to a distinct label)
            - utterance type (human/robot)
        """
        this_folder = os.path.dirname(os.path.abspath(__file__))
        my_file = os.path.join(this_folder, 'files/database.txt')
        file = open(my_file, 'r')
        group = False
        i = 0
        try:
            line = file.readline()
            while line != '':  # The EOF char is an empty string
                if line[0].isdigit():
                    group = Group(i)
                    i += 1
                    self.groups.append(group)
                else:
                    if group:
                        if line[0] == 'H':
                            group.human_utterances.append(line[2:-1])
                        if line[0] == 'R':
                            group.set_robot_utterance(line[2:-1])
                line = file.readline()
            file.close()
        finally:
            examples = []
            for g in self.groups:
                for i in range(len(g.human_utterances)):
                    examples.append(g.human_utterances[i])
                    examples.append(g.robot_id)
            self.human_utterances = np.array(examples[::2])
            self.robot_ids = np.array(examples[1::2])

    def get_all_robot_ids(self):
        """
        Returns a list of all the IDs of the utterances of the robot present in the file
        :return: a list of all the IDs of the robot's phrases
        :rtype: list
        """
        return self.robot_ids

    def get_robot_utterance(self, robot_id):
        """
        Returns the utterance of the robot whose ID corresponds to the ID received
        :param robot_id: the ID of the robot phrase
        :type robot_id: int
        :return: the utterance of the robot
        :rtype: str
        """
        for g in self.groups:
            if g.robot_id == robot_id:
                return g.robot_utterance


class Group:
    """
    A class used to represent a Group of human utterances and the corresponding robot response and ID
    ...

    Methods
    -------
    set_robot_utterance()
        Set the utterance of the robot
    """
    def __init__(self, robot_id):
        """
        :param robot_id: the corresponding ID of the phrase
        :type robot_id: int
        """
        self.human_utterances = []
        self.robot_utterance = ""
        self.robot_id = robot_id

    def set_robot_utterance(self, robot_utterance):
        """
        Set the utterance of the robot
        :param robot_utterance: the phrase uttered by the robot
        :type robot_utterance: str
        """
        self.robot_utterance = robot_utterance
