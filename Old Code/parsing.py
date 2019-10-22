import random
from classes import Group


class File:

    def __init__(self):
        self.examples = []
        file = open('newtextfile.txt', 'r')
        groups = []
        group = False
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
            #TODO HOW MANY RANDOMIZE SHOULD IT DO
            for g in groups:
                self.randomize(g)
                self.randomize(g)
        finally:
            file.close()

    def randomize(self, group):
        for i in range(0, len(group.human)):
            self.examples.append(group.human[i])
            self.examples.append(random.choice(group.robot))






