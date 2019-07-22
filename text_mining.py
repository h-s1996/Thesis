from classes import LSA, Examples, Clustering, NaivesClassifier
from pylab import *
import numpy

###############################################################################
#  Initializing
###############################################################################

utterances = [
        "Bom dia",
        "Bom dia",  # 1
        "Como está o tempo hoje?",
        "Hoje está um tempo fantástico.", #2
        "Boa, queria passear",
        "Faça bom proveito",  # 3

        "Olá",
        "Bom dia",   # 4
        "Hoje está sol?",
        "Sim, está um tempo maravilhoso",   # 5
        "Ainda bem, queria dar um passeio",
        "Então divirta-se",  # 6

        "Bom dia",
        "Bom dia",  # 7
        "Qual é a meteorologia para este dia?",
        "Não existe nuvens no céu, estará sempre solarengo",  # 8
        "Fico contente, queria dar uma volta",
        "Tenha um bom passeio",   # 9

        "Bom dia",
        "Bom dia",  # 10
        "A meteorologia dá sol para hoje?",
        "Sim, para o dia inteiro",  # 11
        "Que alegria, gostaria de sair de casa",
        "Aproveite a boa meteorologia",  # 12

        "Bom dia",
        "Bom dia",  # 13
        "Qual é a meteorologia para hoje?",
        "Hoje está um tempo maravilhoso",  # 14
        "Fico contente, queria dar uma volta",
        "Faça bom proveito",  # 15

        "Olá",
        "Bom dia",  # 16
        "Hoje está sol?",
        "Sim, não existem nuvens no céu",  # 17
        "Ainda bem, gostaria de dar um passeio",
        "Então divirta-se",  # 18

        "Bom dia",
        "Bom dia", #19
        "Hoje vai estar por onde?",
        "Sala de convívio e você?", #20
        "Também vou lá estar a fazer atividades",
        "Vemo-nos lá.",  #21

        "Bom dia",
        "Bom dia",  # 22
        "Vai estar onde durante o dia?",
        "Sala de convívio e você?",  # 23
        "Devo ir até a casa do meu filho",
        "Aproveite.", # 24

        "Bom dia",
        "Bom dia",  # 19
        "Hoje vai estar por onde?",
        "Sala de convívio e você?",  # 20
        "Eu vou lá estar a fazer atividades",
        "Vemo-nos lá."  # 21

]

MIN_FREQ = 2
MAX_GRAM = 3
P_EIG = 0.5

human_utterances = utterances[::2]
robot_utterances = utterances[1::2]

# azure
human_keywords = ['sol', 'meteorologia', 'fico contente', 'volta', 'casa', 'filho', 'atividades', 'alegria', 'passeio']
robot_keywords = ['nuvens no céu', 'sala de convívio', 'vemo', 'meteorologia', 'proveito', 'passeio']

###########################
# LSA
human_lsa = LSA(MAX_GRAM, MIN_FREQ, P_EIG, human_utterances)
human_results = human_lsa.process_examples(human_lsa.manage_keywords(human_keywords))

robot_lsa = LSA(MAX_GRAM, MIN_FREQ, P_EIG, robot_utterances)
robot_results, cluster_results = robot_lsa.process_robot_examples(robot_lsa.manage_keywords(robot_keywords))
###########################

###########################
# ONLY FOR ROBOT UTTERANCES
figure(0)
c = Clustering(cluster_results, len(robot_utterances))
d = c.cluster()
robot_clusters = c.get_clusters(9)
###########################

###########################
# NAIVE BAYES
robot_vectors = Examples(robot_utterances, robot_results)
human_vectors = Examples(human_utterances, human_results)
# robot_vectors.cluster(robot_clusters)  # comment it if you do not want clustering

naives = NaivesClassifier()

naives.train(numpy.array(human_vectors.lsa_s), robot_vectors.get_ids())

print(robot_vectors.search_for_phrase(human_lsa, naives, "Devo ir até à casa do meu filho.", human_keywords))
print(robot_vectors.search_for_phrase(human_lsa, naives, "A meteorologia dá sol para hoje?", human_keywords))
print(robot_vectors.search_for_phrase(human_lsa, naives, "Qual é a meteorologia para este dia?", human_keywords))
print(robot_vectors.search_for_phrase(human_lsa, naives, "Hoje vai estar por onde?", human_keywords))
print(robot_vectors.search_for_phrase(human_lsa, naives, "Hoje está sol?", human_keywords))
print(robot_vectors.search_for_phrase(human_lsa, naives, "Também vou lá estar a fazer atividades.", human_keywords))

###########################
show()
