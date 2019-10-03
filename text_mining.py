from classes import LSA, Examples, NaivesClassifier, SpeakWithTheRobot

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
        "Bom dia",  # 25
        "Hoje vai estar por onde?",
        "Sala de convívio e você?",  # 26
        "Eu vou lá estar a fazer atividades",
        "Vemo-nos lá.",  # 27

        "Bom dia",
        "Bom dia",  # 28
        "Hoje vai fazer o quê?",
        "Hoje é dia de votar.",  # 29
        "Muito bem, eu já votei.",
        "Em quem votou?",  # 30
        "O voto é secreto não lhe posso dizer.",
        "Tem razão.",  # 30

        "Bom dia",
        "Bom dia",  # 28
        "Hoje vai fazer o quê?",
        "Hoje é dia de votar.",  # 29
        "Muito bem, eu já votei.",
        "Em quem votou?",  # 30
        "O voto é secreto não lhe posso dizer.",
        "Tem razão.",  # 30

        "Alô, alô meninos está tudo bem?",
        "Com licença, eu sei isso não me comunico.",

        "Bom tarde",
        "Bom dia",
        "Já passa do meio-dia.",
        "Bom dia é todo o dia puta."
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
# figure(0)
# c = Clustering(cluster_results, len(robot_utterances))
# d = c.cluster()
# robot_clusters = c.get_clusters(9)
# show()
###########################

###########################
# NAIVE BAYES
robot_vectors = Examples(robot_utterances, robot_results)
human_vectors = Examples(human_utterances, human_results)
# robot_vectors.cluster(robot_clusters)  # comment it if you do not want clustering

naives = NaivesClassifier()
naives.train(numpy.array(human_vectors.lsa_s), robot_vectors.get_ids())

###########################
# SPEAKING WITH THE ROBOT
sr = SpeakWithTheRobot(human_lsa, naives, human_keywords, robot_vectors)
sr.speaking_to_the_robot()

###########################
