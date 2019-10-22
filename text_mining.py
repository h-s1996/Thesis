from classes import LSA, Examples, NaivesClassifier, SpeakWithTheRobot, File
import numpy

###############################################################################
#  Initializing
###############################################################################

f = File()

MIN_FREQ = 2
MAX_GRAM = 2
P_EIG = 0.6
human_utterances = f.examples[::2]
robot_utterances = f.examples[1::2]

# azure
human_keywords = ['meteorologia', 'filha', 'casa', 'lojas de roupa', 'centro comercial', 'filho', 'compras', 'comida', 'jantar', 'andar', 'irmão', 'netos', 'supermercado']
print("Data imported.")

###########################
# LSA
human_lsa = LSA(MAX_GRAM, MIN_FREQ, P_EIG, human_utterances)
human_results = human_lsa.process_examples(human_lsa.manage_keywords(human_keywords))
print("LSA vectors created.")
###########################

###########################
# NAIVE BAYES
robot_vectors = Examples(robot_utterances)

naives = NaivesClassifier()
naives.train(numpy.array(human_results), robot_vectors.get_ids())

print("Classifier Trained.")

###########################
# SPEAKING WITH THE ROBOT
sr = SpeakWithTheRobot(human_lsa, naives, human_keywords, robot_vectors)
sr.speaking_to_the_robot()

###########################
