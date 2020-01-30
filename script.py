from classes import File, Examples, LSA, NaivesClassifier
import numpy

f = File()

MIN_FREQ = 2
MAX_GRAM = 2
P_EIG = 0.5
min_freq = [1, 2, 3, 4, 5, 6]
max_gram = [1, 2, 3, 4, 5, 6]
p_eig = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
alphas = [1e-10, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

human_utterances = f.examples[::2]
robot_utterances = f.examples[1::2]
human_keywords = f.keywords
test_human = ["Bom dia", "Como é que anda?", "Está tudo bem?", "Comigo está tudo magnífico.", "Eu estou excelente",
              "Quero ir à casa da minha filha", "Vou almoçar com o meu filho.", "Vou à casa do meu filho",
              "Devo ir ao supermercado", "De manhã vou às lojas do centro comercial",
              "Necessito de ir às compras à tarde", "Está um bom tempo hoje?", "Como está o tempo?",
              "Qual é a meteorologia para hoje?", "Vou dar uma volta de manhã.", "Quero sair hoje ao café.",
              "Amanhã irei numa excursão.", "Adoro uma chávena de café", "Todos os dias amo beber um café.",
              "Amo tomar um chá", "Não um dia que não beba chá"]

print("Data imported.")
robot_vectors = Examples(robot_utterances)
lsa = []

#for mi in min_freq:
#    lsa.append(LSA(MAX_GRAM, mi, P_EIG, human_utterances))
#for ma in max_gram:
#    lsa.append(LSA(ma, MIN_FREQ, P_EIG, human_utterances))
for p in p_eig:
    lsa.append(LSA(MAX_GRAM, MIN_FREQ, p, human_utterances))

for l in lsa:
    results = l.process_examples(l.manage_keywords(human_keywords))

    print("LSAs created.")

    naives = NaivesClassifier(alpha=1e-10)
    # for i in range(len(alphas)):
    #    print(alphas[i])
    #    naives.append(NaivesClassifier(alphas[i]))
    naives.train(numpy.array(results), robot_vectors.get_ids())
    for h in test_human:
        print(robot_vectors.search_for_phrase(l, naives, h, human_keywords))
    print('\n'*5)



