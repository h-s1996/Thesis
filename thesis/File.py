# coding: utf-8
import random
from LSA import LSA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold # import KFold
import numpy as np


class File:

    def __init__(self):
        self.keywords = [   'Sim', 'disposto', 'humor', 'Gosto', 'família', 'casa', 'filha', 'norte do país', 'bola',
                            'escalada', 'toda', 'irmão', 'primos', 'estrangeiro', 'prazer na vida', 'novas memórias', 'netos', 'mora', 'França',
                            'norma gosto', 'amigos em casa', 'casa de amigos', 'casa no quentinho', 'casa rodeada', 'filme', 'parque da cidade', 'pessoa do Verão',
                            'voltas pela cidade', 'sítios novos', 'novos sítios', 'parques da cidade', 'longos passeios pela cidade', 'chuva', 'pessoa friorenta',
                            'série', 'praia', 'parque das nações', 'novos locais', 'Lisboa', 'beira do Tejo', 'café', 'miradouros', 'melhor a natureza', 'sítio favorito',
                            'zonas exteriores', 'sítio preferido', 'amigo a falar', 'novas cidades', 'jardim da Estrela', 'lareira ligada', 'televisão no sofá',
                            'novas culturas', 'zonas interiores', 'restaurantes', 'contacto', 'toda a gente', 'solzinho', 'cerveja', 'praça do comércio', 'livro',
                            'preferência', 'avenida marginal', 'floresta, cogumelos', 'voltinha', 'coisas', 'favorita da zona', 'largo do Rato', 'Deus', 'pouco', 'calor',
                            'caminhadas', 'mundo', 'jardins', 'copos', 'residência', 'alguém', 'dúvida', 'passo', 'poder', 'Gosto de bacalhau', 'Gosto de carne',
                            'gosto de lasanha', 'Gosto de tantas', 'Gosto de dourada', 'Gosto de gelado italiano', 'Pratos de bacalhau', 'carne favorita',
                            'dúvida a lasanha', 'Carne de vaca', 'Carne de Frango', 'gosto de imensas coisas', 'Gosto de truta salmonada', 'sobremesa italiana',
                            'carne preferida', 'bacalhau à brás', 'bacalhau sabe', 'sobremesa favorita', 'melhor sobremesa', 'Carne de porco', 'comida italiana',
                            'tiramissu italiano', 'comida favorita', 'sobremesa preferida', 'particular lasanha', 'dúvida panacota', 'cozinha italiana',
                            'pizza napolitana', 'pergunta difícil', 'variedade de comida', 'gastronomia italiana', 'melhor cozinha', 'comida portuguesa',
                            'frango assado', 'bife vaca', 'massas', 'lasanhas', 'imensos pratos deliciosos', 'falar da massa', 'só prato favorito', 'ragu',
                            'pergunta complicada', 'posta de vitela', 'muitas', 'questão', 'magnífico gelado confeccionado em Itália', 'enorme variedade de pizzas', 
                            'peixe preferido', 'boas', 'gelados', 'queijos', 'parmegianas', 'netinho Henrique', 'caro', 'pesto', 'filetes de pescada', 'Robalo', 'sonhos',
                            'atum', 'Impossível', 'canellonis', 'restaurantes italianos', 'dentinhos', 'filho', 'tiramissu', 'canguru', 'sabor', 'preferência', 'pouco',
                            'jantar a casa', 'comidas das quais', 'Pastéis de nata', 'mãe', 'Bolonha', 'mundo', 'momento', 'pastas', 'nutella', 'risotto', 'broa', 'acordo',
                            'combinação', 'foodie', 'Picanha', 'Salmão', 'Cannoli',' prazer falar', 'Gostei de falar', 'próxima', 'conversa', 'jantar', 'Desculpa', 'oportunidade',
                            'pouco', 'Grande abraço', 'companhia', 'farmácia', 'fome', 'visita', 'coisas', 'tese', 'Lamento', 'pedacinho', 'ensaio', 'correios', 'compromisso', 'aí']
        self.x = []
        self.y = []
        self.robots_vectors = []
        self.splits = []
        file = open('/content/thesis/finaltextfile.txt', 'r')
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
            self.transform(groups)  # transform robot phrases into numbers and save them
            examples = []
            for g in groups:
                for i in range(0, len(g.human)):
                    examples.append(g.human[i])
                    examples.append(g.robot[0])
        finally:
            file.close()
            self.split_sets(examples)

    def transform(self, groups):
        i = 0
        for g in groups:
            self.robots_vectors.append(RobotVector(g.robot[0], i))
            g.robot[0] = i
            i = i+1

    def split_sets(self, examples):
        skf = StratifiedKFold(n_splits = 5, shuffle=True)
        self.x = np.array(examples[::2])
        self.y = np.array(examples[1::2])
        self.splits = skf.split(self.x,self.y)

    def get_ids(self):
        aux = []
        for r in self.robots_vectors:
            aux.append(r.id)
        return aux

    def search_for_phrase(self, lsa, classifier, phrase, keywords):
        lsa_result = lsa.process_phrase(phrase, lsa.manage_keywords(keywords))
        phrase_id = classifier.predict(LSA.normalizer(lsa_result))
        for e in self.robots_vectors:
            if e.id == phrase_id:
                return e.phrase


class Group:
    def __init__(self):
        self.human = []
        self.robot = []


class RobotVector:
    def __init__(self, phrase, number):
        self.id = number
        self.phrase = phrase
