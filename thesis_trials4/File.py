# coding: utf-8
import random
from LSA import LSA
import numpy as np


class File:

    def __init__(self):

        self.keywords = [   'Sim', 'disposto', 'Andas animado', 'humor', 'doente', # 1
                            
                            'falar', 'vontade', 'tomar', 'café', 'Gosto', 'bocadinho', 'disposição', 'bocado', 'conversa', # 3
                            
                            'família', 'casa', 'Jantar', 'filha', 'bola', 'escalada', 'irmão', 'primos', 'prazer na vida', 'netos', # 4
                            
                            'toda', 'mora', 'possibilidade', 'mesma cidade', # 5
                            
                            'norte do país', 'estrangeiro', 'França', # 6
                            
                            'Gosto bastante', 'norma gosto', 'casa rodeada', 'amigos', 'sítios novos', 'novos sítios', 'novos locais', 'longos passeios pela cidade',
                            'natureza', 'banco de jardim', 'voltas pela cidade', 'livro', 'beira', 'novas cidades', 'novas culturas', 'cinema', 'restaurantes', 
                            'contacto', 'pássaros', 'cão', 'jogging', 'cerveja', 'sol', 'miradouros', 'parques', 'voltinha', 'atividade favorita', 'poder', 'vento',
                            'caminhadas', 'pessoa', 'aí', 'Lisboa', 'mundo', 'copos', 'biblioteca', 'compras', # 7
                            
                            'praia', 'parque da cidade', 'parque das nações', 'parques da cidade', 'sítio favorito', 'sítio preferido', 'beira do Tejo', 'melhor a natureza',
                            'zonas exteriores', 'praça do comércio', 'sítios', 'avenida marginal', 'floresta', 'cogumelos', 'favorita da zona', 'jardim da Estrela',
                            'largo do Rato', 'Mercados', 'jardins', 'verão', 'preferência pelo mercado', 'confusão toda', # 8

                            'amigos em casa', 'casa de amigos', 'casa no quentinho', 'frio', 'filme', 'chuva', 'série', 'televisão no sofá', 'lareira ligada', 'amigo a falar',
                            'coisas', 'melhor', 'pouco', 'zonas interiores', 'residência', 'alguém', 'sítio coberto', 'exterior', # 9

                            'pessoa do Verão', 'pessoa friorenta', 'toda a gente', 'solzinho', 'Deus', 'calor', 'dúvida', 'passo', 'coisa', 'roupa', 'clima', 'andar',
                            'sol em vez', 'chuva de inverno', # 10

                            'centro comercial', 'vezes', 'local favorito', 'lojas', 'maior', 'corte inglês', 'par das novidades', 'Colombo', # 11

                            'novas roupas', 'roupas novas', 'vestidos', 'novas modas nas lojas', 'quais', 'cores bonitas', 'fashion', 'montras de roupa', 'bijuteria', # 12

                            'Natal gosto', 'época de Natal', 'melhores prendas', 'roupas', 'mãe', # 13

                            'filme ao cinema', 'vezes ao cinema', 'interesse', 'trocos ao final', # 14

                            'comer pipocas no cinema', 'fome', 'grande fã', 'acaso', 'estômago', 'Saúde', # 15

                            'pergunta difícil', 'Gosto de tantas', 'muitas', 'gosto de imensas coisas', 'pergunta complicada', 'comida favorita', 'comida portuguesa',
                            'variedade de comida', 'pratos bons', 'comidas diferentes', 'comidas favoritas', 'coisas boas', 'comidas das quais', 'questão', 'cabrito assado',
                            'bacalhau', 'natas', 'só prato favorito', 'pizza', 'sushi', 'momento', 'foodie', 'opções', # 16

                            'Carne de Frango', 'Gosto de carne', 'carne preferida', 'carne favorita', 'Carne de porco', 'Carne de aves', 'carne de vaca Arouquesa',
                            'Gosto bastante de frango', 'frango assado', 'bife vaca', 'cabrito', 'bife do lombo', 'posta de vitela', 'melhores bifes', 'dentinhos',
                            'filho', 'canguru', 'sabor', 'preferência', # 17

                            'Gosto de bacalhau', 'bacalhau à brás', 'Pratos de bacalhau', 'Gosto imenso de bacalhau', 'bacalhau sabe', 'bacalhau cozido', 'Gosto de dourada',
                            'grelhado gosto', 'Gosto de truta salmonada', 'peixe preferido', 'tipos de peixe', 'Peixe já', 'peixe favorito', 'filetes de pescada',
                            'caro', 'Robalo', 'atum', 'predilecto', 'Salmão', 'broa', 'forma', 'Linguado', # 18

                            'tiramissu italiano', 'sobremesa italiana', 'sobremesa favorita', 'melhor sobremesa', 'sobremesa preferida', 'Gosto de gelado italiano',
                            'comida italiana', 'dúvida panacota', 'magnífico gelado confeccionado em Itália', 'netinho Henrique', 'sonhos', 'jantar a casa', 'nutella', # 19
                            
                            'dúvida a lasanha', 'melhor lasanha', 'gosto de lasanha', 'particular lasanha', 'imenso de lasanha', 'lasanhas', 'melhor comida',
                            'cozinha italiana', 'melhor cozinha', 'verdadeira pizza napolitana', 'risotto', 'gastronomia italiana', 'massas', 'falar da massa', 'pesto',
                            'tagliatelle ragu', 'pastas', 'enorme variedade de pizzas', 'imensos pratos deliciosos', 'boas', 'Itália', 'gelados', 'resto da vida',
                            'único prato', 'queijos', 'parmegianas', 'Impossível', 'canellonis', 'restaurantes italianos', 'perdição', 'tortellini', 'Bolonha',
                            'combinação', 'acordo', # 20

                            'Gostei de falar', 'grande prazer falar', 'jantar', 'conversa toda', 'Desculpa', 'compromisso', 'indo', 'comida', 'programa da Cristina',
                            'companhia', 'incêndio', 'farmácia', 'visita', 'tese', 'Lamento', 'pedacinho', 'ensaio', 'correios', 'comboio', 'escola', # 21
                            
                            'próxima', 'prazer', 'oportunidade', 'Grande abraço', 'continuação', 'breve', 'Okay', 'resto', # 22
                            ]
        self.x = []
        self.y = []
        self.robots_vectors = []
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
            self.split_utterances(examples)

    def transform(self, groups):
        i = 0
        for g in groups:
            self.robots_vectors.append(RobotVector(g.robot[0], i))
            g.robot[0] = i
            i = i+1

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

    def search_for_phrase(self, classifier, phrase):
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

