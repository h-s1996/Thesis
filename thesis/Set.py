# coding: utf-8
class Set:

    def __init__(self, phrases, class_labels):
        self.phrases = phrases
        self.elements = Set.populate(phrases, class_labels)

    @staticmethod
    def populate(phrases, class_labels):
        aux = []
        for i in range(len(phrases)):
            aux.append(Element(phrases[i], class_labels[i]))
        return aux

    def set_lsa_result(self, index, lsa_result):
        self.elements[index].lsa_result = lsa_result

    def get_lsa_results(self):
        lsa_results = []
        for e in self.elements:
            lsa_results.append(e.lsa_result)
        return lsa_results

    def get_class_labels(self):
        labels = []
        for e in self.elements:
            labels.append(e.class_label)
        return labels


class Element:

    def __init__(self, phrase, class_label):
        self.phrase = phrase
        self.class_label = class_label
        self.lsa_result = []


