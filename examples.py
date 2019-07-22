#TESTE 1

example_pt = [
              "Bom dia. Dormiu bem esta noite?", #1
              "Mas levantou-se muito cedo",
              "Bom dia Como está hoje",
              "Agora o quê é que está a planear para o resto do dia", #4
              "Parece-me um dia em cheio Faça bom proveito",#5
              "Sim está ótimo para passear e apanhar sol",
              "Desfrute porque amanhã o sol vai-se embora e começa a chuva",
              "Boa noite O seu dia correu bem",
              "Passou-se alguma coisa Quer falar sobre isso",
              "Conversa com quem",
              "Não falou com a sua família", #11
              "Durma bem boa noite"]

keywords = ['sol', 'cheio', 'proveito', 'desfrute', 'chuva', 'resto', 'coisa', 'conversa', 'família', 'cedo']

#TESTE 2

example_pt = [ "Machine Learning é super divertido",
               "Python é super, super cool",
               "Estatística também é cool",
               "Data Science é divertido",
               "Python é ótimo para Machine Learning",
               "Eu odeio futebol",
               "Futebol é aborrecido de ver",
               "Hoje o tempo está de chuva",
               "Este tempo anda completamente louco"]
keywords = ['super divertido',
            'super cool',
            'machine learning',
            'python',
            'futebol',
            'data science',
            'estatística',
            'chuva',
            'louco']

#  TESTE 3

example_pt = ["Bom dia.",  #1
              "Não dormi tão bem como queria.",
              "Levantei-me cedo para tomar café com a minha filha.", #3
              "Bom dia",
              "Estou bem disposta. Já estive a passear de manhã e depois fui tomar um café com as minhas amigas",
              "Primeiro vou almoçar, depois devo jogar às cartas e no final do dia vou jantar com os meus filhos.", #6
              "Vou aproveitar que o tempo está fantástico.", #7
              "Exatamente. Este tempo é o meu favorito.",
              "Odeio a chuva, deixa-me triste.", #9
              "Boa noite.",
              "Sim mas estou aborrecido.", #11
              "Não se passou nada, apenas não sai nem de casa e só tive um pouco na conversa",
              "Apenas as pessoas daqui do lar e as auxiliares.",
              "Ninguém me ligou hoje. Amanhã falarei com eles. Agora vou descansar." ] #14


#  TESTE 4

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
        "Aproveite."  # 24

]

# azure
human_keywords = ['sol', 'meteorologia', 'fico contente', 'volta', 'casa', 'filho', 'atividades', 'alegria', 'passeio']
robot_keywords = ['nuvens no céu', 'sala de convívio', 'vemo', 'meteorologia', 'proveito', 'passeio']


# yake

# TUDO JUNTO
# 1-GRAM
#human_keywords = ['olá', 'queria', 'fico', 'meteorologia', 'passeio', 'contente', 'volta', 'gostaria', 'sol', 'passear', 'alegria', 'casa', 'tempo', 'sair']
#robot_keywords = ['tempo', 'faça', 'então', 'proveito', 'maravilhoso', 'divirta-se', 'céu', 'fantástico', 'nuvens', 'tenha', 'aproveite', 'estará', 'solarengo', 'passeio', 'inteiro', 'meteorologia', 'existe', 'existem']
# 2-GRAM
#human_keywords = ['fico contente', 'queria', 'olá', 'fico', 'meteorologia', 'sol', 'queria passear', 'passeio', 'contente', 'volta', 'tempo', 'gostaria', 'passear', 'alegria', 'casa', 'sair']
#robot_keywords = ['então divirta-se', 'tempo maravilhoso', 'tempo fantástico', 'tempo','faça', 'então', 'proveito', 'maravilhoso', 'divirta-se', 'céu', 'fantástico', 'existe nuvens', 'nuvens', 'tenha', 'aproveite', 'estará', 'solarengo', 'passeio', 'inteiro', 'meteorologia', 'existe', 'existem']
# 3-GRAM
#human_keywords = ['fico contente', 'olá', 'queria', 'fico', 'meteorologia', 'sol', 'queria passear', 'passeio', 'contente', 'volta', 'tempo', 'gostaria', 'passear', 'gostaria de sair', 'alegria', 'casa', 'sair', 'sair de casa', 'meteorologia dá sol']
#robot_keywords = ['faça bom proveito', 'então divirta-se', 'tempo maravilhoso', 'tempo fantástico', 'nuvens no céu', 'tempo', 'faça', 'então', 'estará sempre solarengo', 'proveito', 'maravilhoso', 'divirta-se', 'céu', 'fantástico', 'existe nuvens', 'nuvens', 'tenha', 'aproveite', 'estará', 'solarengo', 'passeio', 'inteiro', 'meteorologia', 'existe', 'existem']

# FRASE A FRASE
# 3-GRAM
#human_keywords = ['tempo', 'queria', 'passear', 'queria passear', 'hoje está sol', 'hoje está', 'está sol', 'passeio', 'meteorologia', 'fico contente', 'fico', 'contente', 'sol para hoje', 'para hoje','meteorologia dá sol', 'gostaria de sair', 'sair de casa', 'alegria', 'gostaria']
#robot_keywords = ['tempo', 'fantástico', 'tempo fantástico', 'faça bom proveito', 'faça', 'proveito', 'tempo maravilhoso', 'maravilhoso', 'então', 'divirta-se', 'então divirta-se', 'estará sempre solarengo', 'nuvens no céu', 'existe nuvens', 'tenha', 'passeio', 'inteiro', 'aproveite a boa', 'boa meteorologia', 'aproveite', 'céu']
# 2-GRAM
#human_keywords = ['tempo', 'queria', 'passear', 'queria passear', 'hoje está', 'está sol', 'hoje', 'passeio', 'meteorologia', 'fico contente', 'fico', 'contente', 'para hoje', 'sol para', 'hoje', 'gostaria', 'casa', 'alegria']
#robot_keywords = ['tempo', 'fantástico', 'tempo fantástico', 'faça', 'proveito','tempo maravilhoso', 'maravilhoso', 'então', 'divirta-se', 'então divirta-se', 'nuvens no céu', 'céu', 'solarengo', 'tenha', 'passeio', 'boa meteorologia', 'aproveite', 'meteorologia', 'existem']

# 1-GRAM
#human_keywords = ['tempo', 'queria', 'passear', 'hoje', 'está', 'sol', 'passeio', 'meteorologia', 'fico', 'contente', 'volta', 'hoje', 'para', 'sol', 'gostaria', 'casa', 'alegria']
#robot_keywords = ['tempo', 'fantástico', 'faça', 'proveito', 'maravilhoso', 'então', 'divirta-se', 'solarengo', 'céu', 'estará', 'tenha', 'passeio', 'inteiro', 'boa', 'meteorologia', 'aproveite', 'existem', 'nuvens']


human_keywords = ['fico contente', 'queria', 'olá', 'fico', 'queria passear', 'meteorologia', 'sol', 'gostaria', 'passeio', 'contente', 'volta', 'tempo', 'atividades', 'casa', 'vou', 'devo', 'passear', 'alegria', 'sair', 'filho', 'durante']

human_keywords = ['tempo', 'queria', 'passear', 'queria passear', ]
