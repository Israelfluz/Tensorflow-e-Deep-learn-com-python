# Importação da biblioteca pandas para manipulação, leitura e visualização de dados
import pandas as pd

# Carregamento da base de dado para análise
base = pd.read_csv('census.csv')

base.head()

# Visualizando um registro dentro da base, como por exemplo o income
base['income'].unique()

# Visualizando o núemro de registros que temos nessa base
base.shape

# Realizando uma conversão nos valores do income, quando o valor for <=50K ele vai ser 0
# e quando for >50K eke vai ser 1 
# Criando função para conversão
def converte_class(rotulo):
    if rotulo == ' >50K':
        return 1
    else:
        return 0

# Mudando os valores com a função do pandas (.apply)
base['income'] = base['income'].apply(converte_class)

# Variável X armazena os atributos previsores com o pandas
x = base.drop('income', axis = 1) # Ao criar a variável desse forma estou retirando da base de dados a coluna income

x.head()

# Variável Y recebe a resposta que é a classe com o pandas
y = base['income'] # Ao criar a variável dessa forma ela vem apenas com uma coluna em sua base de dados 

y.head()

type(y)

# ======== Transformar os atributos o número em categórico =========

# Visualizar como estão distribuidos os valores numéricos da coluna age para depois transformá-los em categóricos
base.age.hist()

# Importando a biblioteca tensorflow
import tensorflow as tf

# Variável que recebe os valores numéricos da coluna age
idade = tf.feature_column.numeric_column('age')

# Criando variável e transformando os valores numerico da coluna em atributos categóricos com o bucketized_column
idade_categorica = [tf.feature_column.bucketized_column(idade, boundaries = [20, 30, 40, 50, 60, 70, 80, 90])]

print(idade_categorica)

# Criando variável para receber atributo categórico
nome_colunas_categoricas = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

# Criando a coluna categórica
colunas_categoricas = [tf.feature_column.categorical_column_with_vocabulary_list(key = c, vocabulary_list = x[c].unique()) for c in nome_colunas_categoricas]

print(colunas_categoricas)
      
# Criando variável para receber atributo numérico
nome_colunas_numericas = ['final-weight', 'education-num', 'capital-gain', 'capital-loos', 'hour-per-week']

# Criando a coluna numérica
colunas_numericas = [ tf.feature_column.numeric_column(key = c) for c in nome_colunas_numericas]

print(colunas_numericas)

# Criando colunas
colunas = idade_categorica + colunas_categoricas + colunas_numericas

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.3)

# ============ Construção do modelo de aprendizagem com estimator ==============

# Criando variável
funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = x_treinamento, y = y_treinamento,
                                                        batch_size = 32, num_epochs = None, shuffle = True)

# Criando o classificador
classificador = tf.estimator.LinearClassifier(feature_columns = colunas)

# Realizando o treinamento
classificador.train(input_fn = funcao_treinamento, steps = 10000)

# Criando a função para previsão
funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = x_teste, batch_size = 32, shuffle = False)

previsoes = classificador.predict(input_fn = funcao_previsao)

list(previsoes)

previsoes_final = []
for p in classificador.predict(input_fn = funcao_previsao):
    previsoes_final.append(p['class_ids'])

previsoes_final

# Visualizando a taixa de acerto
from sklearn.metrics import accuracy_score
taxa_acerto = accuracy_score(y_teste, previsoes_final)

taxa_acerto
