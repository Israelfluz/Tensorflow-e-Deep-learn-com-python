# ============ Realizando uma classificação utilizando redes neurais com estimators ============ 


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

# Realizando a divisão da base de dados
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.3)

# Visualizando as dimensões dos atributos previsores
x_treinamento.shape

# Visualizando as dimensões do atributo de teste
x_teste.shape

base.columns

import tensorflow as tf 

# Criando colunas categóricas (** Mais a frente essa coluna categórica terá de ser covertida em coluna embedding)
workclass = tf.feature_column.categorical_column_with_hash_bucket(key = 'workclass', hash_bucket_size = 100)
education = tf.feature_column.categorical_column_with_hash_bucket(key = 'education', hash_bucket_size = 100)
marital_status = tf.feature_column.categorical_column_with_hash_bucket(key = 'marital-status', hash_bucket_size = 100)
occupation = tf.feature_column.categorical_column_with_hash_bucket(key = 'occupation', hash_bucket_size = 100)
relationship = tf.feature_column.categorical_column_with_hash_bucket(key = 'relationship', hash_bucket_size = 100)
race = tf.feature_column.categorical_column_with_hash_bucket(key = 'race', hash_bucket_size = 100)
country = tf.feature_column.categorical_column_with_hash_bucket(key = 'native-country', hash_bucket_size = 100)

base.sex.unique()
sex = tf.feature_column.categorical_column_with_vocabulary_list(key = 'sex', vocabulary_list=[' Male', ' Female'])

# Buscando a média das idades na base de dados
base.age.mean()

# Calculando o desvio padrão
base.age.std()

# De posse do valor da média e do desvio padrão é hora de fazer a padronização de valores com a função abaixo
def padroniza_age(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(38.58)), tf.constant(13.64))

def padroniza_finalweight(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(189778.36)), tf.constant(105549.977))

def padroniza_education(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(10.08)), tf.constant(2.57))

def padroniza_capitalgain(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(1077.64)), tf.constant(7385.29))

def padroniza_capitalloos(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(87.30)), tf.constant(402.96))

def padroniza_hour(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(40.43)), tf.constant(12.34))


# Definindo colunas numéricas
age = tf.feature_column.numeric_column(key = 'age', normalizer_fn = padroniza_age)
final_weight = tf.feature_column.numeric_column(key = 'final-weight', normalizer_fn = padroniza_finalweight)
education_num = tf.feature_column.numeric_column(key = 'education-num', normalizer_fn = padroniza_education)
capital_gain = tf.feature_column.numeric_column(key = 'capital-gain', normalizer_fn = padroniza_capitalgain)
capital_loos = tf.feature_column.numeric_column(key = 'capital-loos', normalizer_fn = padroniza_capitalloos)
hour = tf.feature_column.numeric_column(key = 'hour-per-week', normalizer_fn = padroniza_hour)


len(base.workclass.unique())

# Realizando a conversão ou mapeamento de uma coluna categórica para o tipo de coluna Embedding
embedded_workclass = tf.feature_column.embedding_column(workclass, dimension = 9)
embedded_education = tf.feature_column.embedding_column(education, dimension = len(base.education.unique()))
embedded_marital = tf.feature_column.embedding_column(marital_status, dimension = len(base['marital-status'].unique()))
embedded_occupation = tf.feature_column.embedding_column(occupation, dimension = len(base.occupation.unique()))
embedded_relationship = tf.feature_column.embedding_column(relationship, dimension = len(base.relationship.unique()))
embedded_race = tf.feature_column.embedding_column(race, dimension = len(base.race.unique()))
embedded_sex = tf.feature_column.embedding_column(sex, dimension = len(base.sex.unique()))
embedded_country = tf.feature_column.embedding_column(country, dimension = len(base['native-country'].unique()))


# Definindo uma variável com o nome colunas que receberam a conversão para o tipo embedding
colunas_rna = [age, embedded_workclass, final_weight, embedded_education, education_num,
               embedded_marital, embedded_occupation, embedded_relationship, 
               embedded_race, embedded_sex,
               capital_gain, capital_loos, hour, embedded_country]


# Realizando o treinamento do algoritmo
funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = x_treinamento, y = y_treinamento,
                                                        batch_size = 32, num_epochs = None, shuffle = True)
classificador = tf.estimator.DNNClassifier(hidden_units = [8, 8], feature_columns=colunas_rna, n_classes=2)
classificador.train(input_fn = funcao_treinamento, steps = 10000) # Treinando o algoritmo


# Avaliando o desempenho do algoritmo com a função teste
funcao_teste = tf.estimator.inputs.pandas_input_fn(x = x_teste, y = y_teste, batch_size = 32,
                                                  num_epochs = 1, shuffle = False)

# Efetivando a avaliação
classificador.evaluate(input_fn=funcao_teste)
