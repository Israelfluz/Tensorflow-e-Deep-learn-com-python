# ======= Classificação Multiclasse com XOR utlizando a função softmax caso tenhamos
# problemas com mais valores de classes, usando a base de dados Iris - classificação de plantas  ========
# ======= Essa implemanteção será feita utilizando a low-level API do Tensorflow ======

# Importação da biblioteca sklearn e do pacote de dados
from sklearn import datasets
iris = datasets.load_iris()
iris

# Variável X de entrada que armazena os atributos previsores.
x = iris.data

x

# Variável Y receberá as respostas.
y = iris.target

y

# Aplicar um pré-processamento para padronizar os dados
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

x

# Realizando uma transformação na variável Y com o OneHotEncoder para que cada  
# resposta tenha três atributos igual quantidade de neurônios na camada de saída da rede neural.
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(categorical_features = [0])
y = y.reshape(-1, 1) # Deixando no formato matriz

y.shape

# Aplicando o processo de transformação
y = onehot.fit_transform(y).toarray() # toarray transforma em um tipo de numpy array

y

# Fazendo o processo de divisão da base de dados
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.3)

x_treinamento.shape

x_teste.shape

# Importação de bibliotecas 
import tensorflow as tf
import numpy as np

# Definindo as variáveis para a rede neural
neuronios_entrada = x.shape[1] # O número de neurômios equivale ao número de 
                               # atributos que na entrada são 4, podendo passar x.shape[1]
neuronios_entrada

neuronios_oculta = int(np.ceil((x.shape[1] + y.shape[1]) / 2))

neuronios_oculta

neuronios_saida = y.shape[1]

neuronios_saida

# Criando a variável W que representa os pesos no formato de dicionário
w = {'oculta': tf.Variable(tf.random_normal([neuronios_entrada, neuronios_oculta])),
     'saida': tf.Variable(tf.random_normal([neuronios_oculta, neuronios_saida]))}

# Adcionando a camada BIAS com os seus pesos
b = {'oculta': tf.Variable(tf.random_normal([neuronios_oculta])),
     'saida': tf.Variable(tf.random_normal([neuronios_saida]))}

# Criando os placesholders (previsores) para receber os dados
xph = tf.placeholder('float', [None, neuronios_entrada])

# Criando o placeholder para receber as respostas
yph = tf.placeholder('float', [None, neuronios_saida])

# Criando o modelo por meio de uma função
def mlp(x, w, bias):
    camada_oculta = tf.add(tf.matmul(x, w['oculta']), bias['oculta']) # Calculo da camanda o culta
    camada_oculta_ativacao = tf.nn.relu(camada_oculta) # processo de ativação com Relu que quando 
                                                       # recebe maior que 0 retorna o número, caso 
                                                       # contrário retorna 0.
    camada_saida = tf.add(tf.matmul(camada_oculta_ativacao, w['saida']), b['saida'])
    return camada_saida

# Criando o modelo
modelo = mlp(xph, w, b)
    