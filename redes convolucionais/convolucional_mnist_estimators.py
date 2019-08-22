#
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist/', one_hot = False)

# Carregando os dados da pasta mnist e criando variáveis para treinamento e teste
x_treinamento = mnist.train.images
x_treinamento.shape
x_teste = mnist.test.images
x_teste.shape

y_treinamento = mnist.train.labels
y_treinamento
y_teste = mnist.test.labels


# Visualizando as imagens da base de dados
import matplotlib.pyplot as plt
plt.imshow(x_treinamento[2].reshape((28, 28)), cmap = 'gray') # cmap = 'gray' deixa o fundo preto e o núemro cinza

# Visualizando a classe
plt.title('Classe: ' + str(y_treinamento[2]))

import tensorflow as tf

# Criando a função cria_rede que se encontra nos parâmetros do classificador
def cria_rede(features, labels, mode):
    # Desse ponto em diante é que se cria toda a estrutura da rede neural
    # Mais parâmetros que devem ser passados no tf.reshape o batch_size, largura, altura, canais das imagens da base de dados 
    entrada = tf.reshape(features['x'], [-1, 28, 28, 1]) # Camada de entrada que são as imagens

    # Criandando a primeira camada de covolunção da rede neural 
    # Recebe [batch_size(quantidade de imagens), 28, 28, 1]
    # Retorna [batch_size, 28, 28, 32]
    convolucao1 = tf.layers.conv2d(inputs = entrada, filters = 32, kernel_size = [5, 5], activation = tf.nn.relu,
                                  padding = 'same')
    
    # Criando a primeira camada de max pooling
    # Recebe [batch_size(quantidade de imagens), 28, 28, 32]
    # Retorna [batch_size, 14, 14, 32]
    pooling1 = tf.layers.max_pooling2d(inputs = covolucao1, pool_size = [2, 2], strides = 2)

    # Criandando a segunda camada de covolunção da rede neural 
    # Recebe [batch_size(quantidade de imagens), 14, 14, 32]
    # Retorna [batch_size, 14, 14, 64]
    convolucao2 = tf.layers.conv2d(inputs = pooling1, filters = 64, kernel_size = [5, 5], activation = tf.nn.relu,
                                   padding = 'same')
    
    # Criando a segunda camada de max pooling
    # Recebe [batch_size(quantidade de imagens), 14, 14, 64]
    # Retorna [batch_size, 7, 7, 64]
    pooling2 = tf.layers.max_pooling2d(input = convolucao2, pool_size = [2, 2], strides = 2)
    
    # Aplicando o Flattening e convertendo os dados no formato matriz para um vetor
    # Recebe [batch_size(quantidade de imagens), 7, 7, 64]
    # Retorna [batch_size, 3136]
    flattening = tf.reshape(pooling2, [-1, 7 * 7 * 64])
    
    
    # A rede neural densa terá:
    # 3136 neurônio de entra -> 1024 camada escondida -> 10 camada de saída
    # Recebe [batch_size(quantidade de imagens), 3136]
    # Retorna [batch_size, 1024]
    densa = tf.layers.dense(inputs = flattening, units = 1024, activation = tf.nn.relu)

    # Melhorando a performace da rede neural com a técnica DROPOUT que zera alguns valores das entradas
    dropout = tf.layers.dropout(input = densa, rate = 0.2)
    
    # Criando a camada de saída
    # Recebe [batch_size(quantidade de imagens), 1024]
    # Retorna [batch_size, 10]
    saida = tf.layers.dense(inputs = dropout, units = 10)
    
    
# Criando um classificador
classificador = tf.estimator.Estimator(model_fn = cria_rede)

# Criando uma função para o treinamento
funcao_treinamento = tf.estimator.inputs.numpy_input_fn(x = {'x': x_treinamento}, y_treinamento,
                                                        batch_size = 128, num_epochs = None, shuffle = True)
classificador.train(input_fn = funcao_treinamento, steps = 200)