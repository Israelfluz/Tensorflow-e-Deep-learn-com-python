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

# Realizando a conversão do tipo de dado 
y_treinamento = np.asarray(y_treinamento, dtype = np.int32)
y_teste = np.asarray(y_teste, dtype = np.int32)

# Visualizando as imagens da base de dados
import matplotlib.pyplot as plt
plt.imshow(x_treinamento[2].reshape((28, 28)), cmap = 'gray') # cmap = 'gray' deixa o fundo preto e o núemro cinza

# Visualizando a classe
plt.title('Classe: ' + str(y_treinamento[2]))

import tensorflow as tf

# Criando a função cria_rede que se encontra nos parâmetros do classificador Obs: Essa função faz parte do treinamento
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
    pooling1 = tf.layers.max_pooling2d(inputs = convolucao1, pool_size = [2, 2], strides = 2)

    # Criandando a segunda camada de covolunção da rede neural 
    # Recebe [batch_size(quantidade de imagens), 14, 14, 32]
    # Retorna [batch_size, 14, 14, 64]
    convolucao2 = tf.layers.conv2d(inputs = pooling1, filters = 64, kernel_size = [5, 5], activation = tf.nn.relu,
                                   padding = 'same')
    
    # Criando a segunda camada de max pooling
    # Recebe [batch_size(quantidade de imagens), 14, 14, 64]
    # Retorna [batch_size, 7, 7, 64]
    pooling2 = tf.layers.max_pooling2d(inputs = convolucao2, pool_size = [2, 2], strides = 2)
    
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
    dropout = tf.layers.dropout(inputs = densa, rate = 0.2, 
                                training = mode == tf.estimator.ModeKeys.TRAIN) # Obs: Só executamos esse dropout quando estamos em modo de treinamento
    
    # Criando a camada de saída
    # Recebe [batch_size(quantidade de imagens), 1024]
    # Retorna [batch_size, 10]
    saida = tf.layers.dense(inputs = dropout, units = 10)
    previsoes = tf.argmax(saida, axis = 1)
    
    # Modo de previsão
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = previsoes)
    
    # Fórmula para o calculo do erro
    erro = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = saida)
    
    # Modo de treinamento
    if mode == tf.estimator.ModeKeys.TRAIN:
        otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)
        treinamento = otimizador.minimize(erro, global_step = tf.train.get_global_step())
    # Retornando os resultados do treinamento
        return tf.estimator.EstimatorSpec(mode = mode, loss = erro, train_op = treinamento)
    
    # Modo de avaliação
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics_ops = {'accuracy': tf.metrics.accuracy(labels = labels, predictions = previsoes)} # accurary indica a taixa e acerto
         # Retornando os resultados da avaliação
        return tf.estimator.EstimatorSpec(mode = mode, loss = erro, eval_metric_ops = eval_metrics_ops)
        
# Criando um classificador
classificador = tf.estimator.Estimator(model_fn = cria_rede)

# Criando uma função para o treinamento
funcao_treinamento = tf.estimator.inputs.numpy_input_fn(x = {'x': x_treinamento}, y = y_treinamento,
                                                        batch_size = 128, num_epochs = None, shuffle = True)
classificador.train(input_fn = funcao_treinamento, steps = 200)

# Criando a função para a avaliação ou teste
funcao_teste = tf.estimator.inputs.numpy_input_fn(x = {'x': x_teste}, y = y_teste, num_epochs = 1, shuffle = False)
resultados = classificador.evaluate(input_fn = funcao_teste)
resultados

# Previsão
x_imagem_teste = x_teste[0]
x_imagem_teste.shape

x_imagem_teste = x_imagem_teste.reshape(1, -1)
x_imagem_teste.shape

funcao_previsao = tf.estimator.inputs.numpy_input_fn(x = {'x': x_imagem_teste}, shuffle = False)
pred = list(classificador.predict(input_fn = funcao_previsao))
pred


plt.imshow(x_imagem_teste.reshape((28, 28)), cmap = 'gray')
plt.title('Classe prevista: ' + str(pred[0]))
