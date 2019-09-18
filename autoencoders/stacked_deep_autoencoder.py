# Carregamento da base de dados MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist/', one_hot = True)
x = mnist.train.images  # Tem 55.000 imagens para o treinamento

import matplotlib.pyplot as plt
plt.imshow(x[1].reshape(28,28)) # Mostrando uma imagem da base de dados

# Contruimos uma estrutura com 784 neurônios camada de entrada -> 128 neurônios primeira camada oculta
# -> que seram transformados em 64 neurônios camada -> processo de decoder voltando para 128 -> e volta para 784

# encoder (Até neuronios_oculta2 é a parte do encoder)
neuronios_entrada = 784
neuronios_oculta1 = 128

# dado/imagem codificada
neuronios_oculta2 = 64

# decoder
neuronios_oculta3 = neuronios_oculta1
neuronios_saida = neuronios_entrada


import tensorflow as tf
tf.reset_default_graph()


# Criando o placeholder
xph = tf.placeholder(tf.float32, [None, neuronios_entrada])



# Xavier: sigmoid
# He: relu
inicializador = tf.variance_scaling_initializer() # Deixando os pesos na mesma escala


# 784 -> 128 -> 64 -> 128 -> 784 definindo os pesos manualmente, na estru
W = {'encoder_oculta1': tf.Variable(inicializador([neuronios_entrada, neuronios_oculta1])),
     'encoder_oculta2': tf.Variable(inicializador([neuronios_oculta1, neuronios_oculta2])),
     'decoder_oculta3': tf.Variable(inicializador([neuronios_oculta2, neuronios_oculta3])),
     'decoder_saida': tf.Variable(inicializador([neuronios_oculta3, neuronios_saida]))
}


# Definindo a unidade de baias
b = {'encoder_oculta1': tf.Variable(inicializador([neuronios_oculta1])),
     'encoder_oculta2': tf.Variable(inicializador([neuronios_oculta2])),
     'decoder_oculta3': tf.Variable(inicializador([neuronios_oculta3])),
     'decoder_saida': tf.Variable(inicializador([neuronios_saida]))
}


# Criando as camadas fazendo a multiplicação das matrizes e aplicação da função de ativação
camada_oculta1 = tf.nn.relu(tf.add(tf.matmul(xph, W['encoder_oculta1']), b['encoder_oculta1']))
camada_oculta2 = tf.nn.relu(tf.add(tf.matmul(camada_oculta1, W['encoder_oculta2']), b['encoder_oculta2']))
camada_oculta3 = tf.nn.relu(tf.add(tf.matmul(camada_oculta2, W['decoder_oculta3']), b['decoder_oculta3']))
camada_saida = tf.nn.relu(tf.add(tf.matmul(camada_oculta3, W['decoder_saida']), b['decoder_saida']))


# Função para o calculo do erro
erro = tf.losses.mean_squared_error(xph, camada_saida)

# Criando o atmizador
otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)

# Criando e treinando
treinamento = otimizador.minimize(erro)
batch_size = 128  # O treinamento será feito de 128 em 128


# Processo de treinamento, criando uma sessão para efetivar
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # Inicializando as variáveis
    for epoca in range(50):  # Rodando as épocas, neste caso 50x
        numero_batches = mnist.train.num_examples // batch_size  # treinamento por batches de 128 em 128
        for i in range(numero_batches): # For interno 
            x_batch, _ = mnist.train.next_batch(batch_size)
            custo, _ = sess.run([erro, treinamento], feed_dict = {xph: x_batch})
        print('época: ' + str(epoca + 1) + ' erro: ' + str(custo))
    
    # Buscando as imagens codificadas logo após o For que fez o treinamento
    imagens_codificadas = sess.run(camada_oculta2, feed_dict = {xph: x})
    
    # Buscando as imagens decodificadas
    imagens_decodificadas = sess.run(camada_saida, feed_dict = {xph: x})
    
    
# Visualizando o shape das imagens codificadas    
imagens_codificadas.shape

# Buscando uma imagem
imagens_codificadas[0]

# Visualizando o shape das imagens decodificadas
imagens_decodificadas.shape

# Buscando uma imagem
imagens_decodificadas[0]

## Agora vamos selecionar alguams imagens de amostra e verificar qual é o desenho da imagem
## original, qual é o desenho da imagem codificada e qual é o desenho da imagem decodificada
## para ver se está conseguindo fazer a reconstrução correta 


# 
import numpy as np
numero_imagens = 5 # Trabalhando apenas com cinco amostras de imagens
imagens_teste = np.random.randint(x.shape[0], size = numero_imagens)
imagens_teste


plt.figure(figsize = (18, 18))
for i, indice_imagem in enumerate(imagens_teste):
    #print(i)
    #print(indice_imagem)
    eixo = plt.subplot(10, 5, i + 1)
    plt.imshow(x[indice_imagem].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())
    
    # Visualizando ou gerando a imagem codificada
    eixo = plt.subplot(10, 5, i + 1 + numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(8, 8))
    plt.xticks(())
    plt.yticks(())
    
    # Visualizando ou gerando a imagem decodificada
    eixo = plt.subplot(10, 5, i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())