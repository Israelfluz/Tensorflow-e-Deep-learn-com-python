# Importação da bibliotéca e carregamento da base de dados MNIST
import tensorflow as tf
tf.reset_default_graph()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist/', one_hot = True)

# Importação da biblioteca e visualização 
import matplotlib.pyplot as plt
plt.imshow(mnist.train.images[0].reshape(28, 28), cmap = 'Greys')

# Visualizando a primeira imagens da base de dados do mnist
mnist.train.images[0]

# O gerador recebe número aleatórios, esses números pixes que formam uma imagem
import numpy as np
imagem1 = np.arange(0, 784).reshape(28, 28)
plt.imshow(imagem1)

# Gerando números aleatórios para forma uma imagem
imagem2 = np.random.normal(size = 784).reshape(28, 28)
plt.imshow(imagem2)

# Implemantando uma rede neural
# Criando um placeholder que vai gerar os números aleatórios
ruido_ph = tf.placeholder(tf.float32, [None, 100])


# Criando a função gerador
def gerador(ruido, reuse = None):
    with tf.variable_scope('gerador', reuse = tf.AUTO_REUSE):
        # Criando a estrutura da rede neural que terá 100 neuronios camada_entrada ->
        # 128 neuronios camada_oculta1 -> 128 neuronios camada_oculta2 -> 784 neuronios camada_saida
        camada_oculta1 = tf.nn.relu(tf.layers.dense(inputs = ruido, units = 128))
        camada_oculta2 = tf.nn.relu(tf.layers.dense(inputs = camada_oculta1, units = 128))
        camada_saida = tf.layers.dense(inputs = camada_oculta2, units = 784, activation = tf.nn.tanh)
        return camada_saida

# Definido o gerador faremos uma primeiro teste 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    ruido_teste = np.random.uniform(-1, 1, size = (1, 100))
    amostra = sess.run(gerador(ruido_ph, True), feed_dict = {ruido_ph: ruido_teste})
    
# Visualizando o que foi gerado na amostra
amostra
