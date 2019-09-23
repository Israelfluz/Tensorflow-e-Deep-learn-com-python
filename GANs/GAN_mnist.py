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


# Criando placeholder para a função discriminador 
imagens_reais_ph = tf.placeholder(tf.float32, [None, 784])


# Criando a função discriminador que irá retonar os valores que são ou não reais
def discriminador(x, reuse = None):
    with tf.variable_scope('discriminador', reuse = tf.AUTO_REUSE):
        # Criando a estrutura da rede neural que vai receber imagem com 784 pixes ->
        #128 neuronios camada_oculta1 -> 128 neuronios camada_oculta2 -> 1 neuronio camada_saida
        camada_oculta1 = tf.nn.relu(tf.layers.dense(inputs = x, units = 128))
        camada_oculta2 = tf.nn.relu(tf.layers.dense(inputs = camada_oculta1, units = 128))
        logits = tf.layers.dense(camada_oculta2, units = 1) # logits significa as previsões do modelo que não estão normalizadas
        return logits 

# Definindo as formulas para o treinamento
logits_imagens_reais = discriminador(imagens_reais_ph)
logits_imagens_ruido = discriminador(gerador(ruido_ph), reuse = True)


# Calculo do erro do discriminador
erro_discriminador_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_imagens_reais,
                                                                                 labels = tf.ones_like(logits_imagens_reias) * (0.9)))
erro_discriminador_ruido = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_imagens_ruido,
                                                                                  labels = tf.zeros_like(logits_imagens_ruido)))
erro_discriminador = erro_discriminador_real + erro_discriminador_ruido


# Calculo do erro do gerador
erro_gerador = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_imagens_ruido,
                                                                      labels = tf.ones_like(logits_imagens_ruido))



# Retornando as variáveis passíveis de otmização
variaveis = tf.trainable_variables()
variaveis


# Variáveis do discriminador
variaveis_discriminador = [v for v in variavies if 'discriminador' in v.name]
print([v.name for v in variaveis_discriminador])


# Variáveis do gerador
variaveis_gerador = [v for v in variavies if 'gerador' in v.name]
print([v.name for v in variaveis_gerador])



# Definindo os otimizadores
treinamento_discriminador = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(erro_discriminador,
                                                                                  var_list = variaveis_discriminador)

treinamento_gerador = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(erro_gerador,
                                                                                  var_list = variaveis_gerador)
    
# Definido o gerador, faremos um primeiro teste 
batch_size = 100
amostras_teste = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #ruido_teste = np.random.uniform(-1, 1, size = (1, 100))
    #amostra = sess.run(gerador(ruido_ph, True), feed_dict = {ruido_ph: ruido_teste})
    
    # Visualizando o resultado construido logo abaixo agora dentro da sessão
    #batch = mnist.train.next_batch(100)
    #imagens_batch = batch[0].reshape((100, 784))
    #imagens_batch = imagens_batch * 2 - 1
    #r = sess.run(discriminador(imagens_reais_ph, True), feed_dict = {imagens_reais_ph: imagens_batch})
    
    # Para ter os valores de probabilidades
    #r2 = sess.run(tf.nn.sigmoid(r))
    
    #ex = tf.constant([[1, 2], [3, 4]])
    #print(sess.run(tf.ones_like(ex)))
    
    for epoca in range(500):
        numero_batches = mnist.train.num_examples // batch_size
        for i in range(numero_batches):
            batch = mnist.train.next_batch(batch_size)
            imagens_batch = batch[0].reshape((100, 784))
            imagens_batch = imagens_batch * 2 - 1
            
            batch_ruido = np.random.uniform(-1, 1, size = (batch_size,100))
            
            _, custod = sess.run([treinamento_discriminador, erro_discriminador],
                                 feed_dict = {imagens_reais_ph: imagens_batch, ruido_ph: batch_ruido})
            
            _, custog = sess.run([treinamento_gerador, erro_gerador], feed_dict = {ruido_ph: batch_ruido})
            
            print('epoca: ' + str(epoca + 1) + 'erro D: ' + str(custod) + 'erro G: ' + str(custog))

            ruido_teste = np.random.uniform(-1, 1, size(1, 100))
            imagem_gerada = sess.run(gerador(ruido_ph, reuse = True), feed_dict = {ruido_ph: ruido_teste})
            amostras_teste.append(imagem_gerada)


# Visualizando o que foi gerado na amostra
amostra.shape


# Visualizando a imagem gerada 
plt.imshow(amostra.reshape(28, 28))


# Visualizadno o resultado do def discriminador fora da sessão
batch = mnist.train.next_batch(100)
batch[0].shape
imagens_batch = batch[0].reshape((100, 784))


# Fazendo uma transformação que ajudará no retorno da função de tangente hiperbólica
imagens_batch = imagens_batch * 2 - 1
imagens_batch[0]

# Visualizando os valores de retorno dentro da variável r
r


# Visualizando os valores de propabilidade dentro da variável r2
r2

# Visualizando o gráfico de qual imagem foi gerada
plt.imshow(amostras_teste[0].reshape(28, 28))
