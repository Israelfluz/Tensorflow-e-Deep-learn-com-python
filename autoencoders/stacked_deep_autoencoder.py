from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist/', one_hot = True)
X = mnist.train.images

import matplotlib.pyplot as plt
plt.imshow(X[1].reshape(28,28))

# 784 -> 128 -> 64 -> 128 -> 784

# encoder
neuronios_entrada = 784
neuronios_oculta1 = 128

# dado/imagem codificada
neuronios_oculta2 = 64

# decoder
neuronios_oculta3 = neuronios_oculta1
neuronios_saida = neuronios_entrada


import tensorflow as tf
tf.reset_default_graph()


xph = tf.placeholder(tf.float32, [None, neuronios_entrada])



# Xavier: sigmoid
# He: relu
inicializador = tf.variance_scaling_initializer()


# 784 -> 128 -> 64 -> 128 -> 784
W = {'encoder_oculta1': tf.Variable(inicializador([neuronios_entrada, neuronios_oculta1])),
     'encoder_oculta2': tf.Variable(inicializador([neuronios_oculta1, neuronios_oculta2])),
     'decoder_oculta3': tf.Variable(inicializador([neuronios_oculta2, neuronios_oculta3])),
     'decoder_saida': tf.Variable(inicializador([neuronios_oculta3, neuronios_saida]))
}



b = {'encoder_oculta1': tf.Variable(inicializador([neuronios_oculta1])),
     'encoder_oculta2': tf.Variable(inicializador([neuronios_oculta2])),
     'decoder_oculta3': tf.Variable(inicializador([neuronios_oculta3])),
     'decoder_saida': tf.Variable(inicializador([neuronios_saida]))
}



camada_oculta1 = tf.nn.relu(tf.add(tf.matmul(xph, W['encoder_oculta1']), b['encoder_oculta1']))
camada_oculta2 = tf.nn.relu(tf.add(tf.matmul(camada_oculta1, W['encoder_oculta2']), b['encoder_oculta2']))
camada_oculta3 = tf.nn.relu(tf.add(tf.matmul(camada_oculta2, W['decoder_oculta3']), b['decoder_oculta3']))
camada_saida = tf.nn.relu(tf.add(tf.matmul(camada_oculta3, W['decoder_saida']), b['decoder_saida']))



erro = tf.losses.mean_squared_error(xph, camada_saida)
otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)
treinamento = otimizador.minimize(erro)
batch_size = 128



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoca in range(50):
        numero_batches = mnist.train.num_examples // batch_size
        for i in range(numero_batches):
            X_batch, _ = mnist.train.next_batch(batch_size)
            custo, _ = sess.run([erro, treinamento], feed_dict = {xph: X_batch})
        print('época: ' + str(epoca + 1) + ' erro: ' + str(custo))
    
    imagens_codificadas = sess.run(camada_oculta2, feed_dict = {xph: X})
    imagens_decodificadas = sess.run(camada_saida, feed_dict = {xph: X})
    
    
    
imagens_codificadas.shape


imagens_codificadas[0]


imagens_decodificadas.shape


imagens_decodificadas[0]

import numpy as np
numero_imagens = 5
imagens_teste = np.random.randint(X.shape[0], size = numero_imagens)
imagens_teste


plt.figure(figsize = (18, 18))
for i, indice_imagem in enumerate(imagens_teste):
    #print(i)
    #print(indice_imagem)
    eixo = plt.subplot(10, 5, i + 1)
    plt.imshow(X[indice_imagem].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())
    
    eixo = plt.subplot(10, 5, i + 1 + numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(8, 8))
    plt.xticks(())
    plt.yticks(())
    
    eixo = plt.subplot(10, 5, i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())