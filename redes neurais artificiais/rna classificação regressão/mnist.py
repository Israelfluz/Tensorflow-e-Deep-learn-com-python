# ====== Construção de uma rede neural para fazer a classificação dos dígitos de 0 até 9 =====

# Importando para o tensorflow
from tensorflow.examples.tutorials.mnist import input_data 


mnist = input_data.read_data_sets('mnist/', one_hot = True)


# Carregando os dados da pasta mnist/
x_treinamento = mnist.train.images
x_treinamento.shape
x_teste = mnist.test.images

y_treinamento = mnist.train.labels
y_treinamento[0]
y_teste = mnist.test.labels

 # Visualizando as imagens da base de dados
import matplotlib.pyplot as plt
import numpy as np
plt.imshow(x_treinamento[102].reshape((28, 28)), cmap = 'gray') # cmap = 'gray' deixa o fundo preto e o núemro cinza

# Visualizando a classe
plt.title('Classe: ' + str(np.argmax(y_treinamento[102])))

# Definindo o tamanho do batch
x_batch, y_batch = mnist.train.next_batch(64) # Esse número pode alterar
x_batch.shape

#          ======= Construindo a rede neural ========

# Passando a quantidade de atributos previsores 
neuronios_entrada = x_treinamento.shape[1]
neuronios_entrada

# Definindo os neurônios da camada  oculta, aqui teremos três
neuronios_oculta1 = int((x_treinamento.shape[1] + y_treinamento.shape[1]) / 2)
neuronios_oculta1

neuronios_oculta2 = neuronios_oculta1

neuronios_oculta3 = neuronios_oculta1

# Definindo o neurônio da camada de saída
neuronios_saida = y_treinamento.shape[1]
neuronios_saida

# 784 neurônios na camada de entrada que estaram ligados -> 397 neurônios da primeira camada oculta,
# estaram ligados -> 397 neurônios da segunda camada oculta, que estaram ligas 397 neurônios da terceira camada oculta,
# que estaram ligados -> 10 da camada de saída.

# Construindo a estrutura acima citada
import tensorflow as tf

# Criando a variável W que representa os pesos no formato de dicionário
w = {'oculta1': tf.Variable(tf.random_normal([neuronios_entrada, neuronios_oculta1])),
     'oculta2': tf.Variable(tf.random_normal([neuronios_oculta1, neuronios_oculta2])),
     'oculta3': tf.Variable(tf.random_normal([neuronios_oculta2, neuronios_oculta3])),
     'saida': tf.Variable(tf.random_normal([neuronios_oculta3, neuronios_saida]))
}

# Adcionando a camada BIAS com os seus pesos
b = {'oculta1': tf.Variable(tf.random_normal([neuronios_oculta1])),
     'oculta2': tf.Variable(tf.random_normal([neuronios_oculta2])),
     'oculta3': tf.Variable(tf.random_normal([neuronios_oculta3])),
     'saida': tf.Variable(tf.random_normal([neuronios_saida]))
}

