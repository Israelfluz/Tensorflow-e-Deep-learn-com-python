# ======= Classificação binária com XOR que retonar  valores entre 0 e 1 ========
# ======= Essa implemanteção será feita utilizando a low-level do Tensorflow ======
# ======= Esse é um problema não linearmente separável, ou seja, não da para 
# trassar uma linha separando a classe 0 e a classe 1 ==========


# Importação biblioteca tensorflow
import tensorflow as tf

# Importação da biblioteca para computação científica
import numpy as np

# Variável X de entrada que armazena os atributos previsores do tipo numpy array no formato de matriz
x = np.array([[0,0], [0,1], [1,0], [1,1]])

x 

# Variável Y receberá as respostas que é do tipo numpy array no formato de matriz
y = np.array([[1], [0], [0], [1]])

y

# Definindo algumas variáveis para indicar a estrutura da rede neural
neuronios_entrada = 2
neuronios_oculta = 3
neuronios_saida = 1

# Crianda a variável W que representa os pesos no formato de dicionário
w = {'oculta': tf.Variable(tf.random_normal([neuronios_entrada, neuronios_oculta]), name = 'w_oculta'),
     'saida': tf.Variable(tf.random_normal([neuronios_oculta, neuronios_saida]), name = 'w_saida')}

# Verificando o tipo da variável w
type(w)
