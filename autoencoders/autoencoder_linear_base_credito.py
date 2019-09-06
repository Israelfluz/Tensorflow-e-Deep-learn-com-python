import pandas as pd

# Carregando a base de dados
base = pd.read_csv('credit-data.csv')
base.head()

base.shape

# Apagandoa a coluna i#clientid pois seu conteúdo não ajudará nos algoritmos
base = base.drop('i#clientid', axis = 1)
base.head()


base = base.dropna() # Apagando os registros que não tiveram seus valores preenchidos (NAN)
base.shape

# Realizando o escalonamento e padroizando(standardscaler)
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
base[['income', 'age', 'loan']] = scaler_x.fit_transform(base[['income', 'age', 'loan']])
base.head()

# Variável que recebe os atributos previsores. 
x = base.drop('c#default', axis = 1) # c#default retira a classe (coluna) que mostra quem vai pagar e quem não

# Variável que recebe o C#default que o conjunto de 0 e 1
y = base['c#default']

x.head()

## Construção do autoencoder a partir da definição das variáveis

# Variável onde defino o número de camadas de entrada, aqui como tem três atributos...
neuronios_entrada = 3

# Variável onde defino a camada oculta e é aqui que a redução acontece 3entra 2oculta(era para ter três mas foi reduzido para 2) 3saída
neuronios_oculta = 2

# Variável onde defino a camada de saída
neuronios_saida = neuronios_entrada

import tensorflow as tf

# Criando o primeiro placeholder
xph = tf.placeholder(tf.float32, shape = [None, neuronios_entrada])

## Construindo a estrutura da rede neural para o autoencoder 

from tensorflow.contrib.layers import fully_connected
camada_oculta = fully_connected(inputs = xph, num_outputs = neuronios_oculta, activation_fn = None)
camada_saida = fully_connected(inputs = neuronios_oculta, num_outputs = neuronios_saida)  


# Criando a função para calcular o erro
erro = tf.losses.mean_squared_error(labels = xph, prediction = camada_saida)
otimizador = tf.train.AdamOptimizer(0.01)

# Criando variável para o treinamento
treinamento = otimizador.minimize(erro)


# Efetivando o treinamento
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoca in range(1000):
        custo, _ = sess.run([erro, treinamento], feed_dict = {xph: x})
        if epoca % 100 == 0:
            print('erro: ' + str(custo))