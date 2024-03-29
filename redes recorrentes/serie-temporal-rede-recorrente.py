# Importação da biblioteca pandas para manipulação, leitura e visualização de dados
import pandas as pd

# Carregamento da base de dados para análise
base = pd.read_csv('petr4.csv')
base = base.dropna() # Apagando os registros que não foram informados NAN

# A previsão será realizada no atributo Open
base = base.iloc[:,1].values # O valeus é utilizada para transformar os dados em um numpy

# Visualização do gráfico da série temporal
import matplotlib.pyplot as plt
#%matplotlib inline
plt.plot(base)


# Variável que fará a previsão dos 30 valores futuros utilizando 30 valores passados
periodos = 30
previsao_futura = 1 # Está variável indica os horizontes das previsões

# Base de dados que será utilizada para o treinamento
X = base[0:(len(base) - (len(base) % periodos))]
X_batches = X.reshape(-1, periodos, 1) # Divisão da base de dados para melhor performance da previsão


y = base[1:(len(base) - (len(base) % periodos)) + previsao_futura]
y_batches = y.reshape(-1, periodos, 1)


# Base de dados que será utilizada para teste e avaliação
X_teste = base[-(periodos + previsao_futura):]
X_teste = X_teste[:periodos]
X_teste = X_teste.reshape(-1, periodos, 1)

y_teste = base[-(periodos):] # Aqui vamos encontrar as respostas
y_teste = y_teste.reshape(-1, periodos, 1)


import tensorflow as tf
tf.reset_default_graph() # Este comando serve para que nada fique em memória

# Criando algumas variáveis
entradas = 1   # recebe o valor 1 porque temos um atributo previsor (open)
neuronios_oculta = 100
neuronios_saida = 1

# Criando os placeholders para receber os dados
xph = tf.placeholder(tf.float32, [None, periodos, entradas])
yph = tf.placeholder(tf.float32, [None, periodos, neuronios_saida])

#celula = tf.contrib.rnn.BasicRNNCell(num_units = neuronios_oculta, activation = tf.nn.relu)
#celula = tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)
# Mapeamento para a camada saída
#celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)

## Definindo número de camadas ou celulas.
# Camada oculta 
def cria_uma_celula():
    return tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)

# Camada de saída
def cria_varias_celulas():
    celulas =  tf.nn.rnn_cell.MultiRNNCell([cria_uma_celula() for i in range(4)])
    return tf.contrib.rnn.DropoutWrapper(celulas, output_keep_prob = 0.1)

celula = cria_varias_celulas()
# camada saída
celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)

# Criando a saída ou as resposta para realizar o calculo do erro
saida_rnn, _ = tf.nn.dynamic_rnn(celula, xph, dtype = tf.float32)

# Buscando o valor do erro
erro = tf.losses.mean_squared_error(labels = yph, predictions = saida_rnn)
otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)
treinamento = otimizador.minimize(erro)

# Criando uma sessão 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoca in range(1000):
        _, custo = sess.run([treinamento, erro], feed_dict = {xph: X_batches, yph: y_batches})
        if epoca % 100 == 0:
            print(epoca + 1, ' erro: ', custo)
    
    # Previsão feita logo após o treinamento
    previsoes = sess.run(saida_rnn, feed_dict = {xph: X_teste})
    
import numpy as np
y_teste.shape
y_teste2 = np.ravel(y_teste)

previsoes2 = np.ravel(previsoes)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste2, previsoes2)

# Visualizando gráficos
plt.plot(y_teste2, '*', markersize = 10, label = 'Valor real')
plt.plot(previsoes2, 'o', label = 'Previsões')
plt.legend()

plt.plot(y_teste2, label = 'Valor real')
plt.plot(y_teste2, 'w*', markersize = 10, color = 'red')
plt.plot(previsoes2, label = 'Previsões')
plt.legend()


