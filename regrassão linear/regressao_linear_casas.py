# importação da biblioteca
import pandas as pd

# Importação da base de dados 
base = pd.read_csv('house-prices.csv')

# Variável X que armazena a metragem quadrada
x = base.iloc[:, 5].values
x = x.reshape(-1, 1)
# Visualizando o shape
x.shape

# Variável Y que armazena o valor das casas
y = base.iloc[:, 2:3].values

# Realizando escalonamento dos valores
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()

# Fazendo a transfomação efetivamente da variável X
x = scaler_x.fit_transform(x)

#Escalonando a variável Y
scaler_y = StandardScaler()

# Fazendo a transfomação efetivamente da variável Y
y = scaler_y.fit_transform(y)

# Gráfico que visualiza a disposição dos valores
import matplotlib.pyplot as plt
plt.scatter(x, y)

# Fórmula da regrassão linear simples (manualmente)
# ================ y=b0+b1*x =====================


# Importando a biblioteca do numpy
import numpy as np

np.random.seed(1)
np.random.seed(2)

# Importando a biblioteca do tensorflow
import tensorflow as tf

b0 = tf.Variable(0.41)
b1 = tf.Variable(0.72)

# Criando um placeholder
batch_size = 32
xph = tf.placeholder(tf.float32, [batch_size, 1])
yph = tf.placeholder(tf.float32, [batch_size, 1])

# Criando modelos para as previsoes
y_modelo = b0 + b1 * xph

# Calculando o erro (criando as fórmulas)
erro = tf.losses.mean_squared_error(yph, y_modelo)
otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
treinamento = otimizador.minimize(erro)

# Inicializando as variáveis
init = tf.global_variables_initializer()

# Realizando efetivamente o treinamento dentro de uma sessão
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000): # Definido as épocas
        indices = np.random.randint(len(x), size = batch_size)
        feed = {xph: x[indices], yph: y[indices]}
        sess.run(treinamento, feed_dict = feed)
    b0_final, b1_final = sess.run([b0, b1])
   
 # Visualizando o valor do b0 final e b1_final
b0_final
b1_final

# previsões
previsoes = b0_final + b1_final * x
previsoes

# Criando graficos
plt.plot(x, previsoes, color = 'red')
plt.plot(x, y, 'o')

# Desescalonando
y1 = scaler_y.inverse_transform(y)
y1

previsoes1 = scaler_y.inverse_transform(previsoes)
previsoes1

# Observando a eficiência da regressão com métricas (mean_absolute_error) e (mean_squared_error)
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y1, previsoes1)
mse = mean_squared_error(y1, previsoes1)
