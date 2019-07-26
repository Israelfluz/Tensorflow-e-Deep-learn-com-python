# Importação da biblioteca numpy
import numpy as np

# Definindo variável X com a idade das pessoos com np array
x = np.array([[18],[23],[28],[33],[38], [43],[48],[53],[58],[63]])

# Definindo a variável Y que recebe o valor do plano de saúde com np array
y = np.array([[871],[1132],[1042],[1356],[1488],[1638],[1569],[1754],[1866],[1900]])

# Demonstranso os valores da variável X
x

# Demonstranso os valores da variável Y
y

# No tensorflow é importante fazer o escalonamento dos valores
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()

# Fazendo a transfomação efetivamente da variável X
x = scaler_x.fit_transform(x)

# Visualizando o X escalonado
x

#Escalonando a variável Y
scaler_y = StandardScaler()

# Fazendo a transfomação efetivamente da variável Y
y = scaler_y.fit_transform(y)

# Visualizando o y escalonado

# Gráfico que visualiza a disposição dos valores
import matplotlib.pyplot as plt
plt.scatter(x, y)

# Fórmula da regrassão linear simples (manualmente)
# ================ y=b0+b1*x =====================
import numpy as np
np.random.seed(0)
np.random.seed(2)

# Importando a biblioteca do tensorflow
import tensorflow as tf

b0 = tf.Variable(0.54)
b1 = tf.Variable(0.71)

# Calculando o erro (criando as fórmulas)
erro = tf.losses.mean_squared_error(y, (b0 + b1 * x))
otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
treinamento = otimizador.minimize(erro)

# Inicializando as variáveis
init = tf.global_variables_initializer()

# Realizando efetivamente o treinamento dentro de uma sessão
with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(b0))
    #print(sess.run(b1))
    
# Definido as épocas
    for i in range(1000):
        sess.run(treinamento)
    b0_final, b1_final = sess.run([b0, b1])

# Visualizando o valor do b0 final e b1_final
b0_final
b1_final

# Classificações
previsoes = b0_final + b1_final * x
previsoes

type(previsoes)

# Criando graficos
plt.plot(x, previsoes, color = 'red')
plt.plot(x, y, 'o')

# Realizando uma previsão do valor do plano de saúde para uma pessoa co 40 anos
scaler_x.transform([[40]])
previsao = scaler_y.inverse_transform(b0_final + b1_final * scaler_x.transform([[40]]))
previsao

# Calculo do erro
y1 = scaler_y.inverse_transform(y)
y1

# Desescalonando
previsoes1 = scaler_y.inverse_transform(previsoes)
previsoes1

# Observando a eficiência da regressão com métricas (mean_absolute_error) e (mean_squared_error)
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y1, previsoes1)
mse = mean_squared_error(y1, previsoes1)

# Visualizando o valor de mean_absolute_error
mae

# Visualizando o valor de mean_squared_error
mse
