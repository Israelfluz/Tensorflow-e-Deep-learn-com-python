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

np.random.seed(0)
np.random.seed(2)

# Importando a biblioteca do tensorflow
import tensorflow as tf

b0 = tf.Variable(0.54)
b1 = tf.Variable(0.71)


