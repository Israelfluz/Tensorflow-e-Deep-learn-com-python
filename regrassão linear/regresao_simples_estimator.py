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


x

y

# Importando a biblioteca do tensorflow
import tensorflow as tf

# Criando lista com colunas
colunas = [tf.feature_column.numeric_column('x', shape = [1])]

colunas

# Criando o regressor
regressor = tf.estimator.LinearRegressor(feature_columns= colunas)

# Fazendo a divisão da base de dados entre e Teste e treinamento
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.3)

# Verificando os Shapes
x_treinamento.shape

y_treinamento.shape

x_teste.shape

y_teste.shape

# Criando e definindo funções para o treinamento
funcao_treinamento = tf.estimator.inputs.numpy_input_fn({'x': x_treinamento}, y_treinamento, batch_size = 32, num_epochs = None, shuffle = True)

# Criando e definindo funções para o teste
funcao_teste = tf.estimator.inputs.numpy_input_fn({'x': x_teste}, y_teste, batch_size = 32, num_epochs = 1000, shuffle = False) 

# Realizando o treinamento do regressor
regressor.train(input_fn = funcao_treinamento, steps = 10000)

# Métricas na base de dados treinamento
metricas_treinamento = regressor.evaluate(input_fn = funcao_treinamento,steps = 10000)

# Métricas na base de dados treinamento
metricas_teste = regressor.evaluate(input_fn = funcao_teste,steps = 10000)

# Fazendo um comparativo
metricas_treinamento

metricas_teste

# Realizando as previsões 
import numpy as np

# obs: O objetivo é saber o preço das casas tendo como base a metragem
novas_casas = np.array([[800], [900], [1000]])

# Escalonamento
novas_casas = scaler_x.transform(novas_casas)
novas_casas

# Criando função após o escolonamento
funcao_previsao = tf.estimator.inputs.numpy_input_fn({'x': novas_casas}, shuffle = False)

# Realizando as previsões depois do escalonamento
previsoes = regressor.predict(input_fn = funcao_previsao)

# Mostrando os resultados 
list(previsoes)

# Percorrento o Generator (de previsoes) para saber o resultado em dolares
for p in regressor.predict(input_fn = funcao_previsao):
   # print(p['predictions']) # printou os valores escalonados é preciso fazer a inversão
    print(scaler_y.inverse_transform(p['predictions'])) # Desse forma tenho o valor em moeda