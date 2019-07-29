# importação da biblioteca
import pandas as pd

# Importação da base de dados 
base = pd.read_csv('house-prices.csv')
base.head()

# Visualizando o nome das colunas
base.columns

# Criando uma variável que armazena as strings com os nomes das colunas
colunas_usadas = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']

# Visualizando a variável colunas
colunas_usadas

# Carregando novamente o base de dados apenas com as colunas desejadas
base = pd.read_csv('house-prices.csv', usecols = colunas_usadas)
base.head()

# Realizando escalonamento dos valores (MinMaxScaler - Faz a normalização em uma escala entre 0 e 1)
from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
base [['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']]= scaler_x.fit_transform(base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']])

base.head()

# Fazendo o escalonamento do atributo 'price' preço com uma variável expecífica para a normalização
scaler_y = MinMaxScaler()
base[['price']] = scaler_y.fit_transform(base[['price']])

base.head()

# Variáceis
x = base.drop('price', axis = 1) # Na base X price não vai aparecer
y = base.price # Na base Y somente price vai aparecer

x.head()

type(x)

y.head()

type(y)

# Criando variável e percorrendo as colunas da base de dados
previsores_colunas = colunas_usadas[1:17]
previsores_colunas

# Importação de bibliotecas
import tensorflow as tf

colunas = [tf.feature_column.numeric_column(key = c) for c in previsores_colunas]

print(colunas[10])

# Fazendo a divisão da base de dados entre e Teste e treinamento
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.3)

# Verificando os Shapes
x_treinamento.shape

y_treinamento.shape

x_teste.shape

y_teste.shape

# Criando e definindo funções para o treinamento
funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = x_treinamento, y = y_treinamento, batch_size = 32, num_epochs = None, shuffle = True)

# Criando e definindo funções para o teste
funcao_teste = tf.estimator.inputs.pandas_input_fn(x = x_teste, y = y_teste, batch_size = 32, num_epochs = 10000, shuffle = False) 

# Criando o regressor
regressor = tf.estimator.LinearRegressor(feature_columns = colunas)

# Realizando o treinamento do regressor
regressor.train(input_fn = funcao_treinamento, steps = 10000)

# Métricas na base de dados treinamento
metricas_treinamento = regressor.evaluate(input_fn = funcao_treinamento,steps = 10000)

# Métricas na base de dados treinamento
metricas_teste = regressor.evaluate(input_fn = funcao_teste,steps = 10000)

# Fazendo um comparativo
metricas_treinamento

metricas_teste

# Para as previsões é preciso criar uma função
funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = x_teste, shuffle = False)
 
# Realizando as previsões
previsoes = regressor.predict(input_fn = funcao_previsao)

# Mostrando os resultados 
list(previsoes)

# Percorrendo a variável para joga dentro de uma lista cada um dos valores
valores_previsoes = []
for p in regressor.predict(input_fn = funcao_previsao):
    valores_previsoes.append(p ['predictions'])
    
valores_previsoes

# Calculando o erro
# Importando a biblioteca do numpy
import numpy as np
valores_previsoes = np.asarray(valores_previsoes).reshape(-1, 1 )
   
valores_previsoes

# Realizando a desnormalização
valores_previsoes = scaler_y.inverse_transform(valores_previsoes) 

valores_previsoes

# Criando variável y_teste2 para depois desnormalizar y_teste
y_teste2 = y_teste.values.reshape(-1, 1)
y_teste2.shape

# Realizando a desnormalização do y_teste
y_teste2 = scaler_y.inverse_transform(y_teste2)
y_teste2

 # Observando a eficiência da regressão com métricas (mean_absolute_error) e (mean_squared_error)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste2, valores_previsoes)

mae
