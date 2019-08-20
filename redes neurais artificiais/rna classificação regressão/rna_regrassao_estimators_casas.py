# ======== Implementando Regressão utilizando redes neurais com estimators ========= 



# Importa��o da biblioteca pandas para manipula��o, leitura e visualiza��o de dados
import pandas as pd


# Carregamento da base de dado para an�lise
base = pd.read_csv('house_prices.csv')

# Visualizando alguns registros
base.head()

# Visualizando o nome das colunas
base.columns

# Criando vari�vel com as colunas necess�rias
colunas_usadas = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']
colunas_usadas


# trazendo somante as colunas que v�o ser utlilizadas
base = pd.read_csv('house_prices.csv', usecols = colunas_usadas)
base.head()


# Aplicando a normaliza��o nos atributos previsores 
from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']] = scaler_x.fit_transform(base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']])
       

base.head()

# Aplicando o MinMaxScaler dentro do escalonamento
scaler_y = MinMaxScaler()
base[['price']] = scaler_y.fit_transform(base[['price']])
base.head()

# Vari�vel que contem os atributos previsores com excess�o do valor das casas
x = base.drop('price', axis = 1)

# Vari�vel y que contem o valor das casas
y = base.price


x.head()

y.head()

# Definindo vari�vel com coluna previsores
previsores_colunas = colunas_usadas[1:17]
previsores_colunas

# Importando a biblioteca do tensorflow
import tensorflow as tf

colunas = [tf.feature_column.numeric_column(key = c) for c in previsores_colunas]

# Divis�o da basse de dados para teste e treinamento
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.3)


x_treinamento.shape


x_teste.shape

# Para usar os estimators � preciso criar uma fun��o para realizar o treinamento
funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = x_treinamento, y = y_treinamento, batch_size = 32,
                                                        num_epochs = None, shuffle = True)

# Algoritmo que faz a regress�o usando redes neurais com estimators
regressor = tf.estimator.DNNRegressor(hidden_units = [8, 8, 8], feature_columns=colunas)
regressor.train(input_fn = funcao_treinamento, steps = 20000) # O treinamento


funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = x_teste, shuffle = False)
previsoes = regressor.predict(input_fn=funcao_previsao) # Realizando as previs�es depois do escalonamento
list(previsoes) # Mostrando os resultados 


valores_previsao = []
for p in regressor.predict(input_fn=funcao_previsao): # Percorrento o Generator (de previsoes) para saber o resultado em dolares
    valores_previsao.append(p['predictions'][0]) # Desse forma tenho o valor em moeda
    

valores_previsao # Visualizando 

# Realizando as previs�es e compara��es com os resultados que j� temos
import numpy as np
valores_previsao = np.asarray(valores_previsao).reshape(-1,1) # Realizando a transforma��o dos valores que est�o no formato lista para matrix
valores_previsao = scaler_y.inverse_transform(valores_previsao) # transforma��o inversa
valores_previsao

# A desnormaliza��o � feita tamb�m no valor real para fazemos a compara��o
y_teste2 = y_teste.values.reshape(-1,1)
y_teste2 = scaler_y.inverse_transform(y_teste2)
y_teste2


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste2, valores_previsao)
mae
