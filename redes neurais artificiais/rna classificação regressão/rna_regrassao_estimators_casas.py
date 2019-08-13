# ======== Implementando Regressão utilizando redes neurais com estimators ========= 



import pandas as pd


base = pd.read_csv('house_prices.csv')


base.head()

base.columns

colunas_usadas = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']
colunas_usadas


base = pd.read_csv('house_prices.csv', usecols = colunas_usadas)
base.head()

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

scaler_y = MinMaxScaler()
base[['price']] = scaler_y.fit_transform(base[['price']])
base.head()

X = base.drop('price', axis = 1)
y = base.price

X.head()

y.head()


previsores_colunas = colunas_usadas[1:17]
previsores_colunas

import tensorflow as tf

colunas = [tf.feature_column.numeric_column(key = c) for c in previsores_colunas]

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.3)


X_treinamento.shape


X_teste.shape

funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = X_treinamento, y = y_treinamento, batch_size = 32,
                                                        num_epochs = None, shuffle = True)
regressor = tf.estimator.DNNRegressor(hidden_units = [8, 8, 8], feature_columns=colunas)
regressor.train(input_fn = funcao_treinamento, steps = 20000)


funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = X_teste, shuffle = False)
previsoes = regressor.predict(input_fn=funcao_previsao)
list(previsoes)


valores_previsao = []
for p in regressor.predict(input_fn=funcao_previsao):
    valores_previsao.append(p['predictions'][0])
    

valores_previsao


import numpy as np
valores_previsao = np.asarray(valores_previsao).reshape(-1,1)
valores_previsao = scaler_y.inverse_transform(valores_previsao)
valores_previsao


y_teste2 = y_teste.values.reshape(-1,1)
y_teste2 = scaler_y.inverse_transform(y_teste2)
y_teste2


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste2, valores_previsao)
mae