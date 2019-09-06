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
    x2d_encode = sess.run(camada_oculta, feed_dict = {xph: x})
    x3d_decode = sess.run(camada_saida, feed_dict = {xph: x})
    
    
x2d_encode.shape

x3d_decode.shape



x2 = scaler_x.inverse_transform(x)
x2


x3d_decode2 = scaler_x.inverse_transform(x3d_decode)
x3d_decode2



from sklearn.metrics import mean_absolute_error
mae_income = mean_absolute_error(x2[:,0], x3d_decode2[:,0])
mae_income


mae_age = mean_absolute_error(x2[:,1], x3d_decode2[:,1])
mae_age



mae_loan = mean_absolute_error(x2[:,2], x3d_decode2[:,2])
mae_loan


x_encode = pd.DataFrame({'atributo1': x2d_encode[:,0], 'atributo2': x2d_encode[:,1], 'classe': y})



x_encode.head()



import tensorflow as tf
colunas = [tf.feature_column.numeric_column(key = column) for column in x_encode.columns]
from sklearn.model_selection import train_test_split
x_treinamento, X_teste, y_treinamento, y_teste = train_test_split(x_encode, y, test_size = 0.3, random_state = 0)
funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = x_treinamento,
                                                        y = y_treinamento,
                                                        batch_size = 8, 
                                                        num_epochs = None,
                                                        shuffle = True)
classificador = tf.estimator.DNNClassifier(feature_columns = colunas, hidden_units = [4, 4])
classificador.train(input_fn = funcao_treinamento, steps = 1000)
funcao_teste = tf.estimator.inputs.pandas_input_fn(x = X_teste, y = y_teste,
                                              batch_size = 8, num_epochs = 1000,
                                              shuffle = False)
metricas_teste = classificador.evaluate(input_fn = funcao_teste, steps = 1000)



metricas_teste


