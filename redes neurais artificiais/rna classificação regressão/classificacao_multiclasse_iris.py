# ======= Classificação Multiclasse com XOR utlizando a função softmax caso tenhamos
# problemas com mais valores de classes, usando a base de dados Iris - classificação de plantas  ========
# ======= Essa implemanteção será feita utilizando a low-level API do Tensorflow ======

# Importação da biblioteca sklearn e do pacote de dados
from sklearn import datasets
iris = datasets.load_iris()
iris

# Variável X de entrada que armazena os atributos previsores.
x = iris.data

x

# Variável Y receberá as respostas.
y = iris.target

y

# Aplicar um pré-processamento para padronizar os dados
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

x

# Realizando uma transformação na variável Y com o OneHotEncoder para que cada  
# resposta tenha três atributos igual quantidade de neurônios na camada de saída da rede neural.
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(categorical_features = [0])
y = y.reshape(-1, 1) # Deixando no formato matriz

y.shape

# Aplicando o processo de transformação
y = onehot.fit_transform(y).toarray() # toarray transforma em um tipo de numpy array

y

# Fazendo o processo de divisão da base de dados
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.3)

x_treinamento.shape

x_teste.shape

# Importação de bibliotecas 
import tensorflow as tf
import numpy as np

# Definindo as variáveis para a rede neural
neuronios_entrada = x.shape[1] # O número de neurômios equivale ao número de 
                               # atributos que na entrada são 4, podendo passar x.shape[1]
neuronios_entrada

neuronios_oculta = int(np.ceil((x.shape[1] + y.shape[1]) / 2))

neuronios_oculta

neuronios_saida = y.shape[1]

neuronios_saida

# Criando a variável W que representa os pesos no formato de dicionário
w = {'oculta': tf.Variable(tf.random_normal([neuronios_entrada, neuronios_oculta])),
     'saida': tf.Variable(tf.random_normal([neuronios_oculta, neuronios_saida]))}

# Adcionando a camada BIAS com os seus pesos
b = {'oculta': tf.Variable(tf.random_normal([neuronios_oculta])),
     'saida': tf.Variable(tf.random_normal([neuronios_saida]))}

# Criando os placesholders (previsores) para receber os dados
xph = tf.placeholder('float', [None, neuronios_entrada])

# Criando o placeholder para receber as respostas
yph = tf.placeholder('float', [None, neuronios_saida])

# Criando o modelo por meio de uma função
def mlp(x, w, bias):
    camada_oculta = tf.add(tf.matmul(x, w['oculta']), bias['oculta']) # Calculo da camanda o culta
    camada_oculta_ativacao = tf.nn.relu(camada_oculta) # processo de ativação com Relu que quando 
                                                       # recebe maior que 0 retorna o número, caso 
                                                       # contrário retorna 0.
    camada_saida = tf.add(tf.matmul(camada_oculta_ativacao, w['saida']), b['saida'])
    return camada_saida

# Criando o modelo
modelo = mlp(xph, w, b)

# Criando fórmula para verificar o erro
erro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = modelo, labels = yph))
otimizador = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(erro)

batch_size = 8
batch_total = int(len(x_treinamento) / batch_size)
batch_total

x_batches = np.array_split(x_treinamento, batch_total)
x_batches

# Construção de uma sessão
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoca in range(3000):
            erro_medio = 0.0
            batch_total = int(len(x_treinamento) / batch_size)
            x_batches = np.array_split(x_treinamento, batch_total)
            y_batches = np.array_split(y_treinamento, batch_total)
            for i in range(batch_total):
                x_batches, y_batches = x_batches[i], y_batches[i]
                _, custo = sess.run([otimizador, erro], feed_dict = {xph: x_batches, yph: y_batches})
                erro_medio += custo / batch_total
            if epoca % 500 == 0:
                    print('Época: ' + str((epoca + 1)) + 'erro: ' + str(erro_medio))
        w_final, b_final = sess.run([w, b])
        
# Previsões 
previsoes_teste = mlp(xph, w_final, b_final)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    r1 = sess.run(previsoes_teste, feed_dict = {x_teste})
    r2 = sess.run(tf.nn.softmax(r1))
    r3 = sess.run(tf.argmax(r2, 1))

y_teste2 = np.argmax(y_teste, 1)
y_teste


from sklearn.metrics import accuracy_score
taxa_acerto = acurracy_score(y_teste2, r3)
taxa_acerto
    