# ======= Classificação binária com XOR que retonar  valores entre 0 e 1 ========
# ======= Essa implemanteção será feita utilizando a low-level do Tensorflow ======
# ======= Esse é um problema não linearmente separável, ou seja, não da para 
# trassar uma linha separando a classe 0 e a classe 1 ==========


# Importação biblioteca tensorflow
import tensorflow as tf

# Importação da biblioteca para computação científica
import numpy as np

# Variável X de entrada que armazena os atributos previsores do tipo numpy array no formato de matriz
x = np.array([[0,0], [0,1], [1,0], [1,1]])

x 

# Variável Y receberá as respostas que é do tipo numpy array no formato de matriz
y = np.array([[1], [0], [0], [1]])

y

# Definindo algumas variáveis para indicar a estrutura da rede neural
neuronios_entrada = 2
neuronios_oculta = 3
neuronios_saida = 1

# Crianda a variável W que representa os pesos no formato de dicionário
w = {'oculta': tf.Variable(tf.random_normal([neuronios_entrada, neuronios_oculta]), name = 'w_oculta'),
     'saida': tf.Variable(tf.random_normal([neuronios_oculta, neuronios_saida]), name = 'w_saida')}

# Verificando o tipo da variável w
type(w)


# Fazendo uma inicialização dos pesos de uma Rede neural
distribuicao = np.random.normal(size = 500)
distribuicao

# Criando um gráfico de linha com a biblioteca seaborn
import seaborn as sns
sns.distplot(distribuicao)


# Adcionando a camada BIAS com os seus pesos
b = {'oculta': tf.Variable(tf.random_normal([neuronios_oculta]), name = 'b_oculta'),
     'saida': tf.Variable(tf.random_normal([neuronios_saida]), name = 'b_saida')}


# Criando os placesholders para receber os dados
xph = tf.placeholder(tf.float32, [4, neuronios_entrada], name = 'xph')

# Criando o placeholder para receber as respostas
yph = tf.placeholder(tf.float32, [4, neuronios_saida], name = 'yph')


# Fórmula
camada_oculta = tf.add(tf.matmul(xph, w['oculta']), b['oculta'])
camada_oculta_ativacao = tf.sigmoid(camada_oculta)
camada_saida = tf.add(tf.matmul(camada_oculta_ativacao, w['saida']), b['saida'])
camada_saida_ativacao = tf.sigmoid(camada_saida)

# Fórmula para ajuste automático dos pesos
erro = tf.losses.mean_squared_error(yph, camada_saida_ativacao)
otmizador = tf.train.GradientDescentOptimizer(learning_rate = 0.3).minimize(erro) 

# Inicializando as variáveis
init = tf.global_variables_initializer()

# Visualizando os pesos da variável W dentro de uma sessão
with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(w['oculta']))
    #print(sess.run(w['saida']))
    #print(sess.run(b['oculta']))
    #print('\n')
    #print(sess.run(b['saida']))
    #print(sess.run(camada_oculta, feed_dict = {xph: x}))
    #print(sess.run(camada_oculta_ativacao, feed_dict = {xph: x}))
    #print(sess.run(camada_saida, feed_dict = {xph: x}))
    #print(sess.run(camada_saida_ativacao, feed_dict = {xph: x}))
    
# Esse for dentro da sessão é para realizar a otmização do erro e fazer a aprendizagem
    for epocas in range(20000):
        erro_medio = 0
        _, custo = sess.run([otmizador, erro], feed_dict = { xph: x, yph: y}) # Aqui está o aprendizado que codificação vai realizar
        if epocas % 200 == 0:
            #print(custo)
            erro_medio += custo / 4 
            print(erro_medio)
        w_final, b_final = sess.run([w, b])

w_final

b_final

# Realizando um teste para visualizar os resultados
camada_oculta_teste = tf.add(tf.matmul(xph, w_final['oculta']), b_final['oculta'])
camada_oculta_ativacao_teste = tf.sigmoid(camada_oculta_teste)
camada_saida_teste = tf.add(tf.matmul(camada_oculta_ativacao_teste, w_final['saida']), b_final['saida'])
camada_saida_ativacao_teste = tf.sigmoid(camada_saida_teste)

# Criando sessão para realizar as previsões
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(camada_saida_ativacao_teste, feed_dict = {xph: x})) # Aqui está a previsão
    

        