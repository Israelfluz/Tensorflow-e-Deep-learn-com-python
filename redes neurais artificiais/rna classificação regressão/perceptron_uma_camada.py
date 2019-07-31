# Importando a biblioteca tensorflow
import tensorflow as tf

# Importação da biblioteca para computação científica
import numpy as np

# Variável X de entrada que armazena os atributos previsores do tipo numpy array no formato de matriz
x = np.array([[0.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 1.0]])

x 

# Variável Y receberá as respostas que é do tipo numpy array no formato de matriz
y = np.array([[0.0], [1.0], [1.0], [1.0]])

y

# Definindo os pesos do perceptron.
# Alterando o tf.zeros teremos outro resultado, tipo tf.ones os pesos seram 1
w = tf.Variable(tf.zeros([2, 1], dtype = tf.float64))
type(w)

print(w)

# Aplicando a ativação na camada de entrada
def step(x):
    return tf.cast(tf.to_float(tf.math.greater_equal(x, 1)), tf.float64)

# Inicializando as variáveis
init = tf.global_variables_initializer()

# Variável que realizará a soma dos valores de entra e multiplicação dos pesos
camada_saida = tf.matmul(x, w) 

# Aplicando a ativação na camada de saída
camada_saida_ativacao = step(camada_saida)

## fórmula para o calculo do erro
erro = tf.subtract(y, camada_saida_ativacao)

## Fórmula que atualiza o erro
delta = tf.matmul(x, erro, transpose_a = True)

## Fórmula para a taiza de aprendizagem
treinamento = tf.assign(w, tf.add(w, tf.multiply(delta, 0.1)))

# Criando uma sessão para visualizar o resultado
with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(w)) # esse print foi utilizada para inicializar as variáveis
    #print(sess.run(camada_saida))
    
    #print(sess.run(step(1)))
   # print('\n')
    #print(sess.run(camada_saida_ativacao))
    print('\n')
    #print(sess.run(erro))
    #print(x)
    #print('\n')
    #print(sess.run(tf.transpose(x)))
    #print(sess.run(treinamento))
    epoca = 0
    for i in range(15):
        epoca += 1
        erro_total, _ = sess.run([erro, treinamento])
        erro_soma = tf.reduce_sum(erro_total)
        #print(erro_total)
        print('Época', epoca, 'Erro', sess.run(erro_soma))
        if erro_soma.eval() == 0.0:
            break
        
    w_final = sess.run(w)
    
    w_final
    
# Teste para ver como a rna vai classificar
camada_saida_teste = tf.matmul(x, w_final)
camada_saida_ativacao_teste = step(camada_saida_teste)
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(camada_saida_ativacao_teste))
