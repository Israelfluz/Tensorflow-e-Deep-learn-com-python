# Importando a biblioteca
import tensorflow as tf

# Criando um constante
vetor = tf.constant([5, 10, 15], name = 'vetor')
type(vetor)

print(vetor)

# Criando uma variável
soma = tf.Variable(vetor + 5, name = 'soma')

# Inicializando a variável
init = tf.global_variables_initializer()

# Criando a Sessão
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(soma))
    

# Outro exemplo
    
valor = tf.Variable(0, name = 'valor')

# Inicializando a variável
init2 = tf.global_variables_initializer()

# Criando a Sessão com um FOR para incrementar o valor da variável que é 0
with tf.Session() as sess:
    sess.run(init2)
    for i in range(5):
        valor = valor + 1
        print(sess.run(valor))
        