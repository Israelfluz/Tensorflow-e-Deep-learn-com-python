# Importação da biblioteca
import tensorflow as tf

# Criando as constantes
a = tf.constant([9,8,7], name = 'a')
b = tf.constant([1,2,3], name = 'b')

# Implementando a soma
soma = a + b

# Buscando o tipo
type(a)

# imprimindo o valor da constante
print(a)

# Criando sessão
with tf.Session() as sess:
    print(sess.run(soma))

# ==== Criando matrizes ====

a1 = tf.constant([[1,2,3],[4,5,6]], name = 'a1')

type(a1)

print(a1)


# Visualizando apenas as dimensões 
a1.shape

b1 = tf.constant([[1,2,3],[4,5,6]], name = 'b1')

soma1 = tf.add(a1, b1)

# Criando uma sessão para realizar o calculo
with tf.Session() as sess:
    print(sess.run(a1))
    print('\n')
    print(sess.run(b1))
    print('\n')
    print(sess.run(soma1))
    
  ========================================================  

a2 = tf.constant([[1,2,3],[4,5,6]])
b2 = tf.constant([[1],[2]])
soma2 = tf.add(a2,b2)


with tf.Session() as sess:
    print(sess.run(a2))
    print('\n')
    print(sess.run(b2))
    print('\n')
    print(sess.run(soma2))

