# Importação da biblioteca tensorfolw
import tensorflow as tf

# Criando variáveis com valores constantes
valor1 = tf.constant(2)
valor2 = tf.constant(3)

# Visuializando o tipo da variável constante valor1
type(valor1)

# Imprimindo a classe valor1 
print(valor1)

soma = valor1 + valor2

type(soma)

print(soma)

# Cirando uma sessão para visualizar o valor de soma
with tf.Session() as sess:
    s = sess.run(soma)

print(s)

texto1 = tf.constant('Texto 1 ')
texto2 = tf.constant('Texto 2 ')

type(texto1)

print(texto1)

# Concatenando o Text 1 com o Texto 2 dentro de uma sessão para ser visualizado
with tf.Session() as sess:
    con = sess.run(texto1 + texto2)
    
print(con)
 