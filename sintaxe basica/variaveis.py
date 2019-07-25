# criando variáveis
x = 35
y = x + 35

# Imprimindo 
print(y) 

# Executando o conceito de variáveis(código de cima) no tensorflow
# Importando a biblioteca
import tensorflow as tf

# Definindo a primeira Constante
valor1 = tf.constant(15, name = 'valor1')

# Imprimindo o valor da constante
print(valor1)

# Definindo a variável
soma = tf.Variable(valor1 + 5, name = 'valor1')

print(soma)
type(soma)

# Inicializando as variáveis para que possoam funcionar na sessão
# caso contrário teremos um error
init = tf.global_variables_initializer()

# Criando uma sessão para visualizar o valor da soma no tensorflow
# Quando temos variáveis temos que inicializá-la na sessão
with tf.Session() as sess:
    sess.run(init)
    s = sess.run(soma) # posso imprimir aqui dentro da sessão com print(sess.run(soma))
    
    
s