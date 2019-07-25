# Importação da biblioteca
import tensorflow as tf

tf.reset_default_graph()

#a = tf.add(2, 2, name = 'add')
#b = tf.multiply(a, 3, name = 'mult1')
#c = tf.multiply(b, a, name = 'mult2')

# Escopo
with tf.name_scope('operacoes'):
    with tf.name_scope('Escopo_A'):
        a = tf.add(2, 2, name = 'add')
    with tf.name_scope('Escopo_B'):
        b = tf.multiply(a, 3, name = 'mult1')
        c = tf.multiply(b, a, name = 'mult2')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('output', sess.graph)
    print(sess.run(c))
    writer.close()

# grafo padrão do tensorflow    
tf.get_default_graph()


# Criando novos grafos
grafo1 = tf.get_default_graph()

grafo1

grafo2 = tf.Graph()
grafo2

with grafo2.as_default():
    print(grafo2 is tf.get_default_graph())
    