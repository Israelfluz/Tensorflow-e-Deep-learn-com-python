import pandas as pd
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

y.head()

x.columns

# Importando o framework 
import tensorflow as tf
colunas = [tf.feature_column.numeric_column(key = column) for column in x.columns]

# Visualizando a variável colunas criada
print(colunas[2]) # Basta colocar o número da coluna


# Divisão da base de dados de treinamento e teste
from sklearn.model_selection import train_test_split 
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.3, random_state = 0)


x_treinamento.shape


x_teste.shape

# Criando a função para o treinamento com estimators
funcao_teste = tf.estimator.inputs.pandas_input_fn(x = x_treinamento,
                                                         y = y_treinamento,
                                                         batch_size = 8,
                                                         num_epochs = None,
                                                         shuffle = True) 

# Criando a rede neural
classificador = tf.estimator.DNNClassifier(feature_columns = colunas, hidden_units = [4, 4])
classificador.train(input_fn = funcao_treinamento, steps = 1000)


# Criando a variável para fazer avaliação na base de dados de teste
metricas_teste = classificador.evaluate(input_fn = funcao_teste, steps = 1000)

metricas_teste  
           