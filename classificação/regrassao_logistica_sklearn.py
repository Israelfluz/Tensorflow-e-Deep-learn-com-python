# Importação da biblioteca pandas para manipulação, leitura e visualização de dados
import pandas as pd

# Carregamento da base de dado para análise
base = pd.read_csv('census.csv')

base.head()

# Visualizando um registro dentro da base, como por exemplo o income
base['income'].unique()

# Visualizando o núemro de registros que temos nessa base
base.shape

# Variável X armazena os atributos previsores
x = base.iloc[:, 0:14].values

# Variável Y recebe a resposta que é a classe
y = base.iloc[:, 14].values

# Visualizando os dados que estão dentro da variável x
x

# Visualizando os dados que estão dentro da variável y
y

# Pré-processecamento para transformação dos atributos categóricos em números
# Recuruso da biblioteca de aprendizagem de máquina chamada “Scikit-Learn” o LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Criando variável que recebe o parametro que faz a transfomação
label_encoder = LabelEncoder()

# Passando os atributos que precisam ser transformados em números
x[:, 1] = label_encoder.fit_transform(x[:, 1])
x[:, 3] = label_encoder.fit_transform(x[:, 3])
x[:, 5] = label_encoder.fit_transform(x[:, 5])
x[:, 6] = label_encoder.fit_transform(x[:, 6])
x[:, 7] = label_encoder.fit_transform(x[:, 7])
x[:, 8] = label_encoder.fit_transform(x[:, 8])
x[:, 9] = label_encoder.fit_transform(x[:, 9])
x[:, 13] = label_encoder.fit_transform(x[:, 13])

# Verificando se a transformação ficou correta da variável X
x[0]

# Escalonamento padronizado dos valores da variável x 
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

# Verificando se o escalonameto ficou correto da variável X
x[0]

# Fazendo a divisão da base de treinamento e base de teste
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size = 0.3)

# Visualizando as dimensões dos atributos previsores
x_treinamento.shape

# Visualizando as dimensões da classe
y_treinamento.shape

# Visualizando as dimensões dos atributos de teste
x_teste.shape
y_teste.shape

# Criando o modelo de classificação
from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression(max_iter = 10000)
classificador.fit(x_treinamento, y_treinamento)

# Realizando as previsões para obter respostas
previsoes = classificador.predict(x_teste)
previsoes

# Fazendo o comparativo entre as previsões e a variável y_teste que são as respostas que já tenho
y_teste

from sklearn.metrics import accuracy_score
taxa_acerto = accuracy_score(y_teste, previsoes)
