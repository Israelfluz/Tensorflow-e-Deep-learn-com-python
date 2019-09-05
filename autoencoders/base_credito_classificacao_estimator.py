import pandas as pd
base = pd.read_csv('credit-data.csv')
base.haed()

base.shape

# Apagandoa a coluna i#clientid pois seu conteúdo não ajudará nos algoritmos
base = base.drop('i#clientid', axis = 1)
base.head()


base.dropna() # Apagando os registros que não tiveram seus valores preenchidos (NAN)
base.shape

# Realizando o escalonamento e padroizando(standardscaler)
from sklearn.processing import StandardScaler
scaler_x = StandardScaler()
base[['income', 'age', 'loan']] = scaler_x.fit_transform(base[['income', 'age', 'loan']])
base.haed()

# Criando a variável com os atributos previsores e retirando a classe que mostra quem vai pagar e quem não
x = base.drop('c#default', axis = 1)
y = base['c#default']

x.haed()              