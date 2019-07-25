# Importação da biblioteca numpy
import numpy as np

# Definindo variável X com a idade das pessoos com np array
x = np.array([[18],[23],[28],[33],[38], [43],[48],[53],[58],[63]])

# Demonstranso os valores da variável X
x

# Definindo a variável Y que recebe o valor do plano de saúde com np array
y = np.array([[871],[1132],[1042],[1356],[1488],[1638],[1569],[1754],[1866],[1900]])

# Demonstranso os valores da variável Y
y

# Gráfico que visualiza a disposição dos valores
import matplotlib.pyplot as plt
plt.scatter(x, y)

# Criando um modelo de regressão linear
from sklearn.linear_model import LinearRegression

# Criando o regressor
regressor = LinearRegression()

# Realizando o treinamento

# Visualizando os coeficiente 
# Coeficiênte b0
regressor.intercept_

# Coeficiênte b1
regressor.coef_

# Realizando uma previsão do valor do plano de saúde para uma pessoa co 40 anos
previsao1 = regressor.intercept_ + regressor.coef_ * 40

# Visualizando o valor do plano com a previsão
previsao1

# Outro recursos que o Sklearn nos permite p/ fazer previsão
previsao2 = regressor.predict([[40]])
previsao2

# E por fim este outro recurso
newPerson = np.reshape(40,(1,-1))
previsao2 = regressor.predict(newPerson)
previsao2

# Realizando uma previsão para todos os registros na base de dados X que são as idades
previsoes = regressor.predict(x)
previsoes

# Fazendo um comparativo entre o valor real (Y) e o valor das previsões
resultado = abs(y - previsoes)
resultado

# Observando a média, ou seja, pode ter errado para mais ou para menos %
resultado = abs(y - previsoes).mean()
resultado

# Observando a eficiência da regressão com métricas (mean_absolute_error) e (mean_squared_error)
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y, previsoes)
mse = mean_squared_error(y, previsoes)

# Visualizando o valor de mean_absolute_error
mae

# Visualizando o valor de mean_squared_error
mse

# Gerando um gráfico para vermos a reta
plt.plot(x, y, 'o')

# Visialiando os valores previstos no gráfico
plt.plot(x, previsoes, color = 'green')
plt.title('Regressão linear simples')
plt.xlabel('Idade')
plt.ylabel('Custo')
