# Importação da biblioteca para computação científica
import numpy as np

# transfer function

# Criando função
def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

# Criando função sigmoid (que é muito utilizada para retornar probabilidades)
def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))

# Criando a função tangente hiperbólica
def tahnFunction(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

def reluFunction(soma):
    if soma >= 0:
        return soma
    return 0

def linearFunction(soma):
    return soma

def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

# Realizando o teste da stepFunction
teste = stepFunction(-1)

# Realizando o teste da função sigmoid
teste = sigmoidFunction(-0.358)

# Realizando o teste da função tangente hiperbólica
teste = tahnFunction(-0.358)


teste = reluFunction(0.358)
teste = linearFunction(-0.358)
valores = [7.0, 2.0, 1.3]
print(softmaxFunction(valores))

 