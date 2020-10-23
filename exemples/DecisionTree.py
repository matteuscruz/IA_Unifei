import numpy as np
import pandas as pd
import csv
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt



arquivo_treinamento = pd.read_csv("Treinamento.csv",sep=" ",names=['1','2','3','4','5','6','7','8','9'], header=0)
arquivo_teste = pd.read_csv("Teste.csv",sep=" ",names=['1','2','3','4','5','6','7','8','9'], header=0)
arquivo_validacao = pd.read_csv("Validacao.csv",sep=" ",names=['1','2','3','4','5','6','7','8','9'], header=0)



#Conjunto de Treinamento:
X_treinamento = arquivo_treinamento.drop('7',axis = 1)
X_treinamento = X_treinamento.drop('8',axis = 1)
X_treinamento = X_treinamento.drop('9',axis = 1)

Y_treinamento = pd.DataFrame(columns = ['1','2'])
Y_treinamento['1'] = arquivo_treinamento['7']
Y_treinamento['2'] = arquivo_treinamento['8']
Y_treinamento= Y_treinamento.dropna()

#Conjunto de Teste:
X_teste = arquivo_teste.drop('7',axis = 1)
X_teste = X_teste.drop('8',axis = 1)
X_teste = X_teste.drop('9',axis = 1)

Y_teste = pd.DataFrame(columns = ['1','2'])
Y_teste['1'] = arquivo_teste['7']
Y_teste['2'] = arquivo_teste['8']
Y_teste= Y_teste.dropna()

#Conjunto de Validação:
X_validacao = arquivo_validacao.drop('7',axis = 1)
X_validacao = X_validacao.drop('8',axis = 1)
X_validacao = X_validacao.drop('9',axis = 1)

Y_validacao = pd.DataFrame(columns = ['1','2'])
Y_validacao['1'] = arquivo_validacao['7']
Y_validacao['2'] = arquivo_validacao['8']
Y_validacao= Y_validacao.dropna()

modelo = DecisionTreeClassifier()

tree = modelo.fit(X_treinamento, Y_treinamento)

Prediction = tree.predict(X_teste)

Prediction = modelo.predict(X_teste)
Acc = tree.score(X_validacao, Y_validacao)
print("\nAcurácia:",Acc)

np.set_printoptions(formatter = {'float': lambda x: "{0:0.2f}".format(x)})

for i in range(len(Prediction)):
    if Prediction[i][0] < 0.5:
        Prediction[i][0] = 0
        Prediction[i][1] = 1
    else:
        Prediction[i][0] = 1
        Prediction[i][1] = 0
        
Dif = np.array(Prediction - Y_teste)
ContaErro = 0

for i in range(len(Prediction)):
    if Dif[i][0] != 0:
        ContaErro += 1
print(ContaErro,"previsões erradas")












